from dataclasses import replace

import numpy as np
from scipy.spatial.distance import cdist

from virda.localization.contracts import ElectrodeLocalizer
from virda.models.electrode import Electrode, Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducial, Fiducials
from virda.models.stage3_config import Stage3Config

_OFFSET_SEARCH_MIN_MM = -30.0
_OFFSET_SEARCH_MAX_MM = 30.0
_OFFSET_SEARCH_STEP_MM = 1.0
_OFFSET_REFINE_STEP_MM = 0.25
_OFFSET_REFINE_SPAN_MM = 1.0
_OFFSET_SHIFT_WARN_MM = 2.0

_REFINE_MAX_ITERATIONS = 16
_REFINE_TOLERANCE_MM = 1e-9
_REFINE_MAX_RESIDUAL_MM = 2.0


def _mirror_plane_mask(
    vertices: np.ndarray, fiducial_coords: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    """Return a boolean mask of vertices on the wrong side of the fiducial plane.

    With exactly 3 fiducials, every point has a mirror image across the
    fiducial plane with identical distances to all fiducials.  This mask
    marks the half-space where no real electrodes exist so that the
    brute-force search cannot pick a false mirror candidate.

    The correct side is determined by the vertex normals: the side whose
    normals are more aligned with the fiducial plane normal is the
    ``electrode side'' (normals point outward from the scalp surface,
    so on the electrode side they point away from the plane).
    """
    n_masked = vertices.shape[0]
    if fiducial_coords.shape[0] < 3:
        return np.zeros(n_masked, dtype=bool)

    plane_normal = np.cross(
        fiducial_coords[1] - fiducial_coords[0],
        fiducial_coords[2] - fiducial_coords[0],
    )
    norm_len = np.linalg.norm(plane_normal)
    if norm_len == 0:
        return np.zeros(n_masked, dtype=bool)
    plane_normal /= norm_len

    vertex_side = np.dot(vertices - fiducial_coords[0], plane_normal)
    normal_alignment = np.dot(normals, plane_normal)

    pos_mask = vertex_side >= 0
    neg_mask = ~pos_mask

    if not pos_mask.any() or not neg_mask.any():
        return np.zeros(n_masked, dtype=bool)

    pos_score = float(normal_alignment[pos_mask].sum())
    neg_score = float(normal_alignment[neg_mask].sum())

    correct_side_sign = 1.0 if pos_score > neg_score else -1.0

    wrong_side: np.ndarray = np.sign(vertex_side) != correct_side_sign
    return wrong_side


class BruteForceLocalizer(ElectrodeLocalizer):
    """Localize real electrodes by brute-force search over the ESE point cloud.

    For every electrode the predicted distance from each ESE vertex to the
    fiducials is compared against the measured distances; the vertex minimizing
    the weighted sum of squared differences is chosen (spec Stage 3).

    When ``calibrate_ese_offset`` is enabled, a global shift of the ESE cloud
    along the scalp normals is estimated before localization so that systematic
    reference mismatches (e.g. measurements taken on the scalp while the ESE
    models electrode body centers) do not degrade every electrode.

    After the discrete argmin, the electrode position is refined by Gauss-Newton
    least squares on the sphere equations and snapped back to the nearest
    allowed vertex.  The discrete search alone suffers from tangential slide:
    on a coarsely tessellated surface a distant vertex can fit the measured
    distances better than any vertex near the true intersection point.  The
    continuous refinement removes that bias while keeping the argmin result as
    a fallback whenever refinement does not improve the residual.
    """

    def __init__(self, config: Stage3Config) -> None:
        self._config = config
        super().__init__()

    def _process(
        self,
        ese: ESEMesh,
        fiducials: Fiducials,
        electrodes: Electrodes,
    ) -> Electrodes:
        vertices = np.asarray(ese.vertices)
        normals = np.asarray(ese.normals)
        fiducial_coords = np.asarray([fiducial.coordinates for fiducial in fiducials.items])

        offset_shift = 0.0
        if self._config.calibrate_ese_offset:
            offset_shift = self._calibrate_offset(
                vertices, normals, fiducial_coords, fiducials, electrodes
            )

        search_vertices = vertices + offset_shift * normals
        distances_to_fiducials = cdist(search_vertices, fiducial_coords)
        mirror_mask = _mirror_plane_mask(search_vertices, fiducial_coords, normals)

        localized: list[Electrode] = []
        for electrode in electrodes.items:
            localized.append(
                self._localize_one(
                    electrode,
                    fiducials,
                    ese,
                    fiducial_coords,
                    search_vertices,
                    distances_to_fiducials,
                    mirror_mask,
                )
            )

        localized_count = sum(1 for electrode in localized if electrode.is_localized)
        residuals = [
            electrode.residual_error
            for electrode in localized
            if electrode.residual_error is not None
        ]
        if self._logger:
            if residuals:
                self._logger.info(
                    "Localized %d/%d electrodes, median residual=%.4f mm, flagged=%d",
                    localized_count,
                    len(localized),
                    float(np.median(residuals)),
                    sum(1 for electrode in localized if electrode.flagged),
                )
            else:
                self._logger.info("No electrodes localized (%d provided)", len(localized))
        return Electrodes(
            items=localized,
            calibrated_offset_shift_mm=(
                offset_shift if self._config.calibrate_ese_offset else None
            ),
        )

    def _localize_one(
        self,
        electrode: Electrode,
        fiducials: Fiducials,
        ese: ESEMesh,
        fiducial_coords: np.ndarray,
        search_vertices: np.ndarray,
        distances_to_fiducials: np.ndarray,
        mirror_mask: np.ndarray,
    ) -> Electrode:
        present = [
            fiducial
            for fiducial in fiducials.items
            if fiducial.fiducial_id in electrode.measured_distances
        ]
        if not present:
            if self._logger:
                self._logger.warning(
                    "Electrode %s has no measured distances to any known fiducial; skipped",
                    electrode.electrode_id,
                )
            return electrode

        measured = np.asarray(
            [electrode.measured_distances[fiducial.fiducial_id] for fiducial in present]
        )
        weights = np.asarray([fiducial.weight for fiducial in present])
        present_indices = np.asarray(
            [fiducials.ids.index(fiducial.fiducial_id) for fiducial in present]
        )

        squared_error = (
            weights[None, :] * (distances_to_fiducials[:, present_indices] - measured[None, :]) ** 2
        )
        total_error = squared_error.sum(axis=1)
        if len(present) >= 3:
            total_error[mirror_mask] = np.inf
        best_index = int(np.argmin(total_error))
        best_residual = float(np.sqrt(total_error[best_index]))
        if len(present) >= 3:
            best_index = self._refine_best_index(
                fiducial_coords[present_indices],
                measured,
                weights,
                search_vertices,
                mirror_mask,
                best_index,
                best_residual,
            )
        residual_error = float(np.sqrt(total_error[best_index]))

        return replace(
            electrode,
            ese_coords=np.asarray(ese.vertices)[best_index],
            scalp_coords=np.asarray(ese.scalp_vertices)[best_index],
            residual_error=residual_error,
            confidence=float(ese.quality[best_index]),
            flagged=residual_error > self._config.residual_threshold_mm,
        )

    def _calibrate_offset(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        fiducial_coords: np.ndarray,
        fiducials: Fiducials,
        electrodes: Electrodes,
    ) -> float:
        """Grid-search a global ESE offset shift minimizing median residual.

        A coarse 1 mm sweep over the full range is followed by a fine
        refinement (``_OFFSET_REFINE_STEP_MM``) around the coarse optimum.
        """

        def evaluate(shift: float) -> float:
            cloud = vertices + shift * normals
            distances_to_fiducials = cdist(cloud, fiducial_coords)
            mirror_mask = _mirror_plane_mask(cloud, fiducial_coords, normals)
            residuals = [
                self._best_residual(electrode, fiducials, distances_to_fiducials, mirror_mask)
                for electrode in localizable
            ]
            return float(np.median(residuals))

        localizable = [
            electrode
            for electrode in electrodes.items
            if any(
                fiducial.fiducial_id in electrode.measured_distances for fiducial in fiducials.items
            )
        ]
        if not localizable:
            if self._logger:
                self._logger.warning(
                    "Offset calibration skipped: no electrodes with known fiducials"
                )
            return 0.0

        best_shift = 0.0
        best_score = float(np.inf)
        for shift in np.arange(
            _OFFSET_SEARCH_MIN_MM,
            _OFFSET_SEARCH_MAX_MM + _OFFSET_SEARCH_STEP_MM / 2,
            _OFFSET_SEARCH_STEP_MM,
        ):
            score = evaluate(float(shift))
            if score < best_score:
                best_score = score
                best_shift = float(shift)

        for shift in np.arange(
            best_shift - _OFFSET_REFINE_SPAN_MM,
            best_shift + _OFFSET_REFINE_SPAN_MM + _OFFSET_REFINE_STEP_MM / 2,
            _OFFSET_REFINE_STEP_MM,
        ):
            score = evaluate(float(shift))
            if score < best_score:
                best_score = score
                best_shift = float(shift)

        if self._logger:
            self._logger.info(
                "Offset calibration: best shift=%+.1f mm (median residual %.2f mm)",
                best_shift,
                best_score,
            )
        if abs(best_shift) > _OFFSET_SHIFT_WARN_MM and self._logger:
            self._logger.warning(
                "Measured distances fit best with the ESE shifted by %+.1f mm "
                "from the configured offset; check ese_offset_mm/ese_reference "
                "or the measurement reference point.",
                best_shift,
            )
        return best_shift

    @staticmethod
    def _refine_best_index(
        fiducial_coords: np.ndarray,
        measured: np.ndarray,
        weights: np.ndarray,
        cloud: np.ndarray,
        mirror_mask: np.ndarray,
        best_index: int,
        best_residual: float,
    ) -> int:
        """Refine the argmin vertex via continuous trilateration and re-snap.

        The Gauss-Newton solution is the actual sphere intersection, so the
        nearest allowed vertex to it approximates the true electrode position
        even when the discrete distance-profile fit favored a distant vertex
        (tangential slide on a coarsely tessellated surface).

        Refinement applies only when all involved fiducials share the same
        weight: unequal weights declare some measurements less trustworthy,
        a signal that pure sphere intersection must not override the
        discrete weighted argmin (weighted-fiducial dominance).  On top of
        that, the continuous solution is trusted only when it explains the
        measurements almost perfectly (unweighted residual below
        ``_REFINE_MAX_RESIDUAL_MM``) and strictly better than the argmin
        vertex.  A near-zero residual proves the measured distances are
        mutually consistent and the spheres genuinely intersect near this
        vertex; otherwise the discrete argmin result is kept.
        """
        if not np.all(weights == weights[0]):
            return best_index
        refined = BruteForceLocalizer._gauss_newton(
            fiducial_coords, measured, weights, cloud[best_index]
        )
        if refined is None:
            return best_index
        predicted = np.linalg.norm(fiducial_coords - refined, axis=1)
        refined_residual = float(np.sqrt(((predicted - measured) ** 2).sum()))
        if refined_residual >= best_residual or refined_residual > _REFINE_MAX_RESIDUAL_MM:
            return best_index
        allowed = np.flatnonzero(~mirror_mask)
        squared_distance = ((cloud[allowed] - refined) ** 2).sum(axis=1)
        return int(allowed[int(np.argmin(squared_distance))])

    @staticmethod
    def _gauss_newton(
        fiducial_coords: np.ndarray,
        measured: np.ndarray,
        weights: np.ndarray,
        start: np.ndarray,
    ) -> np.ndarray | None:
        """Gauss-Newton least squares for weighted sphere intersection."""
        x = start.astype(float).copy()
        sqrt_weights = np.sqrt(weights)
        for _ in range(_REFINE_MAX_ITERATIONS):
            diff = x[None, :] - fiducial_coords
            norms = np.linalg.norm(diff, axis=1)
            if not np.all(norms > 1e-9):
                return None
            residuals = (norms - measured) * sqrt_weights
            jacobian = diff / norms[:, None] * sqrt_weights[:, None]
            delta = np.linalg.lstsq(jacobian, -residuals, rcond=None)[0]
            x = x + delta
            if float(np.linalg.norm(delta)) < _REFINE_TOLERANCE_MM:
                break
        if not np.all(np.isfinite(x)):
            return None
        return x

    @staticmethod
    def _best_residual(
        electrode: Electrode,
        fiducials: Fiducials,
        distances_to_fiducials: np.ndarray,
        mirror_mask: np.ndarray,
    ) -> float:
        present: list[Fiducial] = [
            fiducial
            for fiducial in fiducials.items
            if fiducial.fiducial_id in electrode.measured_distances
        ]
        if not present:
            return float(np.inf)
        measured = np.asarray(
            [electrode.measured_distances[fiducial.fiducial_id] for fiducial in present]
        )
        weights = np.asarray([fiducial.weight for fiducial in present])
        present_indices = np.asarray(
            [fiducials.ids.index(fiducial.fiducial_id) for fiducial in present]
        )
        total_error = (
            weights[None, :] * (distances_to_fiducials[:, present_indices] - measured[None, :]) ** 2
        ).sum(axis=1)
        if len(present) >= 3:
            total_error[mirror_mask] = np.inf
        return float(np.sqrt(total_error.min()))
