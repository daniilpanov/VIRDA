import logging
from dataclasses import replace

import numpy as np
from scipy.spatial.distance import cdist

from virda.localization.contracts import ElectrodeLocalizer
from virda.models.electrode import Electrode, Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducial, Fiducials
from virda.models.stage3_config import Stage3Config

logger = logging.getLogger(__name__)

_OFFSET_SEARCH_MIN_MM = -20.0
_OFFSET_SEARCH_MAX_MM = 20.0
_OFFSET_SEARCH_STEP_MM = 1.0
_OFFSET_SHIFT_WARN_MM = 2.0


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
    """

    def __init__(self, config: Stage3Config) -> None:
        self._config = config

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
                self._localize_one(electrode, fiducials, ese, distances_to_fiducials, mirror_mask)
            )

        localized_count = sum(1 for electrode in localized if electrode.is_localized)
        residuals = [
            electrode.residual_error
            for electrode in localized
            if electrode.residual_error is not None
        ]
        if residuals:
            logger.info(
                "Localized %d/%d electrodes, median residual=%.4f mm, flagged=%d",
                localized_count,
                len(localized),
                float(np.median(residuals)),
                sum(1 for electrode in localized if electrode.flagged),
            )
        else:
            logger.info("No electrodes localized (%d provided)", len(localized))
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
        distances_to_fiducials: np.ndarray,
        mirror_mask: np.ndarray,
    ) -> Electrode:
        present = [
            fiducial
            for fiducial in fiducials.items
            if fiducial.fiducial_id in electrode.measured_distances
        ]
        if not present:
            logger.warning(
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
        """Grid-search a global ESE offset shift minimizing median residual."""
        localizable = [
            electrode
            for electrode in electrodes.items
            if any(
                fiducial.fiducial_id in electrode.measured_distances for fiducial in fiducials.items
            )
        ]
        if not localizable:
            logger.warning("Offset calibration skipped: no electrodes with known fiducials")
            return 0.0

        shifts = np.arange(
            _OFFSET_SEARCH_MIN_MM,
            _OFFSET_SEARCH_MAX_MM + _OFFSET_SEARCH_STEP_MM / 2,
            _OFFSET_SEARCH_STEP_MM,
        )
        best_shift = 0.0
        best_score = float(np.inf)
        for shift in shifts:
            cloud = vertices + float(shift) * normals
            distances_to_fiducials = cdist(cloud, fiducial_coords)
            mirror_mask = _mirror_plane_mask(cloud, fiducial_coords, normals)
            residuals = [
                self._best_residual(electrode, fiducials, distances_to_fiducials, mirror_mask)
                for electrode in localizable
            ]
            score = float(np.median(residuals))
            if score < best_score:
                best_score = score
                best_shift = float(shift)

        logger.info(
            "Offset calibration: best shift=%+.1f mm (median residual %.2f mm)",
            best_shift,
            best_score,
        )
        if abs(best_shift) > _OFFSET_SHIFT_WARN_MM:
            logger.warning(
                "Measured distances fit best with the ESE shifted by %+.1f mm "
                "from the configured offset; check ese_offset_mm/ese_reference "
                "or the measurement reference point.",
                best_shift,
            )
        return best_shift

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
