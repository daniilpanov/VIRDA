from dataclasses import replace

import numpy as np
import pytest

from tests.helpers.measurements import make_electrodes, make_ese, make_fiducials
from tests.helpers.pipelines import build_context
from virda.localization.brute_force_localizer import BruteForceLocalizer
from virda.models.electrode import Electrode, Electrodes
from virda.models.fiducial import Fiducials
from virda.models.stage3_config import Stage3Config


def _closest_vertex_index(ese, point) -> int:
    return int(np.linalg.norm(np.asarray(ese.vertices) - point, axis=1).argmin())


def _exact_distances(point, fiducials: Fiducials) -> dict[str, float]:
    return {
        fiducial.fiducial_id: float(np.linalg.norm(point - fiducial.coordinates))
        for fiducial in fiducials.items
    }


def _reweighted(fiducials: Fiducials, weights: dict[str, float]) -> Fiducials:
    return Fiducials(
        items=[
            replace(fiducial, weight=weights.get(fiducial.fiducial_id, 0.01))
            for fiducial in fiducials.items
        ]
    )


def _localize(electrodes: Electrodes, threshold_mm: float = 10.0) -> Electrodes:
    ese = make_ese()
    fiducials = make_fiducials()
    localizer = BruteForceLocalizer(Stage3Config(residual_threshold_mm=threshold_mm))
    return localizer.run(build_context(ese=ese, fiducials=fiducials, electrodes=electrodes))


class TestBruteForceLocalizer:
    def test_exact_recovery_on_sphere(self):
        ese = make_ese()
        fiducials = make_fiducials()
        true_indices = [0, 42, 100]
        result = _localize(make_electrodes(ese.vertices[true_indices], fiducials))

        assert len(result.items) == 3
        for i, electrode in enumerate(result.items):
            assert electrode.is_localized
            assert electrode.ese_coords is not None
            assert electrode.scalp_coords is not None
            assert electrode.residual_error is not None
            assert np.array_equal(electrode.ese_coords, ese.vertices[true_indices[i]])
            assert np.array_equal(electrode.scalp_coords, ese.scalp_vertices[true_indices[i]])
            assert electrode.residual_error < 1e-6
            assert electrode.flagged is False
            assert electrode.confidence == ese.quality[true_indices[i]]

    def test_weighted_fiducial_dominates_when_measurements_conflict(self):
        ese = make_ese()
        fiducials = make_fiducials()
        true_index = 0
        point = ese.vertices[true_index]

        distances = _exact_distances(point, fiducials)
        distances["LPA"] += 100.0
        electrodes = Electrodes(items=[Electrode(electrode_id="E0", measured_distances=distances)])

        weighted = _reweighted(fiducials, weights={"NAS": 1e9, "LPA": 0.01, "RPA": 1.0})
        weighted_result = BruteForceLocalizer(Stage3Config()).run(
            build_context(ese=ese, fiducials=weighted, electrodes=electrodes)
        )
        unweighted_result = _localize(electrodes)

        weighted_electrode = weighted_result.items[0]
        assert weighted_electrode.ese_coords is not None
        assert weighted_electrode.residual_error is not None
        assert unweighted_result.items[0].residual_error is not None

        assert np.linalg.norm(weighted_electrode.ese_coords - point) < 1e-9
        assert weighted_electrode.residual_error == pytest.approx(10.0, abs=1e-6)
        assert unweighted_result.items[0].residual_error > weighted_electrode.residual_error + 30

    def test_localizes_with_subset_of_fiducials(self):
        ese = make_ese()
        fiducials = make_fiducials()
        true_index = 7
        point = ese.vertices[true_index]
        electrodes = Electrodes(
            items=[
                Electrode(
                    electrode_id="E0",
                    measured_distances={"NAS": _exact_distances(point, fiducials)["NAS"]},
                )
            ]
        )

        result = _localize(electrodes)

        assert result.items[0].is_localized
        assert _closest_vertex_index(ese, result.items[0].ese_coords) == true_index

    def test_skips_electrode_without_known_fiducials(self):
        result = _localize(
            Electrodes(
                items=[Electrode(electrode_id="E0", measured_distances={"UNKNOWN_FIDUCIAL": 10.0})]
            )
        )

        assert result.items[0].is_localized is False
        assert result.items[0].residual_error is None

    def test_flags_electrode_over_threshold(self):
        ese = make_ese()
        fiducials = make_fiducials()
        point = ese.vertices[0]
        distances = _exact_distances(point, fiducials)
        distances["LPA"] += 100.0

        result = _localize(
            Electrodes(items=[Electrode(electrode_id="E0", measured_distances=distances)]),
            threshold_mm=10.0,
        )

        assert result.items[0].flagged is True
        assert result.items[0].residual_error is not None
        assert result.items[0].residual_error > 10.0

    def test_rejects_mirror_candidate_across_fiducial_plane(self):
        """Electrode on positive side of fiducial plane must not snap to the
        mirror vertex on the negative side, even though both have nearly
        identical distance profiles."""
        from virda.localization.brute_force_localizer import _mirror_plane_mask

        ese = make_ese()
        fiducials = make_fiducials()
        vertices = np.asarray(ese.vertices)

        fiducial_coords = np.array([f.coordinates for f in fiducials.items])
        plane_normal = np.cross(
            fiducial_coords[1] - fiducial_coords[0],
            fiducial_coords[2] - fiducial_coords[0],
        )
        plane_normal /= np.linalg.norm(plane_normal)

        signed_dist = np.dot(vertices - fiducial_coords[0], plane_normal)

        pos_indices = np.where(signed_dist > 5.0)[0]
        assert len(pos_indices) > 0, "Need at least one vertex on positive side"

        true_index = int(pos_indices[0])
        point = ese.vertices[true_index]
        electrodes = make_electrodes(point[np.newaxis, :], fiducials)

        result = _localize(electrodes)

        result_index = _closest_vertex_index(ese, result.items[0].ese_coords)
        result_sd = signed_dist[result_index]
        msg = f"Localized to wrong side of fiducial plane: signed_dist={result_sd:.1f}"
        assert result_sd > 0, msg

        mirror_mask = _mirror_plane_mask(vertices, fiducial_coords, np.asarray(ese.normals))
        assert not mirror_mask[true_index], "The true vertex must not be masked out"


class TestOffsetCalibration:
    @staticmethod
    def _shifted_electrodes(ese, indices, shift_mm):
        """Electrodes measured at ``vertices + shift_mm * normals`` points."""
        vertices = np.asarray(ese.vertices)
        normals = np.asarray(ese.normals)
        points = vertices[list(indices)] + shift_mm * normals[list(indices)]
        return make_electrodes(points, make_fiducials()), points

    def test_recovers_offset_shift(self):
        ese = make_ese()
        indices = [0, 42, 100]
        # Measurements taken at the scalp surface while the ESE sits 2 mm out.
        electrodes, points = self._shifted_electrodes(ese, indices, shift_mm=-2.0)
        config = Stage3Config(calibrate_ese_offset=True)
        result = BruteForceLocalizer(config).run(
            build_context(ese=ese, fiducials=make_fiducials(), electrodes=electrodes)
        )

        assert result.calibrated_offset_shift_mm == pytest.approx(-2.0, abs=1e-9)
        for i, electrode in enumerate(result.items):
            assert electrode.is_localized
            assert electrode.residual_error is not None
            assert electrode.residual_error < 1e-6
            # The measurement point lies on the calibrated search cloud and
            # maps back to the same scalp vertex.
            assert np.linalg.norm(electrode.scalp_coords - points[i]) < 1e-6
            assert electrode.flagged is False

    def test_disabled_by_default(self):
        ese = make_ese()
        electrodes, _ = self._shifted_electrodes(ese, [0], shift_mm=-2.0)
        result = _localize(electrodes)

        assert result.calibrated_offset_shift_mm is None
        assert result.items[0].is_localized

    def test_zero_shift_when_measurements_match_ese(self):
        ese = make_ese()
        fiducials = make_fiducials()

        vertices = np.asarray(ese.vertices)
        fiducial_coords = np.array([f.coordinates for f in fiducials.items])
        plane_normal = np.cross(
            fiducial_coords[1] - fiducial_coords[0],
            fiducial_coords[2] - fiducial_coords[0],
        )
        plane_normal /= np.linalg.norm(plane_normal)
        signed_dist = np.dot(vertices - fiducial_coords[0], plane_normal)
        indices = np.where(signed_dist > 5.0)[0][:2]
        assert len(indices) == 2

        electrodes = make_electrodes(vertices[indices], fiducials)
        config = Stage3Config(calibrate_ese_offset=True)
        result = BruteForceLocalizer(config).run(
            build_context(ese=ese, fiducials=fiducials, electrodes=electrodes)
        )

        assert result.calibrated_offset_shift_mm == pytest.approx(0.0, abs=1e-9)
        for electrode in result.items:
            assert electrode.residual_error is not None
            assert electrode.residual_error < 1e-6
