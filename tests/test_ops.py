"""Unit tests for the pure high-level ops atoms and their options."""

import numpy as np
import pytest

from tests.helpers.measurements import make_electrodes, make_fiducials
from tests.helpers.meshes import make_sphere
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.ops.atoms import (
    clean,
    decimate,
    generate_ese,
    generate_scalp_surface,
    localize,
    smooth,
)
from virda.ops.options import (
    CleanOptions,
    DecimateOptions,
    EseOptions,
    LocalizeOptions,
    SealingOptions,
    SmoothOptions,
)


@pytest.fixture
def sphere_volume() -> MRIVolume:
    volume_shape = (30, 30, 30)
    center = np.array([15, 15, 15])
    radius = 10
    grid = np.indices(volume_shape)
    inside = np.sum((grid - center.reshape(-1, 1, 1, 1)) ** 2, axis=0) <= radius**2
    data = np.zeros(volume_shape, dtype=np.float32)
    data[inside] = 100.0
    return MRIVolume(
        data=data,
        affine=np.eye(4),
        spacing=(1.0, 1.0, 1.0),
        orientation=("R", "A", "S"),
    )


class TestOptionsDefaults:
    def test_sealing_defaults(self) -> None:
        assert SealingOptions().seal_enabled is True
        assert SealingOptions().seal_radius == 4

    def test_smoothing_defaults(self) -> None:
        assert SmoothOptions().smoother == "laplacian"
        assert SmoothOptions().iterations == 5
        assert SmoothOptions().lamb == 0.5
        assert SmoothOptions().nu == -0.53

    def test_decimation_defaults(self) -> None:
        assert DecimateOptions().density_percent == 100.0

    def test_cleaning_defaults(self) -> None:
        assert CleanOptions().min_component_vertices == 100
        assert CleanOptions().merge_digits == 7

    def test_ese_requires_offset(self) -> None:
        assert EseOptions(ese_offset_mm=2.0).ese_offset_mm == 2.0
        assert EseOptions(ese_offset_mm=2.0).k_neighbors is None
        with pytest.raises(TypeError):
            EseOptions()  # type: ignore[call-arg]

    def test_localize_defaults(self) -> None:
        assert LocalizeOptions().calibrate_ese_offset is True
        assert LocalizeOptions().residual_threshold_mm == 10.0


class TestGenerateScalpSurface:
    def test_segments_and_extracts(self, sphere_volume: MRIVolume) -> None:
        surface = generate_scalp_surface(sphere_volume, sealing=SealingOptions())

        assert surface.mask.mask.dtype == bool
        assert surface.mask.mask.shape == (30, 30, 30)
        assert surface.mask.mask.any()
        assert isinstance(surface.mesh, ScalpMesh)
        assert surface.mesh.vertices.shape[0] > 0
        assert surface.mesh.faces.shape[0] > 0

    def test_without_sealing(self, sphere_volume: MRIVolume) -> None:
        surface = generate_scalp_surface(sphere_volume)

        assert surface.mesh.vertices.shape[0] > 0

    def test_sealing_disabled_option(self, sphere_volume: MRIVolume) -> None:
        surface = generate_scalp_surface(
            sphere_volume, sealing=SealingOptions(seal_enabled=False)
        )

        assert surface.mesh.faces.shape[0] > 0


class TestSmooth:
    def test_none_returns_input_instance(self) -> None:
        mesh = make_sphere()
        assert smooth(mesh, SmoothOptions(smoother="none")) is mesh

    def test_laplacian_changes_vertices(self) -> None:
        mesh = make_sphere()
        result = smooth(mesh, SmoothOptions(smoother="laplacian", iterations=3))
        assert result is not mesh
        assert result.vertices.shape == mesh.vertices.shape
        assert not np.allclose(result.vertices, mesh.vertices, atol=1e-7)

    def test_taubin_uses_nu(self) -> None:
        mesh = make_sphere()
        result = smooth(mesh, SmoothOptions(smoother="taubin", iterations=1, nu=-0.53))
        assert result.vertices.shape == mesh.vertices.shape


class TestDecimate:
    def test_full_density_is_noop(self) -> None:
        mesh = make_sphere()
        assert decimate(mesh, DecimateOptions(density_percent=100.0)) is mesh

    def test_reduces_vertex_count(self) -> None:
        mesh = make_sphere()
        result = decimate(mesh, DecimateOptions(density_percent=50.0))
        assert result.vertices.shape[0] < mesh.vertices.shape[0]

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="density_percent"):
            decimate(make_sphere(), DecimateOptions(density_percent=150.0))


class TestClean:
    def test_returns_watertight_scalp_mesh(self) -> None:
        mesh = make_sphere(subdivisions=1)
        result = clean(mesh, CleanOptions())

        assert isinstance(result, ScalpMesh)
        assert result.faces.shape[0] > 0
        assert result.faces.min() >= 0
        assert result.faces.max() < result.vertices.shape[0]


class TestGenerateEse:
    def test_offsets_along_normals(self) -> None:
        mesh = make_sphere()
        ese = generate_ese(mesh, EseOptions(ese_offset_mm=2.0, k_neighbors=20))

        assert ese.vertices.shape == mesh.vertices.shape
        assert np.allclose(np.linalg.norm(ese.vertices - mesh.vertices, axis=1), 2.0)
        assert np.mean(np.abs(ese.normals[:, 2])) > 0.0


class TestLocalize:
    def test_recovers_exact_positions(self) -> None:
        ese = generate_ese(make_sphere(), EseOptions(ese_offset_mm=2.0, k_neighbors=20))
        fiducials = make_fiducials()
        true_indices = [0, 42, 100]

        result = localize(
            ese,
            fiducials,
            make_electrodes(ese.vertices[true_indices], fiducials),
            LocalizeOptions(calibrate_ese_offset=False),
        )

        assert len(result.items) == 3
        for i, electrode in enumerate(result.items):
            assert electrode.is_localized
            assert np.array_equal(electrode.ese_coords, ese.vertices[true_indices[i]])
            assert electrode.residual_error is not None
            assert electrode.residual_error < 1e-6

    def test_calibrates_offset(self) -> None:
        ese = generate_ese(make_sphere(), EseOptions(ese_offset_mm=2.0, k_neighbors=20))
        fiducials = make_fiducials()
        vertices = np.asarray(ese.vertices)
        normals = np.asarray(ese.normals)
        indices = [0, 42, 100]
        points = vertices[indices] - 2.0 * normals[indices]

        result = localize(
            ese,
            fiducials,
            make_electrodes(points, fiducials),
            LocalizeOptions(calibrate_ese_offset=True),
        )

        assert result.calibrated_offset_shift_mm == pytest.approx(-2.0, abs=1e-9)