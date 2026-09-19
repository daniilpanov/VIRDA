from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import trimesh

from tests.helpers.pipelines import build_context, save_test_fiducials
from virda.mesh.adjacency import build_scalp_mesh
from virda.mesh.density import step_size_for_voxel_size
from virda.mesh.mesh_decimator import QuadraticDecimator
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.models.config import Config
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1PipelineBuilder


@pytest.fixture
def sphere_mask() -> SegmentationMask:
    volume_shape = (30, 30, 30)
    center = np.array([15, 15, 15])
    sphere_radius = 10
    grid_indices = np.indices(volume_shape)
    squared_distance = np.sum((grid_indices - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)

    return SegmentationMask(mask=squared_distance <= sphere_radius**2)


@pytest.fixture
def sphere_volume() -> MRIVolume:
    volume_shape = (30, 30, 30)
    center = np.array([15, 15, 15])
    sphere_radius = 10

    grid_indices = np.indices(volume_shape)
    squared_distance = np.sum((grid_indices - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    image_data = np.zeros(volume_shape, dtype=np.float32)
    image_data[squared_distance <= sphere_radius**2] = 100.0

    return MRIVolume(
        data=image_data,
        affine=np.eye(4),
        spacing=(1.0, 1.0, 1.0),
        orientation=("R", "A", "S"),
    )


@pytest.fixture
def icosphere_mesh() -> ScalpMesh:
    mesh = trimesh.creation.icosphere(subdivisions=3)
    return build_scalp_mesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )


class TestStepSizeForVoxelSize:
    def test_native_spacing_yields_step_one(self) -> None:
        step, real = step_size_for_voxel_size(1.0, (1.0, 1.0, 1.0))

        assert step == 1
        assert real == (1.0, 1.0, 1.0)

    def test_mean_spacing_yields_step_one_anisotropic(self) -> None:
        spacing = (1.0, 1.0, 1.3)
        mean = sum(spacing) / 3.0

        step, real = step_size_for_voxel_size(mean, spacing)

        assert step == 1
        assert real == (1.0, 1.0, 1.3)

    def test_larger_desired_size_scales_step(self) -> None:
        step, real = step_size_for_voxel_size(2.0, (1.0, 1.0, 1.0))

        assert step == 2
        assert real == (2.0, 2.0, 2.0)

    def test_real_voxel_is_step_times_spacing(self) -> None:
        spacing = (1.2, 1.4, 2.0)

        step, real = step_size_for_voxel_size(4.0, spacing)

        assert step == 3
        assert real == pytest.approx((3.6, 4.2, 6.0))

    def test_small_desired_size_clamps_to_one(self) -> None:
        step, real = step_size_for_voxel_size(0.1, (1.0, 1.0, 1.0))

        assert step == 1
        assert real == (1.0, 1.0, 1.0)

    def test_rejects_nonpositive_desired_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            step_size_for_voxel_size(0.0, (1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="positive"):
            step_size_for_voxel_size(-2.0, (1.0, 1.0, 1.0))

    def test_rejects_invalid_spacing(self) -> None:
        with pytest.raises(ValueError, match="spacing"):
            step_size_for_voxel_size(1.0, (1.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="spacing"):
            step_size_for_voxel_size(1.0, (1.0, 1.0))


class TestMarchingCubesDensity:
    def test_no_voxel_size_is_default_step_one(
        self, sphere_mask: SegmentationMask, sphere_volume: MRIVolume
    ) -> None:
        default_mesh = MarchingCubesExtractor().run(
            build_context(SegmentationMask=sphere_mask, MRIVolume=sphere_volume)
        )
        native_mesh = MarchingCubesExtractor(voxel_size_mm=1.0).run(
            build_context(SegmentationMask=sphere_mask, MRIVolume=sphere_volume)
        )

        assert np.array_equal(default_mesh.vertices, native_mesh.vertices)
        assert np.array_equal(default_mesh.faces, native_mesh.faces)

    def test_larger_voxel_size_reduces_vertices(
        self, sphere_mask: SegmentationMask, sphere_volume: MRIVolume
    ) -> None:
        native = MarchingCubesExtractor(voxel_size_mm=1.0).run(
            build_context(SegmentationMask=sphere_mask, MRIVolume=sphere_volume)
        )
        coarse = MarchingCubesExtractor(voxel_size_mm=3.0).run(
            build_context(SegmentationMask=sphere_mask, MRIVolume=sphere_volume)
        )

        assert coarse.vertices.shape[0] < native.vertices.shape[0]
        assert coarse.faces.shape[0] < native.faces.shape[0]

    def test_affine_transform_still_correct(
        self, sphere_mask: SegmentationMask, sphere_volume: MRIVolume
    ) -> None:
        voxel_to_world = np.array(
            [
                [1.5, 0.0, 0.0, 10.0],
                [0.0, 1.5, 0.0, 20.0],
                [0.0, 0.0, 2.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        mesh_world = MarchingCubesExtractor(voxel_size_mm=3.0).run(
            build_context(
                SegmentationMask=sphere_mask,
                MRIVolume=replace(sphere_volume, affine=voxel_to_world),
            )
        )
        mesh_voxel = MarchingCubesExtractor(voxel_size_mm=3.0).run(
            build_context(
                SegmentationMask=sphere_mask,
                MRIVolume=replace(sphere_volume, affine=np.eye(4)),
            )
        )

        expected_world = mesh_voxel.vertices @ voxel_to_world[:3, :3].T + voxel_to_world[:3, 3]
        np.testing.assert_array_almost_equal(mesh_world.vertices, expected_world)


class TestQuadraticDecimator:
    def test_density_100_is_a_noop(self, icosphere_mesh: ScalpMesh) -> None:
        decimated = QuadraticDecimator(density_percent=100.0).run(
            build_context(ScalpMesh=icosphere_mesh)
        )

        assert decimated.vertices is icosphere_mesh.vertices
        assert decimated.faces is icosphere_mesh.faces

    def test_density_50_removes_half_the_vertices(
        self,
        icosphere_mesh: ScalpMesh,
    ) -> None:
        decimated = QuadraticDecimator(density_percent=50.0).run(
            build_context(ScalpMesh=icosphere_mesh)
        )

        original = icosphere_mesh.vertices.shape[0]
        ratio = decimated.vertices.shape[0] / original
        assert 0.35 <= ratio <= 0.65
        assert decimated.faces.shape[0] < icosphere_mesh.faces.shape[0]

    def test_density_low_keeps_valid_mesh(self, icosphere_mesh: ScalpMesh) -> None:
        decimated = QuadraticDecimator(density_percent=15.0).run(
            build_context(ScalpMesh=icosphere_mesh)
        )

        assert isinstance(decimated, ScalpMesh)
        assert decimated.vertices.shape[1] == 3
        assert decimated.faces.shape[1] == 3
        assert decimated.faces.min() >= 0
        assert decimated.faces.max() < decimated.vertices.shape[0]
        assert decimated.vertices.shape[0] < icosphere_mesh.vertices.shape[0]

    def test_density_0_raises(self) -> None:
        with pytest.raises(ValueError, match="density_percent"):
            QuadraticDecimator(density_percent=0.0)

    def test_density_above_100_raises(self) -> None:
        with pytest.raises(ValueError, match="density_percent"):
            QuadraticDecimator(density_percent=150.0)

    def test_density_below_1_raises(self) -> None:
        with pytest.raises(ValueError, match="density_percent"):
            QuadraticDecimator(density_percent=0.5)


@pytest.fixture
def synthetic_nifti_path(tmp_path: Path) -> Path:
    volume_shape = (20, 20, 20)
    center = np.array([10, 10, 10])
    sphere_radius = 8
    grid_indices = np.indices(volume_shape)
    squared_distance = np.sum((grid_indices - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    inside = squared_distance <= sphere_radius**2

    image_data = np.zeros(volume_shape, dtype=np.float32)
    image_data[inside] = 100.0

    nifti_path = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(image_data, np.eye(4)), nifti_path)
    return nifti_path


class TestPipelineDensity:
    def test_from_config_applies_both_density_params(
        self, synthetic_nifti_path: Path, tmp_path: Path
    ) -> None:
        fiducials_path = save_test_fiducials(tmp_path / "fiducials.json")

        dense_config = Config(
            nifti_path=str(synthetic_nifti_path),
            project_dir=str(tmp_path / "dense"),
            fiducials_path=str(fiducials_path),
            closing_radius=0,
            seal_enabled=False,
        )
        dense_result = (
            Stage1PipelineBuilder.from_config(config=dense_config)
            .build()
            .run()
            .get_store_notnull(Stage1Result)
        )

        coarse_config = Config(
            nifti_path=str(synthetic_nifti_path),
            project_dir=str(tmp_path / "coarse"),
            fiducials_path=str(fiducials_path),
            closing_radius=0,
            seal_enabled=False,
            mesh_voxel_size_mm=2.0,
            mesh_density_percent=50.0,
        )
        coarse_result = (
            Stage1PipelineBuilder.from_config(config=coarse_config)
            .build()
            .run()
            .get_store_notnull(Stage1Result)
        )

        assert coarse_result.mesh.vertices.shape[0] > 0
        assert coarse_result.mesh.vertices.shape[0] < dense_result.mesh.vertices.shape[0]