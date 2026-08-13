from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from virda.io.loader.nifti_loader import NiftiLoader
from virda.mesh.contracts import MeshPostprocessor
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1PipelineBuilder
from virda.segmentation.head_segmenter import OtsuHeadSegmenter


@pytest.fixture
def synthetic_nifti_path(tmp_path: Path) -> Path:
    volume_shape = (20, 20, 20)
    center = np.array([10, 10, 10])
    sphere_radius = 8
    grid_indices = np.indices(volume_shape)
    squared_distance = np.sum((grid_indices - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    is_inside_sphere = squared_distance <= sphere_radius**2

    image_data = np.zeros(volume_shape, dtype=np.float32)
    image_data[is_inside_sphere] = 100.0

    voxel_to_world_affine = np.eye(4)

    nifti_image = nib.Nifti1Image(image_data, voxel_to_world_affine)
    nifti_file_path = tmp_path / "synthetic.nii.gz"
    nib.save(nifti_image, nifti_file_path)
    return nifti_file_path


def build_pipeline(nifti_path: Path, postprocessors: list[MeshPostprocessor] | None = None):
    builder = Stage1PipelineBuilder(
        nifti_path=nifti_path,
        mri_loader=NiftiLoader(),
        segmenter=OtsuHeadSegmenter(closing_radius=0),
        extractor=MarchingCubesExtractor(),
    )
    if postprocessors:
        builder.setup_mesh_postprocessors(postprocessors)
    return builder.build()


class TestStage1Pipeline:
    def test_run_returns_stage1_result(self, synthetic_nifti_path: Path) -> None:
        pipeline = build_pipeline(synthetic_nifti_path)
        result = pipeline.run().get_store_notnull(Stage1Result)

        assert isinstance(result, Stage1Result)
        assert result.mri_volume.data.shape == (20, 20, 20)
        assert result.segmentation_mask.mask.shape == (20, 20, 20)
        assert result.segmentation_mask.mask.dtype == bool
        assert result.mesh.vertices.shape[1] == 3
        assert result.mesh.faces.shape[1] == 3
        assert result.mesh.faces.min() >= 0
        assert result.mesh.faces.max() < result.mesh.vertices.shape[0]

    def test_run_with_smoother(self, synthetic_nifti_path: Path) -> None:
        from virda.mesh.laplacian_smoother import LaplacianSmoother

        pipeline = build_pipeline(
            synthetic_nifti_path,
            postprocessors=[TrimeshCleaner(), LaplacianSmoother(iterations=2, lamb=0.5)],
        )
        result = pipeline.run().get_store_notnull(Stage1Result)

        assert isinstance(result, Stage1Result)
        assert result.mesh.vertices.shape[0] > 0

    def test_run_exports_mesh_arrays(self, synthetic_nifti_path: Path, tmp_path: Path) -> None:
        builder = Stage1PipelineBuilder(
            nifti_path=synthetic_nifti_path,
            mri_loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(closing_radius=0),
            extractor=MarchingCubesExtractor(),
            project_dir=tmp_path,
        )
        pipeline = builder.build()

        result = pipeline.run().get_store_notnull(Stage1Result)

        vertices = np.load(tmp_path / "mesh" / "scalp_vertices.npy")

        assert np.array_equal(vertices, result.mesh.vertices)
        assert (tmp_path / "mesh" / "final_mesh.ply").exists()
