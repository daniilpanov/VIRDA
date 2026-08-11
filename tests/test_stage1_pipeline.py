from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from virda.io.loader.nifti_loader import NiftiLoader
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1Pipeline
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


class TestStage1Pipeline:
    def test_run_returns_stage1_result(self, synthetic_nifti_path: Path) -> None:
        loader = NiftiLoader()
        segmenter = OtsuHeadSegmenter()
        cleaner = TrimeshCleaner()

        pipeline = Stage1Pipeline(
            loader=loader, segmenter=segmenter, cleaner=cleaner, smoother=None
        )
        result = pipeline.run(synthetic_nifti_path, closing_radius=0)

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

        loader = NiftiLoader()
        segmenter = OtsuHeadSegmenter()
        cleaner = TrimeshCleaner()
        smoother = LaplacianSmoother(iterations=2, lamb=0.5)

        pipeline = Stage1Pipeline(
            loader=loader, segmenter=segmenter, cleaner=cleaner, smoother=smoother
        )
        result = pipeline.run(synthetic_nifti_path, closing_radius=0)

        assert isinstance(result, Stage1Result)
        assert result.mesh.vertices.shape[0] > 0
