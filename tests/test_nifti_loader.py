import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tests.helpers.pipelines import build_context
from virda.io.loader.nifti_loader import NiftiLoader
from virda.models.mri_volume import MRIVolume
from virda.models.path import NiftiPath


class TestNiftiLoader:
    def test_load_extracts_data_and_spatial_metadata(self, tmp_path: Path) -> None:
        volume_data = np.random.rand(10, 10, 10).astype(np.float32)
        voxel_to_world_affine = np.diag([1.5, 1.5, 2.0, 1.0])

        nifti_image = nib.Nifti1Image(volume_data, voxel_to_world_affine)
        nifti_file_path = tmp_path / "test_diag.nii.gz"
        nib.save(nifti_image, nifti_file_path)

        loader = NiftiLoader()
        loaded_mri = loader.run(build_context(NiftiPath=NiftiPath(nifti_file_path)))

        assert isinstance(loaded_mri, MRIVolume)
        assert loaded_mri.data.shape == (10, 10, 10)
        assert loaded_mri.affine.shape == (4, 4)
        assert loaded_mri.spacing == (1.5, 1.5, 2.0)
        np.testing.assert_array_almost_equal(loaded_mri.data, volume_data)
        np.testing.assert_array_almost_equal(loaded_mri.affine, voxel_to_world_affine)

    def test_load_handles_rotated_affine(self, tmp_path: Path) -> None:
        volume_data = np.zeros((5, 5, 5), dtype=np.float32)

        theta = math.radians(30)
        spacing_x, spacing_y, spacing_z = 1.0, 1.0, 2.0

        # Rotation matrix
        affine_rotated = np.array(
            [
                [
                    spacing_x * math.cos(theta),
                    -spacing_y * math.sin(theta),
                    0,
                    10.0,
                ],  # translation 10mm
                [spacing_x * math.sin(theta), spacing_y * math.cos(theta), 0, 20.0],
                [0, 0, spacing_z, 30.0],
                [0, 0, 0, 1.0],
            ]
        )

        nifti_image = nib.Nifti1Image(volume_data, affine_rotated)
        nifti_file_path = tmp_path / "test_rotated.nii.gz"
        nib.save(nifti_image, nifti_file_path)

        loader = NiftiLoader()
        loaded_mri = loader.run(build_context(NiftiPath=NiftiPath(nifti_file_path)))

        # Assert: Spacing must be (1.0, 1.0, 2.0)
        np.testing.assert_array_almost_equal(
            loaded_mri.spacing,
            (1.0, 1.0, 2.0),
            decimal=5,
            err_msg="Spacing calculated incorrectly for rotated affine",
        )

        # Test matrix
        np.testing.assert_array_almost_equal(loaded_mri.affine, affine_rotated)

    def test_load_raises_on_missing_file(self) -> None:
        loader = NiftiLoader()
        with pytest.raises(FileNotFoundError):
            loader.run(build_context(NiftiPath=NiftiPath(Path("/nonexistent/path/to/scan.nii.gz"))))
