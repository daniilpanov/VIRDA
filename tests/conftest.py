from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from pytest import ExitCode, Session


def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    # If there are no tests, exit with 0 status. Skip tests silently
    if exitstatus == ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = ExitCode.OK


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
