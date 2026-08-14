import numpy as np

from virda.geometry.transforms import fiducials_world_coordinates, mesh_voxel_coordinates
from virda.models.fiducial import Fiducial


def _affine() -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = [[1.2, 0.0, 0.0], [0.0, -1.3, 0.0], [0.0, 0.0, 1.4]]
    affine[:3, 3] = [10.0, -20.0, 30.0]
    return affine


def test_mesh_voxel_round_trip() -> None:
    affine = _affine()
    voxel = np.array([[2.0, 3.0, 4.0], [0.0, 0.0, 0.0], [15.0, 15.0, 15.0]])
    world = voxel @ affine[:3, :3].T + affine[:3, 3]
    back = mesh_voxel_coordinates(world, affine)
    np.testing.assert_allclose(back, voxel, atol=1e-6)


def test_mesh_voxel_known_point() -> None:
    affine = _affine()
    world = np.array([[13.6, -24.3, 36.8]])
    voxel = mesh_voxel_coordinates(world, affine)
    expected = (world[0] - affine[:3, 3]) @ np.linalg.inv(affine[:3, :3]).T
    np.testing.assert_allclose(voxel[0], expected, atol=1e-6)


def test_fiducials_world_coordinates() -> None:
    affine = _affine()
    voxel = Fiducial(
        fiducial_id="NAS",
        name="nasion",
        coordinates=np.array([2.0, 3.0, 4.0]),
        coordinate_system="voxel",
    )
    world = Fiducial(
        fiducial_id="INI",
        name="inion",
        coordinates=np.array([13.6, -24.3, 36.8]),
        coordinate_system="world",
    )
    out = fiducials_world_coordinates([voxel, world], affine)
    expected_voxel = np.array([2.0, 3.0, 4.0]) @ affine[:3, :3].T + affine[:3, 3]
    np.testing.assert_allclose(out[0], expected_voxel, atol=1e-6)
    np.testing.assert_allclose(out[1], world.coordinates, atol=1e-6)
