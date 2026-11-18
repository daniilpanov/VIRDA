import numpy as np

from virda.geometry.transforms import apply_transform


def _affine() -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = [[1.2, 0.0, 0.0], [0.0, -1.3, 0.0], [0.0, 0.0, 1.4]]
    affine[:3, 3] = [10.0, -20.0, 30.0]
    return affine


def test_apply_transform_moves_the_points() -> None:
    affine = _affine()
    points = np.array([[2.0, 3.0, 4.0], [0.0, 0.0, 0.0], [15.0, 15.0, 15.0]])

    moved = apply_transform(points, affine)

    np.testing.assert_allclose(moved, points @ affine[:3, :3].T + affine[:3, 3], atol=1e-6)


def test_apply_transform_of_the_inverse_is_the_identity() -> None:
    affine = _affine()
    points = np.array([[13.6, -24.3, 36.8]])

    back = apply_transform(apply_transform(points, affine), np.linalg.inv(affine))

    np.testing.assert_allclose(back, points, atol=1e-6)
