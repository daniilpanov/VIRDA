"""Affine-frame placement helpers shared by GUI widgets.

Renderable scenes overlay objects that live in different coordinate frames:
a NIfTI volume is addressed by voxel indices but its affine matrix maps them
into world millimeters, while meshes and labeled points come in world
millimeters. These helpers decide which frame a given affine produces and
provide the mappings between them, so that volume-, mesh- and point-consuming
components can place their inputs into one common scene frame:

* an axis-aligned affine with positive spacing keeps the scene in world
  millimeters: the volume gets its spacing and origin from the affine
  diagonal and translation, and mesh data stays in its world coordinates;
* any other affine (rotation, flip) moves the mesh data into voxel index
  space with the inverse affine, so the overlay stays correct.
"""

import numpy as np

ScenePlacement = tuple[np.ndarray, np.ndarray, np.ndarray, bool]


def downsample(data: np.ndarray, affine: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Take every *stride*-th voxel and scale the affine accordingly"""
    if stride < 1:
        raise ValueError(f"downsample must be >= 1, got {stride}")

    if stride == 1:
        return data, affine

    sampled = data[::stride, ::stride, ::stride]
    scaled = affine @ np.diag([stride, stride, stride, 1.0])
    return sampled, scaled


def is_axis_aligned(affine: np.ndarray) -> bool:
    """True when the affine is a positive diagonal (no rotation or flip)."""
    linear = affine[:3, :3]
    return np.allclose(linear, np.diag(np.diag(linear))) and bool(np.all(np.diag(linear) > 0))


def scene_placement(affine: np.ndarray | None) -> ScenePlacement:
    """Return ``(spacing, origin, transform, mm_scene)`` for placing a scene"""
    if affine is not None and is_axis_aligned(affine):
        spacing = np.asarray(np.diag(affine)[:3], dtype=float)
        origin = np.asarray(affine[:3, 3], dtype=float)
        return spacing, origin, np.eye(4), True

    return (
        (np.ones(3), np.zeros(3), np.eye(4), True)
        if affine is None
        else (np.ones(3), np.zeros(3), np.linalg.inv(affine), False)
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine ``transform`` to an (N, 3) array of points."""
    return np.asarray(points @ transform[:3, :3].T + transform[:3, 3], dtype=np.float64)
