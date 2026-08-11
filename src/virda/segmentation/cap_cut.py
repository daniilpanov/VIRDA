"""Inferior cap cut of the head mask below the nose.

The mask is cut with a plane parallel to the LPA-RPA-NAS fiducial plane,
placed ``offset_mm`` below the nasion along the plane normal. Everything below
the plane (neck, jaw, mouth) is dropped. The cut runs on the mask before the
surface mesh is extracted, so marching cubes produces a flat, watertight cap
bottom without any extra capping step.
"""

from collections.abc import Iterable

import numpy as np

from virda.models.fiducial import Fiducial

_PLANE_EPSILON = 1e-6


def cut_plane_from_fiducials(
    fiducials: Iterable[Fiducial], offset_mm: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (normal, point) of the cut plane in world RAS coordinates.

    The plane normal is ``cross(RPA-LPA, NAS-LPA)`` (points towards the
    vertex), and the plane passes ``offset_mm`` below the nasion along the
    normal. ``fiducials`` must include NAS, LPA and RPA in world coordinates.
    """
    world = {fid.fiducial_id.upper(): fid.coordinates for fid in fiducials}
    required = {"NAS", "LPA", "RPA"}
    missing = sorted(required - set(world))
    if missing:
        required_names = ", ".join(sorted(required))
        raise ValueError(f"Cap cut requires fiducials {required_names}; missing {missing}")
    nas = np.asarray(world["NAS"], dtype=np.float64)
    lpa = np.asarray(world["LPA"], dtype=np.float64)
    rpa = np.asarray(world["RPA"], dtype=np.float64)

    normal = np.cross(rpa - lpa, nas - lpa)
    norm = np.linalg.norm(normal)
    if norm < _PLANE_EPSILON:
        raise ValueError("Cap cut fiducials are collinear; cannot build a plane")
    normal = normal / norm
    return normal, nas - offset_mm * normal


def cut_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    fiducials: Iterable[Fiducial],
    offset_mm: float = 30.0,
) -> np.ndarray:
    """Return ``mask`` with everything below the cap-cut plane removed.

    The plane is anchored to the LPA-RPA-NAS fiducial plane (see
    ``cut_plane_from_fiducials``). Kept voxels are those whose world position
    lies on the nasion side of the plane (``normal . (x - point) >= 0``).
    """
    normal_world, point_world = cut_plane_from_fiducials(fiducials, offset_mm)
    normal_voxel = np.linalg.inv(affine[:3, :3]).T @ normal_world
    point_voxel = np.linalg.solve(affine[:3, :3], point_world - affine[:3, 3])

    axes = [
        (np.arange(shape, dtype=np.float64) - point_voxel[axis]).reshape(
            (shape, 1, 1) if axis == 0 else (1, shape, 1) if axis == 1 else (1, 1, shape)
        )
        for axis, shape in enumerate(mask.shape)
    ]
    distance = normal_voxel[0] * axes[0] + normal_voxel[1] * axes[1] + normal_voxel[2] * axes[2]
    result: np.ndarray = mask & (distance >= 0)
    return result
