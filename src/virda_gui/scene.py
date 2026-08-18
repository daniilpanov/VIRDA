"""Shared scene-placement helpers for the pyvista viewer and the HTML export.

Both the interactive ``virda_gui.viewer`` and the ``virda_gui.html_export``
tool place the scalp mesh and the MRI volume into one coordinate frame using
the NIfTI affine:

* an axis-aligned affine with positive spacing keeps the scene in world
  millimeters: the volume gets its spacing and origin from the affine diagonal
  and translation, and the mesh stays in its world coordinates;
* any other affine (rotation, flip) moves the mesh (and the fiducials) into
  voxel index space with the inverse affine, so the overlay stays correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ScenePlacement = tuple[np.ndarray, np.ndarray, np.ndarray, bool]


def downsample(data: np.ndarray, affine: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``data`` with a voxel ``stride`` and scale the affine accordingly."""
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
    if not np.allclose(linear, np.diag(np.diag(linear))):
        return False
    return bool(np.all(np.diag(linear) > 0))


def scene_placement(affine: np.ndarray | None) -> ScenePlacement:
    """Return ``(spacing, origin, transform, mm_scene)`` for placing a scene.

    ``transform`` maps world coordinates into the scene frame: the identity in
    a world-millimeter scene, the inverse affine in a voxel-index scene.
    """
    if affine is not None and is_axis_aligned(affine):
        spacing = np.asarray(np.diag(affine)[:3], dtype=float)
        origin = np.asarray(affine[:3, 3], dtype=float)
        return spacing, origin, np.eye(4), True
    if affine is None:
        return np.ones(3), np.zeros(3), np.eye(4), True
    return np.ones(3), np.zeros(3), np.linalg.inv(affine), False


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine ``transform`` to an (N, 3) array of points."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def load_fiducial_points(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read a fiducials JSON (``{"fiducials": [...]}``) into points and labels."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    points = np.asarray(
        [np.asarray(item["coordinates"], dtype=np.float64) for item in data["fiducials"]],
        dtype=np.float64,
    )
    labels = [f"{item['fiducial_id']} ({item['name']})" for item in data["fiducials"]]
    return points, labels


def percentile_clim(data: np.ndarray) -> tuple[float, float]:
    """Intensity window (3rd, 99.9th percentiles) used by the boosted view."""
    lo, hi = np.percentile(data, (3, 99.9))
    return float(lo), float(hi)
