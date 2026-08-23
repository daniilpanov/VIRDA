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

from pathlib import Path

import numpy as np

from virda.io.fiducial_helpers import load_fiducials

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
    """Read a fiducials JSON file into ``(points, labels)``.

    Delegates to :func:`virda.io.fiducial_helpers.load_fiducials`, so both the
    native ``{"fiducials": [...]}`` format and MNE-style ``coordsystem.json``
    files are supported (with unit conversion applied for the latter).
    """
    fiducials = load_fiducials(Path(path))
    points = np.asarray([item.coordinates for item in fiducials.items], dtype=np.float64)
    labels = [f"{item.fiducial_id} ({item.name})" for item in fiducials.items]
    return points, labels


def percentile_clim(data: np.ndarray) -> tuple[float, float]:
    """Intensity window (3rd, 99.9th percentiles) used by the boosted view."""
    lo, hi = np.percentile(data, (3, 99.9))
    return float(lo), float(hi)


def load_normals(path: str | Path) -> np.ndarray:
    """Load per-vertex normals from a ``.npy`` file as an (N, 3) float64 array."""
    normals = np.load(path)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"Normals must be (N, 3) array, got shape {normals.shape}")
    return normals.astype(np.float64)


def sample_normals(
    normals: np.ndarray,
    density: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Down-sample normals by selecting every *density*-th vertex.

    Returns ``(indices, sampled_normals)`` where ``indices`` are the selected
    vertex indices and ``sampled_normals`` are the corresponding (N, 3) normal
    vectors.
    """
    step = max(1, density)
    indices = np.arange(0, len(normals), step)
    return indices, normals[indices]


def compute_normal_lines(
    points: np.ndarray,
    normals: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute line-segment endpoints for normal visualisation.

    Given *points* (N, 3) and *normals* (N, 3), return
    ``(origins, tips)`` each of shape (N, 3) that can be fed into any
    line-renderer (PyVista, three.js, …).
    """
    origins = points
    tips = points + scale * normals
    return origins, tips
