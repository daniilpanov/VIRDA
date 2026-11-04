"""Export triangular mesh vertices and faces as NumPy arrays."""

from pathlib import Path

import numpy as np


def export_mesh_vertices(path: str | Path, vertices: np.ndarray) -> Path:
    """Write an ``(N, 3)`` float64 vertex array to a NumPy file."""
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"expected (N, 3) vertices, got shape {array.shape}")
    return _save_array(path, array)


def _save_array(path: str | Path, array: np.ndarray) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".npy":
        target = target.with_name(f"{target.name}.npy")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, array, allow_pickle=False)
    return target
