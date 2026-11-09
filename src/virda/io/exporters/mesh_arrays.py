"""Export triangular mesh vertices and faces as NumPy arrays."""

from pathlib import Path

import numpy as np


def export_mesh_vertices(path: str | Path, vertices: np.ndarray) -> Path:
    """Write an ``(N, 3)`` float64 vertex array to a NumPy file."""
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"expected (N, 3) vertices, got shape {array.shape}")
    return _save_array(path, array)


def export_mesh_faces(path: str | Path, faces: np.ndarray) -> Path:
    """Write an ``(M, 3)`` int64 triangular face array to a NumPy file."""
    array = np.asarray(faces)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"expected triangular (M, 3) faces, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        integer_array = np.asarray(array, dtype=np.int64)
        if not np.array_equal(array, integer_array):
            raise ValueError("face indices must be integers")
        array = integer_array
    else:
        array = np.asarray(array, dtype=np.int64)
    return _save_array(path, array)


def _save_array(path: str | Path, array: np.ndarray) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".npy":
        target = target.with_name(f"{target.name}.npy")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, array, allow_pickle=False)
    return target
