"""The affine transform applied to point sets."""

import numpy as np


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine ``transform`` to an (N, 3) array of points."""
    return np.asarray(points @ transform[:3, :3].T + transform[:3, 3], dtype=np.float64)
