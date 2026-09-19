"""Mesh density helpers.

Power the two density knobs of the pipeline:

* pre-extraction ``step_size`` for :func:`skimage.measure.marching_cubes`,
  derived from a desired voxel size in millimetres;
* post-extraction vertex reduction expressed as a percentage (see
  :class:`~virda.mesh.mesh_decimator.QuadraticDecimator`).
"""

import math
from collections.abc import Sequence


def step_size_for_voxel_size(
    desired_mm: float,
    spacing: Sequence[float],
) -> tuple[int, tuple[float, float, float]]:
    """Pick a marching-cubes ``step_size`` matching ``desired_mm``.

    ``marching_cubes`` samples every ``step_size``-th mask voxel, so the
    effective voxel of the resulting mesh is ``step_size * spacing`` along
    each axis.  Returns ``(step_size, real_voxel_mm)`` where
    ``real_voxel_mm`` is that per-axis effective spacing.  A desired size
    equal to the mean native spacing yields ``step_size=1`` (native mesh).
    """
    if not math.isfinite(desired_mm) or desired_mm <= 0:
        raise ValueError(f"desired voxel size must be a positive number, got {desired_mm!r}")
    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        raise ValueError(f"spacing must contain three positive values, got {spacing}")

    mean_spacing = sum(spacing) / 3.0
    step_size = max(1, round(desired_mm / mean_spacing))

    real_voxel_mm = tuple(step_size * float(s) for s in spacing)
    return step_size, (real_voxel_mm[0], real_voxel_mm[1], real_voxel_mm[2])