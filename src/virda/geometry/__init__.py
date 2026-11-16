"""Spatial coordinate helpers."""

from virda.geometry.frames import (
    SLOT_IDS,
    FrameContext,
    convert,
    convert_point,
    match_fiducial_name,
)
from virda.geometry.geometry import Geometry
from virda.geometry.surface import SurfaceProbe, distance_to_surface, nearest_vertex
from virda.geometry.transforms import apply_transform

__all__ = [
    "SLOT_IDS",
    "FrameContext",
    "Geometry",
    "SurfaceProbe",
    "apply_transform",
    "convert",
    "convert_point",
    "distance_to_surface",
    "match_fiducial_name",
    "nearest_vertex",
]
