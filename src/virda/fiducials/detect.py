"""Geometric fiducial auto-detection on a scalp mesh (world RAS, mm).

Heuristics:
  * NAS  - most anterior midline point above the nose bridge (~glabella/nasion).
  * LPA  - most lateral vertex on the left side, at ear-canal height.
  * RPA  - same on the right.
  * INI  - most posterior midline point in the occipital band.

These are approximations for QC/automation. Clinical use requires manual
verification of the detected points.
"""

from collections.abc import Mapping
from typing import cast

import numpy as np

from virda.models.fiducial import Fiducial

MIDLINE_TOLERANCE_MM = 3.0
NOSE_RISE_MM = 8.0
NOSE_BAND_SPAN_MM = 25.0
EAR_Z_ABOVE_MM = 6.0
EAR_Z_BELOW_MM = 6.0
EAR_Y_OFFSET_MM = 72.0
EAR_Y_HALFSPAN_MM = 16.0
INI_Z_ABOVE_MM = -5.0
INI_Z_BELOW_MM = 40.0

DEFAULT_NAMES: dict[str, str] = {
    "NAS": "Nasion",
    "LPA": "Left pre-auricular",
    "RPA": "Right pre-auricular",
    "INI": "Inion",
}


def _midline(vertices: np.ndarray) -> np.ndarray:
    return np.abs(vertices[:, 0]) <= MIDLINE_TOLERANCE_MM


def find_nose_tip(vertices: np.ndarray) -> np.ndarray:
    midline = vertices[_midline(vertices)]
    return cast(np.ndarray, midline[np.argmax(midline[:, 1])])


def find_nasion(vertices: np.ndarray, nose_tip: np.ndarray) -> np.ndarray:
    midline = _midline(vertices)
    z_lo = nose_tip[2] + NOSE_RISE_MM
    z_hi = nose_tip[2] + NOSE_RISE_MM + NOSE_BAND_SPAN_MM
    band = vertices[midline & (vertices[:, 2] >= z_lo) & (vertices[:, 2] <= z_hi)]
    if len(band) == 0:
        raise ValueError("Nasion band is empty; check nose-tip detection")
    z_slices = np.arange(np.floor(band[:, 2].min()), np.ceil(band[:, 2].max()) + 1.0, 1.0)
    anterior: list[np.ndarray] = []
    for z in z_slices:
        slice_mask = np.abs(band[:, 2] - z) <= 0.5
        if not slice_mask.any():
            continue
        row = band[slice_mask]
        row = row[row[:, 1] >= 0]
        if len(row) == 0:
            continue
        anterior.append(cast(np.ndarray, row[np.argmax(row[:, 1])]))
    envelope = np.asarray(anterior)
    return cast(np.ndarray, envelope[np.argmin(envelope[:, 1])])


def find_ears(vertices: np.ndarray, nasion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_lo = nasion[2] - EAR_Z_BELOW_MM
    z_hi = nasion[2] + EAR_Z_ABOVE_MM
    y_center = nasion[1] - EAR_Y_OFFSET_MM
    region = (
        (vertices[:, 2] >= z_lo)
        & (vertices[:, 2] <= z_hi)
        & (np.abs(vertices[:, 1] - y_center) <= EAR_Y_HALFSPAN_MM)
    )
    candidates = vertices[region]
    if len(candidates) == 0:
        raise ValueError("Ear band is empty; check nasion detection")
    left = cast(np.ndarray, candidates[np.argmin(candidates[:, 0])])
    right = cast(np.ndarray, candidates[np.argmax(candidates[:, 0])])
    return left, right


def find_inion(vertices: np.ndarray, nasion: np.ndarray) -> np.ndarray:
    midline = _midline(vertices)
    z_lo = nasion[2] - INI_Z_BELOW_MM
    z_hi = nasion[2] - INI_Z_ABOVE_MM
    band = vertices[midline & (vertices[:, 2] >= z_lo) & (vertices[:, 2] <= z_hi)]
    if len(band) == 0:
        raise ValueError("Inion band is empty; check nasion detection")
    return cast(np.ndarray, band[np.argmin(band[:, 1])])


def find_fiducials(vertices: np.ndarray) -> dict[str, np.ndarray]:
    nose_tip = find_nose_tip(vertices)
    nasion = find_nasion(vertices, nose_tip)
    lpa, rpa = find_ears(vertices, nasion)
    inion = find_inion(vertices, nasion)
    return {
        "NAS": nasion,
        "LPA": lpa,
        "RPA": rpa,
        "INI": inion,
    }


def to_fiducials(
    points: Mapping[str, np.ndarray], names: Mapping[str, str] | None = None
) -> list[Fiducial]:
    names = names or DEFAULT_NAMES
    return [
        Fiducial(
            fiducial_id=key,
            name=names[key],
            coordinates=point.astype(np.float64),
            coordinate_system="world",
            definition_method="auto",
        )
        for key, point in points.items()
    ]
