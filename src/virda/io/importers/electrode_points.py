"""Import a localized-electrode point CSV table into point rows.

Reads the point-variant tables the pipeline writes next to the full
localization result -- ``electrodes_scalp.csv`` / ``electrodes_ese.csv``
(written by :func:`virda.io.exporters.electrodes.export_electrodes_scalp`
and :func:`virda.io.exporters.electrodes.export_electrodes_ese`) -- whose
rows carry the electrode id and one coordinate triple each.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class ElectrodePoint:
    """One localized-electrode point row of a coords CSV table."""

    electrode_id: str
    coordinates: np.ndarray  # (3,) float64
    residual_mm: float | None = None
    flagged: bool = False


def import_electrode_points(path: str | Path) -> list[ElectrodePoint]:
    """Read the ``electrode_id, x, y, z, residual_mm, flagged`` table at *path*.

    Column names are matched case-insensitively and extra columns are
    ignored. Rows without a coordinate triple are skipped (they mark
    non-localized electrodes); a file with no coordinate rows at all, or
    without the coordinate columns, raises ``ValueError``.
    """
    target = Path(path)
    with target.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = _column_map(list(reader.fieldnames or []), target)
        points = [
            point
            for index, row in enumerate(reader)
            if (point := _point(row, index, columns)) is not None
        ]
    if not points:
        raise ValueError(f"no electrode points found in file: {path}")
    return points


def _column_map(fieldnames: list[str], path: Path) -> dict[str, str]:
    """Map the lowercase column names to the file's actual header names."""
    columns = {name.lower().strip(): name for name in fieldnames}
    missing = [axis for axis in _AXES if axis not in columns]
    if not fieldnames or missing:
        raise ValueError(
            f"Electrode point file {path} must carry the x, y, z columns; found: {fieldnames}"
        )
    return columns


def _point(row: dict[str, str], index: int, columns: dict[str, str]) -> ElectrodePoint | None:
    """One point row, or ``None`` when the row carries no coordinates."""
    raw = [row.get(columns[axis]) for axis in _AXES]
    if any(value is None or not value.strip() for value in raw):
        return None
    return ElectrodePoint(
        electrode_id=_electrode_id(row, index, columns),
        coordinates=np.asarray(
            [_coordinate(row[columns[axis]], axis, index) for axis in _AXES],
            dtype=np.float64,
        ),
        residual_mm=_residual(row.get(columns.get("residual_mm", ""))),
        flagged=str(row.get(columns.get("flagged", ""), "")).strip().lower() == "yes",
    )


def _electrode_id(row: dict[str, str], index: int, columns: dict[str, str]) -> str:
    """The row's electrode id, or a sequential fallback when empty."""
    raw = row.get(columns.get("electrode_id", ""))
    return str(raw).strip() if raw and raw.strip() else f"E{index + 1:03d}"


def _coordinate(value: str, axis: str, index: int) -> float:
    """Parse one coordinate; the value is already known to be non-empty."""
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"row {index}: invalid {axis} coordinate {value!r}") from error


def _residual(value: str | None) -> float | None:
    """Parse the residual column; empty cells mean no residual."""
    if value is None or not value.strip():
        return None
    return float(value)
