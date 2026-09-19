"""Load a tabular (TSV/CSV) electrode table into plain arrays.

Mirrors the column convention of the viewer's tabular electrode loader
(``name``/``x``/``y``/``z``, delimiter auto-detection) in the pure ``virda``
layer so headless callers get the same result without importing ``virda_gui``.
"""

import csv
from pathlib import Path
from typing import TypeAlias

import numpy as np

ElectrodesTable: TypeAlias = tuple[
    np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]], list[str]
]


def load_electrodes_table(path: str | Path) -> ElectrodesTable:
    """Read a TSV/CSV electrode table with ``name``, ``x``, ``y``, ``z`` columns.

    Column names are matched case-insensitively and the delimiter is detected
    automatically by :class:`csv.Sniffer`.  Returns ``(positions, residuals,
    flags, measured, names)``: tabular electrode files carry no distance
    information, so residuals are zero, nothing is flagged and ``measured`` is
    a list of empty dictionaries, mirroring the GUI loader's result shape.
    """
    with Path(path).open(encoding="utf-8") as fh:
        sample = fh.read(2048)
        dialect = csv.Sniffer().sniff(sample)
        fh.seek(0)
        reader = csv.DictReader(fh, dialect=dialect)

        if not reader.fieldnames:
            raise ValueError(f"Electrodes file is empty or has no header: {path}")

        col_map = {col.lower().strip(): col for col in reader.fieldnames}

        for required in ("x", "y", "z"):
            if required not in col_map:
                raise ValueError(
                    f"Electrodes file missing required column '{required}', "
                    f"found: {list(reader.fieldnames)}"
                )

        name_col = col_map.get("name")
        points: list[np.ndarray] = []
        names: list[str] = []
        for index, row in enumerate(reader):
            x = float(row[col_map["x"]])
            y = float(row[col_map["y"]])
            z = float(row[col_map["z"]])
            points.append(np.array([x, y, z]))
            raw_name = row.get(name_col) if name_col is not None else None
            names.append(str(raw_name).strip() if raw_name else f"E{index + 1:03d}")

    if not points:
        return np.empty((0, 3)), np.empty(0), np.empty(0, dtype=bool), [], []
    positions = np.asarray(points)
    return (
        positions,
        np.zeros(len(points)),
        np.zeros(len(points), dtype=bool),
        [{} for _ in points],
        names,
    )
