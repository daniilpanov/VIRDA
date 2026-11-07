import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Any

import numpy as np


@dataclass(frozen=True)
class LabeledPoints:
    """An (N, 3) point set with an optional label per point.

    The universal representation of "points drawn over a mesh or a volume":
    fiducials, electrode contact points, and so on.  ``labels`` may be omitted
    for unnamed point sets.
    """

    points: np.ndarray
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must be (N, 3) arrays, got shape {points.shape}")

        labels = self.labels
        if labels is not None:
            labels = tuple(str(label) for label in labels)
            if len(labels) != len(points):
                raise ValueError(
                    f"Got {len(labels)} labels for {len(points)} points; counts must match"
                )

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "labels", labels)

    def __len__(self) -> int:
        return len(self.points)

    @classmethod
    def from_arrays(cls, points: np.ndarray, labels: Sequence[str] | None = None) -> LabeledPoints:
        return cls(points, tuple(labels) if labels is not None else None)

    @classmethod
    def from_json(cls, path: str | Path) -> LabeledPoints:
        """Read a labeled point set from a JSON file.

        Accepts either ``{"points": [...], "labels": [...]}`` or a list of
        records whose coordinates (name) live under common keys such as
        ``coords`` / ``coordinates`` / ``position`` and ``electrode_id`` /
        ``name`` / ``label`` (VIRDA ``electrodes_*.json`` files are covered by
        this).
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "points" in data:
            labels = data.get("labels")
            points = np.asarray(data["points"], dtype=np.float64)
            if labels is None and isinstance(data.get("names"), list):
                labels = data["names"]
            return cls(points, tuple(labels) if labels is not None else None)
        if isinstance(data, dict):
            items = data.get("records") or data.get("items") or data.get("electrodes")
            if items is None:
                raise ValueError(f"Unsupported point JSON structure in {path}")
            return cls._from_records(items)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return cls._from_records(data)
        if isinstance(data, list):
            return cls(np.asarray(data, dtype=np.float64))
        raise ValueError(f"Unsupported point JSON structure in {path}")

    @classmethod
    def _from_records(cls, records: list[Any]) -> LabeledPoints:
        points: list[np.ndarray] = []
        labels: list[str] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict) or record.get("coords") is None:
                coords = None
                for key in _COORD_KEYS:
                    if key in record:
                        coords = record[key]
                        break
                if coords is None:
                    raise ValueError(f"Record {index} has no coordinates")
                record = {"coords": coords, **record}
            name = None
            for key in _NAME_KEYS:
                if record.get(key):
                    name = record[key]
                    break
            points.append(np.asarray(record["coords"], dtype=np.float64))
            labels.append(str(name) if name is not None else f"P{index + 1:03d}")
        return cls(np.asarray(points) if points else np.empty((0, 3)), tuple(labels))

    @classmethod
    def from_table(cls, path: str | Path) -> LabeledPoints:
        """Read a labeled point set from a CSV/TSV table with ``name, x, y, z``."""
        with open(path, encoding="utf-8") as fh:
            sample = fh.read(2048)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error as exc:
                raise ValueError(f"Cannot parse delimiter of {path}: {exc}") from None
            fh.seek(0)
            reader = csv.DictReader(fh, dialect=dialect)
            if not reader.fieldnames:
                raise ValueError(f"Point table is empty or has no header: {path}")
            columns = {name.lower().strip(): name for name in reader.fieldnames}
            for required in ("x", "y", "z"):
                if required not in columns:
                    raise ValueError(
                        f"Point table missing required column '{required}', "
                        f"found: {list(reader.fieldnames)}"
                    )
            name_col = columns.get("name")
            points: list[np.ndarray] = []
            labels: list[str] = []
            for index, row in enumerate(reader):
                points.append(
                    np.array(
                        [
                            float(row[columns["x"]]),
                            float(row[columns["y"]]),
                            float(row[columns["z"]]),
                        ]
                    )
                )
                raw_name = row.get(name_col) if name_col is not None else None
                labels.append(str(raw_name).strip() if raw_name else f"P{index + 1:03d}")
        return cls(np.asarray(points) if points else np.empty((0, 3)), tuple(labels))
