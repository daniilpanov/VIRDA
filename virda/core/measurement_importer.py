"""Measurement importer — load electrode distance measurements."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ElectrodeMeasurement:
    """Distance measurements for one electrode."""

    electrode_id: str
    distances: dict[str, float]
    inter_electrode_distances: Optional[dict[str, float]] = None
    notes: str = ""

    @property
    def fiducial_ids(self) -> list[str]:
        return list(self.distances.keys())

    @property
    def measured_values(self) -> list[float]:
        return list(self.distances.values())


class MeasurementImporter:
    """Import measured distances from electrodes to fiducials."""

    def __init__(self, fiducial_ids: list[str]):
        self.fiducial_ids = list(fiducial_ids)
        self.measurements: dict[str, ElectrodeMeasurement] = {}

    def add_measurement(
        self,
        electrode_id: str,
        distances: dict[str, float],
        inter_electrode_distances: Optional[dict[str, float]] = None,
        notes: str = "",
    ) -> ElectrodeMeasurement:
        """Add measurement for one electrode."""
        meas = ElectrodeMeasurement(
            electrode_id=electrode_id,
            distances=distances,
            inter_electrode_distances=inter_electrode_distances,
            notes=notes,
        )
        self.measurements[electrode_id] = meas
        return meas

    def get_measurement(self, electrode_id: str) -> Optional[ElectrodeMeasurement]:
        """Get measurement for one electrode."""
        return self.measurements.get(electrode_id)

    def get_all_measurements(self) -> dict[str, ElectrodeMeasurement]:
        """Return all measurements."""
        return dict(self.measurements)

    def get_distance_matrix(
        self, electrode_ids: Optional[list[str]] = None
    ) -> tuple[list[str], np.ndarray]:
        """Get distances as a matrix."""
        if electrode_ids is None:
            electrode_ids = list(self.measurements.keys())

        rows = []
        for eid in electrode_ids:
            if eid not in self.measurements:
                continue
            meas = self.measurements[eid]
            row = []
            for fid in self.fiducial_ids:
                row.append(meas.distances.get(fid, np.nan))
            rows.append(row)

        return electrode_ids, np.array(rows, dtype=np.float64)

    def import_csv(self, path: str | Path) -> None:
        """Import measurements from CSV file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            fid_cols = [c for c in header if c != "electrode_id"]

            if not self.fiducial_ids:
                self.fiducial_ids = fid_cols

            for row in reader:
                eid = row["electrode_id"]
                distances = {}
                for fc in fid_cols:
                    val = row.get(fc, "")
                    if val:
                        distances[fc] = float(val)
                self.add_measurement(eid, distances)

        logger.info(
            "Imported %d electrode measurements from %s",
            len(self.measurements),
            path,
        )

    def import_json(self, path: str | Path) -> None:
        """Import measurements from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "fiducial_ids" in data:
            self.fiducial_ids = data["fiducial_ids"]

        for eid, dists in data.get("measurements", {}).items():
            self.add_measurement(eid, dists)

        logger.info(
            "Imported %d measurements from %s (fiducials=%s)",
            len(self.measurements),
            path,
            self.fiducial_ids,
        )

    def save_csv(self, path: str | Path) -> None:
        """Export measurements to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["electrode_id"] + self.fiducial_ids)
            for eid, meas in self.measurements.items():
                row = [eid] + [
                    str(meas.distances.get(fid, "")) for fid in self.fiducial_ids
                ]
                writer.writerow(row)

        logger.info("Saved measurements to %s", path)

    def save_json(self, path: str | Path) -> None:
        """Export measurements to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "fiducial_ids": self.fiducial_ids,
            "measurements": {
                eid: meas.distances for eid, meas in self.measurements.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved measurements to %s", path)
