"""Import and manage electrode-to-fiducial distance measurements."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class MeasurementImporter:
    """Manages measured distances from electrode centers to fiducials.

    Parameters
    ----------
    fiducial_ids : list[str]
        List of fiducial identifiers (e.g. ['NAS', 'LPA', 'RPA']).
    """

    def __init__(self, fiducial_ids: list[str]) -> None:
        self.fiducial_ids = list(fiducial_ids)
        self._measurements: dict[str, dict[str, float]] = {}

    def add_measurement(
        self,
        electrode_id: str,
        distances: dict[str, float],
    ) -> None:
        """Add a distance measurement for one electrode.

        Parameters
        ----------
        electrode_id : str
            Unique electrode identifier.
        distances : dict[str, float]
            Mapping of fiducial_id -> distance in mm.

        Raises
        ------
        ValueError
            If a fiducial_id in distances is not in the configured list.
        """
        unknown = set(distances.keys()) - set(self.fiducial_ids)
        if unknown:
            raise ValueError(
                f"Unknown fiducial IDs: {unknown}. "
                f"Expected one of {self.fiducial_ids}"
            )
        self._measurements[electrode_id] = dict(distances)
        logger.info(
            "Added measurement for %s: %s",
            electrode_id,
            distances,
        )

    def get_measurement(self, electrode_id: str) -> dict[str, float]:
        """Get distance measurement for one electrode."""
        return self._measurements[electrode_id]

    def get_all_measurements(self) -> dict[str, dict[str, float]]:
        """Return all measurements."""
        return dict(self._measurements)

    def import_csv(self, path: Path) -> None:
        """Import measurements from CSV.

        Expected format::

            electrode_id,NAS,LPA,RPA
            E1,45.2,38.7,42.1
        """
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            eid = str(row["electrode_id"])
            dists = {fid: float(row[fid]) for fid in self.fiducial_ids}
            self._measurements[eid] = dists
        logger.info("Imported %d measurements from %s", len(df), path)

    def import_json(self, path: Path) -> None:
        """Import measurements from JSON.

        Expected format::

            {"E1": {"NAS": 45.2, "LPA": 38.7, "RPA": 42.1}, ...}
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for eid, dists in data.items():
            self._measurements[str(eid)] = {
                str(k): float(v) for k, v in dists.items()
            }
        logger.info("Imported %d measurements from %s", len(data), path)

    def export_csv(self, path: Path) -> None:
        """Export measurements to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for eid, dists in sorted(self._measurements.items()):
            row = {"electrode_id": eid}
            row.update(dists)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("Exported %d measurements to %s", len(rows), path)

    def export_json(self, path: Path) -> None:
        """Export measurements to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._measurements, indent=2),
            encoding="utf-8",
        )
        logger.info("Exported %d measurements to %s", len(self._measurements), path)
