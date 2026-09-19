"""Export localized electrodes to JSON and CSV artifacts."""

import csv
import json
from pathlib import Path

import numpy as np

from virda.models.electrode import Electrodes


def export_electrodes(
    path: str | Path,
    electrodes: Electrodes,
    *,
    include_csv: bool = True,
) -> Path:
    """Write *electrodes* to ``path`` as JSON; returns the written path.

    The JSON carries the full localized electrode state (positions in both
    scalp and ESE frame, residuals, confidence and flags).  When
    ``include_csv`` is set, a sibling CSV with the ESE coordinates is written
    alongside the JSON.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            [
                {
                    "electrode_id": electrode.electrode_id,
                    "measured_distances": electrode.measured_distances,
                    "ese_coords": _to_list(electrode.ese_coords),
                    "scalp_coords": _to_list(electrode.scalp_coords),
                    "residual_error": electrode.residual_error,
                    "confidence": electrode.confidence,
                    "flagged": electrode.flagged,
                }
                for electrode in electrodes.items
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    if include_csv:
        csv_path = target.with_suffix(".csv")
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "electrode_id",
                    "x",
                    "y",
                    "z",
                    "residual_error",
                    "confidence",
                    "flagged",
                ],
            )
            writer.writeheader()
            for electrode in electrodes.items:
                writer.writerow(
                    {
                        "electrode_id": electrode.electrode_id,
                        "x": _coordinate(electrode.ese_coords, 0),
                        "y": _coordinate(electrode.ese_coords, 1),
                        "z": _coordinate(electrode.ese_coords, 2),
                        "residual_error": (
                            electrode.residual_error if electrode.is_localized else ""
                        ),
                        "confidence": (electrode.confidence if electrode.is_localized else ""),
                        "flagged": electrode.flagged,
                    }
                )

    return target


def _to_list(coords: np.ndarray | None) -> list[float] | None:
    if coords is None:
        return None
    return [float(value) for value in coords]


def _coordinate(coords: np.ndarray | None, axis: int) -> float | str:
    if coords is None:
        return ""
    return float(coords[axis])