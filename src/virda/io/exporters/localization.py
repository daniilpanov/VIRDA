"""Export the Stage 3 (localization) artifacts into a patient project directory."""

import csv
import json
from pathlib import Path

import numpy as np

from virda.models.electrode import Electrodes


def export_localization(result: Electrodes, project_dir: Path) -> None:
    """Write the electrode JSON files and a CSV table below ``project_dir``."""
    stage3_dir = Path(project_dir) / "localization"
    stage3_dir.mkdir(parents=True, exist_ok=True)
    electrodes = result.items

    (stage3_dir / "electrodes.json").write_text(
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
                for electrode in electrodes
            ],
            indent=2,
        )
    )

    for suffix, attr in [("scalp", "scalp_coords"), ("ese", "ese_coords")]:
        (stage3_dir / f"electrodes_{suffix}.json").write_text(
            json.dumps(
                [
                    {
                        "electrode_id": electrode.electrode_id,
                        "coords": _to_list(getattr(electrode, attr)),
                        "residual_error": electrode.residual_error,
                        "confidence": electrode.confidence,
                        "flagged": electrode.flagged,
                    }
                    for electrode in electrodes
                ],
                indent=2,
            )
        )

    with (stage3_dir / "electrode_coords.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
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
        for electrode in electrodes:
            writer.writerow(
                {
                    "electrode_id": electrode.electrode_id,
                    "x": _coordinate(electrode.ese_coords, 0),
                    "y": _coordinate(electrode.ese_coords, 1),
                    "z": _coordinate(electrode.ese_coords, 2),
                    "residual_error": (electrode.residual_error if electrode.is_localized else ""),
                    "confidence": (electrode.confidence if electrode.is_localized else ""),
                    "flagged": electrode.flagged,
                }
            )


def _to_list(coords: np.ndarray | None) -> list[float] | None:
    if coords is None:
        return None
    return [float(value) for value in coords]


def _coordinate(coords: np.ndarray | None, axis: int) -> float | str:
    if coords is None:
        return ""
    return float(coords[axis])
