import csv
import json
from logging import Logger
from pathlib import Path

import numpy as np

from virda.models.electrode import Electrodes
from virda.models.stage3_config import Stage3Config
from virda.pipeline import Provider


class Stage3Exporter(Provider[Electrodes]):
    """Export Stage 3 (localization) artifacts: electrodes, CSV, summary."""

    def __init__(
        self,
        project_dir: Path,
        stage3_config: Stage3Config,
        logger: Logger | None = None,
    ) -> None:
        self.project = Path(project_dir)
        (self.project / "localization").mkdir(parents=True, exist_ok=True)
        self._stage3_config = stage3_config
        self._logger = logger

    def provide(self, result: Electrodes | None) -> None:
        if not result:
            raise ValueError("There is no result of Stage#3")

        stage3_dir = self.project / "localization"
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
                        "residual_error": (
                            electrode.residual_error if electrode.is_localized else ""
                        ),
                        "confidence": (electrode.confidence if electrode.is_localized else ""),
                        "flagged": electrode.flagged,
                    }
                )

        residuals = [
            electrode.residual_error
            for electrode in electrodes
            if electrode.residual_error is not None
        ]
        (stage3_dir / "localization_summary.json").write_text(
            json.dumps(
                {
                    "n_electrodes": len(electrodes),
                    "n_localized": sum(1 for electrode in electrodes if electrode.is_localized),
                    "n_flagged": sum(1 for electrode in electrodes if electrode.flagged),
                    "median_residual_mm": float(np.median(residuals)) if residuals else None,
                    "residual_threshold_mm": self._stage3_config.residual_threshold_mm,
                    "calibrated_ese_offset_shift_mm": result.calibrated_offset_shift_mm,
                },
                indent=2,
            )
        )


def _to_list(coords: np.ndarray | None) -> list[float] | None:
    if coords is None:
        return None
    return [float(value) for value in coords]


def _coordinate(coords: np.ndarray | None, axis: int) -> float | str:
    if coords is None:
        return ""
    return float(coords[axis])
