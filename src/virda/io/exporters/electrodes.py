"""Export Stage 3 electrodes as JSON files (full, scalp and ese variants)."""

import json
from pathlib import Path

import numpy as np

from virda.models.electrode import Electrodes


def export_electrodes(path: str | Path, electrodes: Electrodes) -> Path:
    """Write the full localization result to ``path`` as a JSON array.

    Every entry carries the electrode id, the measured fiducial distances,
    both the ESE body-center and scalp contact coordinates, the residual
    error, the confidence and the flag; non-localized electrodes carry
    ``null`` coordinates.  Returns the written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "electrode_id": electrode.electrode_id,
            "measured_distances": dict(electrode.measured_distances),
            "ese_coords": _to_list(electrode.ese_coords),
            "scalp_coords": _to_list(electrode.scalp_coords),
            "residual_error": electrode.residual_error,
            "confidence": electrode.confidence,
            "flagged": electrode.flagged,
        }
        for electrode in electrodes.items
    ]
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def export_electrodes_scalp(path: str | Path, electrodes: Electrodes) -> Path:
    """Write the scalp variant: ``coords`` = scalp contact points (RAS mm)."""
    return _export_coords_variant(path, electrodes, "scalp_coords")


def export_electrodes_ese(path: str | Path, electrodes: Electrodes) -> Path:
    """Write the ESE variant: ``coords`` = ESE body-center points (RAS mm)."""
    return _export_coords_variant(path, electrodes, "ese_coords")


def _export_coords_variant(path: str | Path, electrodes: Electrodes, attr: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "electrode_id": electrode.electrode_id,
            "coords": _to_list(getattr(electrode, attr)),
            "residual_error": electrode.residual_error,
            "confidence": electrode.confidence,
            "flagged": electrode.flagged,
        }
        for electrode in electrodes.items
    ]
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def _to_list(coords: np.ndarray | None) -> list[float] | None:
    return None if coords is None else np.asarray(coords, dtype=np.float64).tolist()
