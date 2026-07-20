"""ESE configuration — reference definition and offset parameter."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ESEConfig:
    """Electrode Surface Equivalent configuration."""

    offset_mm: float = 5.0
    reference_point: str = "center_of_external_surface"
    description: str = (
        "The ESE represents the expected location of the electrode external "
        "reference surface center, offset outward from the scalp by the "
        "configured distance."
    )

    def __post_init__(self):
        if self.offset_mm <= 0:
            raise ValueError(f"Offset must be positive, got {self.offset_mm}")

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        logger.info("Saved ESE config to %s (offset=%.2f mm)", path, self.offset_mm)

    @classmethod
    def load(cls, path: str | Path) -> ESEConfig:
        """Load configuration from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
