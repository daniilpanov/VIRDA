"""ESE configuration management."""

from __future__ import annotations

import json
from pathlib import Path

from .types import ESEConfigData


class ESEConfig:
    """Manages ESE configuration with validation and persistence.

    Parameters
    ----------
    offset_mm : float
        Distance from scalp to ESE surface in millimeters.
    reference_point : str
        What point on the electrode the ESE represents.
    description : str
        Optional human-readable description.
    """

    VALID_REFERENCE_POINTS = frozenset({
        "center_of_external_surface",
        "geometric_center",
    })

    def __init__(
        self,
        offset_mm: float = 5.0,
        reference_point: str = "center_of_external_surface",
        description: str = "",
    ) -> None:
        self._data = ESEConfigData(
            offset_mm=offset_mm,
            reference_point=reference_point,
            description=description,
        )
        self.validate()

    @property
    def offset_mm(self) -> float:
        return self._data.offset_mm

    @offset_mm.setter
    def offset_mm(self, value: float) -> None:
        self._data.offset_mm = value
        self.validate()

    @property
    def reference_point(self) -> str:
        return self._data.reference_point

    @property
    def description(self) -> str:
        return self._data.description

    def validate(self) -> bool:
        """Validate configuration parameters.

        Returns
        -------
        bool
            True if valid.

        Raises
        ------
        ValueError
            If offset is not positive or reference_point is unknown.
        """
        if self._data.offset_mm <= 0:
            raise ValueError(
                f"offset_mm must be positive, got {self._data.offset_mm}"
            )
        if self._data.reference_point not in self.VALID_REFERENCE_POINTS:
            raise ValueError(
                f"Unknown reference_point '{self._data.reference_point}'. "
                f"Must be one of {sorted(self.VALID_REFERENCE_POINTS)}"
            )
        return True

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "offset_mm": self._data.offset_mm,
            "reference_point": self._data.reference_point,
            "description": self._data.description,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ESEConfig:
        """Load configuration from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            offset_mm=data["offset_mm"],
            reference_point=data["reference_point"],
            description=data.get("description", ""),
        )

    def __repr__(self) -> str:
        return (
            f"ESEConfig(offset_mm={self.offset_mm}, "
            f"reference_point='{self.reference_point}')"
        )
