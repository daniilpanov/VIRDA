from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Electrode:
    electrode_id: str | None = None
    measured_distances: dict[str, float] = field(default_factory=dict)
    ese_coords: np.ndarray | None = None
    scalp_coords: np.ndarray | None = None
    residual_error: float | None = None
    confidence: float | None = None
    flagged: bool = False

    def __post_init__(self) -> None:
        if self.electrode_id is not None and not self.electrode_id:
            raise ValueError("electrode_id must be None or a non-empty string")
        if not self.measured_distances:
            raise ValueError("measured_distances must contain at least one measurement")
        for fiducial_id, distance in self.measured_distances.items():
            if not fiducial_id:
                raise ValueError("fiducial_id must be a non-empty string")
            if distance <= 0:
                raise ValueError(
                    f"measured distance to {fiducial_id!r} must be positive, got {distance}"
                )
        for name, coords in (("ese_coords", self.ese_coords), ("scalp_coords", self.scalp_coords)):
            if coords is not None and coords.shape != (3,):
                raise ValueError(f"{name} must be a (3,) array, got shape {coords.shape}")

    @property
    def is_localized(self) -> bool:
        return self.ese_coords is not None and self.scalp_coords is not None


@dataclass(frozen=True)
class Electrodes:
    items: list[Electrode]

    def __post_init__(self) -> None:
        ids = [electrode.electrode_id for electrode in self.items if electrode.electrode_id]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Electrode ids must be unique, got {ids}")

    @property
    def ids(self) -> list[str | None]:
        return [electrode.electrode_id for electrode in self.items]

    def get(self, electrode_id: str | None) -> Electrode | None:
        for electrode in self.items:
            if electrode.electrode_id == electrode_id:
                return electrode
        return None
