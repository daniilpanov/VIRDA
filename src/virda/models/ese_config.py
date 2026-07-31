from dataclasses import dataclass


@dataclass(frozen=True)
class ESEConfig:
    n_electrodes: int = 67
    ese_offset_mm: float = 5.0
    ese_reference: str = "electrode_external_surface"

    def __post_init__(self) -> None:
        if self.n_electrodes <= 0:
            raise ValueError(f"n_electrodes must be positive, got {self.n_electrodes}")
        if self.ese_offset_mm <= 0:
            raise ValueError(f"ese_offset_mm must be positive, got {self.ese_offset_mm}")
