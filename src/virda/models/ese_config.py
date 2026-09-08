from dataclasses import dataclass

ESE_REFERENCE_OPTIONS = ("electrode_capsule_center", "electrode_body_center")


@dataclass(frozen=True)
class ESEConfig:
    ese_offset_mm: float
    ese_reference: str

    def __post_init__(self) -> None:
        if self.ese_offset_mm <= 0:
            raise ValueError(f"ese_offset_mm must be positive, got {self.ese_offset_mm}")
        if self.ese_reference not in ESE_REFERENCE_OPTIONS:
            raise ValueError(
                f"ese_reference must be one of {ESE_REFERENCE_OPTIONS}, got {self.ese_reference!r}"
            )
