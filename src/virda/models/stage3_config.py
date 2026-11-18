from dataclasses import dataclass


@dataclass(frozen=True)
class Stage3Config:
    """The settings of the electrode localization step."""

    residual_threshold_mm: float = 10.0
    calibrate_ese_offset: bool = True
    fiducial_tolerance_mm: float = 2.0

    def __post_init__(self) -> None:
        if self.residual_threshold_mm <= 0:
            raise ValueError(
                f"residual_threshold_mm must be positive, got {self.residual_threshold_mm}"
            )
        if not self.fiducial_tolerance_mm >= 0:
            raise ValueError(
                f"fiducial_tolerance_mm must be non-negative, got {self.fiducial_tolerance_mm}"
            )
