from dataclasses import dataclass


@dataclass(frozen=True)
class Stage3Config:
    residual_threshold_mm: float = 10.0
    calibrate_ese_offset: bool = True

    def __post_init__(self) -> None:
        if self.residual_threshold_mm <= 0:
            raise ValueError(
                f"residual_threshold_mm must be positive, got {self.residual_threshold_mm}"
            )
