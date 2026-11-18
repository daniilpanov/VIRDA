"""Validation of the stage-3 localization settings."""

import pytest

from virda.models.stage3_config import Stage3Config


def test_fiducial_tolerance_accepts_zero() -> None:
    assert Stage3Config(fiducial_tolerance_mm=0).fiducial_tolerance_mm == 0


def test_fiducial_tolerance_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        Stage3Config(fiducial_tolerance_mm=-0.1)
