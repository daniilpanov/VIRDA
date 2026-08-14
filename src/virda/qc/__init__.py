"""Automatic quality-control checks for a Stage 1 result (spec §13.1, §16)."""

from virda.qc.checks import (
    Stage1QualityControlStep,
    check_components,
    check_coordinates_mm,
    check_ese_config,
    check_fiducials,
    check_holes,
    check_mesh,
    check_mri,
    check_nifti_mask,
    run_checks,
)

__all__ = [
    "Stage1QualityControlStep",
    "check_components",
    "check_coordinates_mm",
    "check_ese_config",
    "check_fiducials",
    "check_holes",
    "check_mesh",
    "check_mri",
    "check_nifti_mask",
    "run_checks",
]
