"""Fiducial auto-detection and pipeline steps."""

from virda.fiducials.detect import find_fiducials, to_fiducials
from virda.fiducials.detector import AutoFiducialsDetector

__all__ = ["AutoFiducialsDetector", "find_fiducials", "to_fiducials"]
