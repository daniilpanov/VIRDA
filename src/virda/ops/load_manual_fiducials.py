"""Step: load manual fiducials from a JSON file."""

from logging import Logger
from pathlib import Path

from virda.io.fiducial_helpers import load_fiducials
from virda.models.fiducial import ManualFiducials


def load_manual_fiducials(path: Path, *, logger: Logger | None = None) -> ManualFiducials:
    """Load a :class:`ManualFiducials` store from a fiducials JSON file."""
    return ManualFiducials(fiducials=load_fiducials(path, logger))
