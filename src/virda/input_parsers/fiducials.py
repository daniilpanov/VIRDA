"""Parser for fiducials files."""

from pathlib import Path

from virda.io.fiducial_helpers import load_fiducials
from virda.models.fiducial import Fiducials


def parse_fiducials(path: str | Path) -> Fiducials:
    """Load a fiducials JSON file into a :class:`Fiducials` model.

    Delegates to :func:`virda.io.fiducial_helpers.load_fiducials`, which also
    accepts MNE-style ``coordsystem.json`` files and converts them to
    fiducials.
    """
    return load_fiducials(Path(path))