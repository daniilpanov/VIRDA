"""Import fiducials from JSON into typed models.

:func:`import_fiducials` reads a project fiducials JSON (or an MNE-style
``coordsystem.json``); :func:`load_manual_fiducials` wraps the
:mod:`virda.io.fiducial_helpers` reader for the manual input.
"""

from logging import Logger
from pathlib import Path

from virda.io.fiducial_helpers import load_fiducials
from virda.models.fiducial import Fiducials, ManualFiducials


def import_fiducials(path: str | Path) -> Fiducials:
    """Read a JSON file into a :class:`Fiducials` model.

    Accepts both the format written by :func:`virda.io.exporters.export_fiducials`
    and MNE-style ``coordsystem.json`` files (detected via
    :meth:`Coordsystem.is_coordsystem_dict`); the latter are converted via
    :meth:`Coordsystem.to_fiducials`.
    """
    return load_fiducials(Path(path))


def load_manual_fiducials(path: Path, *, logger: Logger | None = None) -> ManualFiducials:
    """Load a :class:`ManualFiducials` from a fiducials JSON file."""
    return ManualFiducials(fiducials=load_fiducials(path, logger))
