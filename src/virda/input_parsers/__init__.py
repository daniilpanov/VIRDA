"""File-path parsers for the VIRDA pipeline inputs.

Each parser takes a file path and returns the corresponding domain object,
delegating to the existing :mod:`virda.io` loaders and :mod:`virda.config`
builder rather than re-implementing any behavior.  The package is pure: it
never imports the GUI stack (``virda_gui`` / PySide6).
"""

from virda.input_parsers.config import parse_config
from virda.input_parsers.coordsystem import parse_coordsystem
from virda.input_parsers.electrodes_table import parse_electrodes_table
from virda.input_parsers.fiducials import parse_fiducials
from virda.input_parsers.measurements import parse_measurements
from virda.input_parsers.mri_volume import parse_mri_volume

__all__ = [
    "parse_config",
    "parse_coordsystem",
    "parse_electrodes_table",
    "parse_fiducials",
    "parse_measurements",
    "parse_mri_volume",
]