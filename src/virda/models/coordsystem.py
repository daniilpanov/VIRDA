"""Pydantic model for MNE-style ``coordsystem.json`` input config files."""

from typing import Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from virda.models.fiducial import Fiducial, Fiducials

FIDUCIAL_ID_MAP: Final[dict[str, str]] = {
    "NASION": "NAS",
    "LPA": "LPA",
    "RPA": "RPA",
}

#: Conversion factors to millimetres for supported ``*CoordinateUnits`` values.
_MM_PER_UNIT: Final[dict[str, float]] = {"mm": 1.0, "m": 1000.0}

#: Coordinate frames whose MRI coordinates are FreeSurfer surface-RAS.
_SURFACE_RAS_FRAMES: Final[set[str]] = {"ras", "surface ras", "fsnative"}


class FiducialCoordinate(BaseModel):
    """One fiducial point in both digitization and MRI coordinates."""

    model_config = ConfigDict(populate_by_name=True)

    head: tuple[float, float, float] = Field(alias="Head")
    mri: tuple[float, float, float] = Field(alias="MRI")


class Coordsystem(BaseModel):
    """Contents of an MNE ``coordsystem.json`` file.

    The file describes the digitization space of an electrode layout: the
    coordinate systems in use, the fiducial positions and the electrode count.
    It is loaded as an input config file and merged into the pipeline ``Config``.
    """

    model_config = ConfigDict(populate_by_name=True)

    coordinate_system: str | None = Field(default=None, alias="CoordinateSystem")
    coordinate_units: str | None = Field(default=None, alias="CoordinateUnits")
    coordinate_system_description: str | None = Field(
        default=None, alias="CoordinateSystemDescription"
    )
    eeg_coordinate_system: str | None = Field(default=None, alias="EEGCoordinateSystem")
    eeg_coordinate_units: str | None = Field(default=None, alias="EEGCoordinateUnits")
    fiducials_coordinates: dict[str, FiducialCoordinate] = Field(
        default_factory=dict, alias="FiducialsCoordinates"
    )
    electrode_count: int | None = Field(default=None, alias="ElectrodeCount")
    electrode_offset_mm: float | None = Field(default=None, alias="ElectrodeOffset")
    electrode_reference: str | None = Field(default=None, alias="ElectrodeReference")
    source: str | None = Field(default=None, alias="Source")

    @classmethod
    def is_coordsystem_dict(cls, data: dict[str, Any]) -> bool:
        """Return True if *data* looks like an MNE ``coordsystem.json`` mapping.

        Files written by MNE-BIDS always carry ``FiducialsCoordinates`` when
        fiducials are present, but ``CoordinateSystem``-only files (no
        digitized fiducials) are also valid coordsystem documents.
        """
        return "FiducialsCoordinates" in data or "CoordinateSystem" in data

    def mri_unit_scale_mm(self) -> float:
        """Return the factor that converts MRI coordinates to millimetres."""
        units = self.coordinate_units or self.eeg_coordinate_units or "m"
        scale = _MM_PER_UNIT.get(units.strip().lower())
        if scale is None:
            raise ValueError(f"Unsupported coordinate units {units!r}: expected 'mm' or 'm'")
        return scale

    def to_fiducials(self) -> Fiducials:
        """Build :class:`Fiducials` from the MRI fiducial coordinates.

        Fiducials are recorded in MRI surface-RAS converted to millimetres,
        which is the world space the scalp mesh is extracted in, so they are
        marked as ``world``.  Coordinates given in metres (the BIDS default)
        are scaled automatically; any other unit is rejected.
        """
        unit_scale = self.mri_unit_scale_mm()
        items = [
            Fiducial(
                fiducial_id=FIDUCIAL_ID_MAP.get(label, label),
                name=label,
                coordinates=np.asarray(coordinate.mri, dtype=np.float64) * unit_scale,
                coordinate_system="world",
                definition_method="imported",
            )
            for label, coordinate in self.fiducials_coordinates.items()
        ]
        return Fiducials(items=items)
