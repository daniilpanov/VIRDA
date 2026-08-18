"""Pydantic model for MNE-style ``coordsystem.json`` input config files."""

from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from virda.models.fiducial import Fiducial, Fiducials

FIDUCIAL_ID_MAP: Final[dict[str, str]] = {
    "NASION": "NAS",
    "LPA": "LPA",
    "RPA": "RPA",
}


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
    source: str | None = Field(default=None, alias="Source")

    def to_fiducials(self) -> Fiducials:
        """Build :class:`Fiducials` from the MRI fiducial coordinates.

        Fiducials are recorded in MRI surface-RAS (mm), which is the world
        space the scalp mesh is extracted in, so they are marked as ``world``.
        """
        items = [
            Fiducial(
                fiducial_id=FIDUCIAL_ID_MAP.get(label, label),
                name=label,
                coordinates=np.asarray(coordinate.mri, dtype=np.float64),
                coordinate_system="world",
                definition_method="imported",
            )
            for label, coordinate in self.fiducials_coordinates.items()
        ]
        return Fiducials(items=items)
