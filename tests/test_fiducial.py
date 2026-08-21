import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from tests.helpers.pipelines import make_fiducials
from virda.io.fiducial_helpers import load_fiducials, save_fiducials
from virda.models.fiducial import (
    CoordinateSystem,
    DefinitionMethod,
    Fiducial,
    Fiducials,
)


class TestFiducialModel:
    def test_fiducial_validates_coordinates_shape(self) -> None:
        with pytest.raises(ValueError, match=r"Coordinates must be \(3,\)"):
            Fiducial(
                fiducial_id="NAS",
                name="Nasion",
                coordinates=np.array([0.0, 88.0]),
                coordinate_system="world",
            )

    def test_fiducial_validates_coordinate_system(self) -> None:
        with pytest.raises(ValueError, match="Invalid coordinate_system"):
            Fiducial(
                fiducial_id="NAS",
                name="Nasion",
                coordinates=np.array([0.0, 88.0, -10.0]),
                coordinate_system=cast(CoordinateSystem, "nonsense"),
            )

    def test_fiducial_validates_definition_method(self) -> None:
        with pytest.raises(ValueError, match="Invalid definition_method"):
            Fiducial(
                fiducial_id="NAS",
                name="Nasion",
                coordinates=np.array([0.0, 88.0, -10.0]),
                coordinate_system="world",
                definition_method=cast(DefinitionMethod, "guessed"),
            )

    def test_fiducials_require_unique_ids(self) -> None:
        coordinates = np.array([0.0, 88.0, -10.0])
        with pytest.raises(ValueError, match="must be unique"):
            Fiducials(
                items=[
                    Fiducial("NAS", "Nasion", coordinates, "world"),
                    Fiducial("NAS", "Nasion", coordinates, "world"),
                ]
            )

    def test_fiducial_defaults_to_unit_weight(self) -> None:
        fiducial = Fiducial(
            fiducial_id="NAS",
            name="Nasion",
            coordinates=np.array([0.0, 88.0, -10.0]),
            coordinate_system="world",
        )
        assert fiducial.weight == 1.0

    def test_fiducial_validates_weight(self) -> None:
        with pytest.raises(ValueError, match="weight must be positive"):
            Fiducial(
                fiducial_id="NAS",
                name="Nasion",
                coordinates=np.array([0.0, 88.0, -10.0]),
                coordinate_system="world",
                weight=0.0,
            )

    def test_fiducials_get_returns_matching_fiducial(self) -> None:
        fiducials = make_fiducials()
        lpa = fiducials.get("LPA")
        assert lpa is not None
        assert lpa.fiducial_id == "LPA"
        assert fiducials.get("RPA") is None


class TestFiducialHelpers:
    def test_round_trip_preserves_fiducials(self, tmp_path: Path) -> None:
        path = tmp_path / "fiducials.json"
        fiducials = make_fiducials()

        save_fiducials(path, fiducials)
        restored = load_fiducials(path)

        assert restored.ids == ["NAS", "LPA"]
        nas = restored.get("NAS")
        assert nas is not None
        assert nas.coordinates.tolist() == [0.0, 88.0, -10.0]
        assert nas.coordinate_system == "world"
        assert nas.definition_method == "manual"
        lpa = restored.get("LPA")
        assert lpa is not None
        assert lpa.name == "Left pre-auricular"
        assert lpa.definition_method == "manual"

    def test_load_fiducials_accepts_mne_coordsystem_json(self, tmp_path: Path) -> None:
        path = tmp_path / "coordsystem.json"
        path.write_text(
            json.dumps(
                {
                    "CoordinateSystem": "RAS",
                    "CoordinateUnits": "mm",
                    "FiducialsCoordinates": {
                        "NASION": {"Head": [0.0, 102.6, 0.0], "MRI": [3.38, 94.66, 32.26]},
                        "LPA": {"Head": [-71.38, 0.0, 0.0], "MRI": [-69.26, 10.59, -25.0]},
                        "RPA": {"Head": [75.27, 0.0, 0.0], "MRI": [77.29, 12.05, -25.4]},
                    },
                }
            ),
            encoding="utf-8",
        )

        fiducials = load_fiducials(path)

        assert fiducials.ids == ["NAS", "LPA", "RPA"]
        nas = fiducials.get("NAS")
        assert nas is not None
        assert nas.coordinates.tolist() == [3.38, 94.66, 32.26]
        assert nas.coordinate_system == "world"
        assert nas.definition_method == "imported"
