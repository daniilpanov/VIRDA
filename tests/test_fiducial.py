import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from tests.helpers.pipelines import make_fiducials
from virda.io.fiducial_helpers import load_fiducials, save_fiducials
from virda.models.coordsystem import Coordsystem
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

    def test_round_trip_preserves_weights(self, tmp_path: Path) -> None:
        path = tmp_path / "fiducials.json"
        base = make_fiducials().items
        fiducials = Fiducials(items=[replace(base[0], weight=2.5), *base[1:]])

        save_fiducials(path, fiducials)
        restored = load_fiducials(path)

        nas = restored.get("NAS")
        assert nas is not None
        assert nas.weight == pytest.approx(2.5)

    def test_load_fiducials_defaults_weight_for_legacy_files(self, tmp_path: Path) -> None:
        path = tmp_path / "fiducials.json"
        path.write_text(
            json.dumps(
                {
                    "fiducials": [
                        {
                            "fiducial_id": "NAS",
                            "name": "Nasion",
                            "coordinates": [0.0, 88.0, -10.0],
                            "coordinate_system": "world",
                            "definition_method": "manual",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        restored = load_fiducials(path)

        nas = restored.get("NAS")
        assert nas is not None
        assert nas.weight == 1.0

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

    def test_load_fiducials_converts_metres_to_mm(self, tmp_path: Path) -> None:
        path = tmp_path / "coordsystem.json"
        path.write_text(
            json.dumps(
                {
                    "CoordinateSystem": "RAS",
                    "CoordinateUnits": "m",
                    "FiducialsCoordinates": {
                        "NASION": {"Head": [0.0, 0.1, 0.0], "MRI": [0.01, 0.09, 0.03]}
                    },
                }
            ),
            encoding="utf-8",
        )

        fiducials = load_fiducials(path)

        nas = fiducials.get("NAS")
        assert nas is not None
        assert nas.coordinates.tolist() == pytest.approx([10.0, 90.0, 30.0])

    def test_load_fiducials_defaults_to_metres_when_units_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "coordsystem.json"
        path.write_text(
            json.dumps(
                {
                    "CoordinateSystem": "RAS",
                    "FiducialsCoordinates": {
                        "NASION": {"Head": [0.0, 0.1, 0.0], "MRI": [0.01, 0.09, 0.03]}
                    },
                }
            ),
            encoding="utf-8",
        )

        fiducials = load_fiducials(path)

        nas = fiducials.get("NAS")
        assert nas is not None
        assert nas.coordinates.tolist() == pytest.approx([10.0, 90.0, 30.0])

    def test_load_fiducials_rejects_unknown_units(self, tmp_path: Path) -> None:
        path = tmp_path / "coordsystem.json"
        path.write_text(
            json.dumps(
                {
                    "CoordinateSystem": "RAS",
                    "CoordinateUnits": "cm",
                    "FiducialsCoordinates": {
                        "NASION": {"Head": [0.0, 0.1, 0.0], "MRI": [1.0, 9.0, 3.0]}
                    },
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Unsupported coordinate units 'cm'"):
            load_fiducials(path)

    def test_load_fiducials_accepts_coordsystem_without_fiducials(self, tmp_path: Path) -> None:
        path = tmp_path / "coordsystem.json"
        path.write_text(
            json.dumps({"CoordinateSystem": "RAS", "CoordinateUnits": "mm"}),
            encoding="utf-8",
        )

        fiducials = load_fiducials(path)

        assert fiducials.items == []

    def test_is_coordsystem_dict(self) -> None:
        assert Coordsystem.is_coordsystem_dict({"FiducialsCoordinates": {}})
        assert Coordsystem.is_coordsystem_dict({"CoordinateSystem": "RAS"})
        assert not Coordsystem.is_coordsystem_dict({"fiducials": []})
        assert not Coordsystem.is_coordsystem_dict({})
