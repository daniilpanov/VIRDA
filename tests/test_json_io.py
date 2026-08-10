from pathlib import Path

import numpy as np
import pytest
from pydantic_settings import BaseSettings

from virda.io.exporter.json_io import load_config, load_fiducials, save_config, save_fiducials
from virda.models.fiducial import Fiducial


class DummyConfig(BaseSettings):
    closing_radius: int = 5
    smoother_iterations: int = 10


@pytest.fixture
def fiducials() -> list[Fiducial]:
    return [
        Fiducial(
            fiducial_id="NAS",
            name="nasion",
            coordinates=np.array([1.0, 2.0, 3.0]),
            coordinate_system="world",
        ),
        Fiducial(
            fiducial_id="LPA",
            name="left preauricular",
            coordinates=np.array([-40.0, 5.0, 0.0]),
            coordinate_system="world",
            definition_method="manual",
        ),
        Fiducial(
            fiducial_id="INI",
            name="inion",
            coordinates=np.array([0.0, -60.0, 10.0]),
            coordinate_system="voxel",
            definition_method="imported",
        ),
    ]


class TestFiducialRoundTrip:
    def test_save_load_round_trip(self, tmp_path: Path, fiducials: list[Fiducial]) -> None:
        path = tmp_path / "fiducials.json"
        save_fiducials(path, fiducials)

        loaded = load_fiducials(path)

        assert len(loaded) == len(fiducials)
        for original, restored in zip(fiducials, loaded, strict=True):
            assert restored.fiducial_id == original.fiducial_id
            assert restored.name == original.name
            assert restored.coordinate_system == original.coordinate_system
            assert restored.definition_method == original.definition_method
            np.testing.assert_allclose(restored.coordinates, original.coordinates)

    def test_save_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        save_fiducials(path, [])

        assert load_fiducials(path) == []

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_fiducials(tmp_path / "missing.json")


class TestConfigRoundTrip:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        config = DummyConfig(closing_radius=7, smoother_iterations=3)

        save_config(path, config)
        loaded = load_config(path, DummyConfig)

        assert isinstance(loaded, DummyConfig)
        assert loaded.closing_radius == 7
        assert loaded.smoother_iterations == 3

    def test_save_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        save_config(path, {"closing_radius": 9})

        loaded = load_config(path, DummyConfig)

        assert loaded.closing_radius == 9
        assert loaded.smoother_iterations == 10
