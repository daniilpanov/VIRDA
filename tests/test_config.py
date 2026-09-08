import json
import os
from typing import Any

import pytest
from pydantic import ValidationError

from virda.config import (
    VirdaSettings,
    build_config,
    load_config_file,
    resolve_config_files,
    resolve_stage3_config,
)
from virda.models.config import Config
from virda.models.coordsystem import Coordsystem
from virda.models.stage3_config import Stage3Config


def sample_coordsystem_dict() -> dict[str, Any]:
    return {
        "CoordinateSystem": "RAS",
        "CoordinateUnits": "mm",
        "CoordinateSystemDescription": "sample",
        "EEGCoordinateSystem": "RAS",
        "EEGCoordinateUnits": "mm",
        "FiducialsCoordinates": {
            "NASION": {"Head": [0.0, 102.6, 0.0], "MRI": [3.379094, 94.659427, 32.259164]},
            "LPA": {"Head": [-71.4, 0.0, 0.0], "MRI": [-69.257414, 10.58946, -25.000859]},
            "RPA": {"Head": [75.3, 0.0, 0.0], "MRI": [77.285621, 12.053672, -30.248822]},
        },
        "ElectrodeOffset": 2.5,
        "Source": "MNE sample dataset",
    }


class TestVirdaSettings:
    def test_rejects_nonpositive_threshold_scale(self) -> None:
        with pytest.raises(ValidationError, match="otsu_threshold_scale"):
            VirdaSettings(otsu_threshold_scale=0)

    def test_loads_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTSU_THRESHOLD_SCALE", raising=False)
        monkeypatch.setenv("OTSU_THRESHOLD_SCALE", "0.42")

        settings = VirdaSettings()
        assert settings.otsu_threshold_scale == pytest.approx(0.42)

    def test_ese_parameters_load_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("N_ELECTRODES", "32")
        monkeypatch.setenv("ESE_OFFSET_MM", "2.5")
        monkeypatch.setenv("ESE_REFERENCE", "electrode_body_center")

        settings = VirdaSettings()
        assert settings.ese_offset_mm == 2.5


class TestResolveConfigFiles:
    def test_legacy_env_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        config_path = tmp_path / "dataset" / ".env.json"
        monkeypatch.setenv("VIRDA_CONFIG_FILE", str(config_path))

        assert resolve_config_files() == [config_path]

    def test_env_list_and_cli_files_keep_order(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        first = tmp_path / "a.json"
        second = tmp_path / "b.json"
        third = tmp_path / "c.json"
        monkeypatch.setenv("VIRDA_CONFIG_FILES", os.pathsep.join([str(first), str(second)]))

        assert resolve_config_files([str(third)]) == [first, second, third]


class TestLoadConfigFile:
    def test_loads_coordsystem_file(self, tmp_path) -> None:
        config_file = tmp_path / "coordsystem.json"
        config_file.write_text(json.dumps(sample_coordsystem_dict()))

        data = load_config_file(config_file)

        assert data["ese_offset_mm"] == 2.5
        coordsystem = data["coordsystem"]
        assert isinstance(coordsystem, Coordsystem)
        assert coordsystem.coordinate_system == "RAS"
        assert coordsystem.electrode_offset_mm == 2.5
        assert coordsystem.fiducials_coordinates["NASION"].mri == (
            3.379094,
            94.659427,
            32.259164,
        )

    def test_loads_plain_config_file(self, tmp_path) -> None:
        config_file = tmp_path / "pipeline_config.json"
        config_file.write_text(json.dumps({"otsu_threshold_scale": 0.42}))

        assert load_config_file(config_file) == {"otsu_threshold_scale": 0.42}

    def test_rejects_non_object_file(self, tmp_path) -> None:
        config_file = tmp_path / "bad.json"
        config_file.write_text("[1, 2, 3]")

        with pytest.raises(ValueError, match="JSON object"):
            load_config_file(config_file)


class TestBuildConfig:
    def test_coordsystem_file_maps_ese_params(self, tmp_path) -> None:
        config_file = tmp_path / "coordsystem.json"
        config_file.write_text(json.dumps(sample_coordsystem_dict()))

        config = build_config(VirdaSettings(), config_files=[config_file])

        assert config.ese_offset_mm == 2.5

    def test_coordsystem_fiducials(self) -> None:
        config = Config(coordsystem=Coordsystem.model_validate(sample_coordsystem_dict()))

        assert config.coordsystem is not None
        fiducials = config.coordsystem.to_fiducials()

        assert fiducials.ids == ["NAS", "LPA", "RPA"]
        assert all(fiducial.definition_method == "imported" for fiducial in fiducials.items)
        assert all(fiducial.coordinate_system == "world" for fiducial in fiducials.items)

    def test_to_stage2_config(self) -> None:
        config = Config(k_neighbors=30, use_weighted_pca=True, pca_sigma_mm=7.0)

        stage2_config = config.to_stage2_config()

        assert stage2_config.k_neighbors == 30
        assert stage2_config.use_weighted_pca is True
        assert stage2_config.pca_sigma_mm == 7.0

    def test_to_stage3_config(self) -> None:
        config = Config(residual_threshold_mm=5.0)

        stage3_config = resolve_stage3_config(config)

        assert isinstance(stage3_config, Stage3Config)
        assert stage3_config.residual_threshold_mm == 5.0
