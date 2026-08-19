import json
import os
from typing import Any

import pytest
from pydantic import ValidationError

from virda.config import VirdaSettings, build_config, load_config_file, resolve_config_files
from virda.models.config import Config
from virda.models.coordsystem import Coordsystem


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
        "ElectrodeCount": 60,
        "ElectrodeOffset": 2.5,
        "ElectrodeReference": "electrode_capsule_center",
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
        assert settings.n_electrodes == 32
        assert settings.ese_offset_mm == 2.5
        assert settings.ese_reference == "electrode_body_center"


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

        assert data["n_electrodes"] == 60
        assert data["ese_offset_mm"] == 2.5
        assert data["ese_reference"] == "electrode_capsule_center"
        coordsystem = data["coordsystem"]
        assert isinstance(coordsystem, Coordsystem)
        assert coordsystem.coordinate_system == "RAS"
        assert coordsystem.electrode_offset_mm == 2.5
        assert coordsystem.electrode_reference == "electrode_capsule_center"
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
    def test_defaults_when_nothing_configured(self) -> None:
        config = build_config(VirdaSettings())

        assert config.otsu_threshold_scale == pytest.approx(0.6)
        assert config.n_electrodes is None
        assert config.coordsystem is None

    def test_config_file_overrides_settings(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("N_ELECTRODES", "10")
        config_file = tmp_path / "dataset.json"
        config_file.write_text(json.dumps({"n_electrodes": 20}))

        config = build_config(VirdaSettings(), config_files=[config_file])

        assert config.n_electrodes == 20

    def test_multiple_config_files_last_wins(self, tmp_path) -> None:
        first = tmp_path / "first.json"
        second = tmp_path / "second.json"
        first.write_text(json.dumps({"n_electrodes": 20}))
        second.write_text(json.dumps({"n_electrodes": 30}))

        config = build_config(VirdaSettings(), config_files=[first, second])

        assert config.n_electrodes == 30

    def test_cli_overrides_config_files(self, tmp_path) -> None:
        config_file = tmp_path / "dataset.json"
        config_file.write_text(json.dumps({"n_electrodes": 30}))

        config = build_config(
            VirdaSettings(),
            config_files=[config_file],
            overrides={"n_electrodes": 40},
        )

        assert config.n_electrodes == 40

    def test_coordsystem_file_maps_electrode_count(self, tmp_path) -> None:
        config_file = tmp_path / "coordsystem.json"
        config_file.write_text(json.dumps(sample_coordsystem_dict()))

        config = build_config(VirdaSettings(), config_files=[config_file])

        assert config.n_electrodes == 60
        assert config.coordsystem is not None
        assert config.coordsystem.electrode_count == 60

    def test_coordsystem_file_maps_ese_params(self, tmp_path) -> None:
        config_file = tmp_path / "coordsystem.json"
        config_file.write_text(json.dumps(sample_coordsystem_dict()))

        config = build_config(VirdaSettings(), config_files=[config_file])

        assert config.ese_offset_mm == 2.5
        assert config.ese_reference == "electrode_capsule_center"

    def test_coordsystem_file_ese_params_without_offset(self, tmp_path) -> None:
        data = sample_coordsystem_dict()
        del data["ElectrodeOffset"]
        config_file = tmp_path / "coordsystem.json"
        config_file.write_text(json.dumps(data))

        config = build_config(VirdaSettings(), config_files=[config_file])

        assert config.ese_offset_mm is None
        assert config.ese_reference == "electrode_capsule_center"

    def test_cli_ese_params_beat_coordsystem(self, tmp_path) -> None:
        config_file = tmp_path / "coordsystem.json"
        config_file.write_text(json.dumps(sample_coordsystem_dict()))

        config = build_config(
            VirdaSettings(),
            config_files=[config_file],
            overrides={"ese_offset_mm": 5.0, "ese_reference": "electrode_body_center"},
        )

        assert config.ese_offset_mm == 5.0
        assert config.ese_reference == "electrode_body_center"

    def test_cli_n_electrodes_beats_coordsystem(self, tmp_path) -> None:
        config_file = tmp_path / "coordsystem.json"
        config_file.write_text(json.dumps(sample_coordsystem_dict()))

        config = build_config(
            VirdaSettings(),
            config_files=[config_file],
            overrides={"n_electrodes": 32},
        )

        assert config.n_electrodes == 32

    def test_coordsystem_fiducials(self) -> None:
        config = Config(coordsystem=Coordsystem.model_validate(sample_coordsystem_dict()))

        assert config.coordsystem is not None
        fiducials = config.coordsystem.to_fiducials()

        assert fiducials.ids == ["NAS", "LPA", "RPA"]
        assert all(fiducial.definition_method == "imported" for fiducial in fiducials.items)
        assert all(fiducial.coordinate_system == "world" for fiducial in fiducials.items)

    def test_to_ese_config_when_configured(self) -> None:
        config = Config(
            n_electrodes=32,
            ese_offset_mm=2.5,
            ese_reference="electrode_body_center",
        )

        ese_config = config.to_ese_config()

        assert ese_config is not None
        assert ese_config.n_electrodes == 32
        assert ese_config.ese_offset_mm == 2.5

    def test_to_ese_config_when_not_configured(self) -> None:
        assert Config().to_ese_config() is None

    def test_to_stage2_config(self) -> None:
        config = Config(k_neighbors=30, use_weighted_pca=True, pca_sigma_mm=7.0)

        stage2_config = config.to_stage2_config()

        assert stage2_config.k_neighbors == 30
        assert stage2_config.use_weighted_pca is True
        assert stage2_config.pca_sigma_mm == 7.0
