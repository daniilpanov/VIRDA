import json

import pytest
from pydantic import ValidationError

from virda.config import (
    VirdaSettings,
    resolve_config_file,
    resolve_ese_config,
    resolve_stage2_config,
)
from virda.models.ese_config import ESEConfig
from virda.models.stage2_config import Stage2Config


class TestResolveConfigFile:
    def test_reads_path_from_environment(self, monkeypatch, tmp_path) -> None:
        config_path = tmp_path / "CTRL_1277" / ".env.json"
        monkeypatch.setenv("VIRDA_CONFIG_FILE", str(config_path))

        assert resolve_config_file() == str(config_path)


# TODO: make normal tests. Maybe config tests are redundant?
class TestVirdaSettings:
    def test_rejects_nonpositive_threshold_scale(self) -> None:
        with pytest.raises(ValidationError, match="otsu_threshold_scale"):
            VirdaSettings(_cli_parse_args=False, otsu_threshold_scale=0)  # type: ignore[call-arg]

    def test_loads_settings_from_dataset_file_via_environment(self, monkeypatch, tmp_path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        config_file = dataset_dir / ".env.json"
        config_file.write_text(json.dumps({"otsu_threshold_scale": 0.42, "otsu_scope": "all"}))
        monkeypatch.setenv("VIRDA_CONFIG_FILE", str(config_file))

        settings = VirdaSettings(_cli_parse_args=False)  # type: ignore[call-arg]
        assert settings.otsu_threshold_scale == pytest.approx(0.42)
        assert settings.otsu_scope == "all"

    def test_environment_overrides_dataset_file(self, monkeypatch, tmp_path) -> None:
        config_file = tmp_path / ".env.json"
        config_file.write_text(json.dumps({"otsu_threshold_scale": 0.42}))
        monkeypatch.setenv("VIRDA_CONFIG_FILE", str(config_file))
        monkeypatch.setenv("OTSU_THRESHOLD_SCALE", "0.9")

        settings = VirdaSettings(_cli_parse_args=False)  # type: ignore[call-arg]
        assert settings.otsu_threshold_scale == pytest.approx(0.9)

    def test_ese_not_configured_yields_no_config(self) -> None:
        settings = VirdaSettings(_cli_parse_args=False)  # type: ignore[call-arg]
        assert resolve_ese_config(settings) is None

    def test_ese_configured_yields_config(self, monkeypatch) -> None:
        monkeypatch.setenv("N_ELECTRODES", "32")
        monkeypatch.setenv("ESE_OFFSET_MM", "2.5")
        monkeypatch.setenv("ESE_REFERENCE", "electrode_body_center")

        settings = VirdaSettings(_cli_parse_args=False)  # type: ignore[call-arg]
        assert settings.n_electrodes == 32
        assert settings.ese_offset_mm == 2.5
        assert settings.ese_reference == "electrode_body_center"

        config = resolve_ese_config(settings)
        assert isinstance(config, ESEConfig)
        assert config == ESEConfig(
            n_electrodes=32, ese_offset_mm=2.5, ese_reference="electrode_body_center"
        )

    def test_stage2_not_configured_without_ese_offset(self) -> None:
        settings = VirdaSettings(_cli_parse_args=False)  # type: ignore[call-arg]
        assert resolve_stage2_config(settings) is None

    def test_stage2_configured_with_defaults(self, monkeypatch) -> None:
        monkeypatch.setenv("ESE_OFFSET_MM", "2.5")
        settings = VirdaSettings(_cli_parse_args=False)  # type: ignore[call-arg]

        config = resolve_stage2_config(settings)
        assert isinstance(config, Stage2Config)
        assert config == Stage2Config()

    def test_stage2_configured_with_custom_fields(self, monkeypatch) -> None:
        monkeypatch.setenv("ESE_OFFSET_MM", "2.5")
        monkeypatch.setenv("K_NEIGHBORS", "20")
        monkeypatch.setenv("NEIGHBORHOOD_RADIUS_MM", "12.0")
        monkeypatch.setenv("PCA_SIGMA_MM", "3.0")
        monkeypatch.setenv("MIN_NEIGHBORS", "7")
        monkeypatch.setenv("USE_WEIGHTED_PCA", "true")

        settings = VirdaSettings(_cli_parse_args=False)  # type: ignore[call-arg]
        config = resolve_stage2_config(settings)
        assert isinstance(config, Stage2Config)
        assert config == Stage2Config(
            neighborhood_radius_mm=12.0,
            k_neighbors=20,
            use_weighted_pca=True,
            pca_sigma_mm=3.0,
            min_neighbors=7,
        )
