import json

import pytest
from pydantic import ValidationError

from virda.config import VirdaSettings, resolve_config_file


class TestResolveConfigFile:
    def test_reads_path_from_environment(self, monkeypatch, tmp_path) -> None:
        config_path = tmp_path / "CTRL_1277" / ".env.json"
        monkeypatch.setenv("VIRDA_CONFIG_FILE", str(config_path))

        assert resolve_config_file() == str(config_path)


class TestVirdaSettings:
    def test_defaults(self) -> None:
        settings = VirdaSettings(_cli_parse_args=False)  # type: ignore[call-arg]

        assert settings.otsu_scope == "all"
        assert settings.otsu_threshold_scale == pytest.approx(0.6)

    def test_rejects_nonpositive_threshold_scale(self) -> None:
        with pytest.raises(ValidationError, match="otsu_threshold_scale"):
            VirdaSettings(_cli_parse_args=False, otsu_threshold_scale=0)  # type: ignore[call-arg]

    def test_rejects_unknown_scope(self) -> None:
        with pytest.raises(ValidationError, match="otsu_scope"):
            VirdaSettings(_cli_parse_args=False, otsu_scope="bogus")  # type: ignore[call-arg, arg-type]

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

    def test_missing_dataset_file_falls_back_to_defaults(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("VIRDA_CONFIG_FILE", str(tmp_path / "absent" / ".env.json"))

        settings = VirdaSettings(_cli_parse_args=False)  # type: ignore[call-arg]

        assert settings.otsu_scope == "all"
        assert settings.otsu_threshold_scale == pytest.approx(0.6)
