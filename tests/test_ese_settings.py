from pydantic_settings import BaseSettings

from virda.config import VirdaSettings
from virda.io.exporter.stage1_exporter import Stage1Exporter
from virda.models.ese_config import ESEConfig


class TestEseSettings:
    def test_defaults(self) -> None:
        settings = VirdaSettings(_cli_parse_args=False)
        assert settings.n_electrodes == 67
        assert settings.ese_offset_mm == 5.0
        assert settings.ese_reference == "electrode_external_surface"

    def test_parse_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("VIRDA_N_ELECTRODES", "128")
        monkeypatch.setenv("VIRDA_ESE_OFFSET_MM", "4.0")
        monkeypatch.setenv("VIRDA_ESE_REFERENCE", "electrode_center")
        settings = VirdaSettings(_cli_parse_args=False)
        assert settings.n_electrodes == 128
        assert settings.ese_offset_mm == 4.0
        assert settings.ese_reference == "electrode_center"


class TestPipelineConfigDedup:
    def test_ese_written_once_from_ese_config(self) -> None:
        class DummySettings(BaseSettings):
            closing_radius: int = 5
            n_electrodes: int = 32
            ese_offset_mm: float = 4.0
            ese_reference: str = "electrode_center"

        exporter = Stage1Exporter(
            settings=DummySettings(),
            ese_config=ESEConfig(ese_offset_mm=7.0, n_electrodes=64),
        )
        config = exporter._pipeline_config()
        assert config["ese"] == {
            "n_electrodes": 64,
            "ese_offset_mm": 7.0,
            "ese_reference": "electrode_external_surface",
        }
        assert "n_electrodes" not in config
        assert "ese_offset_mm" not in config
        assert "ese_reference" not in config
        assert config["closing_radius"] == 5
