from pathlib import Path

from virda.io.exporters.pipeline_config import export_pipeline_config
from virda.models.config import Config


class TestExportPipelineConfig:
    def test_writes_full_merged_config_json(self, tmp_path: Path) -> None:
        config = Config(closing_radius=3, ese_offset_mm=1.5, seal_enabled=False)

        path = export_pipeline_config(tmp_path / "input" / "pipeline_config.json", config)

        content = path.read_text(encoding="utf-8")
        assert '"closing_radius": 3' in content
        assert '"ese_offset_mm": 1.5' in content
        assert '"seal_enabled": false' in content

    def test_written_config_reloads_into_config_model(self, tmp_path: Path) -> None:
        config = Config(closing_radius=7)

        path = export_pipeline_config(tmp_path / "pipeline_config.json", config)

        reloaded = Config.model_validate_json(path.read_text(encoding="utf-8"))
        assert reloaded.closing_radius == 7
        assert reloaded.ese_offset_mm is None

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = export_pipeline_config(tmp_path / "deep" / "dir" / "config.json", Config())

        assert path.is_file()

    def test_overwrites_previous_dump(self, tmp_path: Path) -> None:
        path = export_pipeline_config(tmp_path / "config.json", Config(closing_radius=1))

        export_pipeline_config(path, Config(closing_radius=9))

        assert '"closing_radius": 9' in path.read_text(encoding="utf-8")
