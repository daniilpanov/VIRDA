import json
from pathlib import Path

import pytest

from virda.config import VirdaSettings
from virda.main import build_cleaners
from virda.mesh.air_depth import AirDepthCleaner
from virda.mesh.cleaners import LargestComponentCleaner, MergeCleaner
from virda.mesh.contracts import MeshCleaner
from virda.mesh.hole_fill import HoleFillCleaner


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["virda"])


@pytest.fixture
def clean_env(cli_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.usefixtures("cli_env")
class TestBuildCleaners:
    def test_default_order(self) -> None:
        cleaners = build_cleaners(VirdaSettings())

        assert [type(c) for c in cleaners] == [
            MergeCleaner,
            AirDepthCleaner,
            HoleFillCleaner,
            LargestComponentCleaner,
        ]

    def test_reorder(self) -> None:
        settings = VirdaSettings(cleaner_sequence=["hole_fill", "merge"])

        cleaners = build_cleaners(settings)

        assert [type(c) for c in cleaners] == [HoleFillCleaner, MergeCleaner]

    def test_exclude_all(self) -> None:
        settings = VirdaSettings(cleaner_sequence=[])

        assert build_cleaners(settings) == []

    def test_exclude_single(self) -> None:
        settings = VirdaSettings(cleaner_sequence=["merge", "air_depth"])

        cleaners = build_cleaners(settings)

        assert [type(c) for c in cleaners] == [MergeCleaner, AirDepthCleaner]

    def test_unknown_cleaner_raises(self) -> None:
        settings = VirdaSettings(cleaner_sequence=["bogus"])

        with pytest.raises(ValueError, match="bogus"):
            build_cleaners(settings)

    def test_cleaners_implement_contract(self) -> None:
        cleaners = build_cleaners(VirdaSettings())

        assert all(isinstance(c, MeshCleaner) for c in cleaners)


class TestCleanerSequenceParsing:
    def test_parses_from_json(self, clean_env: Path) -> None:
        (clean_env / ".env.json").write_text(
            json.dumps({"cleaner_sequence": ["merge", "hole_fill"]}), encoding="utf-8"
        )

        assert VirdaSettings().cleaner_sequence == ["merge", "hole_fill"]

    def test_parses_from_yaml(self, clean_env: Path) -> None:
        (clean_env / ".env.yaml").write_text(
            "cleaner_sequence:\n  - hole_fill\n  - merge\n", encoding="utf-8"
        )

        assert VirdaSettings().cleaner_sequence == ["hole_fill", "merge"]

    def test_parses_from_dotenv(self, clean_env: Path) -> None:
        (clean_env / ".env").write_text(
            'CLEANER_SEQUENCE=["hole_fill", "merge"]\n', encoding="utf-8"
        )

        assert VirdaSettings().cleaner_sequence == ["hole_fill", "merge"]

    def test_parses_from_cli(self, clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["virda", "--cleaner_sequence", '["merge", "air_depth"]'])

        assert VirdaSettings().cleaner_sequence == ["merge", "air_depth"]
