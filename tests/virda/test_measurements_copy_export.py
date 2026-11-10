import json
import logging
from pathlib import Path

import pytest

from virda.io.exporters.measurements_copy import export_measurements_copy


def _write_measurements(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"electrodes": [{"electrode_id": "Fz", "measured_distances": {"NAS": 110.5}}]}),
        encoding="utf-8",
    )


class TestExportMeasurementsCopy:
    def test_copies_source_into_input(self, tmp_path: Path) -> None:
        source = tmp_path / "elsewhere" / "my_measurements.json"
        _write_measurements(source)

        target = export_measurements_copy(tmp_path / "proj" / "input" / "measurements.json", source)

        assert target is not None and target.is_file()
        assert json.loads(target.read_text(encoding="utf-8")) == json.loads(
            source.read_text(encoding="utf-8")
        )

    def test_canonical_source_is_not_replaced_by_itself(self, tmp_path: Path) -> None:
        target = tmp_path / "proj" / "input" / "measurements.json"
        _write_measurements(target)

        result = export_measurements_copy(target, target)

        assert result == target
        assert target.is_file()

    def test_replaces_stale_copy(self, tmp_path: Path) -> None:
        target = tmp_path / "input" / "measurements.json"
        target.parent.mkdir(parents=True)
        target.write_text("stale", encoding="utf-8")
        source = tmp_path / "fresh.json"
        _write_measurements(source)

        assert export_measurements_copy(target, source) is not None

        assert (
            json.loads(target.read_text(encoding="utf-8"))["electrodes"][0]["electrode_id"] == "Fz"
        )

    def test_copy_error_warns_and_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        missing = tmp_path / "missing.json"

        with caplog.at_level(logging.WARNING, logger="virda.test"):
            result = export_measurements_copy(
                tmp_path / "input" / "measurements.json",
                missing,
                logger=logging.getLogger("virda.test"),
            )

        assert result is None
        assert any("measurements" in record.message for record in caplog.records)
