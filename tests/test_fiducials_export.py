import json
from pathlib import Path

import numpy as np

from tests.helpers.measurements import make_fiducials
from virda.io.exporters.fiducials import export_fiducials
from virda.io.importers.fiducials import import_fiducials


class TestExportFiducials:
    def test_round_trip(self, tmp_path: Path) -> None:
        fiducials = make_fiducials()

        path = export_fiducials(tmp_path / "out" / "fiducials.json", fiducials)
        restored = import_fiducials(path)

        assert [f.fiducial_id for f in restored.items] == fiducials.ids
        np.testing.assert_array_almost_equal(
            restored.items[0].coordinates, fiducials.items[0].coordinates
        )
        assert restored.items[1].definition_method == fiducials.items[1].definition_method

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = export_fiducials(tmp_path / "deep" / "nested" / "fiducials.json", make_fiducials())

        assert path.is_file()

    def test_json_schema_keys(self, tmp_path: Path) -> None:
        path = export_fiducials(tmp_path / "fiducials.json", make_fiducials())

        data = json.loads(path.read_text(encoding="utf-8"))
        assert list(data) == ["fiducials"]
        assert set(data["fiducials"][0]) == {
            "fiducial_id",
            "name",
            "coordinates",
            "coordinate_system",
            "definition_method",
            "weight",
        }
