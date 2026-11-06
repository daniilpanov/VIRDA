import json
from pathlib import Path

import numpy as np

from virda_gui.viewer.scene import (
    load_fiducial_points,
)


class TestLoadFiducialPoints:
    def test_reads_native_format(self, tmp_path: Path) -> None:
        path = tmp_path / "fiducials.json"
        path.write_text(
            """
            {"fiducials": [
                {"fiducial_id": "NAS", "name": "Nasion",
                 "coordinates": [1.0, 2.0, 3.0],
                 "coordinate_system": "world", "definition_method": "manual"}
            ]}
            """,
            encoding="utf-8",
        )

        points, labels = load_fiducial_points(path)

        np.testing.assert_array_equal(points, [[1.0, 2.0, 3.0]])
        assert labels == ["NAS (Nasion)"]

    def test_reads_mne_coordsystem_json(self, tmp_path: Path) -> None:
        path = tmp_path / "coordsystem.json"
        path.write_text(
            json.dumps(
                {
                    "CoordinateSystem": "RAS",
                    "CoordinateUnits": "m",
                    "FiducialsCoordinates": {
                        "NASION": {"Head": [0.0, 0.0, 0.0], "MRI": [0.01, 0.02, 0.03]}
                    },
                }
            ),
            encoding="utf-8",
        )

        points, labels = load_fiducial_points(path)

        np.testing.assert_allclose(points, [[10.0, 20.0, 30.0]])
        assert labels == ["NAS (NASION)"]
