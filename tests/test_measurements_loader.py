import json

import numpy as np
import pytest

from tests.helpers.measurements import make_measurements_file
from virda.io.importers.measurements import import_measurements


class TestMeasurementsImporter:
    def test_imports_electrodes(self, tmp_path) -> None:
        path = make_measurements_file(tmp_path / "measurements.json", points=np.zeros((2, 3)))

        result = import_measurements(path)

        assert [electrode.electrode_id for electrode in result.items] == ["E0", "E1"]
        assert set(result.items[0].measured_distances) == {"NAS", "LPA", "RPA"}

    def test_imports_distances(self, tmp_path) -> None:
        path = tmp_path / "measurements.json"
        path.write_text(
            json.dumps(
                {
                    "electrodes": [
                        {
                            "electrode_id": "Fz",
                            "measured_distances": {"NAS": 120.5, "LPA": 131.2},
                        }
                    ]
                }
            )
        )

        result = import_measurements(path)

        assert len(result.items) == 1
        electrode = result.items[0]
        assert electrode.electrode_id == "Fz"
        assert electrode.measured_distances == {"NAS": 120.5, "LPA": 131.2}

    def test_generates_ids_when_missing(self, tmp_path) -> None:
        path = tmp_path / "measurements.json"
        path.write_text(
            json.dumps(
                {
                    "electrodes": [
                        {"measured_distances": {"NAS": 120.5}},
                        {"electrode_id": "Fz", "measured_distances": {"NAS": 131.2}},
                        {"measured_distances": {"NAS": 140.0}},
                    ]
                }
            )
        )

        result = import_measurements(path)

        assert [e.electrode_id for e in result.items] == ["E001", "Fz", "E003"]

    def test_empty_id_is_replaced_with_generated(self, tmp_path) -> None:
        path = tmp_path / "measurements.json"
        path.write_text(
            json.dumps({"electrodes": [{"electrode_id": "", "measured_distances": {"NAS": 1.0}}]})
        )

        result = import_measurements(path)

        assert result.items[0].electrode_id == "E001"

    def test_ignores_fiducial_weights(self, tmp_path) -> None:
        """``fiducial_weights`` live on the Fiducials model, not the importer."""
        path = tmp_path / "measurements.json"
        path.write_text(
            json.dumps(
                {
                    "fiducial_weights": {"NAS": 2.0},
                    "electrodes": [
                        {"electrode_id": "Fz", "measured_distances": {"NAS": 1.0}}
                    ],
                }
            )
        )

        result = import_measurements(path)

        assert [electrode.electrode_id for electrode in result.items] == ["Fz"]

    def test_rejects_electrode_without_measurements(self, tmp_path) -> None:
        path = tmp_path / "measurements.json"
        path.write_text(
            json.dumps({"electrodes": [{"electrode_id": "Fz", "measured_distances": {}}]})
        )

        with pytest.raises(
            ValueError, match="measured_distances must contain at least one measurement"
        ):
            import_measurements(path)