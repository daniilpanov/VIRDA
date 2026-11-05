import json

import numpy as np
import pytest

from tests.helpers.measurements import make_fiducials, make_measurements_file
from virda.io.importers.measurements import import_measurements, load_measurements


class TestImportMeasurements:
    def test_returns_only_electrodes(self, tmp_path) -> None:
        path = make_measurements_file(tmp_path / "measurements.json", points=np.zeros((2, 3)))

        electrodes = import_measurements(path)

        assert [electrode.electrode_id for electrode in electrodes.items] == ["E0", "E1"]
        assert set(electrodes.items[0].measured_distances) == {"NAS", "LPA", "RPA"}

    def test_ignores_fiducial_weights(self, tmp_path) -> None:
        path = tmp_path / "measurements.json"
        path.write_text(
            json.dumps(
                {
                    "fiducial_weights": {"NAS": 2.0},
                    "electrodes": [{"electrode_id": "Fz", "measured_distances": {"NAS": 120.5}}],
                }
            )
        )

        electrodes = import_measurements(path)

        assert electrodes.items[0].measured_distances == {"NAS": 120.5}


class TestMeasurementsLoader:
    def test_loads_electrodes(self, tmp_path) -> None:
        path = make_measurements_file(tmp_path / "measurements.json", points=np.zeros((2, 3)))

        result = load_measurements(path)[0]

        assert [electrode.electrode_id for electrode in result.items] == ["E0", "E1"]
        assert set(result.items[0].measured_distances) == {"NAS", "LPA", "RPA"}

    def test_loads_distances(self, tmp_path) -> None:
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

        result = load_measurements(path)[0]

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

        result = load_measurements(path)[0]

        assert [e.electrode_id for e in result.items] == ["E001", "Fz", "E003"]

    def test_empty_id_is_replaced_with_generated(self, tmp_path) -> None:
        path = tmp_path / "measurements.json"
        path.write_text(
            json.dumps({"electrodes": [{"electrode_id": "", "measured_distances": {"NAS": 1.0}}]})
        )

        result = load_measurements(path)[0]

        assert result.items[0].electrode_id == "E001"

    def test_applies_fiducial_weights(self, tmp_path) -> None:
        path = tmp_path / "measurements.json"
        path.write_text(
            json.dumps(
                {
                    "fiducial_weights": {"NAS": 2.0},
                    "electrodes": [],
                }
            )
        )
        _, updated = load_measurements(path, make_fiducials())
        assert updated is not None
        nas = updated.get("NAS")
        lpa = updated.get("LPA")
        assert nas is not None
        assert lpa is not None
        assert nas.weight == pytest.approx(2.0)
        assert lpa.weight == pytest.approx(1.0)

    def test_without_weights_keeps_fiducials(self, tmp_path) -> None:
        path = make_measurements_file(tmp_path / "measurements.json", points=np.zeros((1, 3)))
        fiducials = make_fiducials()
        _, updated = load_measurements(path, fiducials)

        assert updated is fiducials

    def test_rejects_electrode_without_measurements(self, tmp_path) -> None:
        path = tmp_path / "measurements.json"
        path.write_text(
            json.dumps({"electrodes": [{"electrode_id": "Fz", "measured_distances": {}}]})
        )

        with pytest.raises(
            ValueError, match="measured_distances must contain at least one measurement"
        ):
            load_measurements(path)
