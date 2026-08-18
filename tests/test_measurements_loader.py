import json

import numpy as np
import pytest

from tests.helpers.measurements import make_fiducials, make_measurements_file
from tests.helpers.pipelines import build_context
from virda.io.loader.measurements_loader import MeasurementsLoaderFromJson
from virda.models.fiducial import Fiducials
from virda.models.path import MeasurementsPath


class TestMeasurementsLoader:
    def test_loads_electrodes(self, tmp_path) -> None:
        path = make_measurements_file(tmp_path / "measurements.json", points=np.zeros((2, 3)))

        result = MeasurementsLoaderFromJson().run(
            build_context(measurements_path=MeasurementsPath(path))
        )

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

        result = MeasurementsLoaderFromJson().run(
            build_context(measurements_path=MeasurementsPath(path))
        )

        assert len(result.items) == 1
        electrode = result.items[0]
        assert electrode.electrode_id == "Fz"
        assert electrode.measured_distances == {"NAS": 120.5, "LPA": 131.2}

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
        context = build_context(
            fiducials=make_fiducials(),
            measurements_path=MeasurementsPath(path),
        )

        MeasurementsLoaderFromJson().run(context)

        updated = context.get_store_notnull(Fiducials)
        nas = updated.get("NAS")
        lpa = updated.get("LPA")
        assert nas is not None
        assert lpa is not None
        assert nas.weight == pytest.approx(2.0)
        assert lpa.weight == pytest.approx(1.0)

    def test_without_weights_keeps_fiducials(self, tmp_path) -> None:
        path = make_measurements_file(tmp_path / "measurements.json", points=np.zeros((1, 3)))
        fiducials = make_fiducials()
        context = build_context(
            fiducials=fiducials,
            measurements_path=MeasurementsPath(path),
        )

        MeasurementsLoaderFromJson().run(context)

        assert context.get_store_notnull(Fiducials) is fiducials

    def test_rejects_electrode_without_measurements(self, tmp_path) -> None:
        path = tmp_path / "measurements.json"
        path.write_text(
            json.dumps({"electrodes": [{"electrode_id": "Fz", "measured_distances": {}}]})
        )

        with pytest.raises(
            ValueError, match="measured_distances must contain at least one measurement"
        ):
            MeasurementsLoaderFromJson().run(
                build_context(measurements_path=MeasurementsPath(path))
            )
