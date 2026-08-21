import csv
import json

import numpy as np
import pytest

from tests.helpers.measurements import make_electrodes, make_ese, make_fiducials
from tests.helpers.pipelines import build_context
from virda.io.providers.stage3_exporter import Stage3Exporter
from virda.localization.brute_force_localizer import BruteForceLocalizer
from virda.models.electrode import Electrode, Electrodes
from virda.models.stage3_config import Stage3Config


def _localize(electrodes: Electrodes, threshold_mm: float = 10.0) -> Electrodes:
    ese = make_ese()
    fiducials = make_fiducials()
    return BruteForceLocalizer(Stage3Config(residual_threshold_mm=threshold_mm)).run(
        build_context(ese=ese, fiducials=fiducials, electrodes=electrodes)
    )


def _exact_distances(point, fiducials) -> dict[str, float]:
    return {
        fiducial.fiducial_id: float(np.linalg.norm(point - fiducial.coordinates))
        for fiducial in fiducials.items
    }


class TestStage3Exporter:
    def test_exports_electrodes_json(self, tmp_path) -> None:
        ese = make_ese()
        fiducials = make_fiducials()
        result = _localize(make_electrodes(ese.vertices[[0, 42]], fiducials))
        exporter = Stage3Exporter(project_dir=tmp_path, stage3_config=Stage3Config())
        exporter.provide(result)

        data = json.loads((tmp_path / "localization" / "electrodes.json").read_text())
        assert len(data) == 2
        assert data[0]["electrode_id"] == "E0"
        assert data[0]["ese_coords"] is not None
        assert data[0]["scalp_coords"] is not None
        assert data[0]["residual_error"] < 1e-6
        assert data[0]["confidence"] == result.items[0].confidence
        assert data[0]["flagged"] is False
        assert result.items[0].scalp_coords is not None
        assert np.allclose(data[0]["scalp_coords"], result.items[0].scalp_coords.tolist())

    def test_exports_electrodes_scalp_json(self, tmp_path) -> None:
        ese = make_ese()
        fiducials = make_fiducials()
        result = _localize(make_electrodes(ese.vertices[[0, 42]], fiducials))
        exporter = Stage3Exporter(project_dir=tmp_path, stage3_config=Stage3Config())
        exporter.provide(result)

        data = json.loads((tmp_path / "localization" / "electrodes_scalp.json").read_text())
        assert len(data) == 2
        assert data[0]["electrode_id"] == "E0"
        assert data[0]["coords"] is not None
        assert result.items[0].scalp_coords is not None
        assert np.allclose(data[0]["coords"], result.items[0].scalp_coords.tolist())
        assert data[0]["residual_error"] < 1e-6
        assert "ese_coords" not in data[0]

    def test_exports_electrodes_ese_json(self, tmp_path) -> None:
        ese = make_ese()
        fiducials = make_fiducials()
        result = _localize(make_electrodes(ese.vertices[[0, 42]], fiducials))
        exporter = Stage3Exporter(project_dir=tmp_path, stage3_config=Stage3Config())
        exporter.provide(result)

        data = json.loads((tmp_path / "localization" / "electrodes_ese.json").read_text())
        assert len(data) == 2
        assert data[0]["electrode_id"] == "E0"
        assert data[0]["coords"] is not None
        assert result.items[0].ese_coords is not None
        assert np.allclose(data[0]["coords"], result.items[0].ese_coords.tolist())
        assert data[0]["residual_error"] < 1e-6
        assert "scalp_coords" not in data[0]

    def test_exports_csv(self, tmp_path) -> None:
        ese = make_ese()
        fiducials = make_fiducials()
        result = _localize(make_electrodes(ese.vertices[[0, 42]], fiducials))
        exporter = Stage3Exporter(project_dir=tmp_path, stage3_config=Stage3Config())
        exporter.provide(result)

        with (tmp_path / "localization" / "electrode_coords.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["electrode_id"] == "E0"
        assert result.items[0].scalp_coords is not None
        assert result.items[0].ese_coords is not None
        assert float(rows[0]["x"]) == pytest.approx(result.items[0].ese_coords[0])
        assert float(rows[0]["y"]) == pytest.approx(result.items[0].ese_coords[1])
        assert float(rows[0]["z"]) == pytest.approx(result.items[0].ese_coords[2])
        assert float(rows[0]["confidence"]) == pytest.approx(result.items[0].confidence)

    def test_exports_summary(self, tmp_path) -> None:
        ese = make_ese()
        fiducials = make_fiducials()
        result = _localize(make_electrodes(ese.vertices[[0, 42]], fiducials))
        exporter = Stage3Exporter(project_dir=tmp_path, stage3_config=Stage3Config())
        exporter.provide(result)

        summary = json.loads((tmp_path / "localization" / "localization_summary.json").read_text())
        assert summary["n_electrodes"] == 2
        assert summary["n_localized"] == 2
        assert summary["n_flagged"] == 0
        assert summary["residual_threshold_mm"] == 10.0
        assert summary["median_residual_mm"] < 1e-6

    def test_summary_counts_flagged(self, tmp_path) -> None:
        ese = make_ese()
        fiducials = make_fiducials()
        point = ese.vertices[0]
        distances = _exact_distances(point, fiducials)
        distances["LPA"] += 100.0
        result = _localize(
            Electrodes(items=[Electrode(electrode_id="E0", measured_distances=distances)]),
            threshold_mm=10.0,
        )
        exporter = Stage3Exporter(project_dir=tmp_path, stage3_config=Stage3Config())
        exporter.provide(result)

        summary = json.loads((tmp_path / "localization" / "localization_summary.json").read_text())
        assert summary["n_electrodes"] == 1
        assert summary["n_localized"] == 1
        assert summary["n_flagged"] == 1
        assert summary["median_residual_mm"] > 10.0

    def test_raises_without_result(self, tmp_path) -> None:
        exporter = Stage3Exporter(project_dir=tmp_path, stage3_config=Stage3Config())
        with pytest.raises(ValueError, match="no result of Stage#3"):
            exporter.provide(None)
