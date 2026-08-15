import csv
import json

import numpy as np
import pytest

from tests.helpers.measurements import make_electrodes, make_ese, make_fiducials
from virda.localization.brute_force_localizer import BruteForceLocalizer
from virda.models.electrode import Electrode, Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.stage3_config import Stage3Config
from virda.pipelines.stage3 import Stage3PipelineBuilder


def _run_stage3(tmp_path, electrodes: Electrodes, threshold_mm: float = 10.0):
    ese = make_ese()
    fiducials = make_fiducials()
    builder = Stage3PipelineBuilder(
        localizer=BruteForceLocalizer(Stage3Config(residual_threshold_mm=threshold_mm)),
        stage3_config=Stage3Config(residual_threshold_mm=threshold_mm),
        ese_mesh=ese,
        electrodes=electrodes,
        fiducials=fiducials,
        project_dir=tmp_path,
    )
    return builder.build().run(), ese


class TestStage3Pipeline:
    def test_run_builds_localized_electrodes(self, tmp_path) -> None:
        ese = make_ese()
        fiducials = make_fiducials()
        electrodes = make_electrodes(ese.vertices[[0, 42, 100]], fiducials)

        context, _ = _run_stage3(tmp_path, electrodes)

        result = context.get_store_notnull(Electrodes)
        assert len(result.items) == 3
        assert all(electrode.is_localized for electrode in result.items)
        assert context.get_store_notnull(ESEMesh) is not None

    def test_run_exports_artifacts(self, tmp_path) -> None:
        ese = make_ese()
        fiducials = make_fiducials()
        electrodes = make_electrodes(ese.vertices[[0, 42, 100]], fiducials)

        _run_stage3(tmp_path, electrodes)

        stage3_dir = tmp_path / "stage3"
        assert (stage3_dir / "electrodes.json").exists()
        assert (stage3_dir / "electrode_coords.csv").exists()
        assert (stage3_dir / "localization_summary.json").exists()

        summary = json.loads((stage3_dir / "localization_summary.json").read_text())
        assert summary["n_electrodes"] == 3
        assert summary["n_localized"] == 3
        assert summary["n_flagged"] == 0
        assert summary["residual_threshold_mm"] == 10.0

        with (stage3_dir / "electrode_coords.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert [row["electrode_id"] for row in rows] == ["E0", "E1", "E2"]
        assert float(rows[0]["x"]) == pytest.approx(ese.scalp_vertices[0][0])

    def test_run_flags_over_threshold(self, tmp_path) -> None:
        ese = make_ese()
        fiducials = make_fiducials()
        point = ese.vertices[0]
        distances = {
            fiducial.fiducial_id: float(np.linalg.norm(point - fiducial.coordinates))
            for fiducial in fiducials.items
        }
        distances["LPA"] += 100.0
        electrodes = Electrodes(items=[Electrode(electrode_id="E0", measured_distances=distances)])

        _run_stage3(tmp_path, electrodes, threshold_mm=10.0)

        summary = json.loads((tmp_path / "stage3" / "localization_summary.json").read_text())
        assert summary["n_flagged"] == 1
        assert summary["n_localized"] == 1

    def test_run_skips_electrode_without_known_fiducials(self, tmp_path) -> None:
        electrodes = Electrodes(
            items=[Electrode(electrode_id="E0", measured_distances={"UNKNOWN_FIDUCIAL": 10.0})]
        )

        context, _ = _run_stage3(tmp_path, electrodes)

        summary = json.loads((tmp_path / "stage3" / "localization_summary.json").read_text())
        assert summary["n_electrodes"] == 1
        assert summary["n_localized"] == 0
        assert context.get_store_notnull(Electrodes).items[0].is_localized is False
