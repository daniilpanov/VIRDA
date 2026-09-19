"""Unit tests for the pure exporter/importer round trips."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tests.helpers.measurements import make_electrodes, make_fiducials, make_ese
from virda.io.exporters.electrodes import export_electrodes
from virda.io.exporters.ese_mesh import export_ese_mesh
from virda.io.exporters.head_mask import export_head_mask
from virda.io.exporters.measurements import export_measurements
from virda.io.importers.electrodes_table import import_electrodes_table
from virda.io.importers.measurements import import_measurements
from virda.io.importers.scalp_mesh import import_scalp_mesh
from virda.localization.brute_force_localizer import BruteForceLocalizer
from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask


@pytest.fixture
def mri() -> MRIVolume:
    return MRIVolume(
        data=np.zeros((16, 16, 16), dtype=np.float32),
        affine=np.eye(4),
        spacing=(1.0, 1.0, 1.0),
        orientation=("R", "A", "S"),
    )


class TestMeasurementsRoundTrip:
    def test_export_then_import(self, tmp_path: Path) -> None:
        fiducials = make_fiducials()
        electrodes = make_electrodes(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), fiducials)

        path = export_measurements(tmp_path / "measurements.json", electrodes)
        restored = import_measurements(path)

        assert [e.electrode_id for e in restored.items] == ["E0", "E1"]
        assert restored.items[0].measured_distances == electrodes.items[0].measured_distances

    def test_export_carries_weights(self, tmp_path: Path) -> None:
        electrodes = make_electrodes(np.array([[1.0, 2.0, 3.0]]), make_fiducials())

        path = export_measurements(tmp_path / "m.json", electrodes, fiducial_weights={"NAS": 3.0})

        data = json.loads(path.read_text())
        assert data["fiducial_weights"] == {"NAS": 3.0}


class TestElectrodesExporter:
    def test_writes_json_and_csv(self, tmp_path: Path) -> None:
        ese = make_ese()
        fiducials = make_fiducials()
        localized = BruteForceLocalizer(calibrate_ese_offset=False).process(
            ese, fiducials, make_electrodes(ese.vertices[[0, 42]], fiducials)
        )

        json_path = export_electrodes(tmp_path / "electrodes.json", localized)

        loaded = json.loads(json_path.read_text())
        assert len(loaded) == 2
        assert loaded[0]["electrode_id"] == "E0"
        assert loaded[0]["ese_coords"] == np.asarray(ese.vertices[0]).tolist()
        assert loaded[0]["residual_error"] < 1e-6

        table = import_electrodes_table(json_path.with_suffix(".csv"))
        assert table.positions.shape == (2, 3)
        np.testing.assert_allclose(table.positions[0], ese.vertices[0], atol=1e-9)

    def test_json_marks_unlocalized_electrode(self, tmp_path: Path) -> None:
        fiducials = make_fiducials()
        electrodes = make_electrodes(np.array([[999.0, 999.0, 999.0]]), fiducials)
        localized = BruteForceLocalizer(
            residual_threshold_mm=1.0, calibrate_ese_offset=False
        ).process(make_ese(), fiducials, electrodes)

        json_path = export_electrodes(tmp_path / "electrodes.json", localized)

        loaded = json.loads(json_path.read_text())
        assert loaded[0]["flagged"] is True
        assert loaded[0]["ese_coords"] is None
        assert loaded[0]["residual_error"] is None


class TestEseMeshExporter:
    def test_round_trip_preserves_faces(self, tmp_path: Path) -> None:
        ese = make_ese()

        path = export_ese_mesh(tmp_path / "ese.ply", ese)
        loaded = import_scalp_mesh(path)

        assert np.allclose(loaded.faces, ese.faces, atol=1e-9)
        assert np.allclose(loaded.vertices, ese.vertices, atol=1e-5)


class TestHeadMaskExporter:
    def test_writes_matching_nifti(self, tmp_path: Path, mri: MRIVolume) -> None:
        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[4:12, 4:12, 4:12] = True
        segmentation = SegmentationMask(mask=mask)

        path = export_head_mask(tmp_path / "head_mask.nii.gz", segmentation, mri)

        image = nib.load(path)
        stored = np.asanyarray(image.dataobj).astype(bool)
        assert stored.shape == mask.shape
        np.testing.assert_array_equal(stored, mask)
        np.testing.assert_allclose(image.affine, np.eye(4))