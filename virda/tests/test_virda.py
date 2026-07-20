"""Tests for VIRDA core modules."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMRIData:
    def test_shape(self):
        from core.dataclasses import MRIData

        data = np.zeros((50, 60, 70), dtype=np.float64)
        mri = MRIData(data=data, affine=np.eye(4), voxel_size=np.array([1.0, 1.0, 1.0]))
        assert mri.shape == (50, 60, 70)

    def test_voxel_to_world(self):
        from core.dataclasses import MRIData

        affine = np.eye(4) * 2.0
        affine[3, 3] = 1.0
        mri = MRIData(data=np.zeros((10, 10, 10)), affine=affine, voxel_size=np.array([2.0, 2.0, 2.0]))
        world = mri.voxel_to_world(np.array([[5.0, 5.0, 5.0]]))
        np.testing.assert_allclose(world[0, :3], [10.0, 10.0, 10.0])


class TestMeshData:
    def test_empty_mesh(self):
        from core.surface_extractor import MeshData

        mesh = MeshData(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int))
        assert mesh.num_vertices == 0
        assert mesh.num_faces == 0

    def test_simple_mesh(self):
        from core.surface_extractor import MeshData

        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = MeshData(vertices=verts, faces=faces)
        assert mesh.num_vertices == 3
        assert mesh.num_faces == 1


class TestESEConfig:
    def test_default_config(self):
        from core.ese_config import ESEConfig

        cfg = ESEConfig()
        assert cfg.offset_mm == 5.0

    def test_invalid_offset(self):
        from core.ese_config import ESEConfig

        with pytest.raises(ValueError):
            ESEConfig(offset_mm=-1.0)

    def test_save_load(self, tmp_path):
        from core.ese_config import ESEConfig

        cfg = ESEConfig(offset_mm=7.5, reference_point="center")
        cfg.save(tmp_path / "config.json")
        loaded = ESEConfig.load(tmp_path / "config.json")
        assert loaded.offset_mm == 7.5
        assert loaded.reference_point == "center"


class TestFiducialManager:
    def test_add_fiducial(self):
        from core.fiducial_manager import FiducialManager

        mgr = FiducialManager()
        mgr.add_fiducial("NAS", "Nasion", np.array([1.0, 2.0, 3.0]))
        fid = mgr.get_fiducial("NAS")
        assert fid is not None
        assert fid.fiducial_id == "NAS"
        np.testing.assert_array_equal(fid.coordinates, [1.0, 2.0, 3.0])

    def test_remove_fiducial(self):
        from core.fiducial_manager import FiducialManager

        mgr = FiducialManager()
        mgr.add_fiducial("NAS", "Nasion", np.array([1.0, 2.0, 3.0]))
        mgr.remove_fiducial("NAS")
        assert mgr.get_fiducial("NAS") is None

    def test_validate_too_few(self):
        from core.fiducial_manager import FiducialManager

        mgr = FiducialManager()
        warnings = mgr.validate()
        assert any("3" in w for w in warnings)

    def test_save_load(self, tmp_path):
        from core.fiducial_manager import FiducialManager

        mgr = FiducialManager()
        mgr.add_fiducial("NAS", "Nasion", np.array([1.0, 2.0, 3.0]))
        mgr.add_fiducial("LPA", "LPA", np.array([4.0, 5.0, 6.0]))
        mgr.save(tmp_path / "fiducials.json")

        loaded = FiducialManager.load(tmp_path / "fiducials.json")
        assert len(loaded.get_all_fiducials()) == 2
        np.testing.assert_array_equal(loaded.get_fiducial("NAS").coordinates, [1.0, 2.0, 3.0])


class TestMeasurementImporter:
    def test_add_and_get(self):
        from core.measurement_importer import MeasurementImporter

        imp = MeasurementImporter(["NAS", "LPA"])
        imp.add_measurement("E1", {"NAS": 100.0, "LPA": 90.0})
        meas = imp.get_measurement("E1")
        assert meas is not None
        assert meas.distances["NAS"] == 100.0

    def test_csv_import_export(self, tmp_path):
        from core.measurement_importer import MeasurementImporter

        imp = MeasurementImporter(["NAS", "LPA", "RPA"])
        imp.add_measurement("E1", {"NAS": 100.0, "LPA": 90.0, "RPA": 95.0})
        imp.add_measurement("E2", {"NAS": 110.0, "LPA": 85.0, "RPA": 100.0})
        imp.save_csv(tmp_path / "meas.csv")

        imp2 = MeasurementImporter([])
        imp2.import_csv(tmp_path / "meas.csv")
        assert len(imp2.get_all_measurements()) == 2


class TestPCA:
    def test_flat_plane_normal(self):
        from core.surface_extractor import MeshData
        from core.pca_normal_estimator import estimate_normals_pca

        x = np.linspace(0, 20, 10)
        y = np.linspace(0, 20, 10)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)

        verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        n = len(verts)
        faces = []
        for i in range(9):
            for j in range(9):
                idx = i * 10 + j
                faces.append([idx, idx + 1, idx + 10])
                faces.append([idx + 1, idx + 10, idx + 11])

        mesh = MeshData(vertices=verts, faces=np.array(faces, dtype=np.int64))
        result = estimate_normals_pca(mesh, radius_mm=30.0, min_neighbors=3)

        expected_normal = np.array([0.0, 0.0, 1.0])
        for i in range(0, n, 10):
            dot = abs(np.dot(result.normals[i], expected_normal))
            assert dot > 0.9, f"Normal {i} not aligned with Z: dot={dot}"


class TestSegmentation:
    def test_threshold_segmenter(self):
        from core.dataclasses import MRIData
        from core.head_segmenter import HeadSegmenter

        data = np.zeros((50, 50, 50), dtype=np.float64)
        for x in range(50):
            for y in range(50):
                for z in range(50):
                    if np.sqrt((x-25)**2 + (y-25)**2 + (z-25)**2) < 15:
                        data[x, y, z] = 200.0

        mri = MRIData(
            data=data,
            affine=np.eye(4),
            voxel_size=np.array([1.0, 1.0, 1.0]),
        )

        seg = HeadSegmenter(method="threshold", smooth_sigma=0.5, close_radius=2, min_component_size=100)
        result = seg.segment(mri)
        assert result.mask.sum() > 0
        assert result.method_name == "threshold"


class TestQualityControl:
    def test_stage2_qc(self):
        from core.ese_generator import ESEResult
        from core.quality_control import check_stage2

        n = 100
        verts = np.random.randn(n, 3) * 10
        ese_verts = verts + np.array([0, 0, 5.0])
        normals = np.tile([0, 0, 1.0], (n, 1))
        quality = np.full(n, 0.1)

        ese = ESEResult(
            ese_points=[],
            scalp_vertices=verts,
            ese_vertices=ese_verts,
            normals=normals,
            quality=quality,
            head_centroid=verts.mean(axis=0),
        )

        qc = check_stage2(ese=ese)
        assert len(qc.checks) > 0

    def test_stage1_qc_with_affine(self):
        from core.quality_control import check_stage1
        from core.surface_extractor import MeshData

        verts = np.random.randn(100, 3) * 10
        faces = np.random.randint(0, 100, (50, 3))
        mesh = MeshData(vertices=verts, faces=faces)

        qc = check_stage1(mri_affine=np.eye(4), mesh=mesh)
        assert qc.all_passed
