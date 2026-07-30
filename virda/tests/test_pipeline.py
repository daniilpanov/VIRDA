"""Integration test: full pipeline on synthetic sphere."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from virda.core.ese_config import ESEConfig
from virda.core.ese_generator import generate_ese
from virda.core.electrode_localizer import localize_electrodes
from virda.core.fiducial_manager import FiducialManager
from virda.core.head_segmenter import ThresholdSegmenter
from virda.core.measurement_importer import MeasurementImporter
from virda.core.mesh_cleaner import clean_mesh
from virda.core.mri_loader import MRIData
from virda.core.pca_normal_estimator import estimate_normals_pca
from virda.core.quality_control import validate_stage1, validate_stage2
from virda.core.surface_extractor import extract_surface


def _create_sphere_volume(radius: int = 25, size: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic MRI-like volume with a bright sphere on dark background."""
    volume = np.zeros((size, size, size), dtype=np.float64)
    cx = cy = cz = size // 2
    xx, yy, zz = np.mgrid[:size, :size, :size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    volume[dist <= radius] = 800.0
    volume += np.random.default_rng(0).normal(0, 10, volume.shape)
    affine = np.eye(4)
    return volume, affine


class TestPipeline:
    def test_stage1_on_sphere(self):
        volume, affine = _create_sphere_volume()
        mri = MRIData(
            volume=volume,
            affine=affine,
            voxel_size=np.array([1.0, 1.0, 1.0]),
        )

        seg = ThresholdSegmenter()
        seg_result = seg.segment(mri)
        assert seg_result.mask.sum() > 0

        mesh = extract_surface(seg_result.mask, mri.voxel_size, affine=mri.affine)
        assert mesh.num_vertices > 100

        cleaned, stats = clean_mesh(mesh)
        assert cleaned.num_vertices > 0

        msgs = validate_stage1(mri=mri, mesh=cleaned, ese_offset_mm=5.0)
        errors = [m for m in msgs if "ERROR" in m]
        assert len(errors) == 0

    def test_full_pipeline_on_sphere(self):
        volume, affine = _create_sphere_volume()
        mri = MRIData(
            volume=volume,
            affine=affine,
            voxel_size=np.array([1.0, 1.0, 1.0]),
        )

        seg = ThresholdSegmenter()
        seg_result = seg.segment(mri)
        mesh = extract_surface(seg_result.mask, mri.voxel_size, affine=mri.affine)
        mesh, _ = clean_mesh(mesh)

        normal_result = estimate_normals_pca(mesh, radius_mm=8.0, min_neighbors=3)
        assert normal_result.normals.shape == (mesh.num_vertices, 3)

        config = ESEConfig(offset_mm=5.0)
        ese = generate_ese(mesh, normal_result, config)
        assert ese.num_points == mesh.num_vertices

        qc2 = validate_stage2(ese)
        errors2 = [m for m in qc2 if "ERROR" in m]
        assert len(errors2) == 0

        centroid = mesh.vertices.mean(axis=0)
        fiducial_mgr = FiducialManager(
            head_centroid=centroid,
            surface_vertices=mesh.vertices,
        )
        fiducial_mgr.add_fiducial("NAS", "Nasion", centroid + np.array([25.0, 0.0, 0.0]))
        fiducial_mgr.add_fiducial("LPA", "Left", centroid + np.array([0.0, 25.0, 0.0]))
        fiducial_mgr.add_fiducial("RPA", "Right", centroid + np.array([0.0, -25.0, 0.0]))

        fid_coords = fiducial_mgr.get_coordinates_matrix(["NAS", "LPA", "RPA"])
        importer = MeasurementImporter(["NAS", "LPA", "RPA"])
        target_idx = ese.num_points // 2
        for i, label in enumerate(["E1", "E2"]):
            idx = target_idx + i * 10
            if idx < ese.num_points:
                dists = np.linalg.norm(fid_coords - ese.ese_vertices[idx], axis=1)
                importer.add_measurement(label, {
                    "NAS": float(dists[0]),
                    "LPA": float(dists[1]),
                    "RPA": float(dists[2]),
                })

        result = localize_electrodes(
            ese=ese,
            fiducial_mgr=fiducial_mgr,
            measurements=importer.get_all_measurements(),
        )
        assert result.num_electrodes > 0
        for loc in result.electrodes:
            assert loc.residual_error < 1e-6
