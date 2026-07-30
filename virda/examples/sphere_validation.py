"""Sphere validation example — runs the full pipeline on a synthetic sphere.

Usage::

    python -m virda.examples.sphere_validation
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from virda.core.ese_config import ESEConfig
from virda.core.ese_generator import generate_ese
from virda.core.electrode_localizer import localize_electrodes
from virda.core.fiducial_manager import FiducialManager
from virda.core.head_segmenter import ThresholdSegmenter
from virda.core.measurement_importer import MeasurementImporter
from virda.core.mesh_cleaner import clean_mesh
from virda.core.mri_loader import MRIData
from virda.core.pca_normal_estimator import estimate_normals_pca
from virda.core.quality_control import validate_stage1, validate_stage2, validate_stage3
from virda.core.surface_extractor import extract_surface


def create_sphere_volume(radius: int = 25, size: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic MRI-like volume with a bright sphere on dark background."""
    volume = np.zeros((size, size, size), dtype=np.float64)
    cx = cy = cz = size // 2
    xx, yy, zz = np.mgrid[:size, :size, :size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    volume[dist <= radius] = 800.0
    volume += np.random.default_rng(0).normal(0, 10, volume.shape)
    affine = np.eye(4)
    return volume, affine


def main() -> int:
    print("=" * 60)
    print("VIRDA Sphere Validation")
    print("=" * 60)

    # Stage 1
    print("\n[Stage 1] MRI Head Surface Mesh Generation")
    print("  Creating synthetic sphere volume...")
    volume, affine = create_sphere_volume(radius=25, size=60)
    mri = MRIData(
        volume=volume,
        affine=affine,
        voxel_size=np.array([1.0, 1.0, 1.0]),
    )

    print("  Segmenting head...")
    seg = ThresholdSegmenter()
    seg_result = seg.segment(mri)
    print(f"    Segmentation voxels: {seg_result.mask.sum()}")

    print("  Extracting surface mesh...")
    mesh = extract_surface(seg_result.mask, mri.voxel_size, affine=mri.affine)
    print(f"    Vertices: {mesh.num_vertices}, Faces: {mesh.num_faces}")

    print("  Cleaning mesh...")
    mesh, stats = clean_mesh(mesh)
    print(f"    After cleaning: {stats['final_vertices']} vertices, {stats['final_faces']} faces")

    qc1 = validate_stage1(mri=mri, mesh=mesh, ese_offset_mm=5.0)
    if qc1:
        for msg in qc1:
            print(f"    QC: {msg}")
    else:
        print("    QC: All checks passed")

    # Stage 2
    print("\n[Stage 2] ESE Construction")
    print("  Estimating PCA normals (radius=8.0 mm)...")
    normal_result = estimate_normals_pca(mesh, radius_mm=8.0, min_neighbors=3)
    print(f"    Mean quality: {normal_result.quality.mean():.4f}")

    config = ESEConfig(offset_mm=5.0)
    print(f"  Generating ESE (offset={config.offset_mm} mm)...")
    ese = generate_ese(mesh, normal_result, config)

    qc2 = validate_stage2(ese)
    if qc2:
        for msg in qc2:
            print(f"    QC: {msg}")
    else:
        print("    QC: All checks passed")

    # Stage 3
    print("\n[Stage 3] Electrode Localization")
    centroid = mesh.vertices.mean(axis=0)
    fiducial_mgr = FiducialManager(head_centroid=centroid, surface_vertices=mesh.vertices)
    fiducial_mgr.add_fiducial("NAS", "Nasion", centroid + np.array([25.0, 0.0, 0.0]))
    fiducial_mgr.add_fiducial("LPA", "Left Preauricular", centroid + np.array([0.0, 25.0, 0.0]))
    fiducial_mgr.add_fiducial("RPA", "Right Preauricular", centroid + np.array([0.0, -25.0, 0.0]))
    print("  Fiducials: NAS, LPA, RPA")

    fid_coords = fiducial_mgr.get_coordinates_matrix(["NAS", "LPA", "RPA"])
    importer = MeasurementImporter(["NAS", "LPA", "RPA"])

    test_points = [ese.num_points // 4, ese.num_points // 2, 3 * ese.num_points // 4]
    for i, idx in enumerate(test_points):
        if idx < ese.num_points:
            dists = np.linalg.norm(fid_coords - ese.ese_vertices[idx], axis=1)
            importer.add_measurement(f"E{i + 1}", {
                "NAS": float(dists[0]),
                "LPA": float(dists[1]),
                "RPA": float(dists[2]),
            })

    print(f"  Measurements: {len(importer.get_all_measurements())} electrodes")

    print("  Localizing electrodes...")
    result = localize_electrodes(
        ese=ese,
        fiducial_mgr=fiducial_mgr,
        measurements=importer.get_all_measurements(),
    )

    qc3 = validate_stage3(result)
    if qc3:
        for msg in qc3:
            print(f"    QC: {msg}")
    else:
        print("    QC: All checks passed")

    print(f"\n  Results:")
    for loc in result.electrodes:
        print(f"    {loc.electrode_id}: residual={loc.residual_error:.6f} mm, "
              f"coords=({loc.ese_coords[0]:.1f}, {loc.ese_coords[1]:.1f}, {loc.ese_coords[2]:.1f})")

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
