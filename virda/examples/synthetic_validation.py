"""Synthetic validation — test pipeline with a sphere before real MRI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.surface_extractor import MeshData, extract_surface
from core.mesh_cleaner import clean_mesh
from core.pca_normal_estimator import estimate_normals_pca
from core.ese_config import ESEConfig
from core.ese_generator import generate_ese
from core.fiducial_manager import FiducialManager
from core.measurement_importer import MeasurementImporter
from core.electrode_localizer import localize_electrodes
from core.quality_control import check_stage2, check_stage3
from api import exporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_sphere_volume(
    radius_voxels: int = 40,
    center_offset: int = 50,
    voxel_size_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a binary sphere volume for testing."""
    size = 2 * (radius_voxels + center_offset)
    mask = np.zeros((size, size, size), dtype=np.int32)

    cx = cy = cz = size // 2

    for x in range(size):
        for y in range(size):
            for z in range(size):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                if dist <= radius_voxels:
                    mask[x, y, z] = 1

    affine = np.eye(4) * voxel_size_mm
    affine[3, 3] = 1.0

    center_world = np.array([cx * voxel_size_mm, cy * voxel_size_mm, cz * voxel_size_mm])

    return mask, affine, center_world


def test_mesh_generation():
    """Test sphere mesh extraction and cleaning."""
    logger.info("=" * 60)
    logger.info("TEST: Sphere mesh generation")
    logger.info("=" * 60)

    mask, affine, center = create_sphere_volume(radius_voxels=30, voxel_size_mm=1.0)
    voxel_size = np.array([1.0, 1.0, 1.0])

    mesh = extract_surface(mask, voxel_size, affine=affine)
    assert mesh.num_vertices > 0, "Mesh has no vertices"
    assert mesh.num_faces > 0, "Mesh has no faces"
    logger.info("Mesh: %d vertices, %d faces", mesh.num_vertices, mesh.num_faces)

    mesh_clean, report = clean_mesh(mesh)
    logger.info("Cleaned: %d vertices, %d faces", mesh_clean.num_vertices, mesh_clean.num_faces)

    mins = mesh_clean.vertices.min(axis=0)
    maxs = mesh_clean.vertices.max(axis=0)
    extent = maxs - mins
    logger.info("Extent: %.1f x %.1f x %.1f mm", *extent)

    expected_diameter = 60.0
    for i in range(3):
        assert extent[i] > expected_diameter * 0.8, f"Axis {i} extent too small: {extent[i]}"
        assert extent[i] < expected_diameter * 1.5, f"Axis {i} extent too large: {extent[i]}"

    logger.info("PASS: Mesh generation")
    return mesh_clean, center


def test_pca_normals(mesh: MeshData, center: np.ndarray):
    """Test PCA normal estimation on sphere."""
    logger.info("=" * 60)
    logger.info("TEST: PCA normals on sphere")
    logger.info("=" * 60)

    result = estimate_normals_pca(mesh, radius_mm=10.0, min_neighbors=5)

    assert result.normals.shape == (mesh.num_vertices, 3)
    assert result.quality.shape == (mesh.num_vertices,)

    centroid = mesh.vertices.mean(axis=0)
    dot_products = np.sum(result.normals * (mesh.vertices - centroid), axis=1)
    outward_fraction = (dot_products > 0).mean()

    logger.info("Outward fraction: %.3f", outward_fraction)
    assert outward_fraction > 0.8, f"Only {outward_fraction:.1%} normals point outward"

    median_quality = np.median(result.quality)
    logger.info("Median PCA quality: %.4f", median_quality)
    assert median_quality < 0.5, f"Median quality too high: {median_quality}"

    logger.info("PASS: PCA normals")
    return result


def test_ese_generation(mesh: MeshData, normal_result, center: np.ndarray):
    """Test ESE point generation."""
    logger.info("=" * 60)
    logger.info("TEST: ESE generation")
    logger.info("=" * 60)

    config = ESEConfig(offset_mm=5.0)
    ese = generate_ese(mesh, normal_result, config)

    assert ese.num_points == mesh.num_vertices

    scalp_ese_dist = np.linalg.norm(ese.ese_vertices - ese.scalp_vertices, axis=1)
    mean_dist = scalp_ese_dist.mean()
    std_dist = scalp_ese_dist.std()

    logger.info("Scalp->ESE distance: %.2f +/- %.2f mm (expected ~5.0)", mean_dist, std_dist)
    assert abs(mean_dist - 5.0) < 1.0, f"Mean offset {mean_dist} too far from 5.0"
    assert std_dist < 2.0, f"Std offset {std_dist} too high"

    logger.info("PASS: ESE generation")
    return ese


def test_localization(ese, center: np.ndarray):
    """Test electrode localization with synthetic distances."""
    logger.info("=" * 60)
    logger.info("TEST: Electrode localization")
    logger.info("=" * 60)

    fiducial_mgr = FiducialManager(head_centroid=ese.head_centroid, surface_vertices=ese.scalp_vertices)

    fiducial_mgr.add_fiducial("NAS", "Nasion", center + np.array([30.0, 0.0, 0.0]))
    fiducial_mgr.add_fiducial("LPA", "Left Preauricular", center + np.array([0.0, 30.0, 0.0]))
    fiducial_mgr.add_fiducial("RPA", "Right Preauricular", center + np.array([0.0, -30.0, 0.0]))

    fid_coords = fiducial_mgr.get_coordinates_matrix(["NAS", "LPA", "RPA"])

    n_pts = ese.num_points
    idx1 = n_pts // 4
    idx2 = n_pts // 2
    idx3 = 3 * n_pts // 4
    test_points = [
        ese.ese_vertices[idx1].copy(),
        ese.ese_vertices[idx2].copy(),
        ese.ese_vertices[idx3].copy(),
    ]

    importer = MeasurementImporter(["NAS", "LPA", "RPA"])

    for i, point in enumerate(test_points):
        dists = np.linalg.norm(fid_coords - point, axis=1)
        importer.add_measurement(
            f"E{i+1}",
            {"NAS": float(dists[0]), "LPA": float(dists[1]), "RPA": float(dists[2])},
        )

    measurements = importer.get_all_measurements()

    result = localize_electrodes(
        ese=ese,
        fiducial_mgr=fiducial_mgr,
        measurements=measurements,
        max_residual_threshold=5.0,
    )

    assert result.num_electrodes == 3
    logger.info("Mean residual: %.4f mm", result.mean_residual)
    assert result.mean_residual < 5.0, f"Mean residual {result.mean_residual} too high"

    for loc in result.electrodes:
        logger.info(
            "  %s: error=%.4f mm, confidence=%.4f",
            loc.electrode_id,
            loc.residual_error,
            loc.confidence,
        )

    logger.info("PASS: Localization")
    return result


def test_export(project_dir: Path, mesh, normal_result, ese, localization, fiducial_mgr):
    """Test export functionality."""
    logger.info("=" * 60)
    logger.info("TEST: Export")
    logger.info("=" * 60)

    exporter.export_mesh_ply(mesh, project_dir / "mesh" / "scalp.ply")
    exporter.export_vertices_csv(mesh, project_dir / "mesh" / "vertices.csv")
    exporter.export_faces_csv(mesh, project_dir / "mesh" / "faces.csv")
    fiducial_mgr.save(project_dir / "fiducials" / "fiducials.json")
    ESEConfig().save(project_dir / "config" / "parameters.json")
    exporter.export_ese_pairs(ese, project_dir / "mesh" / "ese_pairs.csv")
    exporter.export_normals_csv(normal_result, mesh, project_dir / "mesh" / "normals.csv")
    exporter.export_localization_csv(localization, project_dir / "electrodes.csv")
    exporter.export_localization_json(localization, project_dir / "electrodes.json")

    assert (project_dir / "mesh" / "scalp.ply").exists()
    assert (project_dir / "fiducials" / "fiducials.json").exists()
    assert (project_dir / "config" / "parameters.json").exists()
    assert (project_dir / "mesh" / "ese_pairs.csv").exists()
    assert (project_dir / "electrodes.csv").exists()
    assert (project_dir / "electrodes.json").exists()

    logger.info("All export files created successfully")
    logger.info("PASS: Export")


def main():
    output_dir = Path("examples") / "synthetic_validation"
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    mesh, center = test_mesh_generation()
    normal_result = test_pca_normals(mesh, center)
    ese = test_ese_generation(mesh, normal_result, center)
    localization = test_localization(ese, center)

    fiducial_mgr = FiducialManager(head_centroid=ese.head_centroid)
    fiducial_mgr.add_fiducial("NAS", "Nasion", center + np.array([30.0, 0.0, 0.0]))
    fiducial_mgr.add_fiducial("LPA", "Left Preauricular", center + np.array([0.0, 30.0, 0.0]))
    fiducial_mgr.add_fiducial("RPA", "Right Preauricular", center + np.array([0.0, -30.0, 0.0]))

    test_export(output_dir, mesh, normal_result, ese, localization, fiducial_mgr)

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
