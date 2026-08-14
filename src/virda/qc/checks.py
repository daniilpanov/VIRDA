"""Automatic quality-control checks for a Stage 1 result (spec §13.1, §16)."""

import json
import logging
from collections import defaultdict
from itertools import product
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import label

from virda.geometry.transforms import fiducials_world_coordinates
from virda.models.ese_config import ESEConfig
from virda.models.fiducial import Fiducials
from virda.models.mri_volume import MRIVolume
from virda.models.quality_control_report import QualityControlReport
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage1_result import Stage1Result
from virda.pipeline_context import PipelineContext

CheckStatus = Literal["ok", "warn", "fail", "skip"]
FIDUCIAL_TOLERANCE_MM = 3.0
MAX_HOLE_DIAMETER_MM = 15.0
MIN_COMPONENT_VERTICES = 100
MIN_MESH_VERTICES = 100
ORIENTATION_AXIS_CODES = {"R", "L", "A", "P", "S", "I"}


def _check(name: str, status: CheckStatus, message: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, **details}


def check_mri(mri: MRIVolume) -> dict[str, Any]:
    """MRI affine/spacing/orientation are present and valid."""
    issues: list[str] = []
    if mri.affine.shape != (4, 4):
        issues.append(f"affine must be 4x4, got {mri.affine.shape}")
    elif not np.allclose(mri.affine[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        issues.append("affine bottom row is not [0, 0, 0, 1]")
    if len(mri.spacing) != 3 or any(s <= 0 for s in mri.spacing):
        issues.append(f"spacing must be 3 positive values, got {mri.spacing}")
    if len(mri.orientation) != 3 or any(c not in ORIENTATION_AXIS_CODES for c in mri.orientation):
        issues.append(f"orientation must have 3 axis codes, got {mri.orientation}")
    return _check(
        "mri_metadata",
        "fail" if issues else "ok",
        "; ".join(issues) if issues else "affine, spacing and orientation are valid",
        affine_shape=list(mri.affine.shape),
        spacing=list(mri.spacing),
        orientation=list(mri.orientation),
    )


def check_coordinates_mm(
    mesh: ScalpMesh, mri: MRIVolume, margin_mm: float | None = None
) -> dict[str, Any]:
    """Mesh coordinates are in the MRI world space (millimeters, spec §13.1).

    The scalp mesh is extracted from the MRI segmentation, so its vertices must
    fall inside the MRI volume's world bounding box. If they do not, the mesh is
    in a different unit or the voxel-to-world transform is broken (§16).
    """
    if mesh.vertices.shape[0] == 0:
        return _check(
            "coordinates_mm",
            "ok",
            "no vertices to check",
            mesh_bbox=[],
            volume_bbox=[],
        )
    margin = mri.spacing[0] * 2 if margin_mm is None else margin_mm
    volume_corners = np.array(
        [[voxel[i] * mri.data.shape[i] for i in range(3)] for voxel in product((0, 1), repeat=3)],
        dtype=np.float64,
    )
    volume_world = volume_corners @ mri.affine[:3, :3].T + mri.affine[:3, 3]
    volume_min = volume_world.min(axis=0)
    volume_max = volume_world.max(axis=0)

    mesh_min = np.asarray(mesh.vertices).min(axis=0)
    mesh_max = np.asarray(mesh.vertices).max(axis=0)
    inside = bool(
        np.all(mesh_min >= volume_min - margin) and np.all(mesh_max <= volume_max + margin)
    )
    extent_mm = (mesh_max - mesh_min).round(3).tolist()
    return _check(
        "coordinates_mm",
        "ok" if inside else "fail",
        (
            "mesh coordinates are consistent with the MRI world space"
            if inside
            else (
                "mesh coordinates are inconsistent with the MRI world space"
                " — likely not in millimeters"
            )
        ),
        mesh_bbox=[mesh_min.round(3).tolist(), mesh_max.round(3).tolist()],
        volume_bbox=[volume_min.round(3).tolist(), volume_max.round(3).tolist()],
        extent_mm=extent_mm,
    )


def check_ese_config(ese_config: ESEConfig | None) -> dict[str, Any]:
    """ESE offset is present and positive (spec §13.1).

    Skipped (not failed) when no ESE config is supplied: the offset is an
    ESE/simulation concern and its absence does not invalidate Stage 1 output.
    """
    if ese_config is None:
        return _check("ese_offset", "skip", "ESE config is missing; check skipped")
    return _check(
        "ese_offset",
        "ok" if ese_config.ese_offset_mm > 0 else "fail",
        (
            f"ESE offset is {ese_config.ese_offset_mm} mm"
            if ese_config.ese_offset_mm > 0
            else f"ESE offset must be positive, got {ese_config.ese_offset_mm} mm"
        ),
        ese_offset_mm=ese_config.ese_offset_mm,
        n_electrodes=ese_config.n_electrodes,
        ese_reference=ese_config.ese_reference,
    )


def _boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Undirected edges referenced by exactly one face (open boundaries)."""
    if faces.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).reshape(-1, 2)
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return np.asarray(unique_edges[counts == 1])


def check_mesh(mesh: ScalpMesh, min_vertices: int = MIN_MESH_VERTICES) -> dict[str, Any]:
    """Mesh is non-empty, has valid face indices and a reasonable density.

    Not implemented: spec §16 asks to warn when the mesh is non-manifold
    (an edge shared by more than two faces) — this is not detected here.
    """
    n_vertices = int(mesh.vertices.shape[0])
    n_faces = int(mesh.faces.shape[0])
    issues: list[str] = []
    if n_vertices == 0:
        issues.append("mesh has no vertices")
    if n_faces == 0:
        issues.append("mesh has no faces")
    elif mesh.faces.min() < 0 or mesh.faces.max() >= n_vertices:
        issues.append("face indices reference missing vertices")
    if n_vertices > 0 and n_vertices < min_vertices:
        issues.append(f"mesh is sparse ({n_vertices} < {min_vertices} vertices)")

    boundary_edges = int(_boundary_edges(mesh.faces).shape[0])
    severity: CheckStatus = (
        "fail"
        if any("mesh has no" in s or "face indices reference" in s for s in issues)
        else "warn"
    )
    return _check(
        "mesh",
        severity if issues else "ok",
        (
            "; ".join(issues)
            if issues
            else (
                f"mesh has {n_vertices} vertices and {n_faces} faces; "
                f"open boundary of {boundary_edges} edges (neck opening expected)"
            )
        ),
        n_vertices=n_vertices,
        n_faces=n_faces,
        boundary_edges=boundary_edges,
    )


def check_components(
    mask: np.ndarray, min_component_vertices: int = MIN_COMPONENT_VERTICES
) -> dict[str, Any]:
    """Warn if the segmentation contains several large connected components."""
    labels = label(mask)
    sizes = np.bincount(labels.ravel())
    large = sizes[sizes >= min_component_vertices][1:]
    n_components = int(len(sizes) - 1)
    if n_components == 0:
        return _check(
            "components",
            "fail",
            "segmentation is empty",
            n_components=0,
            large_components=[],
        )
    warnings: list[str] = []
    if len(large) > 1:
        warnings.append(
            f"segmentation has {len(large)} large components "
            f"(sizes {sorted(large.tolist(), reverse=True)[:5]})"
        )
    return _check(
        "components",
        "warn" if warnings else "ok",
        (
            warnings[0]
            if warnings
            else f"segmentation is a single large component ({n_components} total)"
        ),
        n_components=n_components,
        large_components=sorted(large.tolist(), reverse=True)[:5],
    )


def check_fiducials(
    fiducials: Fiducials,
    result: Stage1Result,
    tolerance_mm: float = FIDUCIAL_TOLERANCE_MM,
) -> dict[str, Any]:
    """Distance from each fiducial to the scalp mesh (per spec §13.1 / §16)."""
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not fiducials.items:
        return _check(
            "fiducials_on_surface",
            "warn",
            "no fiducials present",
            checks=checks,
            tolerance_mm=tolerance_mm,
            warnings=["No fiducials present"],
        )
    # Distance to the nearest vertex, not the true surface distance (spec §13.1
    # only asks for fiducials "near the external head surface" without fixing the
    # metric). Nearest-vertex distance is an upper bound on surface distance, so a
    # truly distant fiducial is never missed; the gap equals the local vertex
    # spacing (~1-2 mm on the dense Stage 1 mesh) and is negligible against the
    # 3 mm tolerance.
    tree = cKDTree(np.asarray(result.mesh.vertices, dtype=np.float64))
    for fiducial in fiducials.items:
        world = fiducials_world_coordinates([fiducial], result.mri_volume.affine)[0]
        distance = float(tree.query(world, k=1)[0])
        checks.append(
            {
                "fiducial_id": fiducial.fiducial_id,
                "name": fiducial.name,
                "distance_to_surface_mm": round(distance, 3),
            }
        )
        if distance > tolerance_mm:
            warnings.append(
                f"{fiducial.fiducial_id} is {distance:.1f} mm from the scalp surface "
                f"(tolerance {tolerance_mm} mm)"
            )
    return _check(
        "fiducials_on_surface",
        "warn" if warnings else "ok",
        warnings[0] if warnings else "all fiducials lie on the scalp surface",
        checks=checks,
        tolerance_mm=tolerance_mm,
        warnings=warnings,
    )


def _boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Closed chains of boundary edges; each loop is a list of vertex indices.

    At a non-manifold junction (vertex shared by multiple boundary edges) the
    walker follows ``neighbors[0]``, so loops may split or merge arbitrarily.
    """
    boundary = _boundary_edges(faces)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for start, end in boundary:
        adjacency[int(start)].append(int(end))
        adjacency[int(end)].append(int(start))

    used: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    for start, end in boundary:
        key = (int(start), int(end))
        reverse_key = (int(end), int(start))
        if key in used or reverse_key in used:
            continue
        loop = [int(start)]
        used.add(key)
        current, previous = int(end), int(start)
        while current != int(start):
            loop.append(current)
            neighbors = [v for v in adjacency.get(current, []) if v != previous]
            if not neighbors:
                break
            following = int(neighbors[0])
            used.add((current, following) if current <= following else (following, current))
            previous, current = current, following
        loops.append(loop)
    return loops


def _loop_diameter_mm(vertices: np.ndarray, loop: list[int]) -> float:
    points = vertices[loop]
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def check_holes(mesh: ScalpMesh, max_diameter_mm: float = MAX_HOLE_DIAMETER_MM) -> dict[str, Any]:
    """Warn if a boundary loop wider than ``max_diameter_mm`` sits over the scalp.

    The largest loop is assumed to be the neck opening and is excluded; any
    remaining loop wider than the threshold (spec §13.1) flags a hole over the
    scalp. If a genuine hole is wider than the neck opening it would be
    misclassified as the neck — acceptable for a QC heuristic.
    """
    loops = _boundary_loops(mesh.faces)
    if not loops:
        return _check(
            "holes_over_scalp",
            "ok",
            "mesh is watertight; no boundary loops",
            n_boundary_loops=0,
            diameters=[],
        )
    diameters = sorted((_loop_diameter_mm(mesh.vertices, loop) for loop in loops), reverse=True)
    large_loops = [diameter for diameter in diameters[1:] if diameter > max_diameter_mm]
    non_neck = len(loops) - 1
    return _check(
        "holes_over_scalp",
        "warn" if large_loops else "ok",
        (
            f"{len(large_loops)} non-neck loop(s) wider than {max_diameter_mm} mm"
            if large_loops
            else f"{non_neck} non-neck boundary loop(s); neck (largest) excluded"
        ),
        n_boundary_loops=len(loops),
        diameters=[round(diameter, 3) for diameter in diameters[:5]],
    )


def check_nifti_mask(path: str | Path, mask: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    """Exported NIfTI mask matches the in-memory segmentation."""
    import nibabel as nib

    mask_file = Path(path)
    if not mask_file.is_file():
        return _check("nifti_mask", "fail", f"segmentation mask file not found: {mask_file}")
    image = nib.load(mask_file)
    if not isinstance(image, nib.Nifti1Image):
        return _check(
            "nifti_mask",
            "fail",
            f"file is not a NIfTI image: {mask_file}",
            file=str(mask_file),
        )
    stored = np.asanyarray(image.dataobj)
    issues: list[str] = []
    if stored.shape != mask.shape:
        issues.append(f"shape mismatch: file {stored.shape} vs memory {mask.shape}")
    if not np.allclose(image.affine, affine):
        issues.append("affine mismatch between file and MRI volume")
    if stored.shape == mask.shape and stored.astype(bool).sum() != int(mask.sum()):
        issues.append(
            f"voxel count mismatch: file {int(stored.astype(bool).sum())} "
            f"vs memory {int(mask.sum())}"
        )
    return _check(
        "nifti_mask",
        "fail" if issues else "ok",
        "; ".join(issues) if issues else "exported mask matches the in-memory segmentation",
        file=str(mask_file),
        shape=list(stored.shape),
    )


def run_checks(
    result: Stage1Result,
    *,
    min_mesh_vertices: int = MIN_MESH_VERTICES,
    min_component_vertices: int = MIN_COMPONENT_VERTICES,
    fiducial_tolerance_mm: float = FIDUCIAL_TOLERANCE_MM,
    max_hole_diameter_mm: float = MAX_HOLE_DIAMETER_MM,
    nifti_mask_path: str | Path | None = None,
    ese_config: ESEConfig | None = None,
    coordinate_margin_mm: float | None = None,
) -> dict[str, Any]:
    """Run all automatic QC checks and aggregate a report.

    Each check reports one of the ``CheckStatus`` values: "ok", "warn",
    "fail" or "skip" (a check is skipped when it cannot apply, e.g. the
    missing ESE config). The overall status ignores skipped checks.
    """
    checks = [
        check_mri(result.mri_volume),
        check_coordinates_mm(result.mesh, result.mri_volume, margin_mm=coordinate_margin_mm),
        check_mesh(result.mesh, min_vertices=min_mesh_vertices),
        check_components(result.segmentation_mask.mask, min_component_vertices),
        check_holes(result.mesh, max_diameter_mm=max_hole_diameter_mm),
        check_ese_config(ese_config),
    ]
    if nifti_mask_path is not None:
        checks.append(
            check_nifti_mask(
                nifti_mask_path,
                result.segmentation_mask.mask,
                result.mri_volume.affine,
            )
        )
    fiducials = check_fiducials(result.fiducials, result, tolerance_mm=fiducial_tolerance_mm)

    warnings: list[str] = []
    for check in checks:
        if check["status"] == "warn":
            warnings.append(f"{check['name']}: {check['message']}")
        elif check["status"] == "fail":
            warnings.append(f"FAIL {check['name']}: {check['message']}")
    warnings.extend(fiducials["warnings"])

    statuses: list[CheckStatus] = [c["status"] for c in checks if c["status"] != "skip"] + [
        fiducials["status"]
    ]
    overall = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "ok")
    return {
        "status": overall,
        "checks": checks,
        "fiducials": fiducials,
        "warnings": warnings,
    }


class Stage1QualityControlStep:
    """Run automatic QC after the Stage 1 artifacts are exported and store the report."""

    def __init__(
        self,
        project_dir: Path,
        ese_config: ESEConfig | None = None,
        logger: Logger | None = None,
    ) -> None:
        self._project_dir = project_dir
        self._ese_config = ese_config
        self._logger = logger

    def run(self, context: PipelineContext) -> QualityControlReport:
        result = context.get_store_notnull(Stage1Result)
        report = run_checks(
            result,
            nifti_mask_path=self._project_dir / "segmentation" / "head_mask.nii.gz",
            ese_config=self._ese_config,
        )
        qc_dir = self._project_dir / "quality_control"
        qc_dir.mkdir(parents=True, exist_ok=True)
        (qc_dir / "report.json").write_text(json.dumps(report, indent=2))

        if report["status"] != "ok":
            logger = self._logger or logging.getLogger(__name__)
            logger.warning(
                f"Quality control {report['status'].upper()}: " + "; ".join(report["warnings"])
            )

        return QualityControlReport(report)
