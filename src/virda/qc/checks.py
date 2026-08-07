"""Automatic quality-control checks for a Stage 1 result (spec §13.1, §16)."""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import label

from virda.geometry.transforms import fiducials_world_coordinates
from virda.models.fiducial import Fiducial
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage1_result import Stage1Result

FIDUCIAL_TOLERANCE_MM = 3.0
MIN_MESH_VERTICES = 100
MIN_COMPONENT_VERTICES = 100


def _check(name: str, status: str, message: str, **details: Any) -> dict[str, Any]:
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
    if len(mri.orientation) != 3 or any(not c for c in mri.orientation):
        issues.append(f"orientation must have 3 axis codes, got {mri.orientation}")
    return _check(
        "mri_metadata",
        "fail" if issues else "ok",
        "; ".join(issues) if issues else "affine, spacing and orientation are valid",
        affine_shape=list(mri.affine.shape),
        spacing=list(mri.spacing),
        orientation=list(mri.orientation),
    )


def check_mesh(mesh: ScalpMesh, min_vertices: int = MIN_MESH_VERTICES) -> dict[str, Any]:
    """Mesh is non-empty, has valid face indices and a reasonable density."""
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

    boundary_edges = 0
    if n_faces > 0:
        edges = np.stack(
            [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]
        ).reshape(-1, 2)
        edges = np.sort(edges, axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = int((counts == 1).sum())

    severity = (
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
    fiducials: list[Fiducial],
    result: Stage1Result,
    tolerance_mm: float = FIDUCIAL_TOLERANCE_MM,
) -> dict[str, Any]:
    """Distance from each fiducial to the scalp mesh (per spec §13.1 / §16)."""
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not fiducials:
        return _check(
            "fiducials_on_surface",
            "warn",
            "no fiducials present",
            checks=checks,
            tolerance_mm=tolerance_mm,
            warnings=["No fiducials present"],
        )
    tree = cKDTree(np.asarray(result.mesh.vertices, dtype=np.float64))
    for fiducial in fiducials:
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


def check_nifti_mask(path: str | Path, mask: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    """Exported NIfTI mask matches the in-memory segmentation."""
    import nibabel as nib

    mask_file = Path(path)
    if not mask_file.is_file():
        return _check("nifti_mask", "fail", f"segmentation mask file not found: {mask_file}")
    image = nib.load(mask_file)
    assert isinstance(image, nib.Nifti1Image)
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
    nifti_mask_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run all automatic QC checks and aggregate a report."""
    checks = [
        check_mri(result.mri_volume),
        check_mesh(result.mesh, min_vertices=min_mesh_vertices),
        check_components(result.segmentation_mask, min_component_vertices),
    ]
    if nifti_mask_path is not None:
        checks.append(
            check_nifti_mask(nifti_mask_path, result.segmentation_mask, result.mri_volume.affine)
        )
    fiducials = check_fiducials(result.fiducials, result, tolerance_mm=fiducial_tolerance_mm)

    warnings: list[str] = []
    for check in checks:
        if check["status"] == "warn":
            warnings.append(f"{check['name']}: {check['message']}")
        elif check["status"] == "fail":
            warnings.append(f"FAIL {check['name']}: {check['message']}")
    warnings.extend(fiducials["warnings"])

    statuses = [check["status"] for check in checks] + [fiducials["status"]]
    overall = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "ok")
    return {
        "status": overall,
        "checks": checks,
        "fiducials": fiducials,
        "warnings": warnings,
    }
