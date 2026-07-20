"""Quality control module — automatic and visual checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .surface_extractor import MeshData
from .fiducial_manager import FiducialManager
from .ese_config import ESEConfig
from .ese_generator import ESEResult
from .pca_normal_estimator import NormalResult
from .electrode_localizer import LocalizationResult

logger = logging.getLogger(__name__)


@dataclass
class QCCheck:
    """Single quality check result."""

    name: str
    passed: bool
    message: str
    severity: str = "warning"


@dataclass
class QCReport:
    """Quality control report."""

    checks: list[QCCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def warnings(self) -> list[QCCheck]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def errors(self) -> list[QCCheck]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    def add(self, name: str, passed: bool, message: str, severity: str = "warning"):
        self.checks.append(QCCheck(name=name, passed=passed, message=message, severity=severity))
        status = "PASS" if passed else f"FAIL ({severity})"
        logger.info("QC [%s] %s: %s", status, name, message)

    def summary(self) -> str:
        lines = [f"QC Report: {sum(c.passed for c in self.checks)}/{len(self.checks)} passed"]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{status}] {c.name}: {c.message}")
        return "\n".join(lines)


def check_stage1(
    mri_affine: Optional[np.ndarray] = None,
    mesh: Optional[MeshData] = None,
    fiducial_mgr: Optional[FiducialManager] = None,
    ese_config: Optional[ESEConfig] = None,
    segmentation_mask: Optional[np.ndarray] = None,
) -> QCReport:
    """Run Stage 1 quality checks.

    Parameters
    ----------
    mri_affine : np.ndarray, optional
        MRI affine matrix (4×4). If provided, validates it.
    mesh : MeshData, optional
        Extracted mesh.
    fiducial_mgr : FiducialManager, optional
        Fiducial manager.
    ese_config : ESEConfig, optional
        ESE configuration.
    segmentation_mask : np.ndarray, optional
        Binary mask.
    """
    report = QCReport()

    if mri_affine is not None:
        is_valid = not np.any(np.isnan(mri_affine)) and not np.any(np.isinf(mri_affine))
        report.add("MRI affine valid", is_valid, "Affine matrix contains NaN/Inf", "error")

        has_spatial = np.all(np.abs(np.diag(mri_affine[:3, :3])) > 0)
        report.add("MRI spatial metadata", has_spatial, "Voxel-to-world transform looks valid")

    if mesh is not None:
        has_vertices = mesh.num_vertices > 0
        report.add("Mesh has vertices", has_vertices, f"Vertex count: {mesh.num_vertices}", "error")

        has_faces = mesh.num_faces > 0
        report.add("Mesh has faces", has_faces, f"Face count: {mesh.num_faces}", "error")

        if has_vertices:
            mins = mesh.vertices.min(axis=0)
            maxs = mesh.vertices.max(axis=0)
            extent = maxs - mins
            reasonable = np.all(extent > 0) and np.all(extent < 1000)
            report.add(
                "Mesh extent reasonable",
                reasonable,
                f"Extent: X={extent[0]:.1f}, Y={extent[1]:.1f}, Z={extent[2]:.1f} mm",
            )

        if mesh.num_faces > 0:
            valid_faces = np.all(mesh.faces >= 0) and np.all(mesh.faces < mesh.num_vertices)
            report.add("Face indices valid", valid_faces, "All face indices within vertex range")

    if fiducial_mgr is not None:
        fid_warnings = fiducial_mgr.validate()
        has_enough = len(fiducial_mgr.get_all_fiducials()) >= 3
        report.add(
            "Enough fiducials",
            has_enough,
            f"Defined: {len(fiducial_mgr.get_all_fiducials())} (need >=3)",
            "error",
        )
        for w in fid_warnings:
            report.add("Fiducial check", False, w)

    if ese_config is not None:
        positive_offset = ese_config.offset_mm > 0
        report.add(
            "ESE offset positive",
            positive_offset,
            f"Offset: {ese_config.offset_mm} mm",
            "error",
        )

    if segmentation_mask is not None:
        has_content = segmentation_mask.sum() > 0
        report.add("Segmentation non-empty", has_content, f"Voxels: {int(segmentation_mask.sum())}", "error")

    return report


def check_stage2(
    ese: ESEResult,
    normal_result: Optional[NormalResult] = None,
) -> QCReport:
    """Run Stage 2 quality checks."""
    report = QCReport()

    report.add("ESE has points", ese.num_points > 0, f"Point count: {ese.num_points}", "error")

    if ese.num_points > 0:
        scalp_ese_dist = np.linalg.norm(ese.ese_vertices - ese.scalp_vertices, axis=1)
        mean_offset = scalp_ese_dist.mean()
        std_offset = scalp_ese_dist.std()

        report.add(
            "Offset consistency",
            std_offset < mean_offset * 0.1,
            f"Mean offset: {mean_offset:.2f} mm, std: {std_offset:.2f} mm",
        )

        all_outward = True
        for i in range(ese.num_points):
            vec = ese.ese_vertices[i] - ese.head_centroid
            if np.dot(ese.normals[i], vec) < 0:
                all_outward = False
                break
        report.add("All normals outward", all_outward, "All normals point away from head centroid")

    if normal_result is not None:
        median_quality = float(np.median(normal_result.quality))
        report.add(
            "Normal quality",
            median_quality < 0.5,
            f"Median quality: {median_quality:.4f} (lower = more planar = better)",
        )

        num_outliers = int((normal_result.quality > 0.8).sum())
        report.add(
            "Few normal outliers",
            num_outliers < len(normal_result.quality) * 0.05,
            f"Outliers (quality>0.8): {num_outliers}/{len(normal_result.quality)}",
        )

    return report


def check_stage3(result: LocalizationResult) -> QCReport:
    """Run Stage 3 quality checks."""
    report = QCReport()

    report.add(
        "Electrodes localized",
        result.num_electrodes > 0,
        f"Count: {result.num_electrodes}",
        "error",
    )

    report.add(
        "Mean residual acceptable",
        result.mean_residual < 20.0,
        f"Mean residual: {result.mean_residual:.2f} mm",
    )

    report.add(
        "Max residual acceptable",
        result.max_residual < 50.0,
        f"Max residual: {result.max_residual:.2f} mm",
    )

    num_flagged = len(result.flagged_electrodes)
    report.add(
        "Few flagged electrodes",
        num_flagged < result.num_electrodes * 0.2,
        f"Flagged: {num_flagged}/{result.num_electrodes}",
    )

    if result.num_electrodes > 0:
        confidences = [e.confidence for e in result.electrodes]
        min_conf = min(confidences)
        report.add(
            "Minimum confidence",
            min_conf > 0.01,
            f"Min confidence: {min_conf:.4f}",
        )

    return report
