"""Stage 1 orchestration — MRI → segmentation → mesh → clean.

Returns objects, no file I/O. Use api.exporter to save results.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.dataclasses import MRIData, SegmentationData
from core.head_segmenter import HeadSegmenter
from core.surface_extractor import MeshData, extract_surface
from core.mesh_cleaner import clean_mesh
from core.fiducial_manager import FiducialManager
from core.ese_config import ESEConfig
from core.quality_control import QCReport, check_stage1
from api.mri_loader import load_mri

logger = logging.getLogger(__name__)


def run_stage1(
    mri_path: str,
    *,
    segmentation_method: str = "threshold",
    smooth_sigma: float = 1.0,
    close_radius: int = 3,
    min_component_size: int = 10000,
    mesh_smooth_iterations: int = 0,
    mesh_smooth_lambda: float = 0.5,
    ese_offset_mm: float = 5.0,
    ese_reference_point: str = "center_of_external_surface",
) -> tuple[MeshData, FiducialManager, ESEConfig, SegmentationData, MRIData, QCReport]:
    """Run Stage 1: load MRI, segment head, extract and clean mesh.

    Returns
    -------
    tuple
        (mesh, fiducial_mgr, ese_config, segmentation, mri, qc_report)

        - mesh: cleaned scalp mesh
        - fiducial_mgr: empty fiducial manager (user adds fiducials)
        - ese_config: ESE configuration with offset
        - segmentation: binary segmentation data
        - mri: loaded MRI data (for QC / visualization)
        - qc_report: quality control results
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: MRI Head Surface Mesh Generation")
    logger.info("=" * 60)

    mri = load_mri(mri_path)
    logger.info("MRI loaded: shape=%s, voxel_size=%s", mri.shape, mri.get_voxel_spacing())

    seg = HeadSegmenter(
        method=segmentation_method,
        smooth_sigma=smooth_sigma,
        close_radius=close_radius,
        min_component_size=min_component_size,
    )
    segmentation = seg.segment(mri)
    logger.info("Segmentation complete: %d voxels", int(segmentation.mask.sum()))

    mesh = extract_surface(
        mask=segmentation.mask,
        voxel_size=segmentation.voxel_size,
        affine=mri.affine,
    )
    logger.info("Mesh extracted: %d vertices, %d faces", mesh.num_vertices, mesh.num_faces)

    mesh, _ = clean_mesh(
        mesh,
        smooth_iterations=mesh_smooth_iterations,
        smooth_lambda=mesh_smooth_lambda,
    )
    logger.info("Mesh cleaned: %d vertices, %d faces", mesh.num_vertices, mesh.num_faces)

    fiducial_mgr = FiducialManager(
        head_centroid=mesh.vertices.mean(axis=0),
        surface_vertices=mesh.vertices,
    )

    ese_config = ESEConfig(
        offset_mm=ese_offset_mm,
        reference_point=ese_reference_point,
    )

    qc = check_stage1(
        mri_affine=mri.affine,
        mesh=mesh,
        fiducial_mgr=fiducial_mgr,
        ese_config=ese_config,
        segmentation_mask=segmentation.mask,
    )
    logger.info("\n%s", qc.summary())

    return mesh, fiducial_mgr, ese_config, segmentation, mri, qc
