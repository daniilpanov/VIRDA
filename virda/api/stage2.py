"""Stage 2 orchestration — mesh → PCA normals → ESE.

Returns objects, no file I/O. Use api.exporter to save results.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.surface_extractor import MeshData
from core.ese_config import ESEConfig
from core.pca_normal_estimator import NormalResult, estimate_normals_pca
from core.ese_generator import ESEResult, generate_ese
from core.quality_control import QCReport, check_stage2

logger = logging.getLogger(__name__)


def run_stage2(
    mesh: MeshData,
    *,
    radius_mm: float = 10.0,
    k_neighbors: Optional[int] = None,
    min_neighbors: int = 5,
    weighted: bool = False,
    weight_sigma: float = 5.0,
    ese_config: Optional[ESEConfig] = None,
) -> tuple[ESEResult, NormalResult, QCReport]:
    """Run Stage 2: PCA normals and ESE construction.

    Parameters
    ----------
    mesh : MeshData
        Cleaned scalp mesh from Stage 1.
    radius_mm : float
        Neighborhood radius for PCA in mm.
    k_neighbors : int, optional
        Use k nearest neighbors instead of radius.
    min_neighbors : int
        Minimum neighbors for valid PCA.
    weighted : bool
        Use distance-weighted PCA.
    weight_sigma : float
        Gaussian sigma for weighting.
    ese_config : ESEConfig, optional
        ESE configuration. Defaults to ESEConfig().

    Returns
    -------
    tuple
        (ese, normal_result, qc_report)
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Electrode Surface Equivalent Construction")
    logger.info("=" * 60)

    if ese_config is None:
        from core.ese_config import ESEConfig as _ESEConfig
        ese_config = _ESEConfig()

    normal_result = estimate_normals_pca(
        mesh=mesh,
        radius_mm=radius_mm,
        k_neighbors=k_neighbors,
        min_neighbors=min_neighbors,
        weighted=weighted,
        weight_sigma=weight_sigma,
    )
    logger.info("Normals estimated: median quality=%.4f", float(normal_result.quality.mean()))

    ese = generate_ese(
        mesh=mesh,
        normal_result=normal_result,
        config=ese_config,
    )
    logger.info("ESE generated: %d points", ese.num_points)

    qc = check_stage2(ese=ese, normal_result=normal_result)
    logger.info("\n%s", qc.summary())

    return ese, normal_result, qc
