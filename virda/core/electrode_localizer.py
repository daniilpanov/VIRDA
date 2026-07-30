"""Electrode localization by distance matching on the ESE."""

from __future__ import annotations

import logging

import numpy as np

from .fiducial_manager import FiducialManager
from .types import (
    ElectrodeLocalization,
    ESEResult,
    LocalizationResult,
)

logger = logging.getLogger(__name__)


def localize_electrodes(
    ese: ESEResult,
    fiducial_mgr: FiducialManager,
    measurements: dict[str, dict[str, float]],
    max_residual_threshold: float = 5.0,
    fiducial_ids: list[str] | None = None,
) -> LocalizationResult:
    """Localize electrodes by matching distances to fiducials on the ESE.

    Parameters
    ----------
    ese : ESEResult
        ESE surface with scalp-to-ESE point pairs.
    fiducial_mgr : FiducialManager
        Fiducial manager with coordinates.
    measurements : dict[str, dict[str, float]]
        Mapping: electrode_id -> {fiducial_id: distance_mm}.
    max_residual_threshold : float
        Residual error above which electrodes are flagged.
    fiducial_ids : list[str], optional
        Fiducial IDs to use. If None, uses all available.

    Returns
    -------
    LocalizationResult
        Localization results for all electrodes.
    """
    if fiducial_ids is None:
        fiducial_ids = list(measurements[next(iter(measurements))].keys())

    fid_coords = fiducial_mgr.get_coordinates_matrix(fiducial_ids)
    ese_verts = ese.ese_vertices
    scalp_verts = ese.scalp_vertices

    localizations: list[ElectrodeLocalization] = []

    for eid, dists in measurements.items():
        measured_dist = np.array(
            [dists.get(fid, np.nan) for fid in fiducial_ids],
            dtype=np.float64,
        )

        valid_mask = ~np.isnan(measured_dist)
        if not valid_mask.any():
            logger.warning("Electrode %s: no valid measurements, skipping", eid)
            continue

        valid_fid_coords = fid_coords[valid_mask]
        valid_measured = measured_dist[valid_mask]

        ese_coords, scalp_coords, residual = _localize_single(
            ese_verts, scalp_verts, valid_fid_coords, valid_measured
        )

        confidence = 1.0 / (1.0 + residual) if residual > 0 else 1.0
        flagged = residual > max_residual_threshold

        loc = ElectrodeLocalization(
            electrode_id=eid,
            measured_distances=dists,
            ese_coords=ese_coords,
            scalp_coords=scalp_coords,
            residual_error=residual,
            confidence=confidence,
        )
        localizations.append(loc)

        if flagged:
            logger.warning(
                "Electrode %s: residual=%.2f mm exceeds threshold %.2f mm",
                eid,
                residual,
                max_residual_threshold,
            )

    result = LocalizationResult(
        electrodes=localizations,
        num_electrodes=len(localizations),
        method="brute_force",
    )

    logger.info(
        "Localized %d electrodes (threshold=%.1f mm)",
        result.num_electrodes,
        max_residual_threshold,
    )

    return result


def _localize_single(
    ese_vertices: np.ndarray,
    scalp_vertices: np.ndarray,
    fid_coords: np.ndarray,
    measured_dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Localize a single electrode by brute-force search.

    Parameters
    ----------
    ese_vertices : np.ndarray
        ESE point cloud (N,3).
    scalp_vertices : np.ndarray
        Corresponding scalp points (N,3).
    fid_coords : np.ndarray
        Fiducial coordinates (K,3).
    measured_dist : np.ndarray
        Measured distances to fiducials (K,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Best ESE coords, corresponding scalp coords, and residual error.
    """
    diffs = ese_vertices[:, np.newaxis, :] - fid_coords[np.newaxis, :, :]
    predicted_dist = np.linalg.norm(diffs, axis=2)
    errors = np.sum((predicted_dist - measured_dist[np.newaxis, :]) ** 2, axis=1)
    best_idx = int(np.argmin(errors))

    return (
        ese_vertices[best_idx].copy(),
        scalp_vertices[best_idx].copy(),
        float(errors[best_idx]),
    )
