"""Electrode localizer — search ESE for best-matching positions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ese_generator import ESEResult
from .fiducial_manager import FiducialManager
from .measurement_importer import ElectrodeMeasurement

logger = logging.getLogger(__name__)


@dataclass
class LocalizedElectrode:
    """Result of localizing one electrode."""

    electrode_id: str
    ese_coords: np.ndarray
    scalp_coords: np.ndarray
    measured_distances: dict[str, float]
    predicted_distances: dict[str, float]
    residual_error: float
    confidence: float
    candidate_idx: int


@dataclass
class LocalizationResult:
    """Complete localization result for all electrodes."""

    electrodes: list[LocalizedElectrode]
    mean_residual: float
    max_residual: float
    flagged_electrodes: list[str]

    @property
    def num_electrodes(self) -> int:
        return len(self.electrodes)

    def get_electrode_coords(self) -> tuple[list[str], np.ndarray]:
        """Return electrode IDs and ESE coordinates."""
        ids = [e.electrode_id for e in self.electrodes]
        coords = np.array([e.ese_coords for e in self.electrodes])
        return ids, coords

    def get_scalp_coords(self) -> tuple[list[str], np.ndarray]:
        """Return electrode IDs and scalp coordinates."""
        ids = [e.electrode_id for e in self.electrodes]
        coords = np.array([e.scalp_coords for e in self.electrodes])
        return ids, coords


def localize_electrodes(
    ese: ESEResult,
    fiducial_mgr: FiducialManager,
    measurements: dict[str, ElectrodeMeasurement],
    max_residual_threshold: float = 10.0,
    fiducial_ids: Optional[list[str]] = None,
) -> LocalizationResult:
    """Localize electrodes by matching measured distances to ESE points."""
    ese_verts = ese.get_ese_point_cloud()

    if fiducial_ids is None:
        fiducial_ids = list(fiducial_mgr.get_all_fiducials().keys())

    fiducial_coords = fiducial_mgr.get_coordinates_matrix(fiducial_ids)

    if len(fiducial_coords) == 0:
        raise ValueError("No fiducial coordinates available")

    localized = []
    flagged = []

    for eid, meas in measurements.items():
        measured_dists = []
        used_fids = []
        for fid in fiducial_ids:
            if fid in meas.distances:
                measured_dists.append(meas.distances[fid])
                used_fids.append(fid)

        if len(used_fids) < 2:
            logger.warning(
                "Electrode %s: only %d fiducial distances available, skipping",
                eid,
                len(used_fids),
            )
            continue

        measured_dists = np.array(measured_dists)
        used_fid_coords = fiducial_mgr.get_coordinates_matrix(used_fids)

        dist_matrix = np.linalg.norm(
            ese_verts[:, np.newaxis, :] - used_fid_coords[np.newaxis, :, :],
            axis=2,
        )

        error_per_point = np.sum((dist_matrix - measured_dists[np.newaxis, :]) ** 2, axis=1)

        best_idx = int(np.argmin(error_per_point))
        best_error = float(error_per_point[best_idx])

        predicted = {fid: float(dist_matrix[best_idx, i]) for i, fid in enumerate(used_fids)}

        confidence = 1.0 / (1.0 + best_error)

        loc_electrode = LocalizedElectrode(
            electrode_id=eid,
            ese_coords=ese_verts[best_idx].copy(),
            scalp_coords=ese.get_scalp_point_cloud()[best_idx].copy(),
            measured_distances=meas.distances,
            predicted_distances=predicted,
            residual_error=best_error,
            confidence=confidence,
            candidate_idx=best_idx,
        )

        localized.append(loc_electrode)

        if best_error > max_residual_threshold:
            flagged.append(eid)
            logger.warning(
                "Electrode %s: residual=%.2f mm exceeds threshold",
                eid,
                best_error,
            )

    residuals = [e.residual_error for e in localized]
    mean_res = float(np.mean(residuals)) if residuals else 0.0
    max_res = float(np.max(residuals)) if residuals else 0.0

    logger.info(
        "Localization complete: %d electrodes, mean residual=%.2f mm, max=%.2f mm, flagged=%d",
        len(localized),
        mean_res,
        max_res,
        len(flagged),
    )

    return LocalizationResult(
        electrodes=localized,
        mean_residual=mean_res,
        max_residual=max_res,
        flagged_electrodes=flagged,
    )
