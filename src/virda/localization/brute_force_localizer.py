import logging
from dataclasses import replace

import numpy as np
from scipy.spatial.distance import cdist

from virda.localization.contracts import ElectrodeLocalizer
from virda.models.electrode import Electrode, Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducials
from virda.models.stage3_config import Stage3Config

logger = logging.getLogger(__name__)


class BruteForceLocalizer(ElectrodeLocalizer):
    """Localize real electrodes by brute-force search over the ESE point cloud.

    For every electrode the predicted distance from each ESE vertex to the
    fiducials is compared against the measured distances; the vertex minimizing
    the weighted sum of squared differences is chosen (spec Stage 3).
    """

    def __init__(self, config: Stage3Config) -> None:
        self._config = config

    def _process(
        self,
        ese: ESEMesh,
        fiducials: Fiducials,
        electrodes: Electrodes,
    ) -> Electrodes:
        vertices = np.asarray(ese.vertices)
        fiducial_coords = np.asarray([fiducial.coordinates for fiducial in fiducials.items])
        distances_to_fiducials = cdist(vertices, fiducial_coords)

        localized: list[Electrode] = []
        for electrode in electrodes.items:
            present = [
                fiducial
                for fiducial in fiducials.items
                if fiducial.fiducial_id in electrode.measured_distances
            ]
            if not present:
                logger.warning(
                    "Electrode %s has no measured distances to any known fiducial; skipped",
                    electrode.electrode_id,
                )
                localized.append(electrode)
                continue

            measured = np.asarray(
                [electrode.measured_distances[fiducial.fiducial_id] for fiducial in present]
            )
            weights = np.asarray([fiducial.weight for fiducial in present])
            present_indices = np.asarray(
                [fiducials.ids.index(fiducial.fiducial_id) for fiducial in present]
            )

            squared_error = (
                weights[None, :]
                * (distances_to_fiducials[:, present_indices] - measured[None, :]) ** 2
            )
            total_error = squared_error.sum(axis=1)
            best_index = int(np.argmin(total_error))
            residual_error = float(np.sqrt(total_error[best_index]))

            localized.append(
                replace(
                    electrode,
                    ese_coords=vertices[best_index],
                    scalp_coords=ese.scalp_vertices[best_index],
                    residual_error=residual_error,
                    confidence=float(ese.quality[best_index]),
                    flagged=residual_error > self._config.residual_threshold_mm,
                )
            )

        localized_count = sum(1 for electrode in localized if electrode.is_localized)
        residuals = [
            electrode.residual_error
            for electrode in localized
            if electrode.residual_error is not None
        ]
        if residuals:
            logger.info(
                "Localized %d/%d electrodes, median residual=%.4f mm, flagged=%d",
                localized_count,
                len(localized),
                float(np.median(residuals)),
                sum(1 for electrode in localized if electrode.flagged),
            )
        else:
            logger.info("No electrodes localized (%d provided)", len(localized))
        return Electrodes(items=localized)
