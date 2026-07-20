"""Stage 3 orchestration — ESE + measurements → electrode localization.

Returns objects, no file I/O. Use api.exporter to save results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from core.ese_generator import ESEResult
from core.fiducial_manager import FiducialManager
from core.measurement_importer import MeasurementImporter, ElectrodeMeasurement
from core.electrode_localizer import LocalizationResult, localize_electrodes
from core.quality_control import QCReport, check_stage3

logger = logging.getLogger(__name__)


def run_stage3(
    ese: ESEResult,
    fiducial_mgr: FiducialManager,
    measurements: dict[str, ElectrodeMeasurement],
    *,
    max_residual_threshold: float = 10.0,
    fiducial_ids: Optional[list[str]] = None,
) -> tuple[LocalizationResult, QCReport]:
    """Run Stage 3: electrode localization from distance measurements.

    Parameters
    ----------
    ese : ESEResult
        ESE from Stage 2.
    fiducial_mgr : FiducialManager
        Fiducial coordinates (must have >= 3 fiducials with coordinates).
    measurements : dict
        Mapping of electrode_id → ElectrodeMeasurement.
    max_residual_threshold : float
        Flag electrodes with residual above this threshold (mm).
    fiducial_ids : list[str], optional
        Subset of fiducials to use for localization.

    Returns
    -------
    tuple
        (localization_result, qc_report)
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Electrode Localization")
    logger.info("=" * 60)

    if not measurements:
        raise RuntimeError("No measurements provided for localization")

    logger.info("Localizing %d electrodes", len(measurements))

    result = localize_electrodes(
        ese=ese,
        fiducial_mgr=fiducial_mgr,
        measurements=measurements,
        max_residual_threshold=max_residual_threshold,
        fiducial_ids=fiducial_ids,
    )
    logger.info(
        "Localization complete: %d electrodes, mean residual=%.2f mm",
        result.num_electrodes,
        result.mean_residual,
    )

    qc = check_stage3(result)
    logger.info("\n%s", qc.summary())

    return result, qc


def load_measurements(path: str | Path, fiducial_ids: list[str]) -> dict[str, ElectrodeMeasurement]:
    """Load measurements from CSV or JSON file.

    Parameters
    ----------
    path : str or Path
        Path to measurements file.
    fiducial_ids : list[str]
        Expected fiducial IDs.

    Returns
    -------
    dict
        Mapping of electrode_id → ElectrodeMeasurement.
    """
    path = Path(path)
    importer = MeasurementImporter(fiducial_ids)

    if path.suffix == ".csv":
        importer.import_csv(path)
    elif path.suffix == ".json":
        importer.import_json(path)
    else:
        raise ValueError(f"Unknown measurement format: {path.suffix}")

    return importer.get_all_measurements()
