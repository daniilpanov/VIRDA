"""Live localization service: snapshot building plus debounced runner."""

from virda_gui_new.services.localization.build import (
    FiducialRows,
    MeasurementTable,
    blocked_reason,
    build_electrodes,
    build_fiducials,
    canonical_fiducial_name,
    complete_measurement_table,
    electrodes_from_points,
    is_complete_measurement,
    labeled_points_from_rows,
    measurement_counts,
    measurement_table_from_model,
    points_from_electrodes,
)
from virda_gui_new.services.localization.checks import SurfaceStatus, fiducial_surface_status
from virda_gui_new.services.localization.prediction import (
    predict_missing_measurement,
    predict_missing_measurements,
)
from virda_gui_new.services.localization.runner import LiveLocalizationRunner

__all__ = [
    "FiducialRows",
    "LiveLocalizationRunner",
    "MeasurementTable",
    "blocked_reason",
    "build_electrodes",
    "build_fiducials",
    "canonical_fiducial_name",
    "complete_measurement_table",
    "electrodes_from_points",
    "is_complete_measurement",
    "SurfaceStatus",
    "fiducial_surface_status",
    "labeled_points_from_rows",
    "measurement_counts",
    "measurement_table_from_model",
    "points_from_electrodes",
    "predict_missing_measurement",
    "predict_missing_measurements",
]
