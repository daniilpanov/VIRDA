"""GUI-wide constants shared across tabs, dialogs and workers."""

ELECTRODE_PALETTE = ["yellow", "lime", "magenta", "cyan", "orange", "white"]

DEFAULT_FIDUCIALS_FILENAME = "fiducials.json"
DEFAULT_MEASUREMENTS_FILENAME = "measurements.json"

PROJECT_ARTIFACT_DIRS = [
    "input",
    "segmentation",
    "mesh",
    "fiducials",
    "ese",
    "localization",
    "quality_control",
    "logs",
]

#: GUI-only settings; every knob is controlled in the interface and never
#: persisted to a pipeline config file.
ADVANCED_FIELD_DEFAULTS: dict[str, str] = {
    "seal_enabled": "true",
    "seal_radius": "4",
    "cleaner_min_vertices": "100",
    "cleaner_merge_digits": "7",
    "residual_threshold_mm": "10.0",
    "calibrate_ese_offset": "true",
}