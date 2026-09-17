"""GUI-wide constants shared across tabs, dialogs and workers."""

ELECTRODE_PALETTE = ["yellow", "lime", "magenta", "cyan", "orange", "white"]

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

CONFIG_KEY_TO_INPUT: dict[str, str] = {
    "nifti_path": "nifti_path",
    "project_dir": "project_dir",
    "fiducials_path": "fiducials_path",
    "auto_detect_fiducials": "auto_detect_fiducials",
}
