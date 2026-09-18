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

ADVANCED_FIELD_DEFAULTS: dict[str, str] = {
    "otsu_scope": "all",
    "otsu_threshold_scale": "0.6",
    "closing_radius": "5",
    "seal_enabled": "true",
    "seal_radius": "4",
    "mesh_voxel_size_mm": "",
    "mesh_density_percent": "100",
    "cleaner_min_vertices": "100",
    "cleaner_merge_digits": "7",
    "smoother_type": "laplacian",
    "smoother_iterations": "5",
    "smoother_lamb": "0.5",
    "smoother_nu": "-0.53",
    "ese_offset_mm": "",
    "neighborhood_radius_mm": "10.0",
    "k_neighbors": "",
    "pca_sigma_mm": "5.0",
    "min_neighbors": "5",
    "use_weighted_pca": "false",
    "residual_threshold_mm": "10.0",
    "calibrate_ese_offset": "true",
}

CONFIG_KEY_TO_ADVANCED: dict[str, str] = {
    "otsu_scope": "otsu_scope",
    "otsu_threshold_scale": "otsu_threshold_scale",
    "closing_radius": "closing_radius",
    "seal_enabled": "seal_enabled",
    "seal_radius": "seal_radius",
    "mesh_voxel_size_mm": "mesh_voxel_size_mm",
    "mesh_density_percent": "mesh_density_percent",
    "cleaner_min_vertices": "cleaner_min_vertices",
    "cleaner_merge_digits": "cleaner_merge_digits",
    "smoother_type": "smoother_type",
    "smoother_iterations": "smoother_iterations",
    "smoother_lamb": "smoother_lamb",
    "smoother_nu": "smoother_nu",
    "ese_offset_mm": "ese_offset_mm",
    "neighborhood_radius_mm": "neighborhood_radius_mm",
    "k_neighbors": "k_neighbors",
    "pca_sigma_mm": "pca_sigma_mm",
    "min_neighbors": "min_neighbors",
    "use_weighted_pca": "use_weighted_pca",
    "residual_threshold_mm": "residual_threshold_mm",
    "calibrate_ese_offset": "calibrate_ese_offset",
}

ADVANCED_COMBO_FIELDS: dict[str, list[str]] = {
    "otsu_scope": ["all", "foreground"],
    "smoother_type": ["laplacian", "taubin"],
}
