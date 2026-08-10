from functools import cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class VirdaSettings(BaseSettings):
    nifti_path: str | None = None
    output_dir: str | None = None

    fiducials_path: str | None = None
    skip_fiducials: bool = False

    closing_radius: int = 5

    cleaner_min_vertices: int = 100
    cleaner_merge_digits: int = 7

    remove_internal_faces: bool = True
    internal_face_method: str = "geodesic"
    internal_face_wide_mm: float = 10.0
    internal_face_seed_mm: float = 20.0
    internal_face_flood_mm: float = 12.0
    internal_face_seed_depth_mm: float = 30.0
    internal_face_flood_depth_mm: float = 8.0
    internal_face_ray_length_mm: float = 90.0
    fill_small_holes: bool = True
    fill_small_holes_max_mm: float = 15.0
    subdivide_max_edge: float | None = None

    smoother_type: str = "laplacian"
    smoother_iterations: int = 5
    smoother_lamb: float = 0.5
    smoother_nu: float = -0.53

    n_electrodes: int = 67
    ese_offset_mm: float = 5.0
    ese_reference: str = "electrode_external_surface"

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_file=".env",
        json_file=".env.json",
        yaml_file=".env.yaml",
    )


@cache
def get_virda_settings() -> VirdaSettings:
    return VirdaSettings()
