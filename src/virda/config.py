from functools import cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class VirdaSettings(BaseSettings):
    nifti_path: str | None = None
    output_dir: str | None = None

    closing_radius: int = 5

    cleaner_min_vertices: int = 100
    cleaner_merge_digits: int = 7

    smoother_type: str = "laplacian"
    smoother_iterations: int = 5
    smoother_lamb: float = 0.5
    smoother_nu: float = -0.53

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_file=".env",
        json_file=".env.json",
        yaml_file=".env.yaml",
    )


@cache
def get_virda_settings() -> VirdaSettings:
    return VirdaSettings()
