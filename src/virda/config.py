from functools import cache

from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class VirdaSettings(BaseSettings):
    nifti_path: str | None = None
    output_dir: str | None = None

    fiducials_path: str | None = None
    skip_fiducials: bool = False

    closing_radius: int = 5
    threshold: float | None = None

    cleaner_sequence: list[str] = ["merge", "air_depth", "hole_fill", "largest_component"]
    cleaner_min_vertices: int = 100
    cleaner_merge_digits: int = 7

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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            CliSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            JsonConfigSettingsSource(settings_cls),
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@cache
def get_virda_settings() -> VirdaSettings:
    return VirdaSettings()
