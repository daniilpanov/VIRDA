import argparse
from typing import Any

from virda.config import VirdaSettings, build_config, resolve_config_files
from virda.models.config import Config
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1PipelineBuilder


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _add_option(parser: argparse.ArgumentParser, name: str, value_type: type) -> None:
    parser.add_argument(
        f"--{name.replace('_', '-')}",
        f"--{name}",
        dest=name,
        type=value_type,
    )


def _add_bool_option(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(
        f"--{name.replace('_', '-')}",
        f"--{name}",
        dest=name,
        nargs="?",
        const=True,
        default=None,
        type=_parse_bool,
        metavar="BOOL",
    )


def _parse_cli_args() -> argparse.Namespace:
    """Parse CLI arguments for every pipeline setting (highest priority)."""
    parser = argparse.ArgumentParser(
        prog="virda",
        description="Run the VIRDA electrode localization pipeline.",
    )
    parser.add_argument(
        "--config-file",
        "--config_file",
        dest="config_file",
        action="append",
        type=str,
        help="Input config file (e.g. coordsystem.json); may be repeated.",
    )
    for name in (
        "nifti_path",
        "project_dir",
        "fiducials_path",
        "closing_radius",
        "otsu_scope",
        "otsu_threshold_scale",
        "seal_radius",
        "cleaner_min_vertices",
        "cleaner_merge_digits",
        "smoother_type",
        "smoother_iterations",
        "smoother_lamb",
        "smoother_nu",
        "n_electrodes",
        "ese_offset_mm",
        "ese_reference",
        "neighborhood_radius_mm",
        "k_neighbors",
        "pca_sigma_mm",
        "min_neighbors",
    ):
        value_type: type = (
            int
            if name
            in {
                "closing_radius",
                "seal_radius",
                "cleaner_min_vertices",
                "cleaner_merge_digits",
                "smoother_iterations",
                "n_electrodes",
                "k_neighbors",
                "min_neighbors",
            }
            else (
                float
                if name in {"otsu_threshold_scale", "smoother_lamb", "smoother_nu", "ese_offset_mm"}
                else str
            )
        )
        _add_option(parser, name, value_type)
    for name in ("auto_detect_fiducials", "seal_enabled", "use_weighted_pca"):
        _add_bool_option(parser, name)
    return parser.parse_args()


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """CLI values as overrides, skipping flags that were not provided."""
    return {
        name: value
        for name, value in vars(args).items()
        if value is not None and name != "config_file"
    }


def run(config: Config) -> Stage1Result:
    """Run the VIRDA pipeline for the merged ``config``."""
    return (
        Stage1PipelineBuilder.from_config(config=config)
        .build()
        .run()
        .get_store_notnull(Stage1Result)
    )


def main() -> None:
    args = _parse_cli_args()
    config = build_config(
        settings=VirdaSettings(),
        config_files=resolve_config_files(args.config_file),
        overrides=_cli_overrides(args),
    )
    result = run(config)
    print(f"Stage 1: mesh with {len(result.mesh.vertices)} vertices")
