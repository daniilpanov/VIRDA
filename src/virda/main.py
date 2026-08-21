import argparse
import sys
from pathlib import Path
from typing import Any

from virda.config import VirdaSettings, build_config, resolve_config_files, resolve_stage3_config
from virda.models.config import Config
from virda.models.electrode import Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1PipelineBuilder
from virda.pipelines.stage2 import Stage2PipelineBuilder


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
        "measurements_path",
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
    for name in (
        "auto_detect_fiducials",
        "seal_enabled",
        "use_weighted_pca",
        "calibrate_ese_offset",
    ):
        _add_bool_option(parser, name)
    return parser.parse_args()


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """CLI values as overrides, skipping flags that were not provided."""
    return {
        name: value
        for name, value in vars(args).items()
        if value is not None and name != "config_file"
    }


_ESE_PARAMS = ("n_electrodes", "ese_offset_mm", "ese_reference")

_NEIGHBORHOOD_PARAMS = (
    "neighborhood_radius_mm",
    "k_neighbors",
    "pca_sigma_mm",
    "min_neighbors",
    "use_weighted_pca",
)


def _validate_required_group(
    obj: Any,
    group_name: str,
    group_params: tuple[str, ...],
) -> None:
    """Raise ``SystemExit`` if only a subset of a required parameter group was given."""
    provided = [p for p in group_params if getattr(obj, p) is not None]
    missing = [p for p in group_params if getattr(obj, p) is None]

    if provided and missing:
        provided_fmt = ", ".join(f"--{p.replace('_', '-')}" for p in provided)
        missing_fmt = ", ".join(f"--{p.replace('_', '-')}" for p in missing)
        raise SystemExit(
            f"Error: {provided_fmt} were provided, but the following "
            f"parameters are also required for {group_name}: {missing_fmt}"
        )


def _warn_partial_neighborhood(
    args: argparse.Namespace,
    config: Config,
) -> None:
    """Warn when neighbourhood parameters are set but ESE is not configured."""
    if config.to_ese_config() is not None:
        return

    provided = [p for p in _NEIGHBORHOOD_PARAMS if getattr(args, p) is not None]
    if not provided:
        return

    fmt = ", ".join(f"--{p.replace('_', '-')}" for p in provided)
    missing_ese = ", ".join(
        f"--{p.replace('_', '-')}" for p in _ESE_PARAMS if getattr(config, p) is None
    )
    print(
        f"Warning: {fmt} were provided but ESE is not configured "
        f"({missing_ese} missing). Stage 2 will be skipped.",
        file=sys.stderr,
    )


def run(
    config: Config, measurements_path: str | Path | None = None
) -> tuple[Stage1Result, ESEMesh | None, Electrodes | None]:
    """Run the VIRDA pipeline: Stage 1 -> 2 -> 3.

    Returns the Stage 1 result, the ESE surface when ESE is configured,
    and the localized electrodes when both ESE and measurements are available.
    """
    stage1_result = (
        Stage1PipelineBuilder.from_config(config=config)
        .build()
        .run()
        .get_store_notnull(Stage1Result)
    )

    if config.to_ese_config() is None:
        return stage1_result, None, None

    ese_mesh = (
        Stage2PipelineBuilder.from_config(config=config, scalp_mesh=stage1_result.mesh)
        .build()
        .run()
        .get_store_notnull(ESEMesh)
    )
    electrodes = run_stage3(config, stage1_result, ese_mesh, measurements_path)
    return stage1_result, ese_mesh, electrodes


def run_stage3(
    config: Config,
    stage1_result: Stage1Result,
    ese_mesh: ESEMesh | None,
    measurements_path: str | Path | None = None,
) -> Electrodes | None:
    """Localize electrodes (Stage 3) from precomputed Stage 1/2 results.

    Allows callers that already have the scalp mesh and ESE surface to run
    localization without repeating Stages 1-2.
    """
    from virda.io.loader.measurements_loader import MeasurementsLoaderFromJson
    from virda.localization.brute_force_localizer import BruteForceLocalizer
    from virda.models.fiducial import Fiducials
    from virda.models.path import MeasurementsPath
    from virda.pipeline_context import PipelineContext
    from virda.pipelines.helpers import setup_pipeline_logging
    from virda.pipelines.stage3 import Stage3PipelineBuilder

    if ese_mesh is None:
        return None

    resolved_measurements_path = measurements_path
    if resolved_measurements_path is None:
        return None

    project_dir = config.project_dir
    if project_dir is None:
        raise ValueError(
            "Project directory path not provided. "
            "Pass it as an argument or set the PROJECT_DIR environment variable."
        )
    project = Path(project_dir)

    stage3_config = resolve_stage3_config(config)

    load_context = PipelineContext({})
    load_context.stores[MeasurementsPath] = MeasurementsPath(Path(resolved_measurements_path))
    load_context.stores[Fiducials] = stage1_result.fiducials
    electrodes = MeasurementsLoaderFromJson().run(load_context)
    fiducials = load_context.get_store_notnull(Fiducials)

    stage3_pipeline = Stage3PipelineBuilder(
        localizer=BruteForceLocalizer(stage3_config),
        stage3_config=stage3_config,
        ese_mesh=ese_mesh,
        electrodes=electrodes,
        fiducials=fiducials,
        project_dir=project,
        logger=setup_pipeline_logging(project, "stage_3"),
    ).build()
    context = stage3_pipeline.run()
    return context.get_store_notnull(Electrodes)


def main() -> None:
    args = _parse_cli_args()

    config = build_config(
        settings=VirdaSettings(),
        config_files=resolve_config_files(args.config_file),
        overrides=_cli_overrides(args),
    )

    _validate_required_group(config, "ESE", _ESE_PARAMS)

    _warn_partial_neighborhood(args, config)

    stage1_result, ese_mesh, electrodes = run(config, measurements_path=args.measurements_path)
    print(f"Stage 1: mesh with {len(stage1_result.mesh.vertices)} vertices")
    if ese_mesh is not None:
        print(f"Stage 2: ESE mesh with {len(ese_mesh.vertices)} vertices")
    if electrodes is not None:
        print(
            f"Stage 3: localized "
            f"{sum(electrode.is_localized for electrode in electrodes.items)}/"
            f"{len(electrodes.items)} electrodes"
        )
