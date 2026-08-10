import sys
from collections.abc import Callable
from pathlib import Path

from virda.config import VirdaSettings, get_virda_settings
from virda.fiducials.provider import (
    AutoFiducialProvider,
    FiducialProvider,
    ManualFiducialProvider,
)
from virda.io.exporter.stage1_exporter import Stage1Exporter
from virda.io.loader.nifti_loader import NiftiLoader
from virda.mesh.air_depth import AirDepthCleaner
from virda.mesh.cleaners import LargestComponentCleaner, MergeCleaner
from virda.mesh.contracts import MeshCleaner, MeshSmoother
from virda.mesh.hole_fill import HoleFillCleaner
from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.taubin_smoother import TaubinSmoother
from virda.models.ese_config import ESEConfig
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1Pipeline
from virda.segmentation.head_segmenter import OtsuHeadSegmenter

CLEANER_FACTORIES: dict[str, Callable[[VirdaSettings], MeshCleaner]] = {
    "merge": lambda settings: MergeCleaner(merge_digits=settings.cleaner_merge_digits),
    "air_depth": lambda settings: AirDepthCleaner(),
    "hole_fill": lambda settings: HoleFillCleaner(),
    "largest_component": lambda settings: LargestComponentCleaner(
        min_vertices=settings.cleaner_min_vertices
    ),
}


def build_cleaners(settings: VirdaSettings) -> list[MeshCleaner]:
    unknown = sorted(set(settings.cleaner_sequence) - set(CLEANER_FACTORIES))
    if unknown:
        raise ValueError(
            "Unknown mesh cleaner(s) in cleaner_sequence: "
            f"{', '.join(unknown)}. Available: {', '.join(sorted(CLEANER_FACTORIES))}."
        )
    return [CLEANER_FACTORIES[name](settings) for name in settings.cleaner_sequence]


def run(
    nifti_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    fiducials_path: str | Path | None = None,
    skip_fiducials: bool | None = None,
) -> Stage1Result:
    settings = get_virda_settings()

    resolved_path = nifti_path or settings.nifti_path
    if resolved_path is None:
        raise ValueError(
            "NIfTI path not provided. Pass it as an argument or set the "
            "NIFTI_PATH environment variable."
        )
    resolved_output_dir = output_dir or settings.output_dir
    resolved_fiducials_path = fiducials_path or settings.fiducials_path
    resolved_skip_fiducials = skip_fiducials or settings.skip_fiducials

    fiducial_provider = _build_fiducial_provider(
        resolved_fiducials_path,
        skip_fiducials=resolved_skip_fiducials,
        auto_detect=settings.auto_detect_fiducials,
    )

    loader = NiftiLoader()
    segmenter = OtsuHeadSegmenter()

    cleaners = build_cleaners(settings)

    smoother: MeshSmoother
    if settings.smoother_type == "taubin":
        smoother = TaubinSmoother(
            iterations=settings.smoother_iterations,
            lamb=settings.smoother_lamb,
            nu=settings.smoother_nu,
        )
    else:
        smoother = LaplacianSmoother(
            iterations=settings.smoother_iterations,
            lamb=settings.smoother_lamb,
        )

    exporter = Stage1Exporter(
        settings=settings,
        ese_config=ESEConfig(
            n_electrodes=settings.n_electrodes,
            ese_offset_mm=settings.ese_offset_mm,
            ese_reference=settings.ese_reference,
        ),
        skip_fiducials=resolved_skip_fiducials,
    )

    pipeline = Stage1Pipeline(
        loader=loader,
        segmenter=segmenter,
        cleaners=cleaners,
        smoother=smoother,
        exporter=exporter,
        fiducial_provider=fiducial_provider,
    )
    return pipeline.run(
        resolved_path,
        output_dir=resolved_output_dir,
        closing_radius=settings.closing_radius,
        threshold=settings.threshold,
    )


def _build_fiducial_provider(
    fiducials_path: str | Path | None,
    skip_fiducials: bool,
    auto_detect: bool,
) -> FiducialProvider:
    if skip_fiducials:
        return ManualFiducialProvider(None, skip=True)
    if fiducials_path is not None:
        return ManualFiducialProvider(Path(fiducials_path))
    if auto_detect:
        return AutoFiducialProvider()
    return ManualFiducialProvider(None)


_BOOL_VALUES = {"true", "false", "True", "False", "1", "0"}


def _normalize_cli_flags() -> None:
    """Allow `--skip-fiducials` (and `--skip_fiducials`) as a bare boolean flag."""
    args = sys.argv[1:]
    for flag in ("--skip-fiducials", "--skip_fiducials"):
        if flag not in args:
            continue
        index = args.index(flag)
        has_value = index + 1 < len(args) and args[index + 1] in _BOOL_VALUES
        value = args[index + 1] if has_value else "true"
        drop = 2 if has_value else 1
        sys.argv = ["virda"] + args[:index] + ["--skip_fiducials", value] + args[index + drop :]
        return


if __name__ == "__main__":
    _normalize_cli_flags()
    result = run()
    print(f"Stage 1 done. Mesh: {result.mesh.vertices.shape[0]} vertices")
