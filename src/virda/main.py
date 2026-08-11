from collections.abc import Callable
from pathlib import Path

from virda.config import VirdaSettings, get_virda_settings
from virda.io.loader.nifti_loader import NiftiLoader
from virda.mesh.air_depth import AirDepthCleaner
from virda.mesh.cleaners import LargestComponentCleaner, MergeCleaner
from virda.mesh.contracts import MeshCleaner, MeshSmoother
from virda.mesh.hole_fill import HoleFillCleaner
from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.taubin_smoother import TaubinSmoother
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


def run(nifti_path: str | Path | None = None) -> Stage1Result:
    settings = get_virda_settings()

    resolved_path = nifti_path or settings.nifti_path
    if resolved_path is None:
        raise ValueError(
            "NIfTI path not provided. Pass it as an argument or set the VIRDA_NIFTI_PATH environment variable."
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

    pipeline = Stage1Pipeline(
        loader=loader,
        segmenter=segmenter,
        cleaners=cleaners,
        smoother=smoother,
    )
    return pipeline.run(resolved_path, closing_radius=settings.closing_radius)
