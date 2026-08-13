import logging
from pathlib import Path

from virda.config import VirdaSettings, get_virda_settings
from virda.io.loader.nifti_loader import NiftiLoader
from virda.mesh.contracts import MeshPostprocessor
from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.mesh.taubin_smoother import TaubinSmoother
from virda.models.stage1_result import Stage1Result
from virda.pipeline import PipelineController
from virda.pipelines.stage1 import Stage1PipelineBuilder
from virda.segmentation import SegmentationMaskPostprocessor
from virda.segmentation.head_segmenter import OtsuHeadSegmenter
from virda.segmentation.seal import MaskSealer


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("virda")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_dir / "pipeline.log", mode="w")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def run(
    nifti_path: str | Path | None = None,
    project_dir: str | Path | None = None,
    fiducials_path: str | Path | None = None,
) -> Stage1Result:
    """Run the full VIRDA pipeline: Stage 1 → 2 → 3.

    Parameters
    ----------
    nifti_path
        Path to T1-weighted NIfTI. Falls back to ``settings.nifti_path``.
    project_dir
        Path to output directory
    fiducials_path
        Path to manual fiducials file. Falls back to ``settings.fiducials_path``.

    Returns
    -------
    Stage1Result
    """
    settings = get_virda_settings()

    resolved_path = nifti_path or settings.nifti_path
    if resolved_path is None:
        raise ValueError(
            "NIfTI path not provided."
            "Pass it as an argument or set the NIFTI_PATH environment variable."
        )

    project_dir = project_dir or settings.project_dir
    if project_dir is None:
        raise ValueError(
            "Project directory path not provided. "
            "Pass it as an argument or set the PROJECT_DIR environment variable."
        )

    project = Path(project_dir)
    (project / "logs").mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(project / "logs")

    resolved_fiducials_path = fiducials_path or settings.fiducials_path

    # Stage 1: MRI → Segmentation → Mesh → Fiducials
    loader = NiftiLoader()
    segmenter = OtsuHeadSegmenter(
        closing_radius=settings.closing_radius,
        otsu_scope=settings.otsu_scope,
        threshold_scale=settings.otsu_threshold_scale,
    )
    extractor = MarchingCubesExtractor()

    cleaner = TrimeshCleaner(
        min_component_vertices=settings.cleaner_min_vertices,
        merge_digits=settings.cleaner_merge_digits,
    )

    smoother: MeshPostprocessor
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

    stage1_pipeline: PipelineController = (
        Stage1PipelineBuilder(
            nifti_path=resolved_path,
            mri_loader=loader,
            segmenter=segmenter,
            extractor=extractor,
            project_dir=project,
            logger=logger,
            fiducials_path=resolved_fiducials_path,
            auto_detect_fiducials=settings.auto_detect_fiducials,
        )
        .setup_mask_postprocessors(_build_mask_postprocessors(settings))
        .setup_mesh_postprocessors([cleaner, smoother])
        .build()
    )

    stage1_result = stage1_pipeline.run().get_store_notnull(Stage1Result)

    return stage1_result


def _build_mask_postprocessors(
    settings: VirdaSettings,
) -> list[SegmentationMaskPostprocessor]:
    if not settings.seal_enabled:
        return []
    return [MaskSealer(radius=settings.seal_radius)]


if __name__ == "__main__":
    result = run()
    print(f"Stage 1: mesh with {len(result.mesh.vertices)} vertices")
