from pathlib import Path

from virda.config import get_virda_settings
from virda.io.loader.nifti_loader import NiftiLoader
from virda.mesh.contracts import MeshPostprocessor
from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.mesh.taubin_smoother import TaubinSmoother
from virda.models.stage1_result import Stage1Result
from virda.pipeline import PipelineController
from virda.pipelines.stage1 import Stage1PipelineBuilder
from virda.segmentation.head_segmenter import OtsuHeadSegmenter


def run(nifti_path: str | Path | None = None) -> Stage1Result:
    settings = get_virda_settings()

    resolved_path = nifti_path or settings.nifti_path
    if resolved_path is None:
        raise ValueError(
            "NIfTI path not provided."
            "Pass it as an argument or set the NIFTI_PATH environment variable."
        )

    loader = NiftiLoader()
    segmenter = OtsuHeadSegmenter(closing_radius=settings.closing_radius)
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
        )
        .setup_mesh_postprocessors([cleaner, smoother])
        .build()
    )

    return stage1_pipeline.run().get_store_notnull(Stage1Result)


if __name__ == "__main__":
    run()
