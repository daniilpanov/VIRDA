from pathlib import Path
from typing import Self

from virda.io.loader import MRILoader
from virda.io.providers.mesh_versioning_provider import ScalpMeshVersioningProvider
from virda.mesh import MeshExtractor, MeshPostprocessor
from virda.models.mri_volume import MRIVolume
from virda.models.path import NiftiPath
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask
from virda.models.stage1_result import Stage1Result
from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext
from virda.segmentation import HeadSegmenter


class OutputGenerator:
    def run(self, context: PipelineContext) -> Stage1Result:
        return Stage1Result(
            mri_volume=context.get_store_notnull(MRIVolume),
            segmentation_mask=context.get_store_notnull(SegmentationMask),
            mesh=context.get_store_notnull(ScalpMesh),
        )


class Stage1PipelineBuilder:
    def __init__(
        self,
        nifti_path: str | Path,
        mri_loader: MRILoader,
        segmenter: HeadSegmenter,
        extractor: MeshExtractor,
        project_dir: Path | None = None,
    ) -> None:
        self._nifti_path: Path = Path(nifti_path)
        self._loader: MRILoader = mri_loader
        self._segmenter: HeadSegmenter = segmenter
        self._mesh_extractor: MeshExtractor = extractor
        self._mesh_postprocessors: list[MeshPostprocessor] = []
        self._project_dir: Path | None = project_dir

    def setup_mesh_postprocessors(self, postprocessors: list[MeshPostprocessor]) -> Self:
        """Add mesh cleaners, smoothers, etc."""
        self._mesh_postprocessors.extend(postprocessors)
        return self

    def build(self) -> PipelineController:
        controller = PipelineController()

        # -- Steps --
        controller.register_step(self._loader)
        controller.register_step(self._segmenter)
        controller.register_step(self._mesh_extractor)

        for postprocessor in self._mesh_postprocessors:
            controller.register_step(postprocessor)

        controller.register_step(OutputGenerator())

        # -- Stores --
        controller.register_store(NiftiPath, NiftiPath(self._nifti_path))
        controller.register_store(MRIVolume)
        controller.register_store(SegmentationMask)
        controller.register_store(ScalpMesh)

        # -- Providers --
        if self._project_dir:
            controller.register_provider(
                ScalpMeshVersioningProvider(self._project_dir / "mesh" / "versions"),
                on_store=ScalpMesh,
            )

        return controller
