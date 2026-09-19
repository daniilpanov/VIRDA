"""
Atomic pipeline: generate a scalp mesh from an MRI volume.

Stage 0: load NIfTI -> segment the head -> (optionally) seal the mask ->
extract the scalp mesh with marching cubes.

Stage 1 functions (called separately, without parameters, reusing the pipeline
options): ``clean`` removes small components and merges duplicate vertices,
``smooth`` applies the configured smoother to the current mesh.
"""

from logging import Logger
from pathlib import Path
from typing import Literal

from pydantic import Field

from virda.io.loader.nifti_loader import NiftiLoader
from virda.mesh.contracts import MeshPostprocessor
from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.mesh.taubin_smoother import TaubinSmoother
from virda.models.mri_volume import MRIVolume
from virda.models.path import NiftiPath
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask
from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext
from virda.pipelines.atomic import AtomicPipeline, Stage1Function
from virda.pipelines.contracts import ContractValidationError, PipelineContract
from virda.segmentation.head_segmenter import OtsuHeadSegmenter, OtsuScope
from virda.segmentation.seal import MaskSealer

SMOOTHER_TYPES = Literal["laplacian", "taubin"]


class MeshPipelineContract(PipelineContract):
    """Input contract and options for the scalp-mesh generation pipeline."""

    nifti_path: Path | None = None

    closing_radius: int = Field(default=5, ge=0)
    otsu_scope: OtsuScope = "all"
    otsu_threshold_scale: float = Field(default=0.6, gt=0)

    seal_enabled: bool = True
    seal_radius: int = Field(default=4, ge=0)

    cleaner_min_vertices: int = Field(default=100, ge=1)
    cleaner_merge_digits: int = Field(default=7, ge=0)

    smoother_type: SMOOTHER_TYPES = "laplacian"
    smoother_iterations: int = Field(default=5, ge=1)
    smoother_lamb: float = Field(default=0.5, gt=0)
    smoother_nu: float = -0.53

    mandatory_fields = frozenset({"nifti_path"})


class MeshPipeline(AtomicPipeline[MeshPipelineContract]):
    """Scalp-mesh generation: stage-0 processing plus cleaning/smoothing extras.

    Additional postprocessing is intentionally *not* part of stage 0: ``clean``
    and ``smooth`` are stage-1 functions and can be invoked one by one on the
    produced mesh, or left out entirely.
    """

    name = "mesh_generate"

    def __init__(self, contract: MeshPipelineContract, logger: Logger | None = None) -> None:
        super().__init__(contract)
        self._logger = logger

    # -- Stage 0 --

    def build_stage0(self) -> PipelineController:
        contract = self.contract
        nifti_path = contract.nifti_path
        if nifti_path is None:
            raise ContractValidationError(["missing mandatory input(s): 'nifti_path'"])

        controller = PipelineController(logger=self._logger)

        controller.register_store(NiftiPath, NiftiPath(nifti_path))
        controller.register_store(MRIVolume)
        controller.register_store(SegmentationMask)
        controller.register_store(ScalpMesh)

        controller.register_step(NiftiLoader())
        controller.register_step(
            OtsuHeadSegmenter(
                closing_radius=contract.closing_radius,
                otsu_scope=contract.otsu_scope,
                threshold_scale=contract.otsu_threshold_scale,
            )
        )
        if contract.seal_enabled:
            controller.register_step(MaskSealer(radius=contract.seal_radius))
        controller.register_step(MarchingCubesExtractor())

        return controller

    # -- Stage 1 --

    @property
    def stage1_functions(self) -> dict[str, Stage1Function]:
        return {
            "clean": self._clean,
            "smooth": self._smooth,
        }

    def _clean(self, context: PipelineContext) -> ScalpMesh:
        cleaner: MeshPostprocessor = TrimeshCleaner(
            min_component_vertices=self.contract.cleaner_min_vertices,
            merge_digits=self.contract.cleaner_merge_digits,
        )
        return self._replace_mesh(context, cleaner)

    def _smooth(self, context: PipelineContext) -> ScalpMesh:
        contract = self.contract
        if contract.smoother_type == "taubin":
            smoother: MeshPostprocessor = TaubinSmoother(
                iterations=contract.smoother_iterations,
                lamb=contract.smoother_lamb,
                nu=contract.smoother_nu,
            )
        else:
            smoother = LaplacianSmoother(
                iterations=contract.smoother_iterations,
                lamb=contract.smoother_lamb,
            )
        return self._replace_mesh(context, smoother)

    @staticmethod
    def _replace_mesh(context: PipelineContext, postprocessor: MeshPostprocessor) -> ScalpMesh:
        postprocessed = postprocessor.run(context)
        context.stores[ScalpMesh] = postprocessed
        return postprocessed
