"""Atomic pipeline: build the electrode-skin-entrance (ESE) mesh.

Stage 0: estimate vertex normals on the scalp mesh and offset the scalp
surface by the electrode offset to produce the ESE mesh.

Stage 1 functions (called separately, without parameters, reusing the pipeline
options): ``store_vertices``, ``store_faces`` and ``store_normals`` persist the
corresponding arrays under ``<project_dir>/ese/``, one function per call.
"""

from logging import Logger
from pathlib import Path

import numpy as np
from pydantic import ConfigDict, Field

from virda.ese.contracts import ESEBuilder
from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.models.ese_mesh import ESEMesh
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage2_config import Stage2Config
from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext
from virda.pipelines.atomic import AtomicPipeline, Stage1Function
from virda.pipelines.contracts import PipelineContract


class ESEPipelineContract(PipelineContract):
    """Input contract and options for the ESE mesh generation pipeline."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    scalp_mesh: ScalpMesh | None = None
    ese_offset_mm: float | None = Field(default=None, gt=0)

    project_dir: Path | None = None

    neighborhood_radius_mm: float = Field(default=10.0, gt=0)
    k_neighbors: int | None = None
    use_weighted_pca: bool = False
    pca_sigma_mm: float = Field(default=5.0, gt=0)
    min_neighbors: int = Field(default=5, ge=1)

    mandatory_fields = frozenset({"scalp_mesh", "ese_offset_mm"})

    def _validate_options(self) -> list[str]:
        problems = []
        if self.k_neighbors is not None and self.k_neighbors < 2:
            problems.append(f"'k_neighbors' must be >= 2 when set, got {self.k_neighbors}")
        return problems

    def to_stage2_config(self) -> Stage2Config:
        return Stage2Config(
            neighborhood_radius_mm=self.neighborhood_radius_mm,
            k_neighbors=self.k_neighbors,
            use_weighted_pca=self.use_weighted_pca,
            pca_sigma_mm=self.pca_sigma_mm,
            min_neighbors=self.min_neighbors,
        )


class _ESEGenerationStep:
    def __init__(self, builder: ESEBuilder) -> None:
        self._builder = builder

    def run(self, context: PipelineContext) -> ESEMesh:
        return self._builder.run(context)


class ESEPipeline(AtomicPipeline[ESEPipelineContract]):
    """ESE mesh generation: stage-0 normal estimation plus storage extras."""

    name = "ese"

    def __init__(self, contract: ESEPipelineContract, logger: Logger | None = None) -> None:
        super().__init__(contract)
        self._logger = logger

    # -- Stage 0 --

    def build_stage0(self) -> PipelineController:
        contract = self.contract
        if contract.scalp_mesh is None:
            raise ValueError("Missing mandatory input: 'scalp_mesh'")
        assert contract.ese_offset_mm is not None

        ese_builder: ESEBuilder = PCAESEBuilder(
            config=contract.to_stage2_config(),
            ese_offset_mm=contract.ese_offset_mm,
        )

        controller = PipelineController(logger=self._logger)
        controller.register_store(ScalpMesh, contract.scalp_mesh)
        controller.register_store(ESEMesh)
        controller.register_step(_ESEGenerationStep(ese_builder))
        return controller

    # -- Stage 1 --

    @property
    def stage1_functions(self) -> dict[str, Stage1Function]:
        return {
            "store_vertices": self._store_vertices,
            "store_faces": self._store_faces,
            "store_normals": self._store_normals,
        }

    def _store_vertices(self, context: PipelineContext) -> Path:
        return _save_ese_array(
            self._require_project_dir(),
            "ese_vertices.npy",
            context.get_store_notnull(ESEMesh).vertices,
        )

    def _store_faces(self, context: PipelineContext) -> Path:
        return _save_ese_array(
            self._require_project_dir(),
            "ese_faces.npy",
            context.get_store_notnull(ESEMesh).faces,
        )

    def _store_normals(self, context: PipelineContext) -> Path:
        return _save_ese_array(
            self._require_project_dir(),
            "normals.npy",
            context.get_store_notnull(ESEMesh).normals,
        )

    def _require_project_dir(self) -> Path:
        if self.contract.project_dir is None:
            raise ValueError(
                "Cannot store ESE arrays: 'project_dir' is not set in the ESE pipeline contract"
            )
        return Path(self.contract.project_dir) / "ese"


def _save_ese_array(ese_dir: Path, filename: str, array: np.ndarray) -> Path:
    ese_dir.mkdir(parents=True, exist_ok=True)
    target = ese_dir / filename
    np.save(target, array)
    return target
