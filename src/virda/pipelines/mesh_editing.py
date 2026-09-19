"""
Atomic pipeline: re-apply mesh post-processing to an existing scalp mesh.

Unlike :mod:`virda.pipelines.mesh_generate`, this pipeline never needs a NIfTI
volume: it accepts a pre-built :class:`ScalpMesh` and exposes the same stage-1
post-processing functions (``clean``, ``smooth``, ``decimate``).  This lets the
GUI trial a smoother - or a mesh density level - against the *original* mesh
again and again, without re-running segmentation or reconstruction.

The input mesh is never mutated: every postprocessor builds a new
:class:`ScalpMesh` from the stored one, and :func:`edit_mesh` returns a fresh
object each time, so "start over from the original" is always safe.
"""

from __future__ import annotations

from logging import Logger

from pydantic import ConfigDict

from virda.models.scalp_mesh import ScalpMesh
from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext
from virda.pipelines.contracts import ContractValidationError
from virda.pipelines.mesh_generate import MeshPipeline, MeshPipelineContract


class MeshEditingPipelineContract(MeshPipelineContract):
    """Post-processing options for an already-built scalp mesh.

    Carries all mesh-grow parameters (smoother, decimation) but drops the
    NIfTI input requirement; ``scalp_mesh`` is the single mandatory input.
    """

    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True
    )

    scalp_mesh: ScalpMesh | None = None

    mandatory_fields = frozenset({"scalp_mesh"})


class MeshEditingPipeline(MeshPipeline):
    """Re-apply ``clean``/``smooth``/``decimate`` to an existing scalp mesh.

    Stage 0 simply registers the contract's ``scalp_mesh`` in the context; the
    stage-1 functions are inherited from :class:`MeshPipeline`, so the mesh
    post-processing code is shared and tested in one place.
    """

    name = "mesh_editing"

    def build_stage0(self) -> PipelineController:
        contract = self.contract
        if contract.scalp_mesh is None:
            raise ContractValidationError(["missing mandatory input(s): 'scalp_mesh'"])
        controller = PipelineController(logger=self._logger)
        controller.register_store(ScalpMesh, contract.scalp_mesh)
        return controller


def edit_mesh(contract: MeshEditingPipelineContract) -> ScalpMesh:
    """Return the smoothed (and optionally decimated) result of ``scalp_mesh``.

    Runs the inherited ``smooth`` stage-1 function when ``smoother_type`` is
    not ``"none"``, then ``decimate`` when ``mesh_density_percent`` is below
    100.  The contract's input mesh is left untouched; when no post-processing
    applies the input instance itself is returned.
    """
    pipeline = MeshEditingPipeline(contract)
    context = pipeline.run_stage0()
    if contract.smoother_type != "none":
        pipeline.run_stage1("smooth", context)
    if contract.mesh_density_percent < 100.0:
        pipeline.run_stage1("decimate", context)
    return context.get_store_notnull(ScalpMesh)