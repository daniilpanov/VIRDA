"""
Atomic pipeline: locate real electrodes on an ESE or scalp surface.

Stage 0 runs the brute-force localization over whichever surface is provided
(the ESE mesh carries normals and quality; a plain scalp mesh has neither, so
the localizer derives normals from mesh geometry and uses a neutral quality).

Stage 1 functions (called separately, without parameters): ``compute_quality``
builds a small quantum-of-localization report from the localized electrodes.
Quality control stays disabled by default (Sprint 6 decision), so the extra
must be requested explicitly.
"""

from __future__ import annotations

from logging import Logger
from typing import Any

import numpy as np
from pydantic import ConfigDict, Field

from virda.localization.brute_force_localizer import BruteForceLocalizer
from virda.localization.contracts import ElectrodeLocalizer
from virda.models.electrode import Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducials
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage3_config import Stage3Config
from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext
from virda.pipelines.contracts import PipelineContract

from .atomic import AtomicPipeline, Stage1Function


class Stage3LocalizerStep:
    def __init__(self, localizer: ElectrodeLocalizer) -> None:
        self._localizer = localizer

    def run(self, context: PipelineContext) -> Electrodes:
        return self._localizer.run(context)


class LocalizationPipelineContract(PipelineContract):
    """Input contract and options for the electrode localization pipeline.

    Exactly one surface must be provided — the ESE mesh or a plain scalp
    mesh; electrodes and fiducials are mandatory inputs.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    ese_mesh: ESEMesh | None = None
    scalp_mesh: ScalpMesh | None = None
    electrodes: Electrodes | None = None
    fiducials: Fiducials | None = None

    calibrate_ese_offset: bool = True
    residual_threshold_mm: float = Field(default=10.0, gt=0)

    mandatory_fields = frozenset({"electrodes", "fiducials"})

    def _validate_options(self) -> list[str]:
        problems: list[str] = []
        if self.ese_mesh is None and self.scalp_mesh is None:
            problems.append("Missing mandatory input: provide either 'ese_mesh' or 'scalp_mesh'")
        if self.ese_mesh is not None and self.scalp_mesh is not None:
            problems.append("Provide exactly one of 'ese_mesh' or 'scalp_mesh'")
        return problems

    def to_stage3_config(self) -> Stage3Config:
        return Stage3Config(
            calibrate_ese_offset=self.calibrate_ese_offset,
            residual_threshold_mm=self.residual_threshold_mm,
        )


class LocalizationPipeline(AtomicPipeline[LocalizationPipelineContract]):
    """Localize real electrodes on an ESE or scalp surface (spec Stage 3)."""

    name = "localization"

    def __init__(
        self,
        contract: LocalizationPipelineContract,
        logger: Logger | None = None,
    ) -> None:
        super().__init__(contract)
        self._logger = logger

    def build_stage0(self) -> PipelineController:
        contract = self.contract
        localizer = BruteForceLocalizer(
            calibrate_ese_offset=contract.calibrate_ese_offset,
            residual_threshold_mm=contract.residual_threshold_mm,
        )
        controller = PipelineController(logger=self._logger)

        controller.register_store(ScalpMesh, contract.scalp_mesh)
        controller.register_store(ESEMesh, contract.ese_mesh)

        controller.register_store(Fiducials, contract.fiducials)
        controller.register_store(Electrodes, contract.electrodes)

        controller.register_step(Stage3LocalizerStep(localizer=localizer))
        return controller

    @property
    def stage1_functions(self) -> dict[str, Stage1Function]:
        return {
            "compute_quality": self._compute_quality,
        }

    def _compute_quality(self, context: PipelineContext) -> dict[str, Any]:
        localized = context.get_store_notnull(Electrodes)
        residuals = [
            electrode.residual_error
            for electrode in localized.items
            if electrode.residual_error is not None
        ]
        return {
            "electrode_count": len(localized.items),
            "localized_count": sum(1 for electrode in localized.items if electrode.is_localized),
            "flagged_count": sum(1 for electrode in localized.items if electrode.flagged),
            "median_residual_mm": float(np.median(residuals)) if residuals else None,
        }
