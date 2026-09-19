from logging import Logger
from pathlib import Path
from typing import Self

from virda.io.providers.logging_provider import StoreLoggingProvider
from virda.io.providers.stage3_exporter import Stage3Exporter
from virda.localization.brute_force_localizer import BruteForceLocalizer
from virda.localization.contracts import ElectrodeLocalizer
from virda.models.electrode import Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducials
from virda.models.stage3_config import Stage3Config
from virda.pipeline import PipelineController
from virda.pipelines.localize import (
    LocalizationPipeline,
    LocalizationPipelineContract,
    Stage3LocalizerStep,
)


class Stage3PipelineBuilder:
    def __init__(
        self,
        localizer: ElectrodeLocalizer,
        stage3_config: Stage3Config,
        ese_mesh: ESEMesh,
        electrodes: Electrodes,
        fiducials: Fiducials,
        project_dir: Path,
        logger: Logger | None = None,
    ) -> None:
        self._localizer = localizer
        self._stage3_config = stage3_config
        self._ese_mesh = ese_mesh
        self._electrodes = electrodes
        self._fiducials = fiducials
        self._project_dir = Path(project_dir)
        self._logger = logger
        self._atomic_localization: LocalizationPipeline | None = None

    @classmethod
    def from_config(
        cls,
        stage3_config: Stage3Config,
        ese_mesh: ESEMesh,
        electrodes: Electrodes,
        fiducials: Fiducials,
        project_dir: Path,
        logger: Logger | None = None,
    ) -> Self:
        """Build a Stage 3 pipeline whose localization reuses the atomic
        :class:`~virda.pipelines.localize.LocalizationPipeline`.

        The atomic pipeline derives its own brute-force localizer from the
        stage-3 options and keeps quality control disabled by default (Sprint 6
        decision), so only the store logging and the stage-3 exporter are wired
        on top of its stage-0 controller.
        """
        contract = LocalizationPipelineContract(
            ese_mesh=ese_mesh,
            electrodes=electrodes,
            fiducials=fiducials,
            calibrate_ese_offset=stage3_config.calibrate_ese_offset,
            residual_threshold_mm=stage3_config.residual_threshold_mm,
        )

        builder = cls(
            localizer=BruteForceLocalizer(contract.to_stage3_config()),
            stage3_config=stage3_config,
            ese_mesh=ese_mesh,
            electrodes=electrodes,
            fiducials=fiducials,
            project_dir=project_dir,
            logger=logger,
        )
        builder._atomic_localization = LocalizationPipeline(
            contract=contract,
            logger=logger,
        )
        return builder

    def build(self) -> PipelineController:
        if self._atomic_localization is not None:
            return self._build_atomic()
        return self._build_manual()

    def _build_atomic(self) -> PipelineController:
        assert self._atomic_localization is not None
        controller = self._atomic_localization.build_stage0()

        log_provider = StoreLoggingProvider()
        for store_type in (ESEMesh, Fiducials, Electrodes):
            controller.register_provider(log_provider, on_store=store_type)

        controller.register_provider(
            Stage3Exporter(
                project_dir=self._project_dir,
                stage3_config=self._stage3_config,
            ),
            on_store=Electrodes,
        )

        return controller

    def _build_manual(self) -> PipelineController:
        controller = PipelineController(logger=self._logger)

        controller.register_store(ESEMesh, self._ese_mesh)
        controller.register_store(Electrodes, self._electrodes)
        controller.register_store(Fiducials, self._fiducials)
        controller.register_step(Stage3LocalizerStep(self._localizer))

        log_provider = StoreLoggingProvider()
        for store_type in (ESEMesh, Fiducials, Electrodes):
            controller.register_provider(log_provider, on_store=store_type)

        controller.register_provider(
            Stage3Exporter(
                project_dir=self._project_dir,
                stage3_config=self._stage3_config,
            ),
            on_store=Electrodes,
        )

        return controller
