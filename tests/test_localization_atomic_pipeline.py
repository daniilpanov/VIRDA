"""Tests for the atomic electrode-localization pipeline (spec Stage 3).

Stage 0 localizes the electrodes over the provided surface — the ESE mesh
carries normals and quality; a plain scalp mesh carries neither, so the
localizer derives surface normals from the mesh geometry and uses a neutral
quality.

Stage 1 functions are invoked one at a time on the paused context.  QC
(``compute_quality``) stays disabled by default (Sprint 6 decision) — the
decision lives in the extras layer of the CLI, not in the atomic pipeline — so
here we only check that the function is available but never auto-run.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers.measurements import make_electrodes, make_ese, make_fiducials
from tests.helpers.meshes import make_sphere
from virda.models.electrode import Electrode, Electrodes
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage3_config import Stage3Config
from virda.pipelines.contracts import ContractValidationError
from virda.pipelines.localize import LocalizationPipeline, LocalizationPipelineContract


def _points_on_ese() -> np.ndarray:
    ese = make_ese()
    rng = np.random.default_rng(7)
    n = 24
    indices = rng.integers(0, len(ese.vertices), size=n)
    return np.asarray(ese.vertices[indices])


@pytest.fixture
def scalp_mesh() -> ScalpMesh:
    return make_sphere()


@pytest.fixture
def contract(scalp_mesh: ScalpMesh) -> LocalizationPipelineContract:
    return LocalizationPipelineContract(
        scalp_mesh=scalp_mesh,
        electrodes=make_electrodes(_points_on_ese(), make_fiducials()),
        fiducials=make_fiducials(),
    )


def _build(contract: LocalizationPipelineContract) -> LocalizationPipeline:
    return LocalizationPipeline(contract=contract)


class TestLocalizationPipelineContract:
    def test_requires_surface(self) -> None:
        contract = LocalizationPipelineContract(
            electrodes=make_electrodes(_points_on_ese(), make_fiducials()),
            fiducials=make_fiducials(),
        )

        with pytest.raises(ContractValidationError, match="ese_mesh"):
            contract.validate_for_run()

    def test_rejects_two_surfaces(self) -> None:
        contract = LocalizationPipelineContract(
            ese_mesh=make_ese(),
            scalp_mesh=make_sphere(),
            electrodes=make_electrodes(_points_on_ese(), make_fiducials()),
            fiducials=make_fiducials(),
        )

        with pytest.raises(ContractValidationError, match="ese_mesh"):
            contract.validate_for_run()

    def test_uses_stage3_defaults_when_options_not_set(self) -> None:
        contract = LocalizationPipelineContract(
            scalp_mesh=make_sphere(),
            electrodes=make_electrodes(_points_on_ese(), make_fiducials()),
            fiducials=make_fiducials(),
        )

        stage3 = contract.to_stage3_config()
        assert isinstance(stage3, Stage3Config)
        assert stage3.calibrate_ese_offset is True
        assert stage3.residual_threshold_mm == 10.0


class TestLocalizationPipeline:
    def test_run_stage0_localizes_on_scalp_mesh(self, contract) -> None:
        pipeline = _build(contract)
        context = pipeline.run_stage0()

        localized = context.get_store_notnull(Electrodes)
        assert isinstance(localized, Electrodes)
        assert len(localized.items) == len(contract.electrodes.items) if contract.electrodes else 0
        assert all(isinstance(electrode, Electrode) for electrode in localized.items)

    def test_run_stage0_localizes_on_ese_mesh(self) -> None:
        ese = make_ese()
        contract = LocalizationPipelineContract(
            ese_mesh=ese,
            electrodes=make_electrodes(_points_on_ese(), make_fiducials()),
            fiducials=make_fiducials(),
        )
        assert contract.electrodes is not None

        pipeline = _build(contract)
        context = pipeline.run_stage0()

        localized = context.get_store_notnull(Electrodes)
        assert len(localized.items) == len(contract.electrodes.items)
        assert all(
            electrode.ese_coords is not None
            for electrode in localized.items
            if electrode.electrode_id is not None
        )

    def test_stores_contract_in_context(self, contract) -> None:
        pipeline = _build(contract)
        context = pipeline.run_stage0()

        stored = context.get_store(LocalizationPipelineContract)
        assert stored is contract

    def test_is_localized_flag_set_for_each(self, contract) -> None:
        pipeline = _build(contract)
        context = pipeline.run_stage0()

        localized = context.get_store_notnull(Electrodes)
        assert all(electrode.is_localized for electrode in localized.items)

    def test_quality_function_listed_but_not_auto_run(self, contract) -> None:
        pipeline = _build(contract)
        context = pipeline.run_stage0()

        assert "compute_quality" in pipeline.stage1_names
        quality = pipeline.run_stage1("compute_quality", context)
        assert (
            quality["electrode_count"] == len(contract.electrodes.items)
            if contract.electrodes
            else 0
        )

    def test_unknown_stage1_function_raises(self, contract) -> None:
        pipeline = _build(contract)
        context = pipeline.run_stage0()

        with pytest.raises(ValueError, match="nope"):
            pipeline.run_stage1("nope", context)