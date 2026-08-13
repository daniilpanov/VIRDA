from pathlib import Path

import pytest

from tests.helpers.pipelines import (
    build_context,
    make_auto_fiducials,
    make_fiducials,
    save_test_fiducials,
)
from virda.io.loader.manual_fiducials_loader import ManualFiducialsLoader
from virda.models.fiducial import AutoDetectedFiducials, ManualFiducials
from virda.models.path import FiducialsPath
from virda.pipelines.stage1 import FiducialsRegistrationStep


class TestManualFiducialsLoader:
    def test_loader_reads_fiducials_file(self, tmp_path: Path) -> None:
        path = save_test_fiducials(tmp_path / "fiducials.json")
        loader = ManualFiducialsLoader()

        result = loader.run(build_context(fiducials_path=FiducialsPath(path)))

        assert isinstance(result, ManualFiducials)
        assert result.fiducials.ids == ["NAS", "LPA"]

    def test_loader_raises_when_file_missing(self, tmp_path: Path) -> None:
        loader = ManualFiducialsLoader()
        missing = FiducialsPath(tmp_path / "missing.json")

        with pytest.raises(FileNotFoundError):
            loader.run(build_context(fiducials_path=missing))


class TestFiducialsRegistrationStep:
    def test_uses_manual_fiducials(self) -> None:
        step = FiducialsRegistrationStep()

        result = step.run(build_context(manual=ManualFiducials(fiducials=make_fiducials())))

        assert all(fiducial.definition_method == "manual" for fiducial in result.items)

    def test_falls_back_to_auto_detected(self) -> None:
        step = FiducialsRegistrationStep()

        result = step.run(
            build_context(auto=AutoDetectedFiducials(fiducials=make_auto_fiducials()))
        )

        assert all(fiducial.definition_method == "auto" for fiducial in result.items)

    def test_manual_wins_over_auto_detected(self) -> None:
        step = FiducialsRegistrationStep()

        result = step.run(
            build_context(
                manual=ManualFiducials(fiducials=make_fiducials()),
                auto=AutoDetectedFiducials(fiducials=make_auto_fiducials()),
            )
        )

        assert all(fiducial.definition_method == "manual" for fiducial in result.items)

    def test_raises_without_any_source(self) -> None:
        step = FiducialsRegistrationStep()

        with pytest.raises(ValueError, match="No fiducials available"):
            step.run(build_context())
