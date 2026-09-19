from typing import Any, cast

import pytest

from virda.pipelines.contracts import (
    ContractValidationError,
    PipelineContract,
    is_missing,
)


class FakePipelineContract(PipelineContract):
    nifti_path: str | None = None
    project_dir: str | None = None
    smoothing: bool = True

    mandatory_fields = frozenset({"nifti_path", "project_dir"})

    def _validate_options(self) -> list[str]:
        problems = []
        if self.smoothing is True and self.project_dir is None:
            problems.append("'smoothing' requires a non-empty 'project_dir'")
        return problems


def test_valid_contract_passes() -> None:
    contract = FakePipelineContract(nifti_path="/mri.nii.gz", project_dir="/proj")
    contract.validate_for_run()


def test_missing_mandatory_input_raises_with_names() -> None:
    contract = FakePipelineContract(nifti_path=None, project_dir="/proj")
    with pytest.raises(ContractValidationError) as excinfo:
        contract.validate_for_run()

    assert "missing mandatory input(s): 'nifti_path'" in str(excinfo.value)
    assert "nifti_path" in excinfo.value.errors[0]


def test_blank_string_counts_as_missing() -> None:
    contract = FakePipelineContract(nifti_path="  ", project_dir="/proj")
    with pytest.raises(ContractValidationError) as excinfo:
        contract.validate_for_run()

    assert "nifti_path" in str(excinfo.value)


def test_multiple_missing_inputs_are_all_listed() -> None:
    contract = FakePipelineContract(nifti_path=None, project_dir=None)
    with pytest.raises(ContractValidationError) as excinfo:
        contract.validate_for_run()

    assert "'nifti_path'" in str(excinfo.value)
    assert "'project_dir'" in str(excinfo.value)


def test_option_problems_are_appended_to_missing_inputs() -> None:
    contract = FakePipelineContract(nifti_path=None, project_dir=None)
    with pytest.raises(ContractValidationError) as excinfo:
        contract.validate_for_run()

    assert any("'smoothing'" in error for error in excinfo.value.errors)


def test_mandatory_fields_are_not_pydantic_fields() -> None:
    assert "mandatory_fields" not in FakePipelineContract.model_fields


def test_update_replaces_missing_input(tmp_path) -> None:
    contract = FakePipelineContract(nifti_path=None)
    assert contract.missing_mandatory() == ("nifti_path", "project_dir")

    contract.nifti_path = str(tmp_path)
    assert contract.missing_mandatory() == ("project_dir",)

    with pytest.raises(ContractValidationError):
        contract.validate_for_run()

    contract.project_dir = str(tmp_path)
    assert contract.missing_mandatory() == ()
    contract.validate_for_run()


def test_gui_fields_describe_contract() -> None:
    fields = {item["name"]: item for item in FakePipelineContract.gui_fields()}

    assert fields["nifti_path"]["mandatory"] is True
    assert fields["smoothing"]["mandatory"] is False


def test_extra_fields_are_rejected() -> None:
    payload = cast("Any", {"unknown_option": 1})
    with pytest.raises(ValueError):
        FakePipelineContract(nifti_path="/mri.nii.gz", **payload)


def test_errors_are_retained_on_error_object() -> None:
    contract = FakePipelineContract(nifti_path=None, project_dir=None)
    with pytest.raises(ContractValidationError) as excinfo:
        contract.validate_for_run()

    assert len(excinfo.value.errors) >= 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        ("   ", True),
        ("/mri.nii.gz", False),
        (123, False),
    ],
)
def test_is_missing(value: Any, expected: bool) -> None:
    assert is_missing(value) is expected