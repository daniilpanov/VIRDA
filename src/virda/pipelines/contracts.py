"""Base contract types for atomic pipelines.

An atomic pipeline is a small tool with one responsibility, a declared input
contract and a set of options.  Its contract is a typed pydantic model that
answers two questions before any processing starts:

* which mandatory inputs are missing (the pipeline must refuse to start);
* whether the options obey the per-pipeline rules (ranges, dependencies).

Subclasses declare ``mandatory_fields`` (input field names that must hold a
real value) and may override ``_validate_options`` to reject bad option
combinations.  The base also exposes lightweight field metadata so a GUI can
build the "start pipeline" screen from the contract alone.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ContractValidationError",
    "PipelineContract",
    "is_missing",
]


def is_missing(value: Any) -> bool:
    """Return True when ``value`` does not carry a usable input.

    ``None`` and blank strings are treated as missing so callers do not have
    to distinguish "not provided" from "provided but empty" themselves.
    Path-like objects are never missing here; subclasses convert them first.
    """
    return value is None or (isinstance(value, str) and not value.strip())


class ContractValidationError(ValueError):
    """Raised when an atomic pipeline contract cannot be satisfied.

    Carries the individual problems in :attr:`errors` and formats them into a
    clear, user-facing message.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        bulleted = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"Cannot start the pipeline:\n{bulleted}")


class PipelineContract(BaseModel):
    """Base class for the input contract of an atomic pipeline.

    A contract combines the data the pipeline consumes (``inputs``) and the
    parameters that control processing and output (``options``).  It is a
    plain pydantic model: validation of both parts happens in ``validate_for_run``
    so the pipeline can refuse to start with a clear message.
    """

    model_config = ConfigDict(extra="forbid")

    #: Names of input fields that must hold a usable value for the pipeline
    #: to start.  Subclasses override this with their own mandatory inputs.
    mandatory_fields: ClassVar[frozenset[str]] = frozenset()

    def missing_mandatory(self) -> tuple[str, ...]:
        """Return the mandatory input names that hold no usable value.

        Iterates the pydantic field declaration order so the result (and the
        resulting error message) is deterministic across runs.
        """
        mandatory = type(self).mandatory_fields
        return tuple(
            name
            for name in type(self).model_fields
            if name in mandatory and is_missing(getattr(self, name, None))
        )

    def validate_for_run(self) -> None:
        """Refuse to start by raising ``ContractValidationError`` when the
        contract is not satisfiable.

        Checks mandatory inputs first, then per-pipeline option rules.  The
        check touches only the contract itself, so it stays fast even when
        the surrounding configuration object is large.
        """
        errors: list[str] = []

        missing = self.missing_mandatory()
        if missing:
            quoted = ", ".join(f"{name!r}" for name in missing)
            errors.append(f"missing mandatory input(s): {quoted}")

        errors.extend(self._validate_options())

        if errors:
            raise ContractValidationError(errors)

    def _validate_options(self) -> list[str]:
        """Return per-pipeline option problems as a list of messages.

        Subclasses override this to enforce ranges, formats and dependencies
        that are harder to express through pydantic constraints alone.
        """
        return []

    @classmethod
    def gui_fields(cls) -> list[dict[str, Any]]:
        """Describe the contract fields for a "start pipeline" screen.

        Yields name, type name, description, default and the mandatory flag
        for every field, mirroring the pydantic field metadata.  The GUI can
        build input controls from this list without hard-coding each pipeline.
        """
        fields = []
        for name, field in cls.model_fields.items():
            fields.append(
                {
                    "name": name,
                    "type": field.annotation,
                    "description": field.description,
                    "default": None if field.default is None else field.default,
                    "mandatory": name in cls.mandatory_fields,
                }
            )
        return fields
