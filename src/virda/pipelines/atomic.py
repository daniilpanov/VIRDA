"""
Shared scaffolding for atomic pipelines.

An *atomic pipeline* is a small tool with one responsibility: it runs a fixed
stage-0 processing chain and exposes separately callable stage-1 functions
(additional tasks such as smoothing, cleaning or quality estimation) that reuse
the accumulated result in the pipeline context.

The contract for each atomic pipeline is a :class:`PipelineContract` (see
:mod:`virda.pipelines.contracts`).  Before any step runs, the contract is
validated so a pipeline refuses to start when a mandatory input is missing.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext
from virda.pipelines.contracts import PipelineContract


class Stage1Function(Protocol):
    """A stage-1 function: invoked separately, without extra parameters.

    It reads the current result from the pipeline context, applies one
    additional task and writes the outcome back to the context.  Returns the
    produced store for convenience.
    """

    def __call__(self, context: PipelineContext) -> Any: ...


def missing_stage1_error(pipeline: AtomicPipeline[Any], name: str) -> ValueError:
    known = ", ".join(repr(fn) for fn in pipeline.stage1_names)
    return ValueError(
        f"Unknown stage-1 function {name!r} for pipeline {pipeline.name!r}; "
        f"known functions: {known}"
    )


class AtomicPipeline[ContractT: PipelineContract]:
    """Base class for an atomic pipeline.

    Stage 0 is the main processing chain built by :meth:`build_stage0`.
    Stage 1 functions are listed in :attr:`stage1_functions` and can be
    invoked one at a time with :meth:`run_stage1`.
    """

    name: ClassVar[str]

    def __init__(self, contract: ContractT) -> None:
        self.contract = contract

    def build_stage0(self) -> PipelineController:
        raise NotImplementedError

    def run_stage0(self) -> PipelineContext:
        """Validate the contract, run the stage-0 chain and return the context.

        The contract itself is stored into the context so stage-1 functions
        and later stages can read the options used for this run.
        """
        self.contract.validate_for_run()
        context = self.build_stage0().run()
        context.stores[type(self.contract)] = self.contract
        return context

    @property
    def stage1_functions(self) -> dict[str, Stage1Function]:
        return {}

    @property
    def stage1_names(self) -> tuple[str, ...]:
        return tuple(self.stage1_functions)

    def run_stage1(self, function_name: str, context: PipelineContext) -> Any:
        """Run a single stage-1 function on ``context`` and return its result."""
        functions = self.stage1_functions
        try:
            function = functions[function_name]
        except KeyError:
            raise missing_stage1_error(self, function_name) from None
        return function(context)
