"""Protocols shared by pipeline steps and providers.

Steps read their inputs from and write their results into a
:class:`~virda.pipeline_context.PipelineContext`; providers are side-effect
hooks (logging, export) bound to one store type each.  The orchestration
itself lives in :mod:`virda.run`.
"""

from typing import Any, Protocol, TypeVar

from virda.pipeline_context import PipelineContext

StepOutputStore = TypeVar("StepOutputStore", covariant=True)
StoreOfProvider = TypeVar("StoreOfProvider", contravariant=True)


class Step(Protocol[StepOutputStore]):
    def run(self, context: PipelineContext) -> StepOutputStore: ...


class Provider(Protocol[StoreOfProvider]):
    def provide(self, store: StoreOfProvider | None, context: PipelineContext) -> None: ...
