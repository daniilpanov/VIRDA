"""The provider protocol: a side-effect hook bound to one store type.

Providers (logging, export) run right after a store value is written by
:mod:`virda.run`.  Pipeline steps are plain functions in :mod:`virda.ops`
and share no protocol.
"""

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from virda.pipeline_context import PipelineContext

StoreOfProvider = TypeVar("StoreOfProvider", contravariant=True)


class Provider(Protocol[StoreOfProvider]):
    def provide(
        self,
        store: StoreOfProvider | None,
        context: "PipelineContext",
    ) -> None: ...
