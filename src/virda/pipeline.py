import logging
from collections import defaultdict
from typing import Any, Protocol, TypeVar

from virda.pipeline_context import PipelineContext

Store = TypeVar("Store")
StepOutputStore = TypeVar("StepOutputStore", covariant=True)
StoreOfProvider = TypeVar("StoreOfProvider", contravariant=True)


class Step(Protocol[StepOutputStore]):
    def run(self, context: PipelineContext) -> StepOutputStore: ...


class Provider(Protocol[StoreOfProvider]):
    def provide(self, store: StoreOfProvider | None) -> None: ...


class PipelineController:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._steps: list[Step[Any]] = []
        self._context: PipelineContext = PipelineContext({})
        self._context.logger = logger
        self._providers: dict[type[Any], list[Provider[Any]]] = defaultdict(list)

    def set_logger(self, logger: logging.Logger | None) -> None:
        """Attach the stage logger so steps/providers can read it from context."""
        self._context.logger = logger

    def register_step(self, step: Step[Any]) -> None:
        self._steps.append(step)

    def register_store(self, store_type: type[Store], initial_value: Store | None = None) -> None:
        self._context.stores[store_type] = initial_value

    def register_provider(self, provider: Provider[Store], on_store: type[Store]) -> None:
        self._providers[on_store].append(provider)

    def run(self) -> PipelineContext:
        for step in self._steps:
            new_store = step.run(self._context)

            new_store_type = type(new_store)
            self._context.stores[new_store_type] = new_store

            if new_store_type in self._providers:
                for provider in self._providers[new_store_type]:
                    provider.provide(new_store)

        return self._context
