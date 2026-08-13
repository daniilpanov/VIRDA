from dataclasses import dataclass
from typing import Any, TypeVar

Store = TypeVar("Store")


@dataclass
class PipelineContext:
    stores: dict[type[Any], Any | None]

    def get_store(self, store_type: type[Store]) -> Store | None:
        return self.stores.get(store_type)

    def get_store_notnull(self, store_type: type[Store]) -> Store:
        store = self.get_store(store_type)
        if not store:
            raise KeyError(f"'{store_type.__name__}' store is not initialized")

        return store
