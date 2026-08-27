import logging
from dataclasses import dataclass
from typing import Any, TypeVar

Store = TypeVar("Store")

NULL_LOGGER = logging.getLogger("virda._null")
NULL_LOGGER.addHandler(logging.NullHandler())
NULL_LOGGER.propagate = False


@dataclass
class PipelineContext:
    stores: dict[type[Any], Any | None]
    logger: logging.Logger | None = None

    def get_store(self, store_type: type[Store]) -> Store | None:
        return self.stores.get(store_type)

    def get_store_notnull(self, store_type: type[Store]) -> Store:
        store = self.get_store(store_type)
        if not store:
            raise KeyError(f"'{store_type.__name__}' store is not initialized")

        return store

    def get_logger(self) -> logging.Logger:
        """Return the stage logger, or a no-op logger when none is configured.

        Callers should always log through this so that a missing logger simply
        suppresses the record instead of falling back to a module-level logger.
        """
        return self.logger or NULL_LOGGER
