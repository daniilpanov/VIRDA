import logging
from typing import Any

from virda.pipeline import Provider


class StoreLoggingProvider(Provider[Any]):
    """Provider that logs every store update via a structured logger."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def provide(self, store: Any) -> None:
        name = type(store).__name__
        extra = ""
        if hasattr(store, "shape"):
            extra = f" shape={store.shape}"
        elif hasattr(store, "vertices"):
            extra = f" vertices={len(store.vertices)}"
        elif hasattr(store, "locations"):
            extra = f" locations={len(store.locations)}"
        self.logger.info(f"Store '{name}' updated.{extra}")
