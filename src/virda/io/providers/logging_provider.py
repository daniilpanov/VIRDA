from typing import Any

from virda.pipeline import Provider
from virda.pipeline_context import PipelineContext


class StoreLoggingProvider(Provider[Any]):
    """Provider that logs every store update via the stage logger."""

    def provide(self, store: Any, context: PipelineContext) -> None:
        logger = context.get_logger()
        name = type(store).__name__
        extra = ""
        if hasattr(store, "shape"):
            extra = f" shape={store.shape}"
        elif hasattr(store, "vertices"):
            extra = f" vertices={len(store.vertices)}"
        elif hasattr(store, "locations"):
            extra = f" locations={len(store.locations)}"
        logger.info(f"Store '{name}' updated.{extra}")
