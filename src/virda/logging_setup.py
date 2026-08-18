"""Registration of extra log handlers on the ``virda`` logger tree.

Pipeline stages log through per-stage loggers (``virda.<stage>``) configured
by :func:`virda.pipelines.helpers.setup_pipeline_logging`, and library
modules through ``logging.getLogger(__name__)`` under the same ``virda``
namespace.  Attaching a handler to the parent ``virda`` logger therefore
captures every record produced by the package without touching the
stage-specific file and console handlers.
"""

import logging

_LOGGER_NAME = "virda"

_handlers: list[logging.Handler] = []


def add_log_handler(handler: logging.Handler, *, level: int = logging.INFO) -> None:
    """Register an extra handler that receives every ``virda.*`` log record.

    The handler is attached to the parent logger, so records from all stages
    and library modules propagate to it while the stage file/console handlers
    keep working unchanged.  Registration is idempotent for a given handler
    instance.  The package logger level is lowered to ``level`` when needed so
    INFO records are created at all (child loggers otherwise inherit WARNING).
    """
    if any(existing is handler for existing in _handlers):
        return
    logger = logging.getLogger(_LOGGER_NAME)
    _handlers.append(handler)
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > level:
        logger.setLevel(level)


def remove_log_handler(handler: logging.Handler) -> None:
    """Undo :func:`add_log_handler` for ``handler`` (no-op if absent)."""
    if any(existing is handler for existing in _handlers):
        _handlers.remove(handler)
    logging.getLogger(_LOGGER_NAME).removeHandler(handler)
