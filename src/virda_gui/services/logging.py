"""Registration of extra log handlers on the ``virda`` logger tree.

Pipeline/GUI work logs through ``logging.getLogger(__name__)`` under the
``virda`` namespace, so attaching a handler to the parent ``virda`` logger
captures every record produced by the library and the GUI without touching
root-level configuration.
"""

import logging

_LOGGER_NAME = "virda"

_handlers: list[logging.Handler] = []


def add_log_handler(handler: logging.Handler, *, level: int = logging.INFO) -> None:
    """Register *handler* so it receives every ``virda.*`` log record.

    Registration is idempotent for a given handler instance.  The package
    logger level is lowered to ``level`` when needed so INFO records are
    created at all (child loggers otherwise inherit WARNING).
    """
    if any(existing is handler for existing in _handlers):
        return
    logger = logging.getLogger(_LOGGER_NAME)
    _handlers.append(handler)
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > level:
        logger.setLevel(level)


def remove_log_handler(handler: logging.Handler) -> None:
    """Undo :func:`add_log_handler` for *handler* (no-op if absent)."""
    if any(existing is handler for existing in _handlers):
        _handlers.remove(handler)
    logging.getLogger(_LOGGER_NAME).removeHandler(handler)