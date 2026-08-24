"""Tests for the extra log handler registration API."""

import logging

from virda.logging_setup import add_log_handler, remove_log_handler


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_child_logger_records_reach_parent_handler() -> None:
    handler = _ListHandler()
    add_log_handler(handler)
    try:
        logging.getLogger("virda.segmentation.seal").info("hello from library")
    finally:
        remove_log_handler(handler)

    assert "hello from library" in handler.messages


def test_stage_logger_records_reach_parent_handler() -> None:
    handler = _ListHandler()
    add_log_handler(handler)
    try:
        stage_logger = logging.getLogger("virda.stage1")
        old_handlers = list(stage_logger.handlers)
        stage_logger.handlers = []
        try:
            stage_logger.info("hello from stage")
        finally:
            stage_logger.handlers = old_handlers
    finally:
        remove_log_handler(handler)

    assert "hello from stage" in handler.messages


def test_add_is_idempotent_per_instance() -> None:
    handler = _ListHandler()
    logger = logging.getLogger("virda")
    add_log_handler(handler)
    add_log_handler(handler)
    try:
        attached = sum(1 for existing in logger.handlers if existing is handler)
        assert attached == 1
    finally:
        remove_log_handler(handler)
    assert all(existing is not handler for existing in logger.handlers)


def test_remove_unknown_handler_is_noop() -> None:
    handler = _ListHandler()
    before = list(logging.getLogger("virda").handlers)
    remove_log_handler(handler)
    assert list(logging.getLogger("virda").handlers) == before


def test_level_lowered_to_requested_and_not_restored_below() -> None:
    first = _ListHandler()
    second = _ListHandler()
    logger = logging.getLogger("virda")
    saved_level = logger.level
    try:
        add_log_handler(first, level=logging.DEBUG)
        assert logger.level == logging.DEBUG
        add_log_handler(second, level=logging.WARNING)
        assert logger.level == logging.DEBUG  # never raises back
    finally:
        remove_log_handler(first)
        remove_log_handler(second)
        logger.setLevel(saved_level)
