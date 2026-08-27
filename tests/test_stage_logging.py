import pytest

from virda.pipeline_context import NULL_LOGGER, PipelineContext
from virda.pipelines import helpers
from virda.pipelines.helpers import get_stage_logger


@pytest.fixture(autouse=True)
def _reset_stage_loggers() -> None:
    """Reset the process-global stage-logger cache between tests.

    The pipeline runs a single patient project per process, so the stage loggers
    (``virda.stage<number>``) are canonical per stage. Each test models a fresh
    process, so the cached loggers and their handlers must be torn down first;
    otherwise tests sharing the same stage name would collide on one logger.
    """
    loggers = list(helpers._stage_loggers.values())
    helpers._stage_loggers.clear()
    for logger in loggers:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


class TestStageLogFiles:
    def test_each_stage_writes_to_its_own_file(self, tmp_path) -> None:
        for stage_id, message in [
            ("stage_1", "STAGE1MSG"),
            ("stage_2", "STAGE2MSG"),
            ("stage_3", "STAGE3MSG"),
        ]:
            get_stage_logger(tmp_path, stage_id).info(message)

        for stage_id, message in [
            ("stage1", "STAGE1MSG"),
            ("stage2", "STAGE2MSG"),
            ("stage3", "STAGE3MSG"),
        ]:
            log_file = tmp_path / "logs" / f"{stage_id}.log"
            assert log_file.exists(), f"expected {stage_id}.log"
            content = log_file.read_text()
            assert message in content

    def test_stage_loggers_do_not_overwrite_each_other(self, tmp_path) -> None:
        logger1 = get_stage_logger(tmp_path, "stage_1")
        logger2 = get_stage_logger(tmp_path, "stage_2")

        logger1.info("only-in-1")
        logger2.info("only-in-2")

        log1 = (tmp_path / "logs" / "stage1.log").read_text()
        log2 = (tmp_path / "logs" / "stage2.log").read_text()
        assert "only-in-1" in log1
        assert "only-in-2" not in log1
        assert "only-in-2" in log2
        assert "only-in-1" not in log2

    def test_missing_logger_falls_back_to_noop(self, tmp_path) -> None:
        context = PipelineContext({})
        assert context.logger is None
        assert context.get_logger() is NULL_LOGGER

        context.logger = get_stage_logger(tmp_path, "stage_1")
        assert context.get_logger() is context.logger
