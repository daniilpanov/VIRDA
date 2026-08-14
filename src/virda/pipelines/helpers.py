import logging
from pathlib import Path

# Avoid re-registering logger handlers
# Don't use loggers inside the other processes!
initialized_loggers: dict[str, tuple[logging.Logger, logging.Handler, logging.Handler]] = {}


def setup_pipeline_logging(project_dir: Path, stage_id: str) -> logging.Logger:
    log_dir = project_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f"virda.{stage_id}"
    logger_config = initialized_loggers.get(logger_name)
    if logger_config:
        logger, fh, ch = logger_config
        fh.close()
        logger.removeHandler(fh)
        ch.close()
        logger.removeHandler(ch)
    else:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_dir / "pipeline.log", mode="w")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    initialized_loggers[logger_name] = (logger, fh, ch)

    return logger
