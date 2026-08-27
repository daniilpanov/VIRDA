import logging
from pathlib import Path

# One canonical logger per (stage, project) so re-running a stage reuses the
# same logger/handler instead of duplicating them. Don't use loggers inside
# the other processes!
_stage_loggers: dict[tuple[str, Path], logging.Logger] = {}


def get_stage_logger(project_dir: Path, stage_id: str) -> logging.Logger:
    """Return the canonical logger for ``stage_id`` under ``project_dir``.

    Each stage writes to its own file (``<project>/logs/<stage>.log``, e.g.
    ``stage1.log``) with ``mode="w"``, so stage logs never overwrite each
    other.  The logger propagates to the parent ``virda`` logger, so handlers
    attached there (e.g. the GUI) keep receiving every record.
    """
    key = (stage_id, Path(project_dir))
    cached = _stage_loggers.get(key)
    if cached is not None:
        return cached

    logger = logging.getLogger(f"virda.{stage_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = True

    log_dir = Path(project_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{stage_id.replace('_', '')}.log"

    fh = logging.FileHandler(log_file, mode="w")
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

    _stage_loggers[key] = logger
    return logger
