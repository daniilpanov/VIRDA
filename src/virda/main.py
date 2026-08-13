import logging
from pathlib import Path

from virda.config import get_virda_settings
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1PipelineBuilder


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("virda")
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

    return logger


def run(
    nifti_path: str | Path | None = None,
    project_dir: str | Path | None = None,
    fiducials_path: str | Path | None = None,
) -> Stage1Result:
    """Run the full VIRDA pipeline: Stage 1 → 2 → 3.

    Parameters
    ----------
    nifti_path
        Path to T1-weighted NIfTI. Falls back to ``settings.nifti_path``.
    project_dir
        Path to output directory
    fiducials_path
        Path to manual fiducials file. Falls back to ``settings.fiducials_path``.

    Returns
    -------
    Stage1Result
    """
    settings = get_virda_settings()

    resolved_path = nifti_path or settings.nifti_path
    if resolved_path is None:
        raise ValueError(
            "NIfTI path not provided. "
            "Pass it as an argument or set the NIFTI_PATH environment variable."
        )

    project_dir = project_dir or settings.project_dir
    if project_dir is None:
        raise ValueError(
            "Project directory path not provided. "
            "Pass it as an argument or set the PROJECT_DIR environment variable."
        )

    project = Path(project_dir)
    (project / "logs").mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(project / "logs")

    resolved_fiducials_path = fiducials_path or settings.fiducials_path

    return (
        Stage1PipelineBuilder.from_settings(
            settings=settings,
            nifti_path=resolved_path,
            project_dir=project,
            fiducials_path=resolved_fiducials_path,
            logger=logger,
        )
        .build()
        .run()
        .get_store_notnull(Stage1Result)
    )


if __name__ == "__main__":
    result = run()
    print(f"Stage 1: mesh with {len(result.mesh.vertices)} vertices")
