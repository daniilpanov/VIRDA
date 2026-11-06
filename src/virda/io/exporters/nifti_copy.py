"""Export a byte-for-byte copy of the source MRI NIfTI into a project."""

import shutil
from logging import Logger
from pathlib import Path


def export_nifti_copy(
    path: str | Path,
    source: str | Path,
    *,
    logger: Logger | None = None,
) -> Path | None:
    """Copy the source NIfTI at *source* to *path*, replacing any old copy.

    When *path* already points at *source* itself (e.g. the run reads the
    canonical copy inside ``input/``) nothing is copied and the existing file
    is returned untouched, so the source is never replaced by itself.

    Copy errors are only reported as warnings; the import-side artifacts are
    considered best-effort companions of the run.  Returns the written path,
    or ``None`` when the copy failed.
    """
    target = Path(path)
    try:
        if target.is_file() and Path(source).resolve() == target.resolve():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            target.unlink()
        shutil.copy2(source, target)
        return target
    except OSError:
        if logger is not None:
            logger.warning(
                "Failed to copy source NIfTI ('%s') into patient project ('%s')",
                source,
                target,
                exc_info=True,
            )
        return None
