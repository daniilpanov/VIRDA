"""Parser for tabular (TSV/CSV) electrode tables."""

from pathlib import Path

from virda.io.loader.electrodes_table_loader import (
    ElectrodesTable,
    load_electrodes_table,
)


def parse_electrodes_table(path: str | Path) -> ElectrodesTable:
    """Load a TSV/CSV electrode table with ``name``/``x``/``y``/``z`` columns.

    Returns the result of :func:`virda.io.loader.electrodes_table_loader.load_electrodes_table`:
    ``(positions, residuals, flags, measured, names)``.
    """
    return load_electrodes_table(Path(path))
