"""Preview helpers for the Saved Results tab.

Pure data helpers are kept free of Qt so they can be unit-tested without a
display; the QThread worker and widgets live in :mod:`virda_gui.app`.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

_DELIMITERS = ",;\t"

_TEXT_PREVIEW_SUFFIXES = {".txt", ".log", ".csv", ".tsv", ".md"}
_PREVIEW_MAX_CHARS = 8000
_READ_CHUNK = 8192
_CSV_ROW_BLOCK = 256
_NPY_ROW_BLOCK = 256
_JSON_PREVIEW_MAX_CHARS = 65536


def _always() -> bool:
    return True


def _artifact_header(path: Path) -> str:
    return f"{path}\n{'-' * 60}\n"


def _truncate_text(header: str, body: str) -> str:
    if len(body) > _PREVIEW_MAX_CHARS:
        body = (
            body[:_PREVIEW_MAX_CHARS]
            + f"\n\n... (truncated to first {_PREVIEW_MAX_CHARS} characters)"
        )
    return header + body


def parse_csv_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Parse a CSV/TSV file into ``(headers, rows)``.

    See :func:`parse_csv_tsv_chunked` for the parsing rules; this variant reads
    the whole file at once.
    """
    parsed = parse_csv_tsv_chunked(path, _always)
    assert parsed is not None
    return parsed


def parse_csv_tsv_chunked(
    path: Path, should_continue: Callable[[], bool]
) -> tuple[list[str], list[list[str]]] | None:
    """Parse a CSV/TSV file, abortable between row blocks.

    The first non-empty line is treated as the header; every following row is
    padded/truncated to the header width so a tabular viewer never runs into a
    column-surplus.  Encoding is assumed to be UTF-8 (a BOM is stripped).
    Returns ``None`` when ``should_continue`` fails between blocks.
    """
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        sample = fh.read(_READ_CHUNK)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=_DELIMITERS)
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(fh, dialect)
        header_row = next(reader, None)
        if header_row is None:
            return [], []

        headers = list(header_row)
        width = max(len(headers), 1)
        body: list[list[str]] = []

        while True:
            block: list[list[str]] = []
            for _ in range(_CSV_ROW_BLOCK):
                row = next(reader, None)
                if row is None:
                    break
                block.append(row)

            if not block:
                break

            if not should_continue():
                return None

            for row in block:
                body.append((row + [""] * width)[:width])

    return headers, body


_NPY_COLUMN_MAP = [
    ("*vertices*", ["X", "Y", "Z"]),
    ("*faces*", ["A", "B", "C"]),
    ("*adjacency*", ["Face A", "Face B"]),
    ("*normals*", ["NX", "NY", "NZ"]),
    ("*quality*", ["Quality"]),
]


def npy_columns_for(filename: str) -> list[str] | None:
    """Return column names for a ``.npy`` file matched by filename mask.

    The map is keyed on the file *name* (not the path) so any project layout
    reuses the same headers; returns ``None`` when no mask applies and the
    caller should fall back to generic ``Col N`` headers.
    """
    name = filename.lower()
    for pattern, columns in _NPY_COLUMN_MAP:
        if fnmatch(name, pattern):
            return columns

    return None


def table_from_npy(array: np.ndarray, filename: str) -> tuple[list[str], np.ndarray]:
    """Derive ``(headers, array)`` for tabulation without any file IO.

    Raises :class:`ValueError` for arrays with ``ndim > 2``; 1D arrays get the
    single column name ``Value``.  Masked headers are only kept when their width
    matches the array columns, otherwise generic ``Col N`` names are used.
    """
    if array.ndim > 2:
        raise ValueError(f"{filename} is {array.ndim}D - only 1D/2D arrays can be tabulated")
    if array.ndim == 1:
        return ["Value"], array

    n_columns = int(array.shape[1])
    columns = npy_columns_for(filename) or [f"Col {i}" for i in range(n_columns)]
    if len(columns) != n_columns:
        columns = [f"Col {i}" for i in range(n_columns)]

    return columns, array


def load_npy_table(path: Path) -> tuple[list[str], np.ndarray]:
    """Load a 2D/1D ``.npy`` file into ``(headers, array)``.

    See :func:`table_from_npy` for the tabulation rules.
    """
    return table_from_npy(np.load(path, allow_pickle=False), path.name)


def npy_table_rows(array: np.ndarray) -> list[list[str]]:
    """Convert a 1D/2D numpy array into string rows for a table widget."""
    if array.ndim == 1:
        return [[str(value)] for value in array.tolist()]

    if array.ndim == 0:
        return [[str(array.item())]]

    return [[str(value) for value in row] for row in array.tolist()]


def open_npy_memmap(path: Path) -> np.memmap:
    """Open a ``.npy`` file as a read-only memory map (data is read lazily)."""
    return np.load(path, mmap_mode="r")


def npy_table_rows_chunked(
    array: np.ndarray, should_continue: Callable[[], bool]
) -> list[list[str]] | None:
    """Convert array rows into string rows in blocks, abortable between them.

    Returns ``None`` when ``should_continue`` fails between blocks.
    """
    if array.ndim == 0:
        return [[str(array.item())]]

    if array.ndim == 1:
        rows: list[list[str]] = []
        n = int(array.shape[0])
        for start in range(0, n, _NPY_ROW_BLOCK):
            if not should_continue():
                return None

            block = array[start : start + _NPY_ROW_BLOCK]
            rows.extend([str(value)] for value in block.tolist())
        return rows

    rows = []
    n_rows = int(array.shape[0])
    for start in range(0, n_rows, _NPY_ROW_BLOCK):
        if not should_continue():
            return None

        block = array[start : start + _NPY_ROW_BLOCK]
        for row in block.tolist():
            rows.append([str(value) for value in row])

    return rows


def preview_json_text_chunked(path: Path, should_continue: Callable[[], bool]) -> str | None:
    """Stream a JSON file in chunks, abortable between chunks.

    Accumulates raw text up to ``_JSON_PREVIEW_MAX_CHARS`` and pretty-prints
    whatever parses; an incomplete document falls back to a truncated raw
    snippet.  Returns ``None`` when ``should_continue`` fails.
    """
    pieces: list[str] = []
    length = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        while length < _JSON_PREVIEW_MAX_CHARS:
            if not should_continue():
                return None

            chunk = fh.read(_READ_CHUNK)
            if not chunk:
                break

            pieces.append(chunk)
            length += len(chunk)

    header = _artifact_header(path)
    body = "".join(pieces)
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return _truncate_text(header, body)

    return _truncate_text(header, json.dumps(data, indent=2, ensure_ascii=False))


def preview_artifact_text(path: Path, array: np.ndarray | None = None) -> str:
    """Human-readable preview of a saved artifact (NIfTI/npy/PLY/text/JSON).

    ``array`` avoids reloading a ``.npy`` file whose contents were already
    read by the caller; it is only used for ``.npy`` artifacts.  JSON is
    handled by :func:`preview_json_text_chunked`.
    """
    suffix = path.suffix.lower()
    header = _artifact_header(path)

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return _truncate_text(header, json.dumps(data, indent=2, ensure_ascii=False))

    if path.name.lower().endswith((".nii.gz", ".nii")):
        img: Any = nib.load(str(path))
        nii_header: Any = img.header
        shape = tuple(int(v) for v in nii_header.get_data_shape())
        zooms = tuple(round(float(z), 3) for z in nii_header.get_zooms()[:3])
        return header + f"NIfTI volume\n  shape         : {shape}\n  spacing (mm)  : {zooms}"

    if suffix == ".npy":
        loaded = array if array is not None else np.load(path, allow_pickle=False)
        body = f"NumPy array\n  shape : {loaded.shape}\n  dtype : {loaded.dtype}"
        return _truncate_text(header, body)

    if suffix == ".ply":
        lines: list[str] = []
        with open(path, "rb") as fh:
            for line in fh:
                decoded = line.decode("ascii", errors="replace").rstrip("\r\n")
                lines.append(decoded)
                if len(lines) >= 100 or decoded.strip() == "end_header":
                    break

        return _truncate_text(header, "PLY header:\n" + "\n".join(lines))

    if suffix in _TEXT_PREVIEW_SUFFIXES:
        return _truncate_text(header, path.read_text(encoding="utf-8", errors="replace"))

    size_kb = path.stat().st_size / 1024
    return header + f"(binary file, no preview — {size_kb:.1f} KB)"
