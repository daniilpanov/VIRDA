"""Background artifact preview worker.

Parses a saved artifact off the GUI thread so the interface stays responsive
while a table or text preview is built.  One worker lives for the whole
session on a dedicated :class:`~PySide6.QtCore.QThread`; streaming readers
abort an in-flight parse as soon as a newer request sequence is scheduled.
"""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from .preview import (
    npy_table_rows_chunked,
    open_npy_memmap,
    parse_csv_tsv_chunked,
    preview_artifact_text,
    preview_json_text_chunked,
    table_from_npy,
)


class _PreviewBundle:
    """Result of a background artifact preview: a table or plain text."""

    __slots__ = ("mode", "headers", "rows", "text")

    def __init__(
        self,
        mode: str,
        headers: list[str] | None = None,
        rows: list[list[str]] | None = None,
        text: str | None = None,
    ) -> None:
        self.mode = mode
        self.headers = headers or []
        self.rows = rows or []
        self.text = text or ""


class _PreviewWorker(QObject):
    """Parse a saved artifact off the GUI thread.

    One worker lives for the whole session on a dedicated :class:`QThread`.
    There is no request queue: :meth:`schedule` (called on the GUI thread)
    overwrites the single latest request, and streaming readers abort an
    in-flight parse as soon as a newer ``seq`` is scheduled.  ``ready``/
    ``failed`` are delivered back to the main thread because the widgets (the
    receivers) live there; stale results are dropped by the caller using the
    monotonic ``seq`` token.
    """

    ready = Signal(int, object)  # seq, _PreviewBundle
    failed = Signal(int, str)
    wake = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.wake.connect(self._on_wake)
        self._requested: tuple[int, Path | None] | None = None
        self._processed_seq = 0
        self._aborted = False

    def schedule(self, seq: int, path: Path | None) -> None:
        """Store the latest request, overwriting any earlier one.

        Called on the GUI thread: the single ``_requested`` write is atomic
        under the GIL, so a worker mid-parse sees the superseding ``seq`` at
        the next chunk boundary.  ``path is None`` cancels the active parse
        without scheduling new work.
        """
        self._requested = (seq, path)
        self.wake.emit()

    def stop(self) -> None:
        self._aborted = True

    def _on_wake(self) -> None:
        self._drain()

    def _drain(self) -> None:
        while not self._aborted:
            requested = self._requested
            if requested is None or requested[0] == self._processed_seq:
                return
            seq, path = requested
            if path is None:
                self._processed_seq = seq
                continue
            try:
                bundle = self._build_chunked(seq, path)
            except Exception as exc:
                self.failed.emit(seq, str(exc))
                self._processed_seq = seq
            else:
                if bundle is None:
                    continue  # superseded mid-parse; pick up whatever is newest
                self.ready.emit(seq, bundle)
                self._processed_seq = seq

    def _build_chunked(self, seq: int, path: Path) -> _PreviewBundle | None:
        def is_current() -> bool:
            requested = self._requested
            return not self._aborted and requested is not None and requested[0] == seq

        if path.is_dir():
            n = sum(1 for p in path.rglob("*") if p.is_file())
            return _PreviewBundle("text", text=f"{path}\n\n{n} file(s) in this folder.")

        suffix = path.suffix.lower()

        if suffix in (".csv", ".tsv"):
            parsed = parse_csv_tsv_chunked(path, is_current)
            if parsed is None:
                return None
            headers, rows = parsed
            return _PreviewBundle("table", headers=headers, rows=rows)

        if suffix == ".json":
            text = preview_json_text_chunked(path, is_current)
            if text is None:
                return None
            return _PreviewBundle("text", text=text)

        if suffix == ".npy":
            array = open_npy_memmap(path)
            try:
                try:
                    headers, tabular = table_from_npy(array, path.name)
                except ValueError:
                    return _PreviewBundle("text", text=preview_artifact_text(path, array=array))
                rows = npy_table_rows_chunked(tabular, is_current)
                if rows is None:
                    return None
                return _PreviewBundle("table", headers=headers, rows=rows)
            finally:
                if isinstance(array, np.memmap):
                    array.flush()
                    array._mmap.close()
                    del array

        if not is_current():
            return None
        text = preview_artifact_text(path)
        if not is_current():
            return None
        return _PreviewBundle("text", text=text)
