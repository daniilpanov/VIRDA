"""Preview tab: background-parsed text/table view of a single project file.

Reuses the session-wide streaming worker protocol from
:mod:`virda_gui.preview.preview_worker`: each open preview tab owns one
worker on its own :class:`~PySide6.QtCore.QThread`, so a large JSON/NPY/CSV
artifact never blocks the GUI thread while it is being parsed.
"""

from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QPlainTextEdit,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from virda_gui.preview.preview_worker import _PreviewBundle, _PreviewWorker


class PreviewTab(QWidget):
    """A closable tab streaming a text/table preview for one file."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text_view = QPlainTextEdit(self)
        self._text_view.setReadOnly(True)
        self._table_view = QTableWidget(self)
        self._table_view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table_view.setAlternatingRowColors(True)

        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._text_view)
        self._stack.addWidget(self._table_view)
        self._stack.setCurrentWidget(self._text_view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._stack)

        self._seq = 0
        self._thread = QThread(self)
        self._worker = _PreviewWorker()
        self._worker.moveToThread(self._thread)
        self._thread.start()
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)

    def open(self, path: str | Path) -> None:
        """Start previewing *path*; supersedes any in-flight preview."""
        self._seq += 1
        self._text_view.setPlainText(f"Loading {path}...")
        self._stack.setCurrentWidget(self._text_view)
        self._worker.schedule(self._seq, Path(path))

    def _on_ready(self, seq: int, bundle: _PreviewBundle) -> None:
        if seq != self._seq:
            return
        if bundle.mode == "table":
            self._show_table(bundle.headers, bundle.rows)
        else:
            self._text_view.setPlainText(bundle.text or "")
            self._stack.setCurrentWidget(self._text_view)

    def _on_failed(self, seq: int, message: str) -> None:
        if seq != self._seq:
            return
        self._text_view.setPlainText(f"Preview failed: {message}")
        self._stack.setCurrentWidget(self._text_view)

    def _show_table(self, headers: list[str], rows: list[list[str]]) -> None:
        table = self._table_view
        n_columns = max(len(headers), 1)
        table.clear()
        table.setColumnCount(n_columns)
        table.setHorizontalHeaderLabels(headers if headers else [str(i) for i in range(n_columns)])
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j in range(n_columns):
                table.setItem(i, j, QTableWidgetItem(row[j] if j < len(row) else ""))
        table.resizeColumnsToContents()
        self._stack.setCurrentWidget(table)

    def shutdown(self) -> None:
        """Stop the preview worker and wait for its thread to finish."""
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)
