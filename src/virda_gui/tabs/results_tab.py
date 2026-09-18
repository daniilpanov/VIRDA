"""Saved Results tab: browse saved artifacts and preview them.

Lists the artifact files produced by a run, streams a lightweight preview for
the selected item on a background worker and embeds an interactive
:class:`~virda_gui.viewer.viewer.ViewerWidget` preview when a visual artifact
(mesh/NIfTI) is double-clicked.
"""

import subprocess
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from virda_gui.constants import PROJECT_ARTIFACT_DIRS
from virda_gui.preview.preview_worker import _PreviewBundle, _PreviewWorker
from virda_gui.services.file_manager import open_in_file_manager
from virda_gui.state import AppState
from virda_gui.viewer.viewer import ViewerWidget
from virda_gui.widgets import DirectorySelector


class ResultsTab(QWidget):
    """The "Saved Results" tab of the main window."""

    openViewer = Signal(object)  # noqa: N815  # project: Path
    exportHtml = Signal(object)  # noqa: N815  # project: Path

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state

        self._results_project_dir = DirectorySelector(self, label="Project dir")
        self._results_tree = QTreeWidget(self)
        self._results_summary_label = QLabel("")
        self._results_preview = QPlainTextEdit(self)
        self._results_table = QTableWidget(self)
        self._results_viewer_widget: ViewerWidget | None = None
        self._preview_stack = QStackedWidget(self)
        self._preview_load_seq = 0
        self._preview_thread: QThread | None = None
        self._preview_worker: _PreviewWorker | None = None

        self._build_ui()

    # ---- UI construction ----

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)

        row = QFrame(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        row_layout.addWidget(self._results_project_dir, 1)
        self._results_project_dir.directory_changed.connect(self.refresh)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        row_layout.addWidget(refresh_btn)

        outer.addWidget(row)

        splitter = QSplitter(Qt.Orientation.Vertical, self)

        tree_widget = QWidget(splitter)
        tree_layout = QVBoxLayout(tree_widget)
        tree_layout.setContentsMargins(0, 0, 0, 0)

        self._results_tree.setColumnCount(3)
        self._results_tree.setHeaderLabels(["Artifact", "Size", "Modified"])
        self._results_tree.setColumnWidth(0, 260)
        self._results_tree.itemSelectionChanged.connect(self._on_results_artifact_selected)
        self._results_tree.itemDoubleClicked.connect(self._on_results_artifact_double_clicked)
        tree_layout.addWidget(self._results_tree)

        preview_box = QGroupBox("Preview", splitter)
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self._preview_stack)

        self._results_preview.setReadOnly(True)
        self._preview_stack.addWidget(self._results_preview)

        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results_table.setAlternatingRowColors(True)
        self._preview_stack.addWidget(self._results_table)

        self._preview_stack.setCurrentWidget(self._results_preview)

        splitter.addWidget(tree_widget)
        splitter.addWidget(preview_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

        actions = QFrame(self)
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 4, 0, 0)

        viewer_btn = QPushButton("Open 3D Viewer")
        viewer_btn.clicked.connect(self._on_open_viewer)
        actions_layout.addWidget(viewer_btn)
        actions_layout.addSpacing(8)

        export_btn = QPushButton("Export HTML Viewer")
        export_btn.clicked.connect(self._on_export_html)
        actions_layout.addWidget(export_btn)
        actions_layout.addSpacing(8)

        reveal_btn = QPushButton("Show in Explorer")
        reveal_btn.clicked.connect(self._on_show_in_explorer)
        actions_layout.addWidget(reveal_btn)

        actions_layout.addStretch(1)
        actions_layout.addWidget(self._results_summary_label)

        outer.addWidget(actions)

    # ---- Public API used by the host application ----

    def set_project_dir(self, project: str | Path) -> None:
        self._results_project_dir.set(str(project))

    def selected_project(self) -> Path | None:
        raw = self._results_project_dir.get().strip()
        if not raw:
            return None
        project = Path(raw)
        return project if project.is_dir() else None

    def refresh(self) -> None:
        tree = self._results_tree
        tree.clear()
        self._clear_preview()

        project = self.selected_project()
        if project is None:
            self._results_summary_label.setText("Select a valid project directory.")
            return

        root_item = QTreeWidgetItem([project.name, "<dir>", ""])
        root_item.setExpanded(True)
        root_item.setData(0, Qt.ItemDataRole.UserRole, project)
        tree.addTopLevelItem(root_item)

        known = [name for name in PROJECT_ARTIFACT_DIRS if (project / name).is_dir()]
        extra_dirs = sorted(
            entry.name
            for entry in project.iterdir()
            if entry.is_dir() and entry.name not in PROJECT_ARTIFACT_DIRS
        )
        loose_files = sorted(entry for entry in project.iterdir() if entry.is_file())

        n_files = 0
        for subdir in known + extra_dirs:
            node = project / subdir
            group_item = QTreeWidgetItem([subdir, "<dir>", ""])
            group_item.setData(0, Qt.ItemDataRole.UserRole, node)
            root_item.addChild(group_item)
            for file_path in sorted(node.rglob("*")):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(node).as_posix()
                child = QTreeWidgetItem(
                    [rel, self._format_file_size(file_path), self._format_mtime(file_path)]
                )
                child.setData(0, Qt.ItemDataRole.UserRole, file_path)
                group_item.addChild(child)
                n_files += 1

        for file_path in loose_files:
            child = QTreeWidgetItem(
                [file_path.name, self._format_file_size(file_path), self._format_mtime(file_path)]
            )
            child.setData(0, Qt.ItemDataRole.UserRole, file_path)
            root_item.addChild(child)
            n_files += 1

        self._results_summary_label.setText(
            f"{n_files} file(s)" if n_files else "No saved artifacts found yet."
        )

    def shutdown(self) -> None:
        if self._results_viewer_widget is not None:
            self._results_viewer_widget.shutdown()
        thread = self._preview_thread
        if thread is None:
            return
        if not thread.isRunning():
            self._preview_thread = None
            self._preview_worker = None
            return
        worker = self._preview_worker
        if worker is not None:
            worker.ready.disconnect(self._on_preview_ready)
            worker.failed.disconnect(self._on_preview_failed)
            worker.stop()
            # deleteLater() must be posted while the thread's event loop is
            # still live, otherwise the DeferredDelete event is never processed.
            worker.deleteLater()
        thread.quit()
        thread.wait(3000)
        # Never delete a thread that is still running (wait timed out);
        # destroying a live QThread is undefined behaviour.  In that case the
        # thread keeps running under its parent until it finishes.
        if thread.isFinished():
            thread.deleteLater()
        self._preview_thread = None
        self._preview_worker = None

    # ---- Actions ----

    def _on_open_viewer(self) -> None:
        project = self.selected_project()
        if project is None:
            QMessageBox.warning(
                self,
                "No project directory",
                "Select a valid project directory on the Saved Results tab first.",
            )
            return
        self.openViewer.emit(project)

    def _on_export_html(self) -> None:
        project = self.selected_project()
        if project is None:
            QMessageBox.warning(
                self,
                "No project directory",
                "Select a valid project directory on the Saved Results tab first.",
            )
            return
        self.exportHtml.emit(project)

    def _on_show_in_explorer(self) -> None:
        project = self.selected_project()
        if project is None:
            QMessageBox.warning(
                self,
                "No project directory",
                "Select a valid project directory on the Saved Results tab first.",
            )
            return
        try:
            open_in_file_manager(project)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            QMessageBox.critical(self, "File manager error", f"Could not open the folder:\n{exc}")

    # ---- Artifact selection ----

    def _on_results_artifact_selected(self) -> None:
        items = self._results_tree.selectedItems()
        if not items:
            return
        item = items[0]
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path is None:
            return
        self._start_preview_load(path)

    def _on_results_artifact_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path is None or path.is_dir():
            return
        name = path.name.lower()
        if not (name.endswith(".ply") or name.endswith((".nii.gz", ".nii"))):
            return
        viewer = self._ensure_results_viewer()
        if name.endswith(".ply"):
            viewer.load(mesh_path=str(path))
        else:
            viewer.load(nifti_path=str(path))
        self._preview_stack.setCurrentWidget(viewer)
        self._preview_load_seq += 1
        if self._preview_worker is not None:
            self._preview_worker.schedule(self._preview_load_seq, None)

    def _ensure_results_viewer(self) -> ViewerWidget:
        if self._results_viewer_widget is None:
            self._results_viewer_widget = ViewerWidget(
                self._preview_stack, log=self._state.log_queue.put
            )
            self._results_viewer_widget.sceneFailed.connect(self._on_results_viewer_failed)
            self._preview_stack.addWidget(self._results_viewer_widget)
        return self._results_viewer_widget

    def _on_results_viewer_failed(self, message: str) -> None:
        self._set_results_text(f"3D preview failed: {message}")

    # ---- Background preview ----

    def _start_preview_load(self, path: Path) -> None:
        self._preview_load_seq += 1
        seq = self._preview_load_seq
        self._clear_preview()
        self._set_results_text(f"Loading {path.name}...")
        self._ensure_preview_worker().schedule(seq, path)

    def _ensure_preview_worker(self) -> _PreviewWorker:
        if self._preview_thread is None:
            thread = QThread(self)
            worker = _PreviewWorker()
            worker.moveToThread(thread)
            worker.ready.connect(self._on_preview_ready)
            worker.failed.connect(self._on_preview_failed)
            thread.start()
            self._preview_thread = thread
            self._preview_worker = worker
        assert self._preview_worker is not None
        return self._preview_worker

    def _on_preview_ready(self, seq: int, bundle: _PreviewBundle) -> None:
        if self._state.closed or seq != self._preview_load_seq:
            return
        if bundle.mode == "table":
            self._set_results_table(bundle.headers, bundle.rows)
        else:
            self._set_results_text(bundle.text)

    def _on_preview_failed(self, seq: int, message: str) -> None:
        if self._state.closed or seq != self._preview_load_seq:
            return
        self._set_results_text(f"Failed to read preview:\n{message}")

    def _set_results_text(self, text: str) -> None:
        self._preview_stack.setCurrentWidget(self._results_preview)
        self._results_preview.setPlainText(text)

    def _set_results_table(self, headers: list[str], rows: list[list[str]]) -> None:
        table = self._results_table
        n_columns = max(len(headers), 1)
        table.clear()
        table.setColumnCount(n_columns)
        if headers:
            table.setHorizontalHeaderLabels(headers)
        else:
            table.setHorizontalHeaderLabels([f"Col {i}" for i in range(n_columns)])
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j in range(n_columns):
                table.setItem(i, j, QTableWidgetItem(row[j] if j < len(row) else ""))
        table.resizeColumnsToContents()
        self._preview_stack.setCurrentWidget(table)

    def _clear_preview(self) -> None:
        self._results_preview.setPlainText("")
        self._results_table.clear()
        self._results_table.setRowCount(0)
        self._preview_stack.setCurrentWidget(self._results_preview)

    @staticmethod
    def _format_file_size(path: Path) -> str:
        size = float(path.stat().st_size)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"

    @staticmethod
    def _format_mtime(path: Path) -> str:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
