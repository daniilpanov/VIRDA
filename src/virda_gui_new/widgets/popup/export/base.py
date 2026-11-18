"""Shared base of the export popups: selectors, path picker, result dialogs.

Every export popup offers an optional coordinate-frame selector, a path
line edit with a Browse button prefilled with the kind's canonical
default project path, and an Export button. A missing source surfaces at
the Export click through the error dialog with the reason, so the menu
actions are never disabled. On success the popup closes and a modal
"Export successful" box shows; on failure it stays open behind the
"Export error" box, whose Retry returns without re-running the export.
"""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from virda.models.coordinates import CoordinateSystem
from virda.models.stage3_config import Stage3Config
from virda_gui_new.dto.steps import LOCALIZATION
from virda_gui_new.services.artifact_exporting import (
    ExportKind,
    MeshExportKind,
    default_export_path,
    export_artifact,
)
from virda_gui_new.services.step_config import load_step_config
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.localization.abstract import localization_options
from virda_gui_new.widgets.popup.localization.frame_selector import FrameSelector

_BROWSE_TITLE = "Export to"


class ExportPopupBase(QWidget):
    """Base widget of every export popup; subclasses add kind controls.

    Subclasses set the ``_TITLE``/``_HINT`` class attributes, may add
    kind-specific rows in :meth:`build_form` (added between the frame
    selector and the path row) and may restrict frame availability by
    overriding :meth:`_fiducials_frame_offered`. The base owns the
    frame selector (unless the kind has no frames), the path picker,
    the Export button and both result dialogs.
    """

    exportSucceeded = Signal(Path)  # noqa: N815 - Qt signal API
    exportFailed = Signal(str)  # noqa: N815 - Qt signal API

    _TITLE = "Export"
    _HINT = "Exports the current project artifact."

    #: Whether the popup offers the coordinate-frame selector.
    _has_frames: bool = True

    def __init__(
        self,
        manager: StepsManager,
        kind: ExportKind,
        parent: QWidget | None = None,
        *,
        project: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._kind = kind
        self._project = project
        self._coordinates = manager.geometry
        self._frame_selector: FrameSelector | None = None
        self._mesh_combo: QComboBox | None = None

        self.setWindowTitle(self._TITLE)
        hint = QLabel(self._HINT, self)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")

        self._form_layout = QVBoxLayout()
        if self._has_frames:
            self._frame_selector = FrameSelector(self)
            self._form_layout.addWidget(QLabel("Coordinate frame", self))
            self._form_layout.addWidget(self._frame_selector)
            self._sync_frame_availability()
        self.build_form(self._form_layout)

        self._path_edit = QLineEdit(self)
        browse = QPushButton("Browse", self)
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("File", self))
        path_row.addWidget(self._path_edit, stretch=1)
        path_row.addWidget(browse)

        self._export_button = QPushButton("Export", self)
        self._export_button.clicked.connect(self._on_export)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(self._TITLE, self))
        layout.addWidget(hint)
        layout.addLayout(self._form_layout)
        layout.addLayout(path_row)
        layout.addWidget(self._export_button)
        self._apply_default_path()

    def build_form(self, layout: QVBoxLayout) -> None:
        """Extension hook: add kind-specific rows here."""

    def _fiducials_frame_offered(self) -> bool:
        """Whether the popup offers the fiducials frame at all."""
        return True

    # --- current selections ---

    def _frame(self) -> CoordinateSystem:
        """The selected target frame (``world`` without a selector)."""
        if self._frame_selector is None:
            return CoordinateSystem.WORLD
        return self._frame_selector.frame()

    def _mesh_kind(self) -> MeshExportKind:
        """The selected mesh source (``base`` without a mesh combo)."""
        if self._mesh_combo is None:
            return MeshExportKind.BASE
        kind = self._mesh_combo.currentData()
        return kind if isinstance(kind, MeshExportKind) else MeshExportKind.BASE

    # --- path picker ---

    def _apply_default_path(self) -> None:
        """Prefill the path with the kind's canonical default, if any."""
        default = default_export_path(self._project, self._kind, self._mesh_kind())
        self._path_edit.setText("" if default is None else str(default))

    def _browse(self) -> None:
        """One save path chosen in a file dialog, or keep the current."""
        start = self._path_edit.text().strip()
        path, _ = QFileDialog.getSaveFileName(self, _BROWSE_TITLE, start)
        if path:
            self._path_edit.setText(path)

    # --- frame availability (localization popup's logic, reused) ---

    def _sync_frame_availability(self) -> None:
        """Enable available frames; fall back to head/world when needed.

        The fiducials frame is greyed while unavailable (fewer than
        three fiducials have coordinates) or not offered by the popup;
        the head frame while no scalp mesh exists. A selected frame that
        is not available falls back to head (or world without a scalp
        mesh).
        """
        if self._frame_selector is None:
            return
        head_ok = self._coordinates.is_available(CoordinateSystem.HEAD)
        fiducials_ok = self._fiducials_frame_offered() and self._coordinates.is_available(
            CoordinateSystem.FIDUCIALS
        )
        self._frame_selector.set_frame_enabled(CoordinateSystem.HEAD, head_ok)
        self._frame_selector.set_frame_enabled(CoordinateSystem.FIDUCIALS, fiducials_ok)
        frame = self._frame_selector.frame()
        if not self._coordinates.is_available(frame):
            fallback = CoordinateSystem.HEAD if head_ok else CoordinateSystem.WORLD
            self._frame_selector.setCurrentIndex(self._frame_selector.findData(fallback))

    # --- the localization settings of the project ---

    def _localization_options(self) -> Stage3Config:
        """The localization settings the step stored.

        An export that has to compute the electrodes localizes them the
        way the step does, so it reads the settings the step stored.
        """
        if self._project is None:
            return Stage3Config()
        return localization_options(load_step_config(self._project, LOCALIZATION))

    # --- export click and result dialogs ---

    def _on_export(self) -> None:
        """Run the export service; report through the result dialogs."""
        text = self._path_edit.text().strip()
        if not text:
            self.exportFailed.emit("no target path")
            self._show_error("Choose a target file first: pick or type a path.")
            return
        try:
            written = export_artifact(
                self._manager,
                self._kind,
                Path(text),
                frame=self._frame(),
                mesh_kind=self._mesh_kind(),
                options=self._localization_options(),
            )
        except Exception as error:  # noqa: BLE001 - any export error is user-facing
            self.exportFailed.emit(str(error))
            self._show_error(str(error))
            return
        self.exportSucceeded.emit(written)
        self._report_success(written)

    def _report_success(self, written: Path) -> None:
        """Close the popup host, then show the modal success box.

        On success the popup closes first, then the modal ``Export
        successful`` box shows a single ``Close`` button. Nothing runs
        after the box closes: the closed host window
        (``WA_DeleteOnClose``) may already be destroyed, so this method
        deliberately does not touch ``self`` past the static message-box
        call.
        """
        window = self.window()
        if window is not self:
            window.close()
        QMessageBox.information(
            None,
            "Export successful",
            f"Saved: {written}",
            QMessageBox.StandardButton.Close,
        )

    def _show_error(self, message: str) -> None:
        """Modal ``Export error`` box; Close closes, Retry keeps the popup.

        The popup stays open behind the box either way. ``Retry`` returns
        to the popup without auto-retrying (its settings are untouched);
        ``Close`` closes the popup.
        """
        box = QMessageBox(QMessageBox.Icon.Critical, "Export error", message)
        close_button = box.addButton(QMessageBox.StandardButton.Close)
        retry_button = box.addButton("Retry", QMessageBox.ButtonRole.ActionRole)
        box.setDefaultButton(retry_button)
        box.exec()
        if box.clickedButton() is close_button:
            self._close_host()

    def _close_host(self) -> None:
        """Close the hosting tool window (or this widget when standalone)."""
        window = self.window()
        if window is not self:
            window.close()
        else:
            self.close()
