"""Reusable widgets for the VIRDA GUI application (PySide6).

Provides ``FileSelector``, ``DirectorySelector``, ``CollapsibleSection``,
``LabeledField``, ``LogViewer`` and ``ElectrodeGroupRow`` — thin wrappers
around standard Qt widgets that reduce boilerplate in the main application
window.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

_SELECTOR_FILETYPES = [("All files", "*")]


def _apply_filetypes(dialog_cb: Callable[[str], str], filetypes: list[tuple[str, str]]) -> str:
    """Turn (description, globs) filetypes into a Qt filter string."""
    parts = []
    for description, pattern in filetypes:
        parts.append(f"{description} ({pattern})")
    return dialog_cb(";;".join(parts))


class FileSelector(QFrame):
    """A label + line edit + *Browse* button row for selecting a file."""

    textChanged = Signal(str)  # noqa: N815

    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "",
        filetypes: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self._filetypes = filetypes or _SELECTOR_FILETYPES

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        if label:
            self._label = QLabel(label)
            self._label.setFixedWidth(100)
            layout.addWidget(self._label)

        self._edit = QLineEdit()
        self._edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._edit.textChanged.connect(self.textChanged)
        layout.addWidget(self._edit, 1)

        self._btn = QPushButton("Browse")
        self._btn.setFixedWidth(80)
        self._btn.clicked.connect(self._browse)
        layout.addWidget(self._btn)

    def _browse(self) -> None:
        path = _apply_filetypes(
            lambda flt: QFileDialog.getOpenFileName(self, "Select file", "", flt)[0],
            self._filetypes,
        )
        if path:
            self._edit.setText(path)

    def get(self) -> str:
        """Return the current path (empty string if unset)."""
        return self._edit.text()

    def set(self, value: str) -> None:
        self._edit.setText(value)


class DirectorySelector(QFrame):
    """A label + line edit + *Browse* button row for selecting a directory."""

    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "",
    ) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        if label:
            self._label = QLabel(label)
            self._label.setFixedWidth(100)
            layout.addWidget(self._label)

        self._edit = QLineEdit()
        self._edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._edit, 1)

        self._btn = QPushButton("Browse")
        self._btn.setFixedWidth(80)
        self._btn.clicked.connect(self._browse)
        layout.addWidget(self._btn)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select directory")
        if path:
            self._edit.setText(path)

    def get(self) -> str:
        return self._edit.text()

    def set(self, value: str) -> None:
        self._edit.setText(value)


class CollapsibleSection(QFrame):
    """A titled frame that can be expanded / collapsed by clicking its header.

    Child widgets added via the *body* frame are shown or hidden on toggle.
    """

    def __init__(self, parent: QWidget | None = None, title: str = "") -> None:
        super().__init__(parent)
        self._expanded = True

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)

        self._header = QToolButton()
        self._header.setText(title)
        self._header.setCheckable(True)
        self._header.setChecked(True)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.ArrowType.DownArrow)
        self._header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._header.clicked.connect(self._toggle)
        self._outer.addWidget(self._header)

        self._body = QWidget()
        self._outer.addWidget(self._body)

    @property
    def body(self) -> QWidget:
        """The inner widget where callers place child widgets."""
        return self._body

    def _toggle(self, checked: bool) -> None:
        if checked:
            self._body.show()
            self._header.setArrowType(Qt.ArrowType.DownArrow)
            self._expanded = True
        else:
            self._body.hide()
            self._header.setArrowType(Qt.ArrowType.RightArrow)
            self._expanded = False

    def collapse(self) -> None:
        if self._expanded:
            self._header.setChecked(False)
            self._toggle(False)

    def expand(self) -> None:
        if not self._expanded:
            self._header.setChecked(True)
            self._toggle(True)


class LabeledField(QFrame):
    """A compact label + widget row (entry / combo / check)."""

    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "",
        widget_type: Literal["entry", "combo", "check"] = "entry",
        values: list[str] | None = None,
        default: str | None = None,
    ) -> None:
        super().__init__(parent)

        self._widget_type = widget_type

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._label = QLabel(label)
        self._label.setFixedWidth(160)
        layout.addWidget(self._label)

        if widget_type == "combo":
            self._combo = QComboBox()
            self._combo.setEditable(False)
            if values:
                self._combo.addItems(values)
            if default and default in values:
                self._combo.setCurrentText(default)
            layout.addWidget(self._combo, 1)
        elif widget_type == "check":
            self._check = QCheckBox()
            self._check.setChecked(default == "true")
            layout.addWidget(self._check, 1)
        else:
            self._entry = QLineEdit()
            if default:
                self._entry.setText(default)
            layout.addWidget(self._entry, 1)

    def get(self) -> str:
        if self._widget_type == "combo":
            return self._combo.currentText()
        if self._widget_type == "check":
            return "true" if self._check.isChecked() else "false"
        return self._entry.text()

    def set(self, value: str) -> None:
        if self._widget_type == "combo":
            self._combo.setCurrentText(value)
        elif self._widget_type == "check":
            self._check.setChecked(value == "true")
        else:
            self._entry.setText(value)


class LogViewer(QFrame):
    """A read-only multi-line text area for pipeline log output."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(10000)
        layout.addWidget(self._text)

    def append(self, message: str) -> None:
        """Append a timestamped line to the log."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._text.appendPlainText(f"[{ts}] {message}")

    def clear(self) -> None:
        self._text.clear()


class ElectrodeGroupRow(QFrame):
    """One electrode overlay group row: file selector + color picker + remove.

    Used by the main window to let the user overlay several electrode files
    (Stage 3 ``electrodes.json`` or tabular TSV/CSV tables) in the 3D viewer,
    each with its own color.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        on_remove: Callable[[], None] | None = None,
        color: str = "yellow",
    ) -> None:
        super().__init__(parent)
        self._color = color

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._selector = FileSelector(
            self,
            label="",
            filetypes=[("Electrodes", "*.json *.tsv *.csv *.txt"), ("All files", "*")],
        )
        layout.addWidget(self._selector, 1)

        self._swatch = QPushButton()
        self._swatch.setFixedWidth(40)
        self._swatch.setFixedHeight(24)
        self._update_swatch()
        self._swatch.clicked.connect(self._pick_color)
        layout.addWidget(self._swatch)

        self._remove_btn = QPushButton("✕")
        self._remove_btn.setFixedWidth(28)
        self._remove_btn.clicked.connect(lambda: on_remove() if on_remove else None)
        layout.addWidget(self._remove_btn)

    def _update_swatch(self) -> None:
        color = QColor(self._color)
        self._swatch.setStyleSheet(f"background-color: {color.name()};")

    def _pick_color(self) -> None:
        initial = QColor(self._color)
        color = QColorDialog.getColor(initial, self, "Pick electrode group color")
        if color.isValid():
            self.set_color(color.name())

    def get(self) -> str:
        """Return the selected electrode file path (empty string if unset)."""
        return self._selector.get()

    def get_color(self) -> str:
        """Return the current group color (name or #rrggbb)."""
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color
        self._update_swatch()

    def set(self, value: str) -> None:
        self._selector.set(value)
