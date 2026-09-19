"""Advanced GUI settings dialog.

Holds the :class:`AdvancedSettingsDialog` that lets the user override the
GUI-only mesh-generation and localization parameters.  Everything is
controlled in the interface and never persisted to a pipeline config file.
"""

from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from virda_gui.constants import ADVANCED_FIELD_DEFAULTS
from virda_gui.widgets import LabeledField


class AdvancedSettingsDialog(QDialog):
    """Modal dialog for advanced mesh-generation and localization settings."""

    def __init__(self, parent: QWidget | None, values: dict[str, str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setModal(True)

        self.result_values: dict[str, str] = dict(values)
        self.confirmed: bool = False

        self._fields: dict[str, LabeledField] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._build_mesh_generation_section(container, layout)
        self._build_localization_section(container, layout)

        btn_frame = QFrame(container)
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 8, 0, 0)

        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(100)
        ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addWidget(btn_frame)
        scroll.setWidget(container)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _build_mesh_generation_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Mesh generation", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "seal_enabled", "Seal mask gaps", "check")
        self._add_field(box, layout, "seal_radius", "Seal radius", "entry")
        self._add_field(box, layout, "cleaner_min_vertices", "Min component vertices", "entry")
        self._add_field(box, layout, "cleaner_merge_digits", "Merge digits", "entry")
        outer.addWidget(box)

    def _build_localization_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Localization", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "residual_threshold_mm", "Residual threshold (mm)", "entry")
        self._add_field(box, layout, "calibrate_ese_offset", "Calibrate ESE offset", "check")
        outer.addWidget(box)

    def _add_field(
        self,
        parent: QWidget,
        layout: QVBoxLayout,
        key: str,
        label: str,
        widget_type: str,
    ) -> None:
        field = LabeledField(
            parent,
            label=label,
            widget_type=widget_type,  # type: ignore[arg-type]
            default=self.result_values.get(key, ADVANCED_FIELD_DEFAULTS.get(key, "")),
        )
        layout.addWidget(field)
        self._fields[key] = field

    def _on_ok(self) -> None:
        for key, field in self._fields.items():
            self.result_values[key] = field.get()
        self.confirmed = True
        self.accept()