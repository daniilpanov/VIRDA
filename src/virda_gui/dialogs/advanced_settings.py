"""Advanced pipeline parameters dialog.

Holds the per-field default values, the config-key mapping and the modal
:class:`AdvancedSettingsDialog` that lets the user override the pipeline
parameters that are not part of the main Configuration tab.
"""

from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from virda_gui.constants import (
    ADVANCED_COMBO_FIELDS,
    ADVANCED_FIELD_DEFAULTS,
)
from virda_gui.widgets import LabeledField


class AdvancedSettingsDialog(QDialog):
    """Modal dialog for advanced pipeline parameters."""

    def __init__(
        self,
        parent: QWidget | None,
        values: dict[str, str],
    ) -> None:
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

        self._build_segmentation_section(container, layout)
        self._build_mesh_section(container, layout)
        self._build_ese_section(container, layout)
        self._build_neighborhood_section(container, layout)
        self._build_stage3_section(container, layout)

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

    def _build_segmentation_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Stage 1: Segmentation", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "otsu_scope", "Otsu scope", "combo")
        self._add_field(box, layout, "otsu_threshold_scale", "Threshold scale", "entry")
        self._add_field(box, layout, "closing_radius", "Closing radius", "entry")
        outer.addWidget(box)

    def _build_mesh_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Stage 1: Mesh Processing", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "seal_enabled", "Seal mask gaps", "check")
        self._add_field(box, layout, "seal_radius", "Seal radius", "entry")
        self._add_field(box, layout, "cleaner_min_vertices", "Min component vertices", "entry")
        self._add_field(box, layout, "cleaner_merge_digits", "Merge digits", "entry")
        self._add_field(box, layout, "smoother_type", "Smoother type", "combo")
        self._add_field(box, layout, "smoother_iterations", "Iterations", "entry")
        self._add_field(box, layout, "smoother_lamb", "Lambda", "entry")
        self._add_field(box, layout, "smoother_nu", "Nu (Taubin)", "entry")
        outer.addWidget(box)

    def _build_ese_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Stage 2: ESE Parameters", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "ese_offset_mm", "Offset (mm)", "entry")
        outer.addWidget(box)

    def _build_neighborhood_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Stage 2: Neighborhood", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "neighborhood_radius_mm", "Radius (mm)", "entry")
        self._add_field(box, layout, "k_neighbors", "K neighbors", "entry")
        self._add_field(box, layout, "pca_sigma_mm", "PCA sigma (mm)", "entry")
        self._add_field(box, layout, "min_neighbors", "Min neighbors", "entry")
        self._add_field(box, layout, "use_weighted_pca", "Weighted PCA", "check")
        outer.addWidget(box)

    def _build_stage3_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Stage 3: Localization", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "residual_threshold_mm", "Residual threshold (mm)", "entry")
        self._add_field(box, layout, "calibrate_ese_offset", "Calibrate ESE offset", "check")
        outer.addWidget(box)

    def _add_field(
        self, parent: QWidget, layout: QVBoxLayout, key: str, label: str, widget_type: str
    ) -> None:
        values = ADVANCED_COMBO_FIELDS.get(key)
        field = LabeledField(
            parent,
            label=label,
            widget_type=widget_type,  # type: ignore[arg-type]
            values=values,
            default=self.result_values.get(key, ADVANCED_FIELD_DEFAULTS.get(key, "")),
        )
        layout.addWidget(field)
        self._fields[key] = field

    def _on_ok(self) -> None:
        for key, field in self._fields.items():
            self.result_values[key] = field.get()
        self.confirmed = True
        self.accept()
