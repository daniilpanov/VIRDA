"""Configuration tab: input files, electrode groups and run controls.

The tab builds a :class:`Config` from the visible fields on demand, streams
progress into its own :class:`LogViewer`, and asks the host application to
act through the ``runRequested``/``openViewer``/``exportHtml`` signals.
"""

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from virda.config import load_config_file
from virda.io.fiducial_helpers import load_fiducials
from virda.models.config import Config
from virda.models.coordsystem import Coordsystem
from virda_gui.constants import (
    CONFIG_KEY_TO_ADVANCED,
    CONFIG_KEY_TO_INPUT,
    DEFAULT_PIPELINE_CONFIG_FILENAME,
    ELECTRODE_PALETTE,
)
from virda_gui.dialogs.advanced_settings import AdvancedSettingsDialog
from virda_gui.state import AppState
from virda_gui.widgets import (
    DirectorySelector,
    ElectrodeGroupRow,
    FileSelector,
    LabeledField,
    LogViewer,
)

_MESH_DENSITY_MIN = 1
_MESH_DENSITY_MAX = 100
_MESH_DENSITY_DEFAULT = 100


def density_percent_from_state(advanced: dict[str, str]) -> int:
    """Return the clamped slider percent for the advanced density value.

    The string ``AppState.advanced["mesh_density_percent"]`` is the single
    source of truth for the density slider: this helper parses it (as a
    number) and clamps the result to ``[1, 100]`` so the widget always shows
    a valid percentage, falling back to the default on unparseable input.
    """
    try:
        value = int(round(float(advanced.get("mesh_density_percent", ""))))
    except (ValueError, OverflowError):
        value = _MESH_DENSITY_DEFAULT
    return max(_MESH_DENSITY_MIN, min(_MESH_DENSITY_MAX, value))


def serialize_config_for_save(
    config: Config, advanced: dict[str, str], measurements_path: str | None = None
) -> dict[str, Any]:
    """Serialize a pipeline ``Config`` into the ``pipeline_config.json`` schema.

    The saved file is a flat JSON object whose keys are the ``Config`` field
    names (snake_case, e.g. ``mesh_density_percent``, ``seal_radius``,
    ``smoother_lamb``), produced by ``Config.model_dump(exclude_none=True)``,
    merged with a nested ``"advanced"`` object holding the
    ``AppState.advanced`` string values.  The flat keys are exactly the keys
    :func:`virda.config.build_config` / :func:`virda.config.load_config_file`
    read, so the file round-trips into a ``Config`` with identical field
    values; ``"advanced"`` preserves the GUI-only string settings (such as an
    empty ``mesh_voxel_size_mm`` meaning native NIfTI spacing).  The parsed
    ``coordsystem`` is deliberately not stored: it is a nested model derived
    from the project's ``coordsystem.json`` rather than a flat pipeline field,
    and re-validating it from this file would be lossy.
    """
    data = config.model_dump(exclude_none=True)
    data.pop("coordsystem", None)
    if measurements_path:
        data["measurements_path"] = measurements_path
    data["advanced"] = dict(advanced)
    return data


def write_pipeline_config(path: Path, data: dict[str, Any]) -> None:
    """Write one ``pipeline_config.json`` file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class ConfigTab(QWidget):
    """The "Configuration" tab of the main window."""

    runRequested = Signal()  # noqa: N815
    openViewer = Signal()  # noqa: N815
    exportHtml = Signal()  # noqa: N815

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state

        self._config_file = FileSelector(
            self,
            label="Config file",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
        self._nifti = FileSelector(
            self,
            label="NIfTI scan",
            filetypes=[("NIfTI", "*.nii.gz *.nii"), ("All files", "*")],
        )
        self._project_dir = DirectorySelector(self, label="Project dir")
        self._fiducials = FileSelector(
            self,
            label="Fiducials",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
        self.measurements = FileSelector(
            self,
            label="Measurements",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
        self._auto_detect_fid = LabeledField(
            self,
            label="Auto detect fiducials",
            widget_type="check",
            default="false",
        )

        self.density_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.density_slider.setRange(_MESH_DENSITY_MIN, _MESH_DENSITY_MAX)
        self.density_slider.setValue(_MESH_DENSITY_DEFAULT)
        self.density_value_label = QLabel(self)

        self.run_btn = QPushButton("Run Pipeline")
        self.viewer_btn = QPushButton("Open 3D Viewer")
        self.viewer_btn.setEnabled(False)
        self.export_btn = QPushButton("Export HTML")
        self.export_btn.setEnabled(False)

        self.save_config_btn = QPushButton("Save Config")
        self.save_config_as_btn = QPushButton("Save As...")

        self.electrodes_cras_check = QCheckBox("Force cRAS conversion")

        self.log_viewer = LogViewer(self)

        self._groups_inner = QWidget(self)
        self._groups_layout = QVBoxLayout(self._groups_inner)

        self._syncing_density = False

        self._build_ui()

        self._config_file.textChanged.connect(self._on_config_file_changed)
        self._fiducials.textChanged.connect(self._on_fiducials_path_changed)
        self.density_slider.valueChanged.connect(self._on_density_changed)
        self.run_btn.clicked.connect(self.runRequested)
        self.viewer_btn.clicked.connect(self.openViewer)
        self.export_btn.clicked.connect(self.exportHtml)
        self.save_config_btn.clicked.connect(self._on_save_pipeline_config)
        self.save_config_as_btn.clicked.connect(self._on_save_pipeline_config_as)
        self._project_dir.directory_changed.connect(self._sync_save_buttons)

        self._sync_density_from_state()
        self._sync_save_buttons()

    # ---- UI construction ----

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)

        input_box = QGroupBox("Input Files", self)
        input_layout = QVBoxLayout(input_box)
        input_layout.addWidget(self._config_file)
        input_layout.addWidget(self._nifti)
        input_layout.addWidget(self._project_dir)
        input_layout.addWidget(self._fiducials)
        input_layout.addWidget(self.measurements)
        input_layout.addWidget(self._auto_detect_fid)
        outer.addWidget(input_box)

        density_box = QGroupBox("Mesh Density", self)
        density_layout = QHBoxLayout(density_box)
        density_label = QLabel("Mesh density:", density_box)
        density_label.setFixedWidth(100)
        density_layout.addWidget(density_label)
        density_layout.addWidget(self.density_slider, 1)
        self.density_value_label.setFixedWidth(48)
        self.density_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        density_layout.addWidget(self.density_value_label)
        outer.addWidget(density_box)

        groups_box = QGroupBox("Electrode Groups (viewer overlays)", self)
        groups_layout = QVBoxLayout(groups_box)
        self._groups_layout.setContentsMargins(0, 0, 0, 0)
        self._groups_layout.setSpacing(4)
        groups_layout.addWidget(self._groups_inner)

        groups_btns = QFrame(groups_box)
        groups_btns_layout = QHBoxLayout(groups_btns)
        groups_btns_layout.setContentsMargins(0, 0, 0, 0)

        add_group_btn = QPushButton("Add group")
        add_group_btn.clicked.connect(self._on_add_electrode_group)
        groups_btns_layout.addWidget(add_group_btn)

        groups_btns_layout.addWidget(self.electrodes_cras_check)

        groups_btns_layout.addStretch(1)
        groups_layout.addWidget(groups_btns)

        outer.addWidget(groups_box)

        btn_frame = QFrame(self)
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 4, 0, 4)

        advanced_btn = QPushButton("Advanced Settings")
        advanced_btn.clicked.connect(self._on_show_advanced)
        btn_layout.addWidget(advanced_btn)
        btn_layout.addSpacing(8)

        btn_layout.addWidget(self.run_btn)
        btn_layout.addSpacing(8)

        btn_layout.addWidget(self.viewer_btn)
        btn_layout.addSpacing(8)

        btn_layout.addWidget(self.export_btn)
        btn_layout.addSpacing(8)

        btn_layout.addWidget(self.save_config_btn)
        btn_layout.addSpacing(8)

        btn_layout.addWidget(self.save_config_as_btn)

        btn_layout.addStretch(1)
        outer.addWidget(btn_frame)

        outer.addWidget(self.log_viewer, 1)

    # ---- Config file handling ----

    def _on_config_file_changed(self, _text: str) -> None:
        self._state.coordsystem = None
        path = self._config_file.get()
        if not path:
            return
        try:
            data = load_config_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Config error", f"Invalid config file:\n{exc}")
            self._config_file.set("")
            return
        self._populate_from_config(data)

    def _populate_from_config(self, data: dict[str, Any]) -> None:
        for config_key, attr_name in CONFIG_KEY_TO_INPUT.items():
            if config_key in data:
                widget = getattr(self, f"_{attr_name}", None)
                if widget is not None and not widget.get():
                    widget.set(str(data[config_key]))

        if data.get("measurements_path") and not self.measurements.get():
            self.measurements.set(str(data["measurements_path"]))

        for config_key, adv_key in CONFIG_KEY_TO_ADVANCED.items():
            if config_key in data and not self._state.advanced.get(adv_key):
                self._state.advanced[adv_key] = str(data[config_key])

        self._sync_density_from_state()

        # Keep the parsed MNE coordsystem (its fiducials feed Stage 1).
        coordsystem = data.get("coordsystem")
        if isinstance(coordsystem, Coordsystem):
            self._state.coordsystem = coordsystem
        else:
            if coordsystem is not None:
                self.log_viewer.append(
                    "WARNING: 'coordsystem' entry in the config file was not parsed "
                    "from a coordsystem.json file — its fiducials are ignored."
                )
            self._state.coordsystem = None

    def _on_fiducials_path_changed(self, _text: str) -> None:
        path = self._fiducials.get()
        if not path:
            return
        try:
            load_fiducials(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Fiducials error", f"Invalid fiducials file:\n{exc}")
            self._fiducials.set("")

    def _sync_save_buttons(self) -> None:
        """Enable the config save buttons only when a project dir is present."""
        enabled = bool(self.project_dir() or self._state.last_project_dir)
        self.save_config_btn.setEnabled(enabled)
        self.save_config_as_btn.setEnabled(enabled)

    # ---- Advanced settings ----

    def _on_density_changed(self, value: int) -> None:
        """Keep ``AppState.advanced["mesh_density_percent"]`` in sync.

        The advanced dict is the single source of truth for the density value;
        the slider is a convenience view that writes back to it so
        :meth:`collect_config` always sees the slider's value.  Programmatic
        updates from :meth:`_sync_density_from_state` skip the write-back so a
        persisted fractional value (e.g. ``"57.6"``) is not snapped to the
        integer slider position.
        """
        if not self._syncing_density:
            self._state.advanced["mesh_density_percent"] = str(value)
        self.density_value_label.setText(f"{value}%")

    def _sync_density_from_state(self) -> None:
        """Make the slider (and its label) reflect the advanced density value."""
        value = density_percent_from_state(self._state.advanced)
        self._syncing_density = True
        try:
            self.density_slider.setValue(value)
        finally:
            self._syncing_density = False
        self.density_value_label.setText(f"{value}%")

    def _on_show_advanced(self) -> None:
        dialog = AdvancedSettingsDialog(self, self._state.advanced, nifti_path=self.nifti_path())
        if dialog.exec():
            self._state.advanced = dialog.result_values
            self._sync_density_from_state()

    # ---- Pipeline config save ----

    def _default_config_save_path(self) -> str:
        """Default save location: the project's ``input/pipeline_config.json``."""
        project = self.project_dir() or self._state.last_project_dir
        if project:
            return str(Path(project) / "input" / DEFAULT_PIPELINE_CONFIG_FILENAME)
        return DEFAULT_PIPELINE_CONFIG_FILENAME

    def _on_save_pipeline_config(self) -> None:
        self._save_pipeline_config(Path(self._default_config_save_path()))

    def _on_save_pipeline_config_as(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save pipeline config as",
            self._default_config_save_path(),
            "JSON (*.json);;All files (*)",
        )
        if path:
            self._save_pipeline_config(Path(path))

    def _save_pipeline_config(self, path: Path) -> None:
        """Collect the current form state and write it to *path*.

        The saved file intentionally does not touch the config-file selector,
        so saving never reloads the freshly written file (which would drop a
        parsed ``coordsystem`` from ``AppState``).
        """
        try:
            config = self.collect_config()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Save config", f"Invalid configuration:\n{exc}")
            return
        try:
            write_pipeline_config(
                path,
                serialize_config_for_save(
                    config, self._state.advanced, measurements_path=self.measurements.get()
                ),
            )
        except OSError as exc:
            QMessageBox.critical(self, "Save config", f"Could not write file:\n{exc}")
            return
        self.log_viewer.append(f"Saved pipeline config to {path}")

    # ---- Electrode groups ----

    def _on_add_electrode_group(self, path: str = "", color: str | None = None) -> None:
        if color is None:
            color = ELECTRODE_PALETTE[self._state.palette_index % len(ELECTRODE_PALETTE)]
            self._state.palette_index += 1

        row = ElectrodeGroupRow(
            on_remove=lambda: self._on_remove_electrode_group(row),
            color=color,
        )
        if path:
            row.set(path)
        self._groups_layout.addWidget(row)
        self._state.electrode_rows.append(row)

    def _on_remove_electrode_group(self, row: ElectrodeGroupRow) -> None:
        if row in self._state.electrode_rows:
            self._state.electrode_rows.remove(row)
            self._groups_layout.removeWidget(row)
        row.deleteLater()

    def collect_electrode_specs(self) -> list[tuple[str, str]]:
        """Return (path, color) pairs of all non-empty electrode group rows."""
        specs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for row in self._state.electrode_rows:
            path = row.get().strip()
            if not path or path in seen:
                continue
            seen.add(path)
            specs.append((path, row.get_color()))
        return specs

    def ensure_stage3_electrodes_group(self, project_dir: str | Path | None = None) -> str | None:
        """Add the Stage 3 output as a group unless already present.

        Returns the added path or None when there is nothing to add.
        """
        resolved = project_dir or self._state.last_project_dir
        if not resolved:
            return None
        electrodes_path = Path(resolved) / "localization" / "electrodes.json"
        if not electrodes_path.is_file():
            return None
        path_str = str(electrodes_path)
        if any(row.get().strip() == path_str for row in self._state.electrode_rows):
            return None
        self._on_add_electrode_group(path=path_str)
        return path_str

    # ---- Config collection ----

    def nifti_path(self) -> str:
        return self._nifti.get()

    def project_dir(self) -> str:
        return self._project_dir.get().strip()

    def set_project_dir(self, project: str | Path) -> None:
        self._project_dir.set(str(project))

    def prefill_from_project(self, project: str | Path) -> None:
        """Fill the run fields from the project's canonical input artifacts.

        The nifti scan, fiducials, measurements and config JSON live under the
        project's ``input/`` directory; the config file is set last so its
        loader back-fills any still-empty fields from the saved config.
        """
        root = Path(project)
        self.set_project_dir(root)

        nifti = next(iter(sorted(root.glob("input/*.nii.gz"))), None)
        if nifti is None:
            nifti = next(iter(sorted(root.glob("input/*.nii"))), None)
        if nifti is not None:
            self._nifti.set(str(nifti))

        fiducials = root / "input" / "fiducials.json"
        if fiducials.is_file():
            self._fiducials.set(str(fiducials))

        measurements = root / "input" / "measurements.json"
        if measurements.is_file():
            self.measurements.set(str(measurements))

        config = next(
            (
                candidate
                for candidate in (
                    root / "input" / "pipeline_config.json",
                    root / "input" / "config.json",
                )
                if candidate.is_file()
            ),
            None,
        )
        if config is not None:
            self._config_file.set(str(config))

    def collect_config(self) -> Config:
        nifti = self._nifti.get() or None
        project = self._project_dir.get() or None
        fiducials = self._fiducials.get() or None

        if not nifti:
            raise ValueError("NIfTI scan path is required.")
        if not project:
            raise ValueError("Project directory is required.")

        adv = self._state.advanced

        def _int(val: str, default: int | None = None, *, key: str = "value") -> int | None:
            val = val.strip()
            if not val:
                return default
            try:
                return int(val)
            except ValueError:
                raise ValueError(f"{key}: expected an integer, got {val!r}") from None

        def _float(val: str, default: float | None = None, *, key: str = "value") -> float | None:
            val = val.strip()
            if not val:
                return default
            try:
                return float(val)
            except ValueError:
                raise ValueError(f"{key}: expected a number, got {val!r}") from None

        def _mesh_voxel_size(val: str) -> float | None:
            if not val.strip():
                return None
            size = _float(val, key="mesh_voxel_size_mm")
            if size is None:
                raise ValueError("mesh_voxel_size_mm: expected a number")
            if size <= 0:
                raise ValueError("mesh_voxel_size_mm: must be positive")
            return size

        def _mesh_density(val: str) -> float:
            density = _float(val, 100.0, key="mesh_density_percent")
            if density is None:
                raise ValueError("mesh_density_percent: expected a number")
            if not 1 <= density <= 100:
                raise ValueError("mesh_density_percent: must be within [1, 100]")
            return density

        return Config(
            nifti_path=nifti,
            project_dir=project,
            fiducials_path=fiducials or None,
            auto_detect_fiducials=self._auto_detect_fid.get() == "true",
            coordsystem=self._state.coordsystem,
            closing_radius=_int(adv["closing_radius"], 5, key="closing_radius"),
            otsu_scope=adv["otsu_scope"] or "all",  # type: ignore[arg-type]
            otsu_threshold_scale=_float(
                adv["otsu_threshold_scale"], 0.6, key="otsu_threshold_scale"
            ),
            seal_enabled=adv["seal_enabled"] == "true",
            seal_radius=_int(adv["seal_radius"], 4, key="seal_radius"),
            mesh_voxel_size_mm=_mesh_voxel_size(adv["mesh_voxel_size_mm"]),
            mesh_density_percent=_mesh_density(adv["mesh_density_percent"]),
            cleaner_min_vertices=_int(adv["cleaner_min_vertices"], 100, key="cleaner_min_vertices"),
            cleaner_merge_digits=_int(adv["cleaner_merge_digits"], 7, key="cleaner_merge_digits"),
            smoother_type=adv["smoother_type"] or "laplacian",
            smoother_iterations=_int(adv["smoother_iterations"], 5, key="smoother_iterations"),
            smoother_lamb=_float(adv["smoother_lamb"], 0.5, key="smoother_lamb"),
            smoother_nu=_float(adv["smoother_nu"], -0.53, key="smoother_nu"),
            ese_offset_mm=_float(adv["ese_offset_mm"], key="ese_offset_mm"),
            neighborhood_radius_mm=_float(
                adv["neighborhood_radius_mm"], 10.0, key="neighborhood_radius_mm"
            ),
            k_neighbors=_int(adv["k_neighbors"], key="k_neighbors"),
            use_weighted_pca=adv["use_weighted_pca"] == "true",
            pca_sigma_mm=_float(adv["pca_sigma_mm"], 5.0, key="pca_sigma_mm"),
            min_neighbors=_int(adv["min_neighbors"], 5, key="min_neighbors"),
            residual_threshold_mm=_float(
                adv["residual_threshold_mm"],  # type: ignore[arg-type]
                10.0,
                key="residual_threshold_mm",
            ),
            calibrate_ese_offset=adv["calibrate_ese_offset"] == "true",
        )
