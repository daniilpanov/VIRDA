"""The four concrete export popups of the File -> Export menu."""

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QFileDialog, QLabel, QVBoxLayout, QWidget

from virda.models.coordinates import CoordinateSystem
from virda_gui_new.services.artifact_exporting import (
    ExportKind,
    MeshExportKind,
    default_export_path,
)
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.export.base import ExportPopupBase

_MESH_ITEMS = (
    ("Base", MeshExportKind.BASE),
    ("Postprocessed", MeshExportKind.POSTPROCESSED),
    ("ESE", MeshExportKind.ESE),
)


class MeasurementsExportPopup(ExportPopupBase):
    """Export the stored measurements (no coordinates, no frame selector).

    Exports the measurements of the last successful localization run or
    the seeded measurements file (the global store ``measurements``): an
    edited-but-not-yet-run popup table is deliberately not exported.
    """

    _TITLE = "Export measurements"
    _HINT = (
        "Exports the measurements of the last successful localization run "
        "(or the seeded measurements file). Edits not yet run stay out."
    )
    _has_frames = False

    def __init__(
        self,
        manager: StepsManager,
        parent: QWidget | None = None,
        *,
        project: Path | None = None,
    ) -> None:
        super().__init__(manager, ExportKind.MEASUREMENTS, parent, project=project)


class ElectrodesExportPopup(ExportPopupBase):
    """Export the full electrode table in the chosen frame.

    The export is the full pipeline CSV (ids, ESE/scalp coordinates,
    measured distances, residual, flag) of the last successful
    localization run; the point CSVs written by the localization step
    itself are not the export target.
    """

    _TITLE = "Export electrodes"
    _HINT = (
        "Exports the full electrode table (ids, ESE and scalp coordinates, "
        "measured distances, residual, flag) of the last successful "
        "localization run, in the chosen frame."
    )

    def __init__(
        self,
        manager: StepsManager,
        parent: QWidget | None = None,
        *,
        project: Path | None = None,
    ) -> None:
        super().__init__(manager, ExportKind.ELECTRODES, parent, project=project)


class FiducialsExportPopup(ExportPopupBase):
    """Export the global-store fiducials in the world, head or voxel frame.

    The ``fiducials`` frame is permanently greyed out: fiducials define
    that frame, so they are only offered relative to another frame.
    """

    _TITLE = "Export fiducials"
    _HINT = (
        "Exports the current fiducials from the localization popup store "
        "in the chosen frame (world, head or voxel)."
    )

    def __init__(
        self,
        manager: StepsManager,
        parent: QWidget | None = None,
        *,
        project: Path | None = None,
    ) -> None:
        super().__init__(manager, ExportKind.FIDUCIALS, parent, project=project)

    def _offered_frames(self) -> frozenset[CoordinateSystem]:
        return frozenset({CoordinateSystem.WORLD, CoordinateSystem.HEAD, CoordinateSystem.VOXEL})


class MeshExportPopup(ExportPopupBase):
    """Export one mesh source in the chosen frame.

    The mesh type combo picks the source (base = step-1 scalp surface,
    postprocessed = step-2 mesh, ese = step-3 ESE mesh); switching it
    resets the path field to the new source's canonical default path,
    because source and default path change together. PLY carries no frame
    metadata, so the voxel frame is not offered for meshes.
    """

    _TITLE = "Export mesh"
    _HINT = (
        "Exports the chosen mesh in the chosen coordinate frame. "
        "Switching the mesh type resets the path to its default."
    )

    def __init__(
        self,
        manager: StepsManager,
        parent: QWidget | None = None,
        *,
        project: Path | None = None,
    ) -> None:
        super().__init__(manager, ExportKind.MESH, parent, project=project)

    def build_form(self, layout: QVBoxLayout) -> None:
        """Add the mesh type combo that drives source and default path."""
        self._mesh_combo = QComboBox(self)
        for label, kind in _MESH_ITEMS:
            self._mesh_combo.addItem(label, kind)
        layout.addWidget(QLabel("Mesh type", self))
        layout.addWidget(self._mesh_combo)
        self._mesh_combo.currentIndexChanged.connect(self._on_mesh_kind_changed)

    def _offered_frames(self) -> frozenset[CoordinateSystem]:
        return frozenset(
            {CoordinateSystem.WORLD, CoordinateSystem.HEAD, CoordinateSystem.FIDUCIALS}
        )

    def _on_mesh_kind_changed(self, _index: int) -> None:
        """Follow the new mesh source with its canonical default path."""
        self._apply_default_path()


class MeshArraysExportPopup(ExportPopupBase):
    """Export mesh vertices and faces as NumPy arrays in a chosen frame."""

    _TITLE = "Export mesh arrays"
    _HINT = (
        "Exports the chosen mesh as vertices and faces NumPy arrays in the selected "
        "coordinate frame."
    )

    def __init__(
        self,
        manager: StepsManager,
        parent: QWidget | None = None,
        *,
        project: Path | None = None,
    ) -> None:
        super().__init__(manager, ExportKind.MESH_ARRAYS, parent, project=project)

    def build_form(self, layout: QVBoxLayout) -> None:
        self._mesh_combo = QComboBox(self)
        for label, kind in _MESH_ITEMS:
            self._mesh_combo.addItem(label, kind)
        layout.addWidget(QLabel("Mesh type", self))
        layout.addWidget(self._mesh_combo)
        self._mesh_combo.currentIndexChanged.connect(self._on_mesh_kind_changed)

    def _offered_frames(self) -> frozenset[CoordinateSystem]:
        return frozenset(CoordinateSystem)

    def _path_label_text(self) -> str:
        return "Directory"

    def _apply_default_path(self) -> None:
        """Use the canonical scalp or ESE folder when a project is open."""
        default = default_export_path(self._project, self._kind, self._mesh_kind())
        self._path_edit.setText("" if default is None else str(default))

    def _browse(self) -> None:
        """Choose the directory that will receive the two NPY files."""
        start = self._path_edit.text().strip()
        path = QFileDialog.getExistingDirectory(self, "Export to", start)
        if path:
            self._path_edit.setText(path)

    def _on_mesh_kind_changed(self, _index: int) -> None:
        self._apply_default_path()
