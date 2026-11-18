"""Export popup widgets: File -> Export dialogs over the export service.

Each popup is a plain :class:`~PySide6.QtWidgets.QWidget`. The shared
base in :mod:`virda_gui_new.widgets.popup.export.base` owns the common
controls (frame selector, path picker, export button) and the
result-dialog flow; the subclasses in
:mod:`virda_gui_new.widgets.popup.export.popups` only add their kind
specific controls. All heavy work is delegated to
:mod:`virda_gui_new.services.artifact_exporting` -- the widgets hold no
business logic.
"""

from virda_gui_new.widgets.popup.export.base import ExportPopupBase
from virda_gui_new.widgets.popup.export.popups import (
    ElectrodesExportPopup,
    FiducialsExportPopup,
    MeasurementsExportPopup,
    MeshExportPopup,
)

__all__ = [
    "ElectrodesExportPopup",
    "ExportPopupBase",
    "FiducialsExportPopup",
    "MeasurementsExportPopup",
    "MeshExportPopup",
]
