"""Live localization popup widgets."""

from virda_gui_new.widgets.popup.localization.abstract import LocalizationPopupBase
from virda_gui_new.widgets.popup.localization.electrodes_table import ElectrodesTable
from virda_gui_new.widgets.popup.localization.fiducials_table import FiducialsTable
from virda_gui_new.widgets.popup.localization.measurements_table import MeasurementsTable
from virda_gui_new.widgets.popup.localization.popup import LocalizationPopup

__all__ = [
    "ElectrodesTable",
    "FiducialsTable",
    "LocalizationPopup",
    "LocalizationPopupBase",
    "MeasurementsTable",
]
