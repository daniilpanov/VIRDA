"""Steps navigator widget: one clickable row per pipeline step.

The navigator reflects the live state of a :class:`StepsManager` and runs
step 0 (NIfTI import) itself; later steps only emit
``stepRequested`` for a future popup dispatcher to handle.
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from virda_gui_new.dialogs.warnings import confirm_step_redo, show_warning
from virda_gui_new.dto.steps import STEPS
from virda_gui_new.services import artifact_loading
from virda_gui_new.services.steps_manager import StepsManager

_DONE = " [done]"
_CURRENT = " [current]"
_NIFTI_TITLE = "Import NIfTI..."
_NIFTI_FILTER = "NIfTI files (*.nii.gz *.nii)"


class StepsMenu(QWidget):
    """Clickable list of the pipeline steps with statuses pulled on refresh.

    Rows read ``"{index}. {name}"`` plus ``[done]`` for an earlier completed
    step or ``[current]`` for the working step; later steps are pending.
    A row is clickable only when its step exists in the manager. Clicking
    an already-satisfied step asks for a redo confirmation (only when later
    step results would be discarded) and, on agree, discards them. Step 0
    re-imports a NIfTI in place; a fresh step 0 import auto-advances to
    step 1 and emits ``stepRequested``.
    """

    stepRequested = Signal(int)  # noqa: N815
    stepsInvalidated = Signal(int)  # noqa: N815

    def __init__(self, manager: StepsManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._items: tuple[QListWidgetItem, ...]

        self._list = QListWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)

        items: list[QListWidgetItem] = []
        for step in STEPS:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, step.index)
            self._list.addItem(item)
            items.append(item)
        self._items = tuple(items)
        self._list.itemClicked.connect(self._on_item_clicked)

        manager.subscribe(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """Re-read statuses and enablement for every row from the manager."""
        current = min(self._manager.get_current_step(), len(STEPS) - 1)
        for index, step in enumerate(STEPS):
            item = self._items[index]
            suffix = ""
            if index < current and self._manager.is_step_done(index):
                suffix = _DONE
            elif index == current:
                suffix = _CURRENT
            flags = item.flags()
            if index <= current:
                item.setText(f"{index}. {step.name}{suffix}")
                item.setFlags(flags | Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setText(f"{index}. {step.name}")
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsEnabled)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(index, int):
            self._on_step_clicked(index)

    def _on_step_clicked(self, step_n: int) -> None:
        current = self._manager.get_current_step()
        if step_n > current:  # disabled row; guarded for programmatic clicks
            return
        if step_n < current or self._manager.is_step_done(step_n):
            last = min(current + 1, len(STEPS))
            discarded = [STEPS[later].name for later in range(step_n + 1, last)]
            if discarded and not confirm_step_redo(self, STEPS[step_n].name, discarded):
                return
            if discarded:
                self._manager.invalidate_steps_after(step_n)
                self.stepsInvalidated.emit(step_n)
        if step_n == 0:
            self._import_volume()
        else:
            self.stepRequested.emit(step_n)

    def _import_volume(self) -> None:
        """Run the step-0 flow: pick a NIfTI, store it, auto-advance."""
        path, _ = QFileDialog.getOpenFileName(self, _NIFTI_TITLE, "", _NIFTI_FILTER)
        if not path:
            return
        try:
            volume = artifact_loading.load_volume(Path(path))
        except Exception as error:  # any parse/validation failure is user-facing
            show_warning(self, _NIFTI_TITLE, str(error))
            return
        self._manager.set_step_context_item(0, "volume", volume)
        if self._manager.get_current_step() == 0:
            self._manager.next_step()
            self.stepRequested.emit(1)
