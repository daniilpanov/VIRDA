"""Shared collapsible container for less commonly used step settings."""

from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QWidget


class AdvancedSettings(QGroupBox):
    """A checkable group box whose contents collapse when unchecked."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Advanced settings", parent)
        self.setObjectName("advanced_settings")
        self.setCheckable(True)
        self.setChecked(False)

        self._content = QWidget(self)
        self._content.hide()
        self.content_layout = QVBoxLayout(self._content)
        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(self._content)
        self.toggled.connect(self._content.setVisible)
