"""Shared application state object.

``AppState`` is the single mutable state object of the GUI: instead of
scattered ad-hoc instance attributes on ``virda_gui.main_window.IdeWindow``,
every view/worker touches the same dataclass so ownership is explicit.
"""

from dataclasses import dataclass, field


@dataclass
class AppState:
    """Mutable state shared across the GUI tabs and background workers.

    Owned by :class:`virda_gui.main_window.IdeWindow` and passed to any
    component that must read or update it (tabs, preview worker).
    """

    viewer_loading: bool = False
    last_project_dir: str | None = None
    advanced: dict[str, str] = field(default_factory=dict)
    electrode_rows: list[tuple[str, str]] = field(default_factory=list)
    palette_index: int = 0
    electrodes_cras: bool = False
    closed: bool = False