"""Shared application state object.

``AppState`` is the single mutable state object of the GUI: instead of
scattered ad-hoc instance attributes on ``virda_gui.app.VirdaApp``, every
view/worker touches the same dataclass so ownership is explicit.
"""

import queue
from dataclasses import dataclass, field
from typing import Any

from virda.models.coordsystem import Coordsystem
from virda_gui.widgets import ElectrodeGroupRow


@dataclass
class AppState:
    """Mutable state shared across the GUI tabs and background workers.

    Owned by :class:`virda_gui.app.VirdaApp` and passed to any component that
    must read or update it (tabs, pipeline runner, preview worker).
    """

    log_queue: queue.Queue[str | None] = field(default_factory=queue.Queue)
    viewer_loading: bool = False
    last_project_dir: str | None = None
    advanced: dict[str, str] = field(default_factory=dict)
    electrode_rows: list[ElectrodeGroupRow] = field(default_factory=list)
    palette_index: int = 0
    stage3_summary: dict[str, Any] | None = None
    coordsystem: Coordsystem | None = None
    closed: bool = False
