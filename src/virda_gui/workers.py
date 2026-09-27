"""Shared background workers for the GUI tabs.

:class:`BackgroundWorker` runs a pure callable off the GUI thread; ``done``
and ``failed`` are delivered back to the main thread because the receiving
tab lives there.  Used by the mesh processing and ESE tabs.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, Signal


class BackgroundWorker(QObject):
    """Run a pure callable off the GUI thread."""

    done = Signal(int, object)  # noqa: N815 - seq, result
    failed = Signal(int, str)  # noqa: N815 - seq, error message

    def __init__(self, fn: Callable[[], object], seq: int) -> None:
        super().__init__()
        self._fn = fn
        self._seq = seq

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(self._seq, str(exc))
        else:
            self.done.emit(self._seq, result)
