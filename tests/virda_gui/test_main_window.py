"""Unit tests for :class:`virda_gui_new.windows.main_window` glue logic.

Runs under the offscreen Qt platform. The step/viewer machinery is stubbed
like in :mod:`tests.virda_gui.test_step_popup_base`, so only the wiring of
the dispatcher-adjacent pieces is exercised.
"""

import os
from pathlib import Path

import pytest

from virda_gui_new.widgets.filetree import FileTree
from virda_gui_new.windows.main_window import MainWindow


class _Host:
    """Duck-typed stand-in carrying only the file-tree glue state."""

    def __init__(self) -> None:
        self._project_path: Path | None = None
        self._filetree: FileTree | None = FileTree()
        self.logs: list[str] = []

    def _log(self, message: str) -> None:
        self.logs.append(message)

    _forget_filetree = MainWindow._forget_filetree
    _refresh_filetree = MainWindow._refresh_filetree
    _on_steps_invalidated = MainWindow._on_steps_invalidated


@pytest.fixture
def qapp():
    """Yield an offscreen ``QApplication``, skipping without a Qt install."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    return app


def test_steps_invalidated_refreshes_file_tree(qapp, tmp_path: Path) -> None:
    host = _Host()
    host._project_path = tmp_path
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.nii").write_text("")

    MainWindow._on_steps_invalidated(host, 1)

    tree = host._filetree
    assert tree is not None
    assert tree._list.count() == 1  # noqa: SLF001 - row check on the stub
    assert host.logs == ["Steps after 1 invalidated."]


def test_steps_invalidated_after_filetree_destroyed_is_guarded(qapp) -> None:
    """A closed file tree window leaves a dropped reference, not a crash."""
    host = _Host()
    host._forget_filetree()

    MainWindow._on_steps_invalidated(host, 1)

    assert host.logs == ["Steps after 1 invalidated."]


def test_refresh_filetree_rescans_the_project(qapp, tmp_path: Path) -> None:
    """A fresh scan picks up artifacts written since the last refresh."""
    host = _Host()
    host._project_path = tmp_path
    (tmp_path / "segmentation").mkdir()
    (tmp_path / "segmentation" / "head_mask.nii.gz").write_text("")

    MainWindow._refresh_filetree(host)

    tree = host._filetree
    assert tree is not None
    assert [tree._list.item(i).text() for i in range(tree._list.count())] == [
        "segmentation/head_mask.nii.gz"
    ]


def test_forget_filetree_wiring(qapp) -> None:
    """The window ``destroyed`` signal drops the host's file tree."""
    from virda_gui_new.windows.widget_windows import open_widget_window

    host = _Host()
    assert host._filetree is not None
    window = open_widget_window(host._filetree, title="File tree")
    window.destroyed.connect(host._forget_filetree)

    window.close()
    qapp.processEvents()

    assert host._filetree is None
