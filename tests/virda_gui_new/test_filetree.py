"""Unit tests for :class:`virda_gui_new.widgets.filetree.FileTree`.

Runs under the offscreen Qt platform; the project files are fed
directly, so no scanning or project layout is involved. Real temp files
back only the double-click activation check.
"""

import os
from pathlib import Path

import pytest

from virda_gui_new.widgets.filetree import FileTree


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


@pytest.fixture
def tree(qapp):
    from virda_gui_new.widgets.filetree import FileTree

    widget = FileTree()
    yield widget
    widget.deleteLater()


def _row_texts(widget: FileTree) -> list[str]:
    return [widget._list.item(i).text() for i in range(widget._list.count())]


def _project(tmp_path: Path, names: tuple[str, ...]) -> list[Path]:
    """Create the given project files and return them."""
    files = []
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        files.append(path)
    return files


def test_no_files_show_no_rows(qapp, tree) -> None:
    tree.set_files([])

    assert _row_texts(tree) == []


def test_the_rows_show_the_path_inside_the_project(qapp, tree, tmp_path) -> None:
    files = _project(
        tmp_path,
        (
            "input/head.nii.gz",
            "input/fiducials.json",
            "mesh/final_mesh.ply",
            "ese/ese_mesh.ply",
            "localization/electrodes_scalp.csv",
        ),
    )

    tree.set_files(files, tmp_path)

    assert _row_texts(tree) == [
        "ese/ese_mesh.ply",
        "input/fiducials.json",
        "input/head.nii.gz",
        "localization/electrodes_scalp.csv",
        "mesh/final_mesh.ply",
    ]


def test_every_project_file_is_listed(qapp, tree, tmp_path) -> None:
    """The tree browses the project, not only the known artifacts."""
    from virda_gui_new.services.project_scan import project_files

    _project(tmp_path, ("input/head.nii.gz", "notes.txt", "config/base_mesh.json"))

    tree.set_files(project_files(tmp_path), tmp_path)

    assert _row_texts(tree) == [
        "config/base_mesh.json",
        "input/head.nii.gz",
        "notes.txt",
    ]


def test_double_click_on_existing_file_emits_path(qapp, tree, tmp_path) -> None:
    nifti = tmp_path / "head.nii.gz"
    nifti.touch()
    tree.set_files([nifti], tmp_path)
    activated: list[Path] = []
    tree.fileActivated.connect(activated.append)

    tree._list.itemDoubleClicked.emit(tree._list.item(0))

    assert activated == [nifti]


def test_deleted_file_double_click_is_guarded(qapp, tree, tmp_path) -> None:
    tree.set_files([tmp_path / "head.nii.gz"], tmp_path)
    activated: list[tuple[Path, ...]] = []
    tree.fileActivated.connect(activated.append)

    tree._list.itemDoubleClicked.emit(tree._list.item(0))

    assert activated == []


def test_set_files_rebuild_resets_rows(qapp, tree, tmp_path) -> None:
    first = tmp_path / "a.nii"
    first.touch()
    tree.set_files([first], tmp_path)
    tree._list.setCurrentRow(0)

    mesh = tmp_path / "mesh" / "final_mesh.ply"
    mesh.parent.mkdir()
    mesh.touch()
    tree.set_files([mesh], tmp_path)

    assert _row_texts(tree) == ["mesh/final_mesh.ply"]
    assert tree._list.currentRow() == -1
