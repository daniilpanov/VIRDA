"""Unit tests for :class:`virda_gui_new.widgets.filetree.FileTree`.

Runs under the offscreen Qt platform; artifacts are fed directly, so no
scanning or project layout is involved. Real temp files back only the
double-click activation check.
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


def test_empty_artifacts_show_no_rows(qapp, tree) -> None:
    from virda_gui_new.services.project_scan import ProjectArtifacts

    tree.set_artifacts(ProjectArtifacts())

    assert _row_texts(tree) == []


def test_rows_group_in_stable_order_with_plain_text(qapp, tree, tmp_path) -> None:
    from virda_gui_new.services.project_scan import ProjectArtifacts

    files = [
        tmp_path / "input" / "head.nii.gz",
        tmp_path / "input" / "fiducials.json",
        tmp_path / "mesh" / "final_mesh.ply",
        tmp_path / "ese" / "ese_mesh.ply",
        tmp_path / "localization" / "electrodes_scalp.json",
        tmp_path / "localization" / "electrodes_ese.json",
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    tree.set_artifacts(
        ProjectArtifacts(
            nifti=files[0],
            fiducials=files[1],
            scalp_mesh=files[2],
            ese_mesh=files[3],
            scalp_points=files[4],
            ese_points=files[5],
        )
    )

    assert _row_texts(tree) == [
        "input/head.nii.gz",
        "input/fiducials.json",
        "mesh/final_mesh.ply",
        "ese/ese_mesh.ply",
        "localization/electrodes_scalp.json",
        "localization/electrodes_ese.json",
    ]


def test_missing_artifacts_are_absent(qapp, tree) -> None:
    from virda_gui_new.services.project_scan import ProjectArtifacts

    tree.set_artifacts(
        ProjectArtifacts(
            nifti=Path("input/missing.nii.gz"),
            ese_points=Path("localization/missing.json"),
        )
    )

    assert _row_texts(tree) == []


def test_new_artifact_rows_use_real_subdirectories(qapp, tree, tmp_path) -> None:
    from virda_gui_new.services.project_scan import ProjectArtifacts

    measurements = tmp_path / "input" / "measurements.json"
    config = tmp_path / "somewhere" / "pipeline_config.json"
    head_mask = tmp_path / "segmentation" / "head_mask.nii.gz"
    base_mesh = tmp_path / "scalp" / "base_mesh.ply"
    final_mesh = tmp_path / "scalp" / "final_mesh.ply"
    for path in (measurements, config, head_mask, base_mesh, final_mesh):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    tree.set_artifacts(
        ProjectArtifacts(
            measurements=measurements,
            pipeline_config=config,
            head_mask=head_mask,
            base_mesh=base_mesh,
            scalp_mesh=final_mesh,
        )
    )

    assert _row_texts(tree) == [
        "input/measurements.json",
        "somewhere/pipeline_config.json",
        "segmentation/head_mask.nii.gz",
        "scalp/base_mesh.ply",
        "scalp/final_mesh.ply",
    ]


def test_double_click_on_existing_file_emits_path(qapp, tree, tmp_path) -> None:
    from virda_gui_new.services.project_scan import ProjectArtifacts

    nifti = tmp_path / "head.nii.gz"
    nifti.touch()
    tree.set_artifacts(ProjectArtifacts(nifti=nifti))
    activated: list[Path] = []
    tree.fileActivated.connect(activated.append)

    tree._list.itemDoubleClicked.emit(tree._list.item(0))

    assert activated == [nifti]


def test_deleted_file_double_click_is_guarded(qapp, tree, tmp_path) -> None:
    from virda_gui_new.services.project_scan import ProjectArtifacts

    tree.set_artifacts(ProjectArtifacts(nifti=tmp_path / "head.nii.gz"))
    activated: list[tuple[Path, ...]] = []
    tree.fileActivated.connect(activated.append)

    tree._list.itemDoubleClicked.emit(tree._list.item(0))

    assert activated == []


def test_set_artifacts_rebuild_resets_rows(qapp, tree, tmp_path) -> None:
    from virda_gui_new.services.project_scan import ProjectArtifacts

    tree.set_artifacts(ProjectArtifacts(nifti=tmp_path / "a.nii"))
    tree._list.setCurrentRow(0)

    mesh = tmp_path / "mesh" / "final_mesh.ply"
    mesh.parent.mkdir()
    mesh.touch()
    tree.set_artifacts(ProjectArtifacts(scalp_mesh=mesh))

    assert _row_texts(tree) == ["mesh/final_mesh.ply"]
    assert tree._list.currentRow() == -1
