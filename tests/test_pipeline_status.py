"""Unit tests for the Qt-free pipeline step model."""

from virda_gui.pipeline_status import pipeline_steps


def _base(**overrides):
    args = {
        "nifti_saved": False,
        "mesh_saved": False,
        "ese_saved": False,
        "mesh_in_memory": False,
        "ese_in_memory": False,
        "fiducials_filled": 0,
        "measurement_rows": 0,
    }
    args.update(overrides)
    return pipeline_steps(**args)  # type: ignore[arg-type]


def _states(steps):
    return [(step.key, step.state) for step in steps]


def _enabled(steps):
    return [(step.key, step.enabled) for step in steps]


def test_empty_project_only_scan_enabled() -> None:
    assert _enabled(_base()) == [
        ("scan", True),
        ("mesh", False),
        ("ese", False),
        ("points", False),
    ]


def test_steps_unlock_in_order() -> None:
    assert _enabled(_base(nifti_saved=True)) == [
        ("scan", True),
        ("mesh", True),
        ("ese", False),
        ("points", False),
    ]
    assert _enabled(_base(nifti_saved=True, mesh_in_memory=True)) == [
        ("scan", True),
        ("mesh", True),
        ("ese", True),
        ("points", False),
    ]
    assert _enabled(_base(nifti_saved=True, mesh_saved=True, ese_saved=True)) == [
        ("scan", True),
        ("mesh", True),
        ("ese", True),
        ("points", True),
    ]


def test_empty_project_first_step_active() -> None:
    assert _states(_base()) == [
        ("scan", "active"),
        ("mesh", "waiting"),
        ("ese", "waiting"),
        ("points", "waiting"),
    ]


def test_full_project_all_done() -> None:
    steps = _base(
        nifti_saved=True,
        mesh_saved=True,
        ese_saved=True,
        fiducials_filled=3,
        measurement_rows=2,
    )
    assert [state for _, state in _states(steps)] == ["done"] * 4


def test_memory_only_mesh_is_unsaved() -> None:
    steps = _base(nifti_saved=True, mesh_in_memory=True)
    assert _states(steps)[1] == ("mesh", "unsaved")
    assert "Save" in steps[1].detail


def test_memory_only_ese_is_unsaved() -> None:
    steps = _base(nifti_saved=True, mesh_saved=True, ese_in_memory=True)
    assert _states(steps)[2] == ("ese", "unsaved")


def test_points_need_three_fiducials_and_one_row() -> None:
    steps = _base(nifti_saved=True, mesh_saved=True, ese_saved=True, fiducials_filled=2)
    assert _states(steps)[3] == ("points", "active")
    assert "NAS" in steps[3].detail
    done = _base(
        nifti_saved=True,
        mesh_saved=True,
        ese_saved=True,
        fiducials_filled=3,
        measurement_rows=1,
    )
    assert _states(done)[3] == ("points", "done")
