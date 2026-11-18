"""Unit tests for :mod:`virda_gui_new.services.step_config`."""

import json
from pathlib import Path

from virda_gui_new.dto.steps import BASE_MESH, ESE_MESH
from virda_gui_new.services.step_config import (
    config_path,
    load_step_config,
    save_step_config,
)


def test_config_path_lives_under_config(tmp_path: Path) -> None:
    assert config_path(tmp_path, BASE_MESH) == tmp_path / "config" / "base_mesh.json"


def test_missing_configuration_is_an_empty_mapping(tmp_path: Path) -> None:
    assert load_step_config(tmp_path, BASE_MESH) == {}


def test_saved_configuration_is_read_back(tmp_path: Path) -> None:
    save_step_config(tmp_path, BASE_MESH, {"seal_enabled": False, "seal_radius": 6})

    assert load_step_config(tmp_path, BASE_MESH) == {"seal_enabled": False, "seal_radius": 6}


def test_the_configuration_of_a_step_is_its_own(tmp_path: Path) -> None:
    save_step_config(tmp_path, BASE_MESH, {"seal_radius": 6})

    assert load_step_config(tmp_path, ESE_MESH) == {}


def test_the_configuration_is_written_as_json(tmp_path: Path) -> None:
    written = save_step_config(tmp_path, BASE_MESH, {"seal_radius": 6})

    assert json.loads(written.read_text(encoding="utf-8")) == {"seal_radius": 6}


def test_a_broken_configuration_file_falls_back_to_the_defaults(tmp_path: Path) -> None:
    target = config_path(tmp_path, BASE_MESH)
    target.parent.mkdir(parents=True)
    target.write_text("not json", encoding="utf-8")

    assert load_step_config(tmp_path, BASE_MESH) == {}


def test_a_configuration_file_with_another_shape_is_ignored(tmp_path: Path) -> None:
    target = config_path(tmp_path, BASE_MESH)
    target.parent.mkdir(parents=True)
    target.write_text("[1, 2, 3]", encoding="utf-8")

    assert load_step_config(tmp_path, BASE_MESH) == {}
