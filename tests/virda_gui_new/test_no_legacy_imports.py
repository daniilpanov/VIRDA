"""The new GUI must not depend on the legacy GUI package."""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_NEW_GUI = _ROOT / "src" / "virda_gui_new"
_NEW_TESTS = _ROOT / "tests" / "virda_gui_new"
_LEGACY = "virda_gui"


def _imports(path: Path) -> list[str]:
    """The module names a file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return names


@pytest.mark.parametrize("root", [_NEW_GUI, _NEW_TESTS], ids=["src", "tests"])
def test_the_new_gui_does_not_import_the_legacy_package(root: Path) -> None:
    """No module of the new GUI may import the legacy one."""
    offenders = [
        f"{path.relative_to(_ROOT)}: {name}"
        for path in sorted(root.rglob("*.py"))
        for name in _imports(path)
        if name == _LEGACY or name.startswith(f"{_LEGACY}.")
    ]

    assert offenders == []
