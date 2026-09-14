# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone VIRDA GUI executable.

Bundles the PySide6 (Qt) app, gathering the Qt plugins and translations it
needs at runtime.
"""

import importlib.util
import os
import sys
import warnings
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_submodules,
    copy_metadata,
)

# ==========================================================
# Application settings
# ==========================================================

ENTRY_SCRIPT = "src/virda_gui/__main__.py"
APP_NAME = "virda"

PATH_EX: list[str] = []

# The GUI ships its own log pane; a console window would be a duplicate.
CONSOLE = False
DEBUG = False

# ==========================================================
# Exclusions from the production bundle
#
# mypy is a dev dependency and not needed at runtime. pyvista imports
# pyvista.typing.mypy_plugin, but that only sets up mypy when mypy is
# installed (importlib.util.find_spec('mypy')). Without mypy in the bundle
# that block is skipped. This also avoids ModuleNotFoundError: No module
# named '<hash>__mypyc' — PyInstaller bundles mypy's compiled (mypyc)
# modules without their hash-named shared extensions.
# ==========================================================

EXCLUDED_MODULES = [
    "mypy",
]

# ==========================================================
# PyInstaller base lists
# ==========================================================

datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = []
runtime_hooks: list[str] = []

# ==========================================================
# PySide6 / Qt
# ==========================================================

try:
    qt_data, qt_binaries, qt_hidden = collect_all("PySide6")
    datas += qt_data
    binaries += qt_binaries
    hiddenimports += qt_hidden
except Exception as exc:
    warnings.warn(f"collect_all('PySide6') failed: {exc}", stacklevel=1)

try:
    datas += copy_metadata("PySide6")
except Exception as exc:
    warnings.warn(f"copy_metadata('PySide6') failed: {exc}", stacklevel=1)

hiddenimports += ["PySide6"]

# ==========================================================
# Don't bundle the system C++ runtime
#
# PyInstaller copies libstdc++.so.6 / libgcc_s.so.1 from the build machine
# into the bundle, and the bootloader prepends _MEIPASS to LD_LIBRARY_PATH,
# so that copy shadows the target system's. A CI build (ubuntu-22.04) then
# ships an older libstdc++ than the user's distro, and the Mesa/GPU drivers
# loaded by the system libGL fail to dlopen inside the frozen process
# ("Could not find a decent config" -> EGL/OSMesa fallback -> segfault in
# the 3D viewer). Both libraries exist in any desktop Linux base system,
# so rely on the target's own runtime instead.
# ==========================================================

SYSTEM_RUNTIME_LIBS = {"libstdc++.so.6", "libgcc_s.so.1"}

# ==========================================================
# Analysis
# ==========================================================

a = Analysis(
    [ENTRY_SCRIPT],
    pathex=PATH_EX,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=EXCLUDED_MODULES,
    noarchive=False,
    optimize=0,
)

a.binaries = [
    entry
    for entry in a.binaries
    if os.path.basename(entry[0]) not in SYSTEM_RUNTIME_LIBS
]

pyz = PYZ(a.pure)

# ==========================================================
# One-file executable
#
# upx=False: UPX compression regularly triggers false positives in antivirus
# scanners and can slow down startup; the size gain is not worth it here.
# ==========================================================

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=APP_NAME,
    debug=DEBUG,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
