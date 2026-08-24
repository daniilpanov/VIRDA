# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone VIRDA GUI executable.

Bundles the tkinter app together with the Tcl/Tk libraries and script
libraries it needs at runtime. The autodetection below locates libtcl/libtk
binaries and the tcl/tk data directories of the interpreter that runs this
spec, and fails the build loudly when they cannot be found.
"""

import importlib.util
import os
import platform
import re
import subprocess
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
# ttkbootstrap
# ==========================================================

tmp_ret = collect_all("ttkbootstrap")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

try:
    datas += copy_metadata("ttkbootstrap")
except Exception as exc:
    warnings.warn(f"copy_metadata('ttkbootstrap') failed: {exc}", stacklevel=1)

# ==========================================================
# tkinter / ttk
# ==========================================================

hiddenimports += collect_submodules("tkinter")

hiddenimports += [
    "tkinter",
    "tkinter.font",
    "tkinter.ttk",
    "tkinter.messagebox",
    "tkinter.filedialog",
    "tkinter.colorchooser",
    "tkinter.simpledialog",
]

# ==========================================================
# Pillow / PIL for ttkbootstrap ImageTk
# ==========================================================

try:
    tmp_ret = collect_all("PIL")
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
except Exception as exc:
    warnings.warn(f"collect_all('PIL') failed: {exc}", stacklevel=1)

try:
    datas += copy_metadata("Pillow")
except Exception as exc:
    warnings.warn(f"copy_metadata('Pillow') failed: {exc}", stacklevel=1)

pil_extra_hiddenimports = [
    "PIL",
    "PIL.Image",
    "PIL.ImageTk",
    "PIL._tkinter_finder",
    "PIL._imagingtk",
    "PIL._imaging",
]

for mod_name in pil_extra_hiddenimports:
    try:
        if importlib.util.find_spec(mod_name) is not None:
            hiddenimports.append(mod_name)
    except Exception as exc:
        warnings.warn(f"find_spec({mod_name!r}) failed: {exc}", stacklevel=1)

# ==========================================================
# Runtime hook for TCL_LIBRARY / TK_LIBRARY
#
# PyInstaller registers this file (created under build/) as a runtime hook.
# ==========================================================

try:
    hook_dir = Path(WORKPATH)
except NameError:
    hook_dir = Path("build")

hook_dir.mkdir(parents=True, exist_ok=True)
tcl_tk_runtime_hook = hook_dir / "tcl_tk_env.py"

TCL_TK_RUNTIME_HOOK_CODE = """
import os
import sys


def _get_base_dir():
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
        if base and os.path.isdir(base):
            return base

        exe_dir = os.path.dirname(sys.executable)

        # PyInstaller 6 onedir bundles usually place resources in _internal.
        internal_dir = os.path.join(exe_dir, "_internal")
        if os.path.isdir(internal_dir):
            return internal_dir

        return exe_dir

    return os.path.dirname(os.path.abspath(__file__))


base_dir = _get_base_dir()

for env_name, dir_name in (("TCL_LIBRARY", "tcl"), ("TK_LIBRARY", "tk")):
    candidate = os.path.join(base_dir, dir_name)
    current = os.environ.get(env_name)

    if os.path.isdir(candidate) and (not current or not os.path.isdir(current)):
        os.environ[env_name] = candidate
"""

tcl_tk_runtime_hook.write_text(TCL_TK_RUNTIME_HOOK_CODE, encoding="utf-8")
runtime_hooks.append(str(tcl_tk_runtime_hook))

# ==========================================================
# Tcl/Tk library and data autodetection
# ==========================================================

seen_binaries: set[str] = set()
seen_datas: set[str] = set()

lib_dirs: list[Path] = []
roots: list[Path] = []


def add_root(path):
    if not path:
        return

    try:
        p = Path(path).resolve()
        if p.is_dir() and p not in roots:
            roots.append(p)
    except Exception:
        pass


def add_lib_dir(path):
    try:
        p = Path(path)
        if p.is_dir() and p not in lib_dirs:
            lib_dirs.append(p)
    except Exception:
        pass


def add_binary_file(path):
    try:
        p = Path(path)

        if p.exists() or p.is_symlink():
            key = str(p)

            if key not in seen_binaries:
                binaries.append((key, "."))
                seen_binaries.add(key)

            # If this is a symlink, also add the real file. This matters for
            # cases like libtcl9.0.so -> libtcl9.0.so.0.
            if p.is_symlink():
                target = p.resolve()
                if target.exists():
                    target_key = str(target)
                    if target_key not in seen_binaries:
                        binaries.append((target_key, "."))
                        seen_binaries.add(target_key)

    except Exception:
        pass


def add_data_dir(path, dest_name):
    try:
        p = Path(path)
        if p.is_dir():
            key = f"{p}->{dest_name}"
            if key not in seen_datas:
                datas.append((str(p), dest_name))
                seen_datas.add(key)
    except Exception:
        pass


def is_tcl_binary_name(name):
    # Examples:
    # Linux: libtcl9.0.so, libtcl8.6.so, libtcl.so
    # Windows: tcl90.dll, tcl86t.dll, libtcl9.dll
    name = Path(name).name.lower()
    return (
        re.match(r"^(lib)?tcl\d", name, re.IGNORECASE) is not None
        or name in {"libtcl.so", "tcl.dll"}
    )


def is_tk_binary_name(name):
    # Examples:
    # Linux: libtk9.0.so, libtk8.6.so, libtk.so
    # Windows: tk90.dll, tk86t.dll, libtk9.dll
    name = Path(name).name.lower()
    return (
        re.match(r"^(lib)?tk\d", name, re.IGNORECASE) is not None
        or name in {"libtk.so", "tk.dll"}
    )


def is_tcl_data_dir(path):
    # Examples: tcl, tcl8.6, tcl9.0
    name = Path(path).name
    return re.match(r"^tcl(\d.*)?$", name, re.IGNORECASE) is not None


def is_tk_data_dir(path):
    # Examples: tk, tk8.6, tk9.0
    # Note: must not match "tkinter".
    name = Path(path).name
    return re.match(r"^tk(\d.*)?$", name, re.IGNORECASE) is not None


system = platform.system()

# Scan only the interpreter that runs this spec — the same environment the
# build was made for. Broad system-wide roots would risk picking up
# mismatched Tcl/Tk builds.
add_root(sys.base_prefix)
add_root(sys.prefix)

# sysconfig usually points into the same installation and knows about
# multiarch layouts.
try:
    import sysconfig

    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        add_root(libdir)

    datadir = sysconfig.get_path("data")
    if datadir:
        add_root(datadir)
except Exception as exc:
    warnings.warn(f"sysconfig lookup failed: {exc}", stacklevel=1)

# Add typical subdirectories that hold shared libraries.
for root in roots:
    add_lib_dir(root)
    add_lib_dir(root / "lib")
    add_lib_dir(root / "lib64")
    add_lib_dir(root / "bin")
    add_lib_dir(root / "DLLs")
    add_lib_dir(root / "Library" / "bin")
    add_lib_dir(root / "Library" / "lib")
    add_lib_dir(root / "Library" / "share")

    # Libraries may live in multiarch subdirectories such as
    # lib/x86_64-linux-gnu, lib/aarch64-linux-gnu, etc.
    for base_lib in (root / "lib", root / "lib64"):
        if base_lib.is_dir():
            try:
                for child in base_lib.iterdir():
                    if child.is_dir():
                        add_lib_dir(child)
            except OSError as exc:
                warnings.warn(f"cannot list {base_lib}: {exc}", stacklevel=1)

# On Linux additionally ask ldd which libtcl/libtk the _tkinter extension
# links against, then bundle exactly those files.
if system == "Linux":
    tkinter_ext_spec = importlib.util.find_spec("_tkinter")

    if tkinter_ext_spec and tkinter_ext_spec.origin:
        tkinter_ext_path = Path(tkinter_ext_spec.origin)

        if tkinter_ext_path.exists():
            ldd_output = subprocess.check_output(
                ["ldd", str(tkinter_ext_path)],
                text=True,
                stderr=subprocess.DEVNULL,
            )

            for line in ldd_output.splitlines():
                if "=>" not in line:
                    continue

                right = line.split("=>", 1)[1].strip()

                if not right.startswith("/"):
                    continue

                lib_path = right.split()[0]
                lib_name = Path(lib_path).name

                if is_tcl_binary_name(lib_name) or is_tk_binary_name(lib_name):
                    add_binary_file(lib_path)


# Library glob patterns for the manual directory scan below.
if system == "Windows":
    binary_patterns = [
        "tcl*.dll",
        "tk*.dll",
        "libtcl*.dll",
        "libtk*.dll",
    ]
elif system == "Darwin":
    binary_patterns = [
        "libtcl*.dylib",
        "libtk*.dylib",
        "libtcl*.so*",
        "libtk*.so*",
    ]
else:
    binary_patterns = [
        "libtcl*.so*",
        "libtk*.so*",
    ]


def scan_binary_dirs():
    for lib_dir in lib_dirs:
        if not lib_dir.is_dir():
            continue

        for pattern in binary_patterns:
            try:
                for f in lib_dir.glob(pattern):
                    name = f.name

                    if is_tcl_binary_name(name) or is_tk_binary_name(name):
                        add_binary_file(f)
            except OSError as exc:
                warnings.warn(f"cannot scan {lib_dir}: {exc}", stacklevel=1)


scan_binary_dirs()

# ==========================================================
# Locate the Tcl/Tk data directories
# ==========================================================

tcl_data_dir = None
tk_data_dir = None

# First ask Tcl itself where its script library lives. Typically something
# like:
#   /path/to/lib/tcl9.0
# or:
#   C:\path\to\tcl\tcl9.0
try:
    import tkinter

    tcl_interp = tkinter.Tcl()
    tcl_library = tcl_interp.eval("info library").strip("{}")

    if tcl_library:
        p = Path(tcl_library)
        if p.is_dir():
            tcl_data_dir = p

except Exception as exc:
    warnings.warn(
        f"Tcl/Tk autodetect: tkinter.Tcl 'info library' failed: {exc}",
        stacklevel=1,
    )


# Once the Tcl data dir is found, add it and inspect neighbouring dirs too.
if tcl_data_dir:
    add_data_dir(tcl_data_dir, "tcl")

    parent = tcl_data_dir.parent
    grandparent = parent.parent

    add_lib_dir(parent)
    add_lib_dir(grandparent)

    for base in (parent, grandparent):
        add_lib_dir(base / "bin")
        add_lib_dir(base / "lib")
        add_lib_dir(base / "lib64")
        add_lib_dir(base / "DLLs")
        add_lib_dir(base / "Library" / "bin")
        add_lib_dir(base / "Library" / "lib")

    # Search for libraries again after adding the new directories.
    scan_binary_dirs()

    # Guess the tk data dir location. It normally sits next to the Tcl one:
    # tcl9.0 -> tk9.0
    m = re.search(r"(\d+(?:\.\d+)?)", tcl_data_dir.name)
    candidates = []

    if m:
        version = m.group(1)
        major = version.split(".")[0]

        candidates.append(parent / f"tk{version}")
        candidates.append(parent / f"tk{major}")

    candidates.append(parent / "tk")

    try:
        candidates.extend(sorted(p for p in parent.glob("tk*") if is_tk_data_dir(p)))
    except OSError as exc:
        warnings.warn(f"cannot list {parent}: {exc}", stacklevel=1)

    for candidate in candidates:
        if Path(candidate).is_dir():
            tk_data_dir = Path(candidate)
            break


# Fall back to searching typical directories manually.
data_search_dirs = []

for root in roots:
    for rel in (
        "lib",
        "share",
        "tcl",
        Path("Library") / "lib",
        Path("Library") / "share",
    ):
        d = root / rel
        if d.is_dir():
            data_search_dirs.append(d)

if tcl_data_dir:
    data_search_dirs.insert(0, tcl_data_dir.parent)

if not tcl_data_dir:
    for d in data_search_dirs:
        try:
            for p in sorted(d.glob("tcl*")):
                if p.is_dir() and is_tcl_data_dir(p):
                    tcl_data_dir = p
                    break
        except OSError as exc:
            warnings.warn(f"cannot list {d}: {exc}", stacklevel=1)

        if tcl_data_dir:
            break

if not tk_data_dir:
    for d in data_search_dirs:
        try:
            for p in sorted(d.glob("tk*")):
                if p.is_dir() and is_tk_data_dir(p):
                    tk_data_dir = p
                    break
        except OSError as exc:
            warnings.warn(f"cannot list {d}: {exc}", stacklevel=1)

        if tk_data_dir:
            break

if tcl_data_dir:
    add_data_dir(tcl_data_dir, "tcl")

if tk_data_dir:
    add_data_dir(tk_data_dir, "tk")


# ==========================================================
# Fail loudly when Tcl/Tk could not be located
# ==========================================================

missing = []
if tcl_data_dir is None:
    missing.append("the Tcl script library directory (tcl8.6 / tcl9.0 ...)")
if tk_data_dir is None:
    missing.append("the Tk script library directory (tk8.6 / tk9.0 ...)")
if missing:
    raise SystemExit(
        "Tcl/Tk autodetect failed: could not locate "
        + " and ".join(missing)
        + f" under {roots}. The frozen GUI cannot start without them.\n"
        "Make sure the build environment has a Python with tkinter installed "
        "(e.g. 'uv sync --extra dev' plus the python3-tk system package on "
        "Debian/Ubuntu), or set TCL_LIBRARY/TK_LIBRARY explicitly."
    )


# ==========================================================
# Autodetection summary
# ==========================================================

print("=" * 80)
print("Tcl/Tk autodetect results")
print("=" * 80)

print("Roots:")
for r in roots:
    print(" ", r)

print("\nBinary dirs searched:")
for d in lib_dirs:
    print(" ", d)

print("\nAdded binaries:")
for b in binaries:
    if is_tcl_binary_name(Path(b[0]).name) or is_tk_binary_name(Path(b[0]).name):
        print(" ", b)

print("\nAdded datas:")
for d in datas:
    if Path(d[0]).name.lower().startswith(("tcl", "tk")):
        print(" ", d)

print("=" * 80)

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
