# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import os
import re
import sys
import platform
import subprocess
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_submodules,
    copy_metadata,
)

# ==========================================================
# Настройки приложения
# ==========================================================

ENTRY_SCRIPT = "src/virda_gui/__main__.py"
APP_NAME = "virda"

# Если у вас src-layout, оставляем src.
# Если не нужно, можно поставить []
# PATH_EX = ["src"]
PATH_EX = []

CONSOLE = True
DEBUG = False

# ==========================================================
# Исключения из прод-сборки
#
# mypy — dev-зависимость и в рантайме не нужна. pyvista
# импортирует pyvista.typing.mypy_plugin, но тот поднимает mypy
# только если он установлен (importlib.util.find_spec('mypy')).
# Без mypy в бандле этот блок просто пропускается. Заодно это
# чинит ModuleNotFoundError: No module named '<hash>__mypyc' —
# скомпилированные (mypyc) модули mypy PyInstaller собирает
# без их hash-named общих расширений.
# ==========================================================

EXCLUDED_MODULES = [
    "mypy",
]

# ==========================================================
# Базовые списки PyInstaller
# ==========================================================

datas = []
binaries = []
hiddenimports = []
runtime_hooks = []

# ==========================================================
# ttkbootstrap
# ==========================================================

tmp_ret = collect_all("ttkbootstrap")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

try:
    datas += copy_metadata("ttkbootstrap")
except Exception:
    pass

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
# Pillow / PIL для ttkbootstrap ImageTk
# ==========================================================

try:
    tmp_ret = collect_all("PIL")
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
except Exception as exc:
    print("WARNING: collect_all('PIL') failed:", exc)

try:
    datas += copy_metadata("Pillow")
except Exception:
    pass

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
    except Exception:
        pass

# ==========================================================
# Runtime hook для TCL_LIBRARY / TK_LIBRARY
#
# PyInstaller во время сборки создаст файл внутри build/
# и подключит его как runtime hook.
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

        # PyInstaller 6 для onedir часто кладет ресурсы в _internal
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
# Автоопределение Tcl/Tk библиотек и данных
# ==========================================================

seen_binaries = set()
seen_datas = set()

lib_dirs = []
roots = []


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

            # Если это symlink, добавим также реальный файл.
            # Это полезно для случаев libtcl9.0.so -> libtcl9.0.so.0
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
    # Примеры:
    # Linux: libtcl9.0.so, libtcl8.6.so, libtcl.so
    # Windows: tcl90.dll, tcl86t.dll, libtcl9.dll
    name = Path(name).name.lower()
    return (
        re.match(r"^(lib)?tcl\d", name, re.IGNORECASE) is not None
        or name in {"libtcl.so", "tcl.dll"}
    )


def is_tk_binary_name(name):
    # Примеры:
    # Linux: libtk9.0.so, libtk8.6.so, libtk.so
    # Windows: tk90.dll, tk86t.dll, libtk9.dll
    name = Path(name).name.lower()
    return (
        re.match(r"^(lib)?tk\d", name, re.IGNORECASE) is not None
        or name in {"libtk.so", "tk.dll"}
    )


def is_tcl_data_dir(path):
    # Например: tcl, tcl8.6, tcl9.0
    name = Path(path).name
    return re.match(r"^tcl(\d.*)?$", name, re.IGNORECASE) is not None


def is_tk_data_dir(path):
    # Например: tk, tk8.6, tk9.0
    # Важно: не должно матчить tkinter
    name = Path(path).name
    return re.match(r"^tk(\d.*)?$", name, re.IGNORECASE) is not None


system = platform.system()

# Корневые каталоги, где может находиться Python из uv / venv / conda
add_root(sys.base_prefix)
add_root(sys.prefix)
add_root(os.environ.get("CONDA_PREFIX"))
add_root(os.environ.get("UV_PYTHON_DIR"))

# Дополнительно используем sysconfig, если он дает нормальные пути
try:
    import sysconfig

    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        add_root(libdir)

    datadir = sysconfig.get_path("data")
    if datadir:
        add_root(datadir)
except Exception:
    pass

# Добавляем типовые подкаталоги с библиотеками
for root in roots:
    add_lib_dir(root)
    add_lib_dir(root / "lib")
    add_lib_dir(root / "lib64")
    add_lib_dir(root / "bin")
    add_lib_dir(root / "DLLs")
    add_lib_dir(root / "Library" / "bin")
    add_lib_dir(root / "Library" / "lib")
    add_lib_dir(root / "Library" / "share")

    # Иногда библиотеки лежат в multiarch-подкаталогах:
    # lib/x86_64-linux-gnu, lib/aarch64-linux-gnu и т.п.
    for base_lib in (root / "lib", root / "lib64"):
        if base_lib.is_dir():
            try:
                for child in base_lib.iterdir():
                    if child.is_dir():
                        add_lib_dir(child)
            except Exception:
                pass

# На Linux дополнительно спрашиваем ldd, к каким libtcl/libtk линкуется _tkinter
if system == "Linux":
    try:
        import importlib.util

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

    except Exception as exc:
        print("Tcl/Tk autodetect: ldd check failed:", exc)


# Паттерны библиотек для поиска вручную
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
            except Exception:
                pass


scan_binary_dirs()

# ==========================================================
# Определяем папки с данными Tcl/Tk
# ==========================================================

tcl_data_dir = None
tk_data_dir = None

# Сначала пробуем спросить у самого Tcl, где лежит его библиотека скриптов.
# Обычно это что-то вроде:
#   /path/to/lib/tcl9.0
# или:
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
    print("Tcl/Tk autodetect: tkinter.Tcl info library failed:", exc)


# Если нашли Tcl data dir, добавляем его и заодно осматриваем соседние каталоги.
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

    # После добавления новых каталогов снова ищем библиотеки
    scan_binary_dirs()

    # Пытаемся угадать расположение tk data dir.
    # Обычно он лежит рядом: tcl9.0 -> tk9.0
    m = re.search(r"(\d+(?:\.\d+)?)", tcl_data_dir.name)
    candidates = []

    if m:
        version = m.group(1)
        major = version.split(".")[0]

        candidates.append(parent / f"tk{version}")
        candidates.append(parent / f"tk{major}")

    candidates.append(parent / "tk")

    try:
        candidates.extend(
            sorted(p for p in parent.glob("tk*") if is_tk_data_dir(p))
        )
    except Exception:
        pass

    for candidate in candidates:
        if Path(candidate).is_dir():
            tk_data_dir = Path(candidate)
            break


# Если что-то не нашлось через tkinter, ищем вручную по типовым каталогам.
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
        except Exception:
            pass

        if tcl_data_dir:
            break

if not tk_data_dir:
    for d in data_search_dirs:
        try:
            for p in sorted(d.glob("tk*")):
                if p.is_dir() and is_tk_data_dir(p):
                    tk_data_dir = p
                    break
        except Exception:
            pass

        if tk_data_dir:
            break

if tcl_data_dir:
    add_data_dir(tcl_data_dir, "tcl")

if tk_data_dir:
    add_data_dir(tk_data_dir, "tk")


# ==========================================================
# Печать результатов автоопределения
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

pyz = PYZ(a.pure)

# ==========================================================
# One-file executable
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
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
