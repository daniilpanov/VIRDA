"""Configuration loading and merging for the VIRDA pipeline.

Re-exports the public API of the former :mod:`virda.config` module so that
``from virda.config import ...`` call sites keep working unchanged after the
split into :mod:`virda.config.settings`, :mod:`virda.config.files` and
:mod:`virda.config.merge`.
"""

from virda.config.files import load_config_file, resolve_config_files
from virda.config.merge import build_config, resolve_stage3_config
from virda.config.settings import VirdaSettings

__all__ = [
    "VirdaSettings",
    "resolve_config_files",
    "load_config_file",
    "build_config",
    "resolve_stage3_config",
]