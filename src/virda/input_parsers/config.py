"""Parser for pipeline config files."""

from pathlib import Path

from virda.config import VirdaSettings, build_config
from virda.models.config import Config


def parse_config(path: str | Path) -> Config:
    """Build the pipeline :class:`Config` from one input config file.

    Delegates to :func:`virda.config.build_config` with default
    :class:`VirdaSettings` (environment and ``.env``) as the base and the
    given file as the only config source; CLI overrides are not applied.
    """
    return build_config(VirdaSettings(), config_files=[Path(path)])
