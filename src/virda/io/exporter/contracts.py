from pathlib import Path
from typing import Protocol, runtime_checkable

from virda.models.stage1_result import Stage1Result


@runtime_checkable
class Exporter(Protocol):
    def export(self, result: Stage1Result, output_dir: str | Path) -> Path: ...
