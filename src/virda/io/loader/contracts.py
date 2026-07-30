from pathlib import Path
from typing import Protocol, runtime_checkable

from virda.models.mri_volume import MRIVolume


@runtime_checkable
class MRILoader(Protocol):
    def load(self, path: str | Path) -> MRIVolume: ...
