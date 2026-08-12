from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NiftiPath:
    nifti_path: Path
