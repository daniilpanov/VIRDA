from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NiftiPath:
    nifti_path: Path


@dataclass(frozen=True)
class FiducialsPath:
    fiducials_path: Path
