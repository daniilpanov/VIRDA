"""Fiducial providers: where the pipeline gets its fiducial points from."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from virda.fiducials.detect import find_fiducials, to_fiducials
from virda.io.exporter.json_io import load_fiducials
from virda.models.fiducial import Fiducial
from virda.models.scalp_mesh import ScalpMesh


@runtime_checkable
class FiducialProvider(Protocol):
    def fiducials(self, mesh: ScalpMesh) -> list[Fiducial]: ...


class ManualFiducialProvider:
    """Loads fiducials from a JSON file (or skips them entirely)."""

    def __init__(self, path: Path | None, skip: bool = False) -> None:
        self._path = path
        self._skip = skip

    def fiducials(self, mesh: ScalpMesh) -> list[Fiducial]:
        if self._skip:
            return []
        if self._path is None:
            raise ValueError(
                "No fiducials provided. Pass --fiducials_path (or set FIDUCIALS_PATH), "
                "use --auto_detect_fiducials, or use --skip_fiducials to run without "
                "fiducial-dependent steps."
            )
        fiducials = load_fiducials(self._path)
        if not fiducials:
            raise ValueError(f"No fiducials found in {self._path}")
        return fiducials


class AutoFiducialProvider:
    """Detects fiducial points geometrically on the scalp mesh (QC approximations)."""

    def fiducials(self, mesh: ScalpMesh) -> list[Fiducial]:
        return to_fiducials(find_fiducials(mesh.vertices))
