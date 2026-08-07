import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic_settings import BaseSettings
from scipy.spatial import cKDTree

from virda.io.exporter.json_io import save_config, save_fiducials, save_json
from virda.io.exporter.ply_exporter import export_ply
from virda.models.ese_config import ESEConfig
from virda.models.fiducial import Fiducial
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage1_result import Stage1Result

FIDUCIAL_TOLERANCE_MM = 3.0


class Stage1Exporter:
    def __init__(
        self,
        settings: BaseSettings,
        ese_config: ESEConfig | None = None,
        fiducials: list[Fiducial] | None = None,
        skip_fiducials: bool = False,
    ) -> None:
        self._settings = settings
        self._ese_config = ese_config
        self._fiducials = fiducials or []
        self._skip_fiducials = skip_fiducials

    def export(self, result: Stage1Result, output_dir: str | Path) -> Path:
        project_dir = Path(output_dir) / "patient_project"
        project_dir.mkdir(parents=True, exist_ok=True)
        export_ply(project_dir / "mesh.ply", result.mesh)

        fiducials_qc: dict[str, Any] | None = None
        if self._skip_fiducials:
            print(
                "[info] --skip-fiducials: fiducial-dependent steps disabled "
                "(fiducials.json not written)",
                file=sys.stderr,
            )
        else:
            save_fiducials(project_dir / "fiducials.json", self._fiducials)
            fiducials_qc = fiducial_qc(self._fiducials, result)
            for warning in fiducials_qc["warnings"]:
                print(f"[warning] {warning}", file=sys.stderr)

        save_json(
            project_dir / "stage1_result.json",
            _stage1_result_to_dict(
                result,
                fiducials=self._fiducials,
                fiducials_qc=fiducials_qc,
                skipped=self._skip_fiducials,
            ),
        )
        save_config(project_dir / "pipeline_config.json", self._pipeline_config())
        return project_dir

    def _pipeline_config(self) -> dict[str, Any]:
        config: dict[str, Any] = dict(self._settings.model_dump())
        if self._ese_config is not None:
            config["ese"] = asdict(self._ese_config)
        return config


def fiducial_qc(
    fiducials: list[Fiducial], result: Stage1Result, tolerance_mm: float = FIDUCIAL_TOLERANCE_MM
) -> dict[str, Any]:
    """Distance from each fiducial to the scalp mesh (per TZ 13.1 / 16)."""
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not fiducials:
        return {"checks": checks, "tolerance_mm": tolerance_mm, "warnings": warnings}
    tree = cKDTree(_mesh_world_vertices(result))
    for fiducial in fiducials:
        world = _fiducial_world_coordinates(fiducial, result)
        distance = float(tree.query(world, k=1)[0])
        checks.append(
            {
                "fiducial_id": fiducial.fiducial_id,
                "name": fiducial.name,
                "distance_to_surface_mm": round(distance, 3),
            }
        )
        if distance > tolerance_mm:
            warnings.append(
                f"{fiducial.fiducial_id} is {distance:.1f} mm from the scalp surface "
                f"(tolerance {tolerance_mm} mm)"
            )
    return {"checks": checks, "tolerance_mm": tolerance_mm, "warnings": warnings}


def _fiducial_world_coordinates(fiducial: Fiducial, result: Stage1Result) -> np.ndarray:
    coordinates = np.asarray(fiducial.coordinates, dtype=np.float64)
    if fiducial.coordinate_system == "voxel":
        affine = result.mri_volume.affine
        return cast(np.ndarray, coordinates @ affine[:3, :3].T + affine[:3, 3])
    return coordinates


def _mesh_world_vertices(result: Stage1Result) -> np.ndarray:
    mesh: ScalpMesh = result.mesh
    return np.asarray(mesh.vertices, dtype=np.float64)


def _stage1_result_to_dict(
    result: Stage1Result,
    fiducials: list[Fiducial] | None = None,
    fiducials_qc: dict[str, Any] | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    fiducial_count = len(fiducials or [])
    fiducials_section: dict[str, Any]
    if skipped:
        fiducials_section = {
            "skipped": True,
            "note": "--skip-fiducials: fiducial-dependent steps disabled",
        }
    elif fiducials_qc is not None:
        fiducials_section = {"count": fiducial_count, "qc": fiducials_qc}
    else:
        fiducials_section = {"count": fiducial_count}

    mri = result.mri_volume
    return {
        "mri_volume": {
            "shape": list(mri.data.shape),
            "affine": mri.affine.tolist(),
            "spacing": list(mri.spacing),
            "orientation": list(mri.orientation),
            "metadata": mri.metadata,
        },
        "segmentation_mask": {
            "shape": list(result.segmentation_mask.shape),
            "voxel_count": int(result.segmentation_mask.sum()),
        },
        "mesh": {
            "n_vertices": int(result.mesh.vertices.shape[0]),
            "n_faces": int(result.mesh.faces.shape[0]),
            "vertices_min": result.mesh.vertices.min(axis=0).tolist(),
            "vertices_max": result.mesh.vertices.max(axis=0).tolist(),
        },
        "fiducials": fiducials_section,
    }
