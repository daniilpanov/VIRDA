import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from virda.config import VirdaSettings
from virda.io.exporter.json_io import save_config, save_fiducials, save_json
from virda.io.exporter.ply_exporter import export_ply
from virda.models.ese_config import ESEConfig
from virda.models.fiducial import Fiducial
from virda.models.stage1_result import Stage1Result


class Stage1Exporter:
    def __init__(
        self,
        settings: VirdaSettings,
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

        if self._skip_fiducials:
            print(
                "[info] --skip-fiducials: fiducial-dependent steps disabled "
                "(fiducials.json not written)",
                file=sys.stderr,
            )
        else:
            save_fiducials(project_dir / "fiducials.json", self._fiducials)

        save_json(
            project_dir / "stage1_result.json",
            _stage1_result_to_dict(
                result,
                fiducials=self._fiducials,
                skipped=self._skip_fiducials,
            ),
        )
        save_config(project_dir / "pipeline_config.json", self._pipeline_config())
        return project_dir

    def _pipeline_config(self) -> dict[str, Any]:
        config: dict[str, Any] = self._settings.model_dump(
            exclude={"n_electrodes", "ese_offset_mm", "ese_reference"}
        )
        if self._ese_config is not None:
            config["ese"] = asdict(self._ese_config)
        return config


def _stage1_result_to_dict(
    result: Stage1Result,
    fiducials: list[Fiducial] | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    fiducial_count = len(fiducials or [])
    fiducials_section: dict[str, Any]
    if skipped:
        fiducials_section = {
            "skipped": True,
            "note": "--skip-fiducials: fiducial-dependent steps disabled",
        }
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
