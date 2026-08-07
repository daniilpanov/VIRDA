import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic_settings import BaseSettings

from virda.io.exporter.json_io import save_config, save_fiducials, save_json
from virda.io.exporter.nifti_exporter import export_segmentation
from virda.io.exporter.ply_exporter import export_ply
from virda.models.ese_config import ESEConfig
from virda.models.fiducial import Fiducial
from virda.models.stage1_result import Stage1Result
from virda.qc.checks import run_checks
from virda.visualization import write_visual_artifacts


class Stage1Exporter:
    def __init__(
        self,
        settings: BaseSettings,
        ese_config: ESEConfig | None = None,
        skip_fiducials: bool = False,
    ) -> None:
        self._settings = settings
        self._ese_config = ese_config
        self._skip_fiducials = skip_fiducials
        self._qc_html = bool(getattr(settings, "qc_html", False))

    def export(self, result: Stage1Result, output_dir: str | Path) -> Path:
        project_dir = Path(output_dir) / "patient_project"
        mesh_dir = project_dir / "mesh"
        segmentation_dir = project_dir / "segmentation"
        fiducials_dir = project_dir / "fiducials"
        config_dir = project_dir / "config"
        quality_control_dir = project_dir / "quality_control"
        input_mri_dir = project_dir / "input_mri"
        for directory in (
            mesh_dir,
            segmentation_dir,
            fiducials_dir,
            config_dir,
            quality_control_dir,
            input_mri_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        log = _configure_logging(project_dir / "logs" / "stage1.log")

        mesh_path = mesh_dir / "scalp.ply"
        export_ply(mesh_path, result.mesh)
        if result.mesh.face_adjacency is not None:
            np.save(mesh_dir / "scalp_face_adjacency.npy", result.mesh.face_adjacency)

        fiducials = result.fiducials
        fiducials_qc: dict[str, Any] | None = None
        if self._skip_fiducials:
            log.info(
                "--skip-fiducials: fiducial-dependent steps disabled (fiducials.json not written)"
            )
        else:
            save_fiducials(fiducials_dir / "fiducials.json", fiducials)

        mask_path = segmentation_dir / "head_mask.nii.gz"
        export_segmentation(mask_path, result.segmentation_mask, result.mri_volume.affine)

        report = run_checks(result, nifti_mask_path=mask_path)
        fiducials_qc = report["fiducials"]
        for warning in report["warnings"]:
            log.warning(warning)
        save_json(quality_control_dir / "report.json", report)

        save_json(
            project_dir / "stage1_result.json",
            _stage1_result_to_dict(
                result,
                fiducials=fiducials,
                fiducials_qc=fiducials_qc,
                skipped=self._skip_fiducials,
            ),
        )
        save_config(config_dir / "pipeline_config.json", self._pipeline_config())
        save_json(input_mri_dir / "provenance.json", _input_provenance(result))
        write_visual_artifacts(
            result, quality_control_dir, mesh_path=mesh_path, with_html=self._qc_html
        )
        return project_dir

    def _pipeline_config(self) -> dict[str, Any]:
        config = self._settings.model_dump(
            exclude={"n_electrodes", "ese_offset_mm", "ese_reference"}
        )
        if self._ese_config is not None:
            config["ese"] = asdict(self._ese_config)
        return config


def _configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("virda.stage1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _input_provenance(result: Stage1Result) -> dict[str, Any]:
    mri = result.mri_volume
    return {
        "source": mri.metadata.get("source"),
        "shape": list(mri.data.shape),
        "spacing": list(mri.spacing),
        "orientation": list(mri.orientation),
        "affine": mri.affine.tolist(),
    }


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
    adjacency_edges = (
        int(result.mesh.face_adjacency.shape[0]) if result.mesh.face_adjacency is not None else None
    )
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
            "n_adjacency_edges": adjacency_edges,
            "coordinate_system": result.mesh.coordinate_system,
            "vertices_min": result.mesh.vertices.min(axis=0).tolist(),
            "vertices_max": result.mesh.vertices.max(axis=0).tolist(),
        },
        "fiducials": fiducials_section,
    }
