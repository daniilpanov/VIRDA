"""Export the Stage 1 artifacts into a patient project directory.

Writes the final scalp mesh, segmentation mask, fiducials, the processing
config and a copy of the source NIfTI.
"""

import json
import shutil
from logging import Logger
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh

from virda.io.fiducial_helpers import save_fiducials
from virda.models.config import Config
from virda.models.stage1_result import Stage1Result


def export_stage1(
    result: Stage1Result,
    *,
    project_dir: Path,
    config: Config,
    nifti_path: Path | None = None,
    logger: Logger | None = None,
) -> None:
    """Write every Stage 1 artifact below ``project_dir``."""
    project = Path(project_dir)
    for subdir in ("input", "mesh", "segmentation"):
        (project / subdir).mkdir(parents=True, exist_ok=True)

    # 1. Final mesh (copy of the latest ScalpMesh version)
    mesh = result.mesh
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    tm.export(str(project / "mesh" / "final_mesh.ply"))
    np.save(str(project / "mesh" / "scalp_vertices.npy"), mesh.vertices)
    np.save(str(project / "mesh" / "scalp_faces.npy"), mesh.faces)
    np.save(str(project / "mesh" / "scalp_face_adjacency.npy"), mesh.face_adjacency)

    # 2. Segmentation mask as NIfTI (preserves MRI affine)
    seg = result.segmentation_mask
    mri = result.mri_volume
    seg_nii = nib.Nifti1Image(seg.mask.astype(np.uint8), mri.affine)
    nib.save(seg_nii, str(project / "segmentation" / "head_mask.nii.gz"))

    # 3. Fiducials as JSON
    save_fiducials(project / "input" / "fiducials.json", result.fiducials)

    # 4. Processing config as JSON
    (project / "input" / "pipeline_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2)
    )

    # 5. Source NIfTI copy
    if nifti_path is not None:
        nifti_path = Path(nifti_path)
        target_path = project / "input" / nifti_path.name
        try:
            shutil.copy2(nifti_path, target_path)
        except OSError:
            if logger is not None:
                logger.warning(
                    "Failed to copy source NIfTI ('%s') into patient project ('%s')",
                    nifti_path,
                    target_path,
                    exc_info=True,
                )
