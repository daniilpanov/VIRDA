import json
import logging
import shutil
from dataclasses import asdict
from logging import Logger
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh

from virda.io.fiducial_helpers import save_fiducials
from virda.models.config import Config
from virda.models.ese_config import ESEConfig
from virda.models.stage1_result import Stage1Result


class Stage1Exporter:
    """
    Export Stage 1 artifacts:
        final mesh
        segmentation mask
        fiducials
        ESE config
        config
    """

    def __init__(
        self,
        project_dir: Path,
        ese_config: ESEConfig | None = None,
        config: Config | None = None,
        nifti_path: Path | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.project = Path(project_dir)
        for subdir in ("input", "mesh", "segmentation", "fiducials", "config"):
            (self.project / subdir).mkdir(parents=True, exist_ok=True)

        self._ese_config = ese_config
        self._config = config
        self._nifti_path = Path(nifti_path) if nifti_path else None
        self._logger = logger

    def provide(self, result: Stage1Result | None) -> None:
        if not result:
            raise ValueError("There is no result of Stage#1")

        # 1. Final mesh (copy of the latest ScalpMesh version)
        mesh = result.mesh
        tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        tm.export(str(self.project / "mesh" / "final_mesh.ply"))
        np.save(str(self.project / "mesh" / "scalp_vertices.npy"), mesh.vertices)
        np.save(str(self.project / "mesh" / "scalp_faces.npy"), mesh.faces)
        np.save(str(self.project / "mesh" / "scalp_face_adjacency.npy"), mesh.face_adjacency)
        (self.project / "mesh" / "n_adjacency_edges.json").write_text(
            f'{{"n_adjacency_edges": {mesh.face_adjacency.shape[0]}}}'
        )

        # 2. Segmentation mask as NIfTI (preserves MRI affine)
        seg = result.segmentation_mask
        mri = result.mri_volume
        seg_nii = nib.Nifti1Image(seg.mask.astype(np.uint8), mri.affine)
        nib.save(seg_nii, str(self.project / "segmentation" / "head_mask.nii.gz"))

        # 3. Fiducials as JSON
        save_fiducials(self.project / "fiducials" / "fiducials.json", result.fiducials)

        # 4. Processing ESE config as JSON
        if self._ese_config is not None:
            pipeline_config = {"ese": asdict(self._ese_config)}
            (self.project / "config" / "ese.json").write_text(json.dumps(pipeline_config, indent=2))

        # 5. Processing config as JSON
        if self._config is not None:
            (self.project / "input" / "pipeline_config.json").write_text(
                json.dumps(self._config.model_dump(mode="json"), indent=2)
            )

        # 6. Source NIfTI copy
        if self._nifti_path is not None:
            target_path = self.project / "input" / self._nifti_path.name
            try:
                shutil.copy2(self._nifti_path, target_path)
            except OSError:
                logger = self._logger or logging.getLogger(__name__)
                logger.warning(
                    f"Failed to copy source NIfTI ('{self._nifti_path}')"
                    f" into patient project ('{target_path}')",
                    exc_info=True,
                )
