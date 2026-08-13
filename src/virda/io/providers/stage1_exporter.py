from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh

from virda.io.fiducial_helpers import save_fiducials
from virda.models.stage1_result import Stage1Result


class Stage1Exporter:
    """Export Stage 1 artifacts: final mesh, segmentation mask, fiducials."""

    def __init__(self, project_dir: Path) -> None:
        self.project = Path(project_dir)
        for subdir in ("mesh", "segmentation", "fiducials"):
            (self.project / subdir).mkdir(parents=True, exist_ok=True)

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
