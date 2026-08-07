"""QC overlay: MRI slices with the scalp mesh contour drawn over them."""

from pathlib import Path

import numpy as np

from virda.io.qc.geometry import mesh_voxel_coordinates
from virda.models.stage1_result import Stage1Result


def overlay_slices(result: Stage1Result, output_dir: str | Path) -> Path:
    """Sagittal/coronal/axial MRI slices with the scalp mesh contour overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = result.mri_volume.data
    vox = mesh_voxel_coordinates(result.mesh.vertices, result.mri_volume.affine)
    plane_names = {0: "sagittal", 1: "coronal", 2: "axial"}
    for plane, name in plane_names.items():
        n = data.shape[plane]
        idxs = np.linspace(0, n - 1, 6, dtype=int)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, s in zip(axes.ravel(), idxs, strict=False):
            if plane == 0:
                sl = np.rot90(data[s, :, :])
                pts = np.stack([vox[:, 1], vox[:, 2]], 1)
                hit = np.abs(vox[:, 0] - s) < 1.0
            elif plane == 1:
                sl = np.rot90(data[:, s, :])
                pts = np.stack([vox[:, 0], vox[:, 2]], 1)
                hit = np.abs(vox[:, 1] - s) < 1.0
            else:
                sl = np.rot90(data[:, :, s])
                pts = np.stack([vox[:, 0], vox[:, 1]], 1)
                hit = np.abs(vox[:, 2] - s) < 1.0
            ax.imshow(sl, cmap="gray")
            ax.scatter(pts[hit, 0], pts[hit, 1], s=0.05, c="r", linewidths=0)
            ax.set_title(f"{name} slice {s}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"Scalp mesh over {name} slices", fontsize=13)
        fig.tight_layout()
        fig.savefig(out / f"qc_overlay_{name}.png", dpi=90)
        plt.close(fig)
    return out
