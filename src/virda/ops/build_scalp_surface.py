"""Step: segment the head, seal the mask and extract the scalp surface."""

from virda.models.mri_volume import MRIVolume
from virda.models.scalp_surface import ScalpSurface
from virda.ops.extract_mesh import extract_mesh
from virda.ops.seal_mask import seal_mask
from virda.ops.segment_head import segment_head


def generate_scalp_surface(
    mri: MRIVolume,
    *,
    seal_enabled: bool = True,
    seal_radius: int = 4,
    voxel_size_mm: float | None = None,
) -> ScalpSurface:
    """Segment the head, optionally seal the mask, and extract the scalp mesh.

    ``voxel_size_mm`` is the desired effective voxel size of the extracted
    mesh; ``None`` keeps the native NIfTI resolution (``step_size=1``).
    """
    mask = segment_head(mri)

    if seal_enabled:
        mask = seal_mask(mask, radius=seal_radius)

    return ScalpSurface(mask=mask, mesh=extract_mesh(mask, mri, voxel_size_mm=voxel_size_mm))
