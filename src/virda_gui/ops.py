"""GUI-side facades over the ``virda`` core step functions.

The core stays a flat set of typed functions; the facades bundle their
keyword arguments into :mod:`virda_gui.options` values so tabs call
``clean(mesh, CleanOptions())``-shaped APIs instead of raw keyword lists.

No Qt here: these are plain callables, runnable off the GUI thread.
"""

from dataclasses import dataclass
from logging import Logger

from virda.mesh.mesh_decimator import decimate_mesh
from virda.models.electrode import Electrodes
from virda.models.fiducial import Fiducials
from virda.models.mesh import Mesh, Normals
from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask
from virda.models.stage2_config import Stage2Config
from virda.models.stage3_config import Stage3Config
from virda.ops.build_ese import build_ese
from virda.ops.clean_mesh import clean_mesh
from virda.ops.extract_mesh import extract_mesh
from virda.ops.localize_electrodes import localize_electrodes
from virda.ops.seal_mask import seal_mask
from virda.ops.segment_head import segment_head
from virda.ops.smooth_mesh import smooth_mesh
from virda_gui.options import (
    CleanOptions,
    DecimateOptions,
    EseOptions,
    LocalizeOptions,
    SealingOptions,
    SmoothOptions,
)


@dataclass(frozen=True)
class ScalpSurface:
    """Segmentation mask together with the mesh extracted from it."""

    mask: SegmentationMask
    mesh: Mesh


def generate_scalp_surface(
    mri: MRIVolume,
    sealing: SealingOptions | None = None,
    voxel_size_mm: float | None = None,
) -> ScalpSurface:
    """Segment the head, optionally seal the mask, and extract the scalp mesh.

    ``voxel_size_mm`` is the desired effective voxel size of the extracted
    mesh; ``None`` keeps the native NIfTI resolution (``step_size=1``).
    """
    mask = segment_head(mri)

    if sealing is not None and sealing.seal_enabled:
        mask = seal_mask(mask, radius=sealing.seal_radius)

    return ScalpSurface(mask=mask, mesh=extract_mesh(mask, mri, voxel_size_mm=voxel_size_mm))


def clean(mesh: Mesh, options: CleanOptions) -> Mesh:
    """Merge near-coincident vertices, split into components and keep the main body."""
    return clean_mesh(
        mesh,
        min_component_vertices=options.min_component_vertices,
        merge_digits=options.merge_digits,
    )


def decimate(mesh: Mesh, options: DecimateOptions) -> Mesh:
    """Reduce the mesh density by ``options.density_percent``."""
    return decimate_mesh(mesh, density_percent=options.density_percent)


def smooth(mesh: Mesh, options: SmoothOptions) -> Mesh:
    """Smooth a scalp mesh with the selected Laplacian or Taubin kernel."""
    if options.smoother == "none":
        return mesh
    return smooth_mesh(
        mesh,
        smoother_type=options.smoother,
        iterations=options.iterations,
        lamb=options.lamb,
        nu=options.nu,
    )


def generate_ese(mesh: Mesh, options: EseOptions) -> tuple[Mesh, Normals]:
    """Estimate outward normals and offset the scalp surface into the ESE mesh."""
    return build_ese(
        mesh,
        config=Stage2Config(
            neighborhood_radius_mm=options.neighborhood_radius_mm,
            k_neighbors=options.k_neighbors,
            use_weighted_pca=options.use_weighted_pca,
            pca_sigma_mm=options.pca_sigma_mm,
            min_neighbors=options.min_neighbors,
        ),
        ese_offset_mm=options.ese_offset_mm,
    )


def localize(
    ese: Mesh,
    normals: Normals,
    scalp_mesh: Mesh,
    fiducials: Fiducials,
    electrodes: Electrodes,
    options: LocalizeOptions,
    logger: Logger | None = None,
) -> Electrodes:
    """Localize electrodes on the ESE surface by brute-force search."""
    return localize_electrodes(
        ese,
        normals,
        scalp_mesh,
        fiducials,
        electrodes,
        config=Stage3Config(
            residual_threshold_mm=options.residual_threshold_mm,
            calibrate_ese_offset=options.calibrate_ese_offset,
        ),
        logger=logger,
    )
