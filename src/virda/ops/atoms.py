"""Pure high-level operations over the VIRDA core processors.

Each atom is a thin function orchestrating the pure core classes (segmentation,
mesh processing, ESE estimation and localization) with explicit options.
"""

from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.localization.brute_force_localizer import BruteForceLocalizer
from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.mesh.mesh_decimator import QuadraticDecimator
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.mesh.taubin_smoother import TaubinSmoother
from virda.models.electrode import Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducials
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.ops.options import (
    CleanOptions,
    DecimateOptions,
    EseOptions,
    LocalizeOptions,
    SealingOptions,
    SmoothOptions,
)
from virda.ops.scalp_surface import ScalpSurface
from virda.segmentation.head_segmenter import OtsuHeadSegmenter
from virda.segmentation.seal import MaskSealer


def generate_scalp_surface(
    mri: MRIVolume,
    sealing: SealingOptions | None = None,
) -> ScalpSurface:
    """Segment the head, optionally seal the mask, and extract the scalp mesh."""
    segmenter = OtsuHeadSegmenter()
    mask = segmenter.process(mri)

    if sealing is not None and sealing.seal_enabled:
        sealer = MaskSealer(radius=sealing.seal_radius)
        mask = sealer.process(mask)

    extractor = MarchingCubesExtractor()
    mesh = extractor.process(mask=mask, mri_volume=mri)

    return ScalpSurface(mask=mask, mesh=mesh)


def smooth(mesh: ScalpMesh, options: SmoothOptions) -> ScalpMesh:
    """Smooth a scalp mesh with the selected Laplacian or Taubin kernel."""
    if options.smoother == "none":
        return mesh
    if options.smoother == "taubin":
        smoother = TaubinSmoother(
            iterations=options.iterations,
            lamb=options.lamb,
            nu=options.nu,
        )
    else:
        smoother = LaplacianSmoother(iterations=options.iterations, lamb=options.lamb)
    return smoother.process(mesh)


def decimate(mesh: ScalpMesh, options: DecimateOptions) -> ScalpMesh:
    """Reduce the mesh density by ``options.density_percent``."""
    decimator = QuadraticDecimator(density_percent=options.density_percent)
    return decimator.process(mesh)


def clean(mesh: ScalpMesh, options: CleanOptions) -> ScalpMesh:
    """Split, filter and merge the mesh into a watertight scalp surface."""
    cleaner = TrimeshCleaner(
        min_component_vertices=options.min_component_vertices,
        merge_digits=options.merge_digits,
    )
    return cleaner.process(mesh)


def generate_ese(mesh: ScalpMesh, options: EseOptions) -> ESEMesh:
    """Estimate outward normals and offset the scalp surface into the ESE mesh."""
    builder = PCAESEBuilder(
        ese_offset_mm=options.ese_offset_mm,
        neighborhood_radius_mm=options.neighborhood_radius_mm,
        k_neighbors=options.k_neighbors,
        use_weighted_pca=options.use_weighted_pca,
        pca_sigma_mm=options.pca_sigma_mm,
        min_neighbors=options.min_neighbors,
    )
    return builder.process(mesh)


def localize(
    surface: ESEMesh | ScalpMesh,
    fiducials: Fiducials,
    electrodes: Electrodes,
    options: LocalizeOptions,
) -> Electrodes:
    """Localize real electrodes on an ESE or scalp surface by brute-force search."""
    localizer = BruteForceLocalizer(
        calibrate_ese_offset=options.calibrate_ese_offset,
        residual_threshold_mm=options.residual_threshold_mm,
    )
    return localizer.process(
        surface=surface,
        fiducials=fiducials,
        electrodes=electrodes,
    )