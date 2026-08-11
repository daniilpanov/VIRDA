import logging
from pathlib import Path

import numpy as np

from virda.fiducials.provider import FiducialProvider
from virda.io.exporter.contracts import Exporter
from virda.io.loader.contracts import MRILoader
from virda.mesh.contracts import MeshCleaner, MeshExtractor, MeshSmoother
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.models.fiducial import Fiducial
from virda.models.stage1_result import Stage1Result
from virda.segmentation.cap_cut import cut_mask, cut_plane_from_fiducials
from virda.segmentation.contracts import HeadSegmenter
from virda.segmentation.seal import seal_mask

logger = logging.getLogger(__name__)


class Stage1Pipeline:
    def __init__(
        self,
        loader: MRILoader,
        segmenter: HeadSegmenter,
        extractor: MeshExtractor | None = None,
        cleaners: list[MeshCleaner] | None = None,
        smoother: MeshSmoother | None = None,
        exporter: Exporter | None = None,
        fiducial_provider: FiducialProvider | None = None,
        seal: bool = False,
        seal_radius: int = 4,
        cutoff: bool = False,
        cutoff_below_nasion_mm: float = 30.0,
    ):
        self._loader = loader
        self._segmenter = segmenter
        self._extractor = extractor or MarchingCubesExtractor()
        self._cleaners = list(cleaners or [])
        self._smoother = smoother
        self._exporter = exporter
        self._fiducial_provider = fiducial_provider
        self._seal = seal
        self._seal_radius = seal_radius
        self._cutoff = cutoff
        self._cutoff_below_nasion_mm = cutoff_below_nasion_mm

    def run(
        self,
        path: str | Path,
        output_dir: str | Path | None = None,
        closing_radius: int = 5,
        threshold: float | None = None,
    ) -> Stage1Result:
        mri_volume = self._loader.load(path)
        segmentation_mask = self._segmenter.segment(
            mri_volume, closing_radius=closing_radius, threshold=threshold
        )
        if self._seal:
            segmentation_mask = seal_mask(segmentation_mask, self._seal_radius)
        anchor_fiducials: list[Fiducial] | None = None
        if self._cutoff:
            segmentation_mask, anchor_fiducials = self._apply_cutoff(
                segmentation_mask, mri_volume.affine
            )

        raw_mesh = self._extractor.extract(segmentation_mask, mri_volume.affine)
        mesh = raw_mesh
        for cleaner in self._cleaners:
            mesh = cleaner.clean(mesh, mask=segmentation_mask, affine=mri_volume.affine)
        if self._smoother is not None:
            mesh = self._smoother.smooth(mesh)
        fiducials: list[Fiducial] = []
        if self._fiducial_provider is not None:
            if anchor_fiducials is not None:
                fiducials = anchor_fiducials
            else:
                fiducials = self._fiducial_provider.fiducials(mesh)
        result = Stage1Result(
            mri_volume=mri_volume,
            segmentation_mask=segmentation_mask,
            mesh=mesh,
            fiducials=fiducials,
        )
        if output_dir is not None:
            if self._exporter is None:
                raise ValueError("output_dir requires an exporter to be configured on the pipeline")
            self._exporter.export(result, output_dir)
        return result

    def _apply_cutoff(
        self, segmentation_mask: np.ndarray, affine: np.ndarray
    ) -> tuple[np.ndarray, list[Fiducial] | None]:
        """Drop everything below the LPA-RPA-NAS cap-cut plane.

        The plane needs fiducials, so they are detected on a cheap raw mesh of
        the sealed mask first. If the fiducial provider is unavailable or the
        required fiducials are missing, the mask is returned unchanged with a
        warning. The detected fiducials are returned alongside so the pipeline
        reuses them as the result fiducials (the flat cut bottom confuses
        detection on the final mesh).
        """
        if self._fiducial_provider is None:
            logger.warning("Cap cut enabled but no fiducial provider configured; skipping cut")
            return segmentation_mask, None
        try:
            raw_mesh = self._extractor.extract(segmentation_mask, affine)
            anchor_fiducials = self._fiducial_provider.fiducials(raw_mesh)
            cut_plane_from_fiducials(anchor_fiducials, self._cutoff_below_nasion_mm)
        except ValueError as exc:
            logger.warning("Cap cut skipped: %s", exc)
            return segmentation_mask, None
        cut_mask_result = cut_mask(
            segmentation_mask,
            affine,
            anchor_fiducials,
            self._cutoff_below_nasion_mm,
        )
        return cut_mask_result, anchor_fiducials
