from pathlib import Path

from virda.io.exporter.contracts import Exporter
from virda.io.loader.contracts import MRILoader
from virda.mesh.contracts import MeshCleaner, MeshExtractor, MeshSmoother
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.models.stage1_result import Stage1Result
from virda.segmentation.contracts import HeadSegmenter


class Stage1Pipeline:
    def __init__(
        self,
        loader: MRILoader,
        segmenter: HeadSegmenter,
        extractor: MeshExtractor | None = None,
        cleaners: list[MeshCleaner] | None = None,
        smoother: MeshSmoother | None = None,
        exporter: Exporter | None = None,
    ):
        self._loader = loader
        self._segmenter = segmenter
        self._extractor = extractor or MarchingCubesExtractor()
        self._cleaners = list(cleaners or [])
        self._smoother = smoother
        self._exporter = exporter

    def run(
        self,
        path: str | Path,
        output_dir: str | Path | None = None,
        closing_radius: int = 5,
    ) -> Stage1Result:
        mri_volume = self._loader.load(path)
        segmentation_mask = self._segmenter.segment(mri_volume, closing_radius=closing_radius)
        raw_mesh = self._extractor.extract(segmentation_mask, mri_volume.affine)
        mesh = raw_mesh
        for cleaner in self._cleaners:
            mesh = cleaner.clean(mesh, mask=segmentation_mask, affine=mri_volume.affine)
        if self._smoother is not None:
            mesh = self._smoother.smooth(mesh)
        result = Stage1Result(
            mri_volume=mri_volume,
            segmentation_mask=segmentation_mask,
            mesh=mesh,
        )
        if output_dir is not None:
            if self._exporter is None:
                raise ValueError("output_dir requires an exporter to be configured on the pipeline")
            self._exporter.export(result, output_dir)
        return result
