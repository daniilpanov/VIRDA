from pathlib import Path

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
        cleaner: MeshCleaner | None = None,
        smoother: MeshSmoother | None = None,
    ):
        self._loader = loader
        self._segmenter = segmenter
        self._extractor = extractor or MarchingCubesExtractor()
        self._cleaner = cleaner
        self._smoother = smoother

    def run(self, path: str | Path, closing_radius: int = 5) -> Stage1Result:
        mri_volume = self._loader.load(path)
        segmentation_mask = self._segmenter.segment(mri_volume, closing_radius=closing_radius)
        raw_mesh = self._extractor.extract(segmentation_mask, mri_volume.affine)
        mesh = raw_mesh
        if self._cleaner is not None:
            mesh = self._cleaner.clean(mesh)
        if self._smoother is not None:
            mesh = self._smoother.smooth(mesh)
        return Stage1Result(
            mri_volume=mri_volume,
            segmentation_mask=segmentation_mask,
            mesh=mesh,
        )
