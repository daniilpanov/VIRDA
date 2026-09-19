"""Parser for NIfTI MRI volume files."""

from pathlib import Path

from virda.io.loader.nifti_loader import NiftiLoader
from virda.models.mri_volume import MRIVolume
from virda.models.path import NiftiPath
from virda.pipeline_context import PipelineContext


def parse_mri_volume(path: str | Path) -> MRIVolume:
    """Load a NIfTI file into an :class:`MRIVolume`.

    Delegates to :class:`virda.io.loader.nifti_loader.NiftiLoader`, the same
    loader the Stage 1 pipeline uses.
    """
    context = PipelineContext({})
    context.stores[NiftiPath] = NiftiPath(Path(path))
    return NiftiLoader().run(context)