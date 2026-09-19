"""Parser for Stage 3 measurements JSON files."""

from pathlib import Path

from virda.io.loader.measurements_loader import MeasurementsLoaderFromJson
from virda.models.electrode import Electrodes
from virda.models.fiducial import Fiducials
from virda.models.path import MeasurementsPath
from virda.pipeline_context import PipelineContext


def parse_measurements(
    path: str | Path,
    fiducials: Fiducials | None = None,
) -> Electrodes:
    """Load a Stage 3 measurements JSON into an :class:`Electrodes` object.

    Delegates to :class:`MeasurementsLoaderFromJson` with a fresh pipeline
    context.  When the file carries ``fiducial_weights``, pass the reference
    :class:`Fiducials` so the weights are applied.
    """
    context = PipelineContext({})
    context.stores[MeasurementsPath] = MeasurementsPath(Path(path))
    if fiducials is not None:
        context.stores[Fiducials] = fiducials
    return MeasurementsLoaderFromJson().run(context)
