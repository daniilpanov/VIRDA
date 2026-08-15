import json
from dataclasses import replace
from typing import Any, cast

from virda.io.loader.contracts import MeasurementsLoader
from virda.models.electrode import Electrode, Electrodes
from virda.models.fiducial import Fiducials
from virda.models.path import MeasurementsPath
from virda.pipeline_context import PipelineContext


class MeasurementsLoaderFromJson(MeasurementsLoader):
    """Load a Stage 3 measurements JSON into an :class:`Electrodes` store.

    Format::

        {
          "electrodes": [
            {"electrode_id": "Fz", "measured_distances": {"NAS": 120.5, "LPA": 131.2}}
          ],
          "fiducial_weights": {"NAS": 1.5}
        }

    ``fiducial_weights`` is optional; when present it overrides the weights of
    the :class:`Fiducials` store used by the localizer.
    """

    def _process(self, context: PipelineContext, path: MeasurementsPath) -> Electrodes:
        data = cast(
            dict[str, Any],
            json.loads(path.measurements_path.read_text(encoding="utf-8")),
        )
        self._apply_weights(context, data)
        return Electrodes(
            items=[
                Electrode(
                    electrode_id=item["electrode_id"],
                    measured_distances={
                        str(fiducial_id): float(distance)
                        for fiducial_id, distance in item["measured_distances"].items()
                    },
                )
                for item in data["electrodes"]
            ]
        )

    @staticmethod
    def _apply_weights(context: PipelineContext, data: dict[str, Any]) -> None:
        raw = data.get("fiducial_weights")
        if not raw:
            return
        fiducials = context.get_store_notnull(Fiducials)
        context.stores[Fiducials] = Fiducials(
            items=[
                replace(
                    fiducial,
                    weight=(
                        float(raw[fiducial.fiducial_id])
                        if fiducial.fiducial_id in raw
                        else fiducial.weight
                    ),
                )
                for fiducial in fiducials.items
            ]
        )
