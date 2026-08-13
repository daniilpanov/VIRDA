from typing import Any

from virda.pipeline_context import PipelineContext


def build_context(**stores: Any) -> PipelineContext:
    return PipelineContext({type(value): value for value in stores.values()})
