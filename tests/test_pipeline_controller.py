import pytest

from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext


class InputStore:
    def __init__(self, value: int) -> None:
        self.value = value


class FirstOutput:
    def __init__(self, value: int) -> None:
        self.value = value


class FinalOutput:
    def __init__(self, value: int) -> None:
        self.value = value


class Loader:
    def run(self, context: PipelineContext) -> FirstOutput:
        initial = context.get_store_notnull(InputStore)
        return FirstOutput(initial.value)


class Transformer:
    def run(self, context: PipelineContext) -> FinalOutput:
        first = context.get_store_notnull(FirstOutput)
        return FinalOutput(first.value * 2)


class ResultGenerator:
    def run(self, context: PipelineContext) -> FinalOutput:
        return context.get_store_notnull(FinalOutput)


class RecordingProvider:
    def __init__(self) -> None:
        self.received: list[object] = []

    def provide(self, store: object, context: PipelineContext) -> None:
        self.received.append(store)


class TestPipelineController:
    def test_run_runs_steps_in_order_and_returns_output(self) -> None:
        controller = PipelineController()
        controller.register_store(InputStore, InputStore(5))
        controller.register_step(Loader())
        controller.register_step(Transformer())
        controller.register_step(ResultGenerator())

        result = controller.run().get_store_notnull(FinalOutput)

        assert isinstance(result, FinalOutput)
        assert result.value == 10

    def test_register_provider_notified_when_store_type_produced(self) -> None:
        provider = RecordingProvider()
        controller = PipelineController()
        controller.register_store(InputStore, InputStore(3))
        controller.register_provider(provider, on_store=FirstOutput)
        controller.register_step(Loader())
        controller.register_step(Transformer())
        controller.register_step(ResultGenerator())

        controller.run()

        assert len(provider.received) == 1
        produced = provider.received[0]
        assert isinstance(produced, FirstOutput)
        assert produced.value == 3

    def test_output_generator_receives_populated_context(self) -> None:
        seen: list[PipelineContext] = []

        class CapturingGenerator:
            def run(self, context: PipelineContext) -> FinalOutput:
                seen.append(context)
                return context.get_store_notnull(FinalOutput)

        controller = PipelineController()
        controller.register_store(InputStore, InputStore(2))
        controller.register_step(Loader())
        controller.register_step(Transformer())
        controller.register_step(CapturingGenerator())

        controller.run()

        assert seen
        first_store = seen[0].stores[FirstOutput]
        assert first_store is not None
        assert first_store.value == 2
        final_store = seen[0].stores[FinalOutput]
        assert final_store is not None
        assert final_store.value == 4

    def test_pre_registered_store_initial_value_visible_to_steps(self) -> None:
        controller = PipelineController()
        controller.register_store(InputStore, InputStore(42))
        controller.register_step(Loader())

        result = controller.run().get_store_notnull(FirstOutput)

        assert result.value == 42


class TestPipelineContext:
    def test_get_store_notnull_raises_on_missing_store(self) -> None:
        context = PipelineContext({})

        with pytest.raises(KeyError):
            context.get_store_notnull(InputStore)

    def test_get_store_notnull_raises_on_none_value(self) -> None:
        context = PipelineContext({InputStore: None})

        with pytest.raises(KeyError):
            context.get_store_notnull(InputStore)

    def test_get_store_returns_none_for_missing_store(self) -> None:
        context = PipelineContext({})

        assert context.get_store(InputStore) is None
