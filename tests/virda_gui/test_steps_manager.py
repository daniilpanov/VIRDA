"""Unit tests for :class:`virda_gui_new.services.steps_manager.StepsManager`."""

import pytest

from virda_gui_new.services.steps_manager import StepsManager


def test_init_creates_step_zero_context() -> None:
    manager = StepsManager((("a",), ("b",)))

    assert manager.get_current_step() == 0
    assert manager.get_step_context_item(0, "a") is None


def test_init_without_required_keys_starts_at_step_zero() -> None:
    manager = StepsManager()

    assert manager.get_current_step() == 0
    assert manager.is_step_done(0) is False


def test_next_step_advances_current_step() -> None:
    manager = StepsManager([(), ()])

    manager.next_step()

    assert manager.get_current_step() == 1
    manager.next_step()
    assert manager.get_current_step() == 2


def test_set_and_get_context_item() -> None:
    manager = StepsManager()

    manager.set_step_context_item(0, "volume", "data")

    assert manager.get_step_context_item(0, "volume") == "data"
    assert manager.get_step_context_item(0, "other") is None


def test_step_done_requires_all_keys() -> None:
    manager = StepsManager((("a", "b"), ()))

    manager.set_step_context_item(0, "a", 1)
    assert manager.is_step_done(0) is False

    manager.set_step_context_item(0, "b", 2)
    assert manager.is_step_done(0) is True


def test_step_done_false_for_step_without_required_keys_entry() -> None:
    # Five contexts exist but only two steps carry required keys.
    manager = StepsManager((("a",), ("b",)))
    for _ in range(4):
        manager.next_step()

    assert manager.is_step_done(3) is False
    assert manager.is_step_done(4) is False


def test_invalidate_steps_after_truncates_contexts() -> None:
    manager = StepsManager([(), (), ()])
    manager.set_step_context_item(0, "a", 1)
    manager.next_step()
    manager.set_step_context_item(1, "b", 2)
    manager.next_step()

    manager.invalidate_steps_after(0)

    assert manager.get_current_step() == 0
    assert manager.get_step_context_item(0, "a") == 1
    with pytest.raises(ValueError, match="out of bounds"):
        manager.get_step_context_item(1, "b")


def test_context_accessors_reject_out_of_bounds_steps() -> None:
    manager = StepsManager()

    for call in ("get", "set"):
        with pytest.raises(ValueError, match="out of bounds"):
            if call == "get":
                manager.get_step_context_item(1, "a")
            else:
                manager.set_step_context_item(1, "a", 1)

    with pytest.raises(ValueError, match="out of bounds"):
        manager.is_step_done(1)

    with pytest.raises(ValueError, match="out of bounds"):
        manager.invalidate_steps_after(1)

    with pytest.raises(ValueError, match="out of bounds"):
        manager.get_step_context_item(-1, "a")


def test_all_contexts_returns_ordered_copies() -> None:
    manager = StepsManager([(), ()])
    manager.set_step_context_item(0, "a", 1)
    manager.next_step()
    manager.set_step_context_item(1, "b", 2)

    contexts = manager.all_contexts()

    assert contexts == [{"a": 1}, {"b": 2}]
    contexts[0]["a"] = 999
    assert manager.get_step_context_item(0, "a") == 1
    assert manager.all_contexts() == [{"a": 1}, {"b": 2}]


def test_subscriber_notified_on_every_mutation() -> None:
    calls: list[int] = []
    manager = StepsManager()
    manager.subscribe(lambda: calls.append(len(calls)))

    assert calls == []
    manager.set_step_context_item(0, "a", 1)
    assert len(calls) == 1

    manager.next_step()
    assert len(calls) == 2

    manager.invalidate_steps_after(0)
    assert len(calls) == 3
