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


def test_next_step_refuses_to_leave_an_unfinished_step() -> None:
    """The current step must hold its required keys to be left."""
    manager = StepsManager((("a",),))

    with pytest.raises(ValueError, match="missing required keys"):
        manager.next_step()

    assert manager.get_current_step() == 0


def test_next_step_advances_once_the_step_is_finished() -> None:
    manager = StepsManager((("a",),))
    manager.set_step_context_item(0, "a", 1)

    manager.next_step()

    assert manager.get_current_step() == 1


def test_missing_required_keys_lists_the_gaps() -> None:
    manager = StepsManager((("a", "b"), ()))
    manager.set_step_context_item(0, "a", 1)

    assert manager.missing_required_keys(0) == ("b",)

    manager.set_step_context_item(0, "b", 2)
    assert manager.missing_required_keys(0) == ()


def test_can_advance_follows_the_current_step() -> None:
    manager = StepsManager((("a",),))

    assert manager.can_advance() is False

    manager.set_step_context_item(0, "a", 1)
    assert manager.can_advance() is True


def test_a_step_without_required_keys_never_blocks_the_advance() -> None:
    """The live step has no required keys, so it is always left behind."""
    manager = StepsManager(((),))

    manager.next_step()

    assert manager.get_current_step() == 1
    assert manager.can_advance() is True
    assert manager.missing_required_keys(1) == ()


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
    manager.set_step_context_item(0, "a", 1)
    manager.next_step()
    manager.set_step_context_item(1, "b", 2)
    manager.next_step()
    manager.next_step()
    manager.next_step()

    assert manager.is_step_done(3) is False
    assert manager.is_step_done(4) is False


def test_step_done_false_for_empty_required_keys_entry() -> None:
    """A step with a ``()`` keys entry (live step) is never done."""
    manager = StepsManager((("a",), ()))
    manager.set_step_context_item(0, "a", 1)
    manager.next_step()

    assert manager.is_step_done(1) is False


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


# --- global store ---


def test_set_and_get_global_item() -> None:
    manager = StepsManager()

    manager.set_global_item("fiducials", "points")

    assert manager.get_global_item("fiducials") == "points"
    assert manager.get_global_item("other") is None


def test_global_item_survives_invalidation_and_next_step() -> None:
    """The store holds project-level inputs; step contexts do not."""
    manager = StepsManager()
    manager.set_global_item("fiducials", "points")
    manager.next_step()

    manager.invalidate_steps_after(0)

    assert manager.get_global_item("fiducials") == "points"
    assert manager.get_current_step() == 0


def test_setting_global_item_to_none_clears_it() -> None:
    manager = StepsManager()
    manager.set_global_item("fiducials", "points")

    manager.set_global_item("fiducials", None)

    assert manager.get_global_item("fiducials") is None


def test_global_item_mutation_notifies_subscribers() -> None:
    calls: list[int] = []
    manager = StepsManager()
    manager.subscribe(lambda: calls.append(len(calls)))

    manager.set_global_item("fiducials", "points")
    assert len(calls) == 1

    manager.set_global_item("fiducials", None)
    assert len(calls) == 2
