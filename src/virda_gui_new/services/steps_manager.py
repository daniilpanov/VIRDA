"""Per-step pipeline contexts with completion state and change notification."""

from collections.abc import Callable, Sequence
from typing import Any

from lib.forward_list import ForwardList


class StepsManager:
    """Tracks one context dict per pipeline step and which steps are done.

    Step 0's empty context is created at construction, so
    ``get_current_step()`` is always >= 0 and step 0 accepts writes
    immediately. Context accessors and ``invalidate_steps_after`` raise
    ``ValueError`` for step numbers outside ``[0, current step]``;
    ``is_step_done`` follows the same rule and additionally returns
    ``False`` for valid steps that have no ``required_keys`` entry.
    Listeners are pull-model callbacks of no arguments, re-reading the
    manager after each mutation.

    ``next_step`` refuses to leave an unfinished step: every required
    key of the current step must be present in its context, or it
    raises ``ValueError``. A step without required keys (the live
    localization step) never blocks the advance. The invariant behind
    the rule: every step before the current one is finished.
    """

    def __init__(self, required_keys_by_step: Sequence[tuple[str, ...]] = ()) -> None:
        self._required_keys: tuple[tuple[str, ...], ...] = tuple(required_keys_by_step)
        self._step_contexts: ForwardList[dict[str, Any]] = ForwardList()
        self._listeners: list[Callable[[], None]] = []
        self._step_contexts.push_back({})

    def _validate_step(self, step_n: int) -> None:
        step_count = len(self._step_contexts)
        if step_n < 0 or step_n >= step_count:
            raise ValueError(
                f"Step {step_n} is out of bounds; valid range is [0, {step_count - 1}]"
            )

    def get_current_step(self) -> int:
        return len(self._step_contexts) - 1

    def get_step_context_item(self, step_n: int, key: str) -> Any | None:
        self._validate_step(step_n)
        return self._step_contexts[step_n].get(key)

    def set_step_context_item(self, step_n: int, key: str, item: Any) -> None:
        self._validate_step(step_n)
        self._step_contexts[step_n][key] = item
        self._notify()

    def is_step_done(self, step_n: int) -> bool:
        """True when every required context key of ``step_n`` is present.

        A step without required keys (or without a keys entry at all)
        is never done.
        """
        self._validate_step(step_n)
        if step_n >= len(self._required_keys):
            return False

        required = self._required_keys[step_n]
        if not required:
            # If a step has no required keys, it can't be done
            return False

        context = self._step_contexts[step_n]
        return all(key in context for key in required)

    def missing_required_keys(self, step_n: int) -> tuple[str, ...]:
        """Required keys of ``step_n`` that are absent from its context.

        Empty when the step needs no keys (or has no keys entry at
        all) and when every required key is present. Unlike
        :meth:`is_step_done`, a step without required keys counts as
        complete: it never blocks an advance.
        """
        self._validate_step(step_n)
        if step_n >= len(self._required_keys):
            return ()
        context = self._step_contexts[step_n]
        return tuple(key for key in self._required_keys[step_n] if key not in context)

    def can_advance(self) -> bool:
        """True when the current step holds all its required keys."""
        return not self.missing_required_keys(self.get_current_step())

    def next_step(self) -> None:
        """Advance to the next step.

        Raises ``ValueError`` when the current step is not finished:
        all of its required keys must be present in its context.
        """
        step_n = self.get_current_step()
        missing = self.missing_required_keys(step_n)
        if missing:
            raise ValueError(f"Step {step_n} is missing required keys: {', '.join(missing)}")
        self._step_contexts.push_back({})
        self._notify()

    def invalidate_steps_after(self, step_n: int) -> None:
        self._validate_step(step_n)
        self._step_contexts.remove_after(step_n)
        self._notify()

    def all_contexts(self) -> list[dict[str, Any]]:
        """Copies of the step contexts in step order."""
        return [dict(context) for context in self._step_contexts]

    def subscribe(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()
