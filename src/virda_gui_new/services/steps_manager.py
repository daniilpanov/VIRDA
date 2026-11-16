"""Per-step pipeline contexts with completion state and change notification."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from lib.forward_list import ForwardList
from virda.models.mesh import Mesh
from virda_gui_new.dto.steps import BASE_MESH, MESH_POSTPROCESSING, RUN_INPUTS


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

    def set_step_context_items(self, step_n: int, items: Mapping[str, Any]) -> None:
        """Write several context keys of a step and notify once.

        The listeners rebuild what they show from the contexts, so one
        write of several keys means one rebuild instead of one per key.
        """
        self._validate_step(step_n)
        for key, item in items.items():
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

    def missing_inputs(self, step_n: int) -> tuple[str, ...]:
        """The inputs ``step_n`` needs to run that the manager does not hold.

        An empty tuple means the step can run. The step menu and the
        step popups read this answer, so they cannot disagree about
        whether a step is ready.
        """
        self._validate_step(step_n)
        missing: list[str] = []
        for requirement in RUN_INPUTS.get(step_n, ()):
            if not any(self._has_context(step, key) for step, key in requirement):
                missing.append(requirement[0][1])
        return tuple(missing)

    def _has_context(self, step_n: int, key: str) -> bool:
        """Whether the context of ``step_n`` holds a value for ``key``."""
        try:
            return self.get_step_context_item(step_n, key) is not None
        except ValueError:
            return False

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

    def active_scalp_mesh(self) -> Mesh | None:
        """The scalp mesh of the current state.

        The postprocessed step-2 mesh replaces the raw step-1 scalp
        surface, so it is the one the scene shows and the one the
        localization and the export use. A step-2 value that is not a
        mesh leaves the step without a mesh: it is not taken for the
        raw scalp surface.
        """
        try:
            final = self.get_step_context_item(MESH_POSTPROCESSING.index, "mesh")
        except ValueError:
            final = None
        if final is not None:
            return final if isinstance(final, Mesh) else None
        try:
            base = self.get_step_context_item(BASE_MESH.index, "scalp_surface")
        except ValueError:
            return None
        return base if isinstance(base, Mesh) else None

    def all_contexts(self) -> list[dict[str, Any]]:
        """Copies of the step contexts in step order."""
        return [dict(context) for context in self._step_contexts]

    def subscribe(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[], None]) -> None:
        """Drop a listener; one that died must not be called again."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()
