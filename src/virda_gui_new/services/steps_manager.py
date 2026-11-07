from typing import Any

from lib.forward_list import ForwardList


class StepsManager:
    def __init__(self, steps_count: int):
        self._steps_count: int = steps_count
        self._step_contexts: ForwardList[dict[str, Any]] = ForwardList()

    def get_current_step(self) -> int:
        return len(self._step_contexts) - 1

    def get_step_context_item(self, step_n: int, key: str) -> Any | None:
        return self._step_contexts[step_n].get(key)

    def set_step_context_item(self, step_n: int, key: str, item: Any):
        self._step_contexts[step_n][key] = item

    def next_step(self):
        self._step_contexts.push_back({})

    def invalidate_steps_after(self, step_n: int):
        self._step_contexts.remove_after(step_n)
