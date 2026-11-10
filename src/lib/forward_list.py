from typing import Any, Self, cast


class Node[T = Any]:
    def __init__(self, payload: T):
        self.payload: T = payload
        self.next_node: Node[T] | None = None


class ForwardListNodeIterator[T = Any]:
    def __init__(self, root_node: Node[T] | None):
        self._current_node: Node[T] | None = root_node

    def __iter__(self) -> ForwardListNodeIterator[T]:
        return self

    def __next__(self) -> Node[T]:
        if self._current_node is None:
            raise StopIteration

        node = self._current_node
        self._current_node = self._current_node.next_node
        return node


class ForwardListIterator[T = Any]:
    def __init__(self, node_iterator: ForwardListNodeIterator[T]):
        self._node_iterator: ForwardListNodeIterator[T] = node_iterator

    def __iter__(self) -> ForwardListIterator[T]:
        return self

    def __next__(self) -> T:
        return next(self._node_iterator).payload


class ForwardList[T = Any]:
    @classmethod
    def from_list(cls, data: list[T]) -> Self:
        fl = cls()
        fl.init_from_list(data)
        return fl

    def __init__(self) -> None:
        self._root_node: Node[T] | None = None
        self._last_node: Node[T] | None = None
        self._length: int = 0

    def __iter__(self) -> ForwardListIterator[T]:
        return ForwardListIterator(ForwardListNodeIterator(self._root_node))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> T:
        return self.get_by_index(index)

    def _get_node_by_index(self, index: int) -> Node[T]:
        if index < 0 or index >= self._length:
            raise IndexError("Index out of range")

        iterator = ForwardListNodeIterator(self._root_node)

        current_node = None
        for _ in range(index + 1):
            current_node = next(iterator)

        return cast(Node[T], current_node)

    def get_by_index(self, index: int) -> T:
        return self._get_node_by_index(index).payload

    def _push_root_node(self, item: T) -> None:
        self._last_node = self._root_node = Node(item)
        self._length = 1

    def init_from_list(self, data: list[T]) -> None:
        for item in data:
            self.push_back(item)

    def push_front(self, item: T) -> None:
        if not self._root_node:
            return self._push_root_node(item)

        node = Node(item)
        node.next_node = self._root_node
        self._root_node = node
        self._length += 1

    def push_back(self, item: T) -> None:
        if not self._last_node:
            return self._push_root_node(item)

        node = Node(item)
        self._last_node.next_node = node
        self._last_node = node
        self._length += 1

    def remove_after(self, index: int) -> None:
        new_last_node = self._get_node_by_index(index)
        new_last_node.next_node = None
        self._last_node = new_last_node
        self._length = index + 1
