import unittest

from lib.forward_list import ForwardList, ForwardListNodeIterator, Node


class TestForwardList(unittest.TestCase):
    def setUp(self):
        self.fl = ForwardList()

    def test_push_back_and_length(self):
        self.assertEqual(self.fl._length, 0)

        self.fl.push_back("A")
        self.assertEqual(self.fl._length, 1)
        self.assertEqual(self.fl._root_node.payload, "A")  # type: ignore[union-attr]
        self.assertEqual(self.fl._last_node.payload, "A")  # type: ignore[union-attr]

        self.fl.push_back("B")
        self.assertEqual(self.fl._length, 2)
        self.assertEqual(self.fl._root_node.payload, "A")  # type: ignore[union-attr]
        self.assertEqual(self.fl._last_node.payload, "B")  # type: ignore[union-attr]

    def test_push_front(self):
        self.fl.push_front("B")
        self.fl.push_front("A")

        self.assertEqual(self.fl._length, 2)
        self.assertEqual(self.fl._root_node.payload, "A")  # type: ignore[union-attr]
        self.assertEqual(self.fl._last_node.payload, "B")  # type: ignore[union-attr]

    def test_from_list(self):
        data = [10, 20, 30, 40]
        self.fl.init_from_list(data)

        self.assertEqual(self.fl._length, 4)
        self.assertEqual(list(self.fl), data)
        self.assertEqual(self.fl._root_node.payload, 10)  # type: ignore[union-attr]
        self.assertEqual(self.fl._last_node.payload, 40)  # type: ignore[union-attr]

    def test_forward_list_iterator(self):
        self.fl.init_from_list(["X", "Y", "Z"])
        result = list(self.fl)
        self.assertEqual(result, ["X", "Y", "Z"])

    def test_node_iterator_internals(self):
        self.fl.init_from_list([1, 2])
        node_iter = ForwardListNodeIterator(self.fl._root_node)

        node1 = next(node_iter)
        self.assertIsInstance(node1, Node)
        self.assertEqual(node1.payload, 1)

        node2 = next(node_iter)
        self.assertIsInstance(node2, Node)
        self.assertEqual(node2.payload, 2)

        with self.assertRaises(StopIteration):
            next(node_iter)

    def test_get_by_index_valid(self):
        self.fl.init_from_list(["First", "Second", "Third"])

        self.assertEqual(self.fl.get_by_index(0), "First")
        self.assertEqual(self.fl.get_by_index(1), "Second")
        self.assertEqual(self.fl.get_by_index(2), "Third")

    def test_get_by_index_out_of_range(self):
        self.fl.init_from_list([1, 2, 3])

        with self.assertRaises(IndexError):
            self.fl.get_by_index(-1)

        with self.assertRaises(IndexError):
            self.fl.get_by_index(3)

    def test_remove_after(self):
        self.fl.init_from_list([10, 20, 30, 40, 50])

        self.fl.remove_after(2)

        self.assertEqual(self.fl._length, 3)
        self.assertEqual(list(self.fl), [10, 20, 30])
        self.assertEqual(self.fl._last_node.payload, 30)  # type: ignore[union-attr]
        self.assertIsNone(self.fl._last_node.next_node)  # type: ignore[union-attr]
