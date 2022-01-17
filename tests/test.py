import unittest
import numpy as np

from tsrrt.data_structure import Node, Tree, StateSpace

class StateSpaceTest(unittest.TestCase):
    def setUp(self):
        print("setup")
        self.ss = StateSpace(Node)

    def test_random(self):
        a = self.ss.random()
        self.assertEqual(type(a), Node)

class TreeTest(unittest.TestCase):
    def setUp(self):
        print("setup")
        self.ss = StateSpace(Node)
        root = self.ss.random()
        self.tree = Tree(root, state_space=self.ss)

    def test_distance(self):
        node1 = self.ss.random()
        node2 = self.ss.random()
        self.ss.distance(node1, node2)

class RRTTest(unittest.TestCase):
    def setUp(self):
        pass
    #ss = StateSpace(Node)

if __name__ == '__main__':
    unittest.main()