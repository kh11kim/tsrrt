import numpy as np
from spatial_math_mini import SO3, SE3
import kdtree
from collections.abc import Sequence

""" TODO : consider removing dependency of spatial_math_mini
"""

class StateSpace:
    pos_ll = [-1, -1, 0]
    pos_ul = [1, 1, 1]
    def __init__(self):
        pass
    
    @staticmethod
    def _get_sample():
        t = np.zeros(3)
        for i in range(3):
            ll, ul = StateSpace.pos_ll[i], StateSpace.pos_ul[i]
            t[i] = np.random.uniform(low=ll, high=ul)
        qtn = SO3._uniform_sampling_quaternion()
        return SE3(qtn, t)
    
    def is_valid(self):
        return True
    
class Node(StateSpace, Sequence):
    def __init__(self, SE3_):
        self.pos = SE3_.t
        self.ori = SE3_._qtn
        self.idx = None
        super().__init__()
    
    @staticmethod
    def random():
        return Node(StateSpace._get_sample())
    
    def __getitem__(self, i):
        return self.pos[i]
    
    def __len__(self):
        return len(self.pos)
    
    def __repr__(self):
        return "pos:{} ori:{}".format(self.pos, self.ori)

    
class Tree:
    def __init__(self, root):
        #init
        self._kdtree = kdtree.create(dimensions=3)
        self._parent_of = {0:None} # parent of 0(root) is None
        root.idx = 0
        self._data = [root]
        self._kdtree.add(root)
        self._num = 1
        
        #config
        self._rebalance_term = 20
    
    def add(self, parent, child):
        child.idx = self._num
        self._data.append(child)
        self._kdtree.add(child)
        self._parent_of[child.idx] = parent.idx
        self._num += 1
        
        if self._num % self._rebalance_term == 0:
            root = self._kdtree.rebalance()
    
    def nearest(self, node):
        node, dist = self._kdtree.search_nn(node)
        return node.data