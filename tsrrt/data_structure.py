import numpy as np
from spatial_math_mini import SO3, SE3
from abc import ABC, abstractmethod
from tsrrt.kinematics import PandaKinematics
import heapq
#from collections.abc import Sequence

class Node:
    """
    This class defines the structure of a node.
    """
    def __init__(self, pos, ori, q=None):
        self.pos = pos
        self.ori = ori
        self.q = q
        self.idx = None # this is used for indexing in a tree

    def __repr__(self):
        return "pos:{}, ori:{}".format(self.pos, self.ori)


class StateSpace:
    """
    This class defines the node space and generate random node samples
    example:
        ss = StateSpace(Node)
    """
    # the weight for translate when calculating distance. trans_weight=1
    rot_weight = 0.2

    def __init__(self, node_class):
        self.pos_ll = [-1, -1, 0]
        self.pos_ul = [1, 1, 1]
        self._node_class = node_class
    
    @staticmethod
    def twist(node_from, node_to, is_weighted=False):
        SE3_from = SE3(node_from.ori, node_from.pos)
        SE3_to = SE3(node_to.ori, node_to.pos)
        screw, angle = (SE3_from.inv() @ SE3_to).to_twistangle()
        twist = screw * angle # [w, v]
        if is_weighted:
            twist[:3] *= StateSpace.rot_weight
        return twist
        
    @staticmethod
    def distance(node1, node2, is_weighted=False):
        twist = StateSpace.twist(node1, node2, is_weighted)
        return np.linalg.norm(twist)
    
    def _get_random_pos(self):
        pos = np.empty(3)
        for i in range(3):
            ll, ul = self.pos_ll[i], self.pos_ul[i]
            pos[i] = np.random.uniform(low=ll, high=ul)
        return pos
    
    def _get_random_ori(self):
        return  SO3._uniform_sampling_quaternion()
    
    def _get_random_q():
        q = np.empty(7)
        for i in range(7):
            ll, ul = StateSpace.q_ll[i], StateSpace.q_ul[i]
            q[i] = np.random.uniform(low=ll, high=ul)
        return q
    
    def random(self, cspace=False):
        if cspace:
            #make C-space random variable
            raise NotImplementedError()
        pos, ori = self._get_random_pos(), self._get_random_ori()
        return self._node_class(pos, ori)
    
    def is_valid(self, node):
        for i in range(3):
            if not self.pos_ll[i] <= node.pos[i] <= self.pos_ul[i]:
                return False
        return True
    
class Tree:
    """
    This class defines the node space and generate random node samples
    example:
        tree = Tree(root, state_space=ss)
    """
    def __init__(self, root, state_space):
        #init
        self._ss = state_space
        self._parent_of = {0:None} # parent of 0(root) is None
        root.idx = 0
        self._data = [root]
        self._num = 1

    def add(self, node, parent):
        node.idx = self._num
        self._data.append(node)
        self._parent_of[node.idx] = parent.idx
        self._num += 1
    
    def nearest(self, node, is_weighted=True):
        ## TODO : implement kd-tree for SE3
        heap = []
        for i, node_tree in enumerate(self._data):
            distance = self._ss.distance(node, node_tree, is_weighted=is_weighted)
            heapq.heappush(heap, (distance, i))
        return self._data[heap[0][1]]
    
    def parent(self, node):
        assert node.idx is not None
        return self._data[self._parent_of[node.idx]]
    
    def backtrack(self, node_last):
        path = [node_last]
        child_idx = node_last.idx
        while True:
            parent_idx = self._parent_of[child_idx]
            if parent_idx is None:
                break
            path.append(self._data[parent_idx])
            child_idx = parent_idx
        return path[::-1]


    

