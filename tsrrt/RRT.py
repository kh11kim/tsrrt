import numpy as np
from spatial_math_mini import SO3, SE3
import datetime
from tsrrt.data_structure import Node, StateSpace, Tree
import copy
from tsrrt.util import *
import time

class RRT:
    def __init__(self, start, goal, checker, bias=None, debug=False):
        self._ss = StateSpace(Node)
        self._tree = Tree(start, self._ss)
        self._start = start
        self._goal = goal
        self._checker = checker
        self._last_node = None
        self._is_debug = debug
        self._bias = bias

        #configuration
        self._goal_bias = 0.05
        self.eps = 0.05
        #self.stuck_eps = self.eps/10
        self.q_delta_max = 0.2 # local planner max dist
        self.DLS_damping = 0.2
    
    def control(self, node1, node2):
        """
        local planner using DLS method
        """
        dist = self._ss.distance(node1, node2, is_weighted=True)
        is_reached = is_advanced = is_collision = False
        curr_dist = dist
        node_curr = node1
        for i in range(20):
            if is_advanced | is_reached:
                return Node(pos, ori, q_new)
            if is_collision:
                return None

            twist = self._ss.twist(node_curr, node2, is_weighted=True)
            jac = self._checker.get_body_jacobian(node_curr.q)
            U, s, VT = np.linalg.svd(jac)
            dls = np.zeros((7,6))
            for i in range(6):
                u, v = U[:,i], VT[i,:]
                dls += s[i]/(s[i]**2+self.DLS_damping**2) * np.outer(v, u)
            q_delta = self._clamp_max(dls @ twist, self.q_delta_max)
            q_new = q_delta + node_curr.q
            pos, ori = self._checker.FK(q_new)
            
            #update
            node_new = Node(pos, ori, q_new)
            curr_dist = self._ss.distance(node_new, node2, is_weighted=True)
            is_advanced = dist - curr_dist > self.eps
            is_collision = self._checker.is_self_collision(q_new)
            is_reached = curr_dist < self.eps
            node_curr = node_new
            #print("dist:{}, curr:{}".format(dist, curr_dist))

            #debug
            if self._is_debug:
                self._checker.set_joint_positions(q_new)
                time.sleep(0.1)
        #return node_curr

    def extend(self, node_rand):
        node_tree = self._tree.nearest(node_rand)
        node_new = self.control(node_tree, node_rand)
        if node_new is not None:
            self._tree.add(node_new, node_tree)
            if self._ss.distance(node_new, self._goal, is_weighted=True) < self.eps:
                self._last_node = node_new
                return
        return None
    
    def plan(self):
        success = False
        for i in range(1000):

            if np.random.random() < self._goal_bias: 
                node_rand = self._goal
            else:
                if self._bias == "normal":
                    node_rand = self.sample_normal_bias()
                else:
                    node_rand = self._ss.random()
            view_node(node_rand)
            self.extend(node_rand)
            if self.is_goal_reached():
                print("goal!")
                success = True
                return success, self._tree.backtrack(self._last_node)
    
    def is_goal_reached(self):
        return self._last_node is not None

    def sample_free_space(self):
        q_curr = self._checker.get_arm_positions()
        while True:
            node = self._ss.random()
            _, q = self._checker.IK(node.pos, node.ori)
            self._checker.set_joint_positions(q)
            pos, ori = self._checker.get_ee_pose()
            is_collision = self._checker.is_self_collision()
            if not is_collision:
                break
        self._checker.set_joint_positions(q_curr)
        return Node(pos, ori, q)

    def sample_normal_bias(self):
        for i in range(100):
            pos = np.empty(3)
            for i in range(3):
                ll, ul = self._ss.pos_ll[i], self._ss.pos_ul[i]
                pos[i] = np.random.normal(loc=self._goal.pos[i], scale=((ul-ll)/6))
            ori = self._ss._get_random_ori()
            node_rand = Node(pos, ori)
            if self._ss.is_valid(node_rand):
                return node_rand
        raise Exception("random generation over 100")

    @staticmethod
    def _clamp_max(qs, qmax):
        mag = np.linalg.norm(qs)
        if mag < qmax:
            return qs
        else:
            return qmax * qs/mag

    @staticmethod
    def _clamp_max_abs(qs, qmax):
        mag = np.linalg.norm(qs,np.inf)
        if mag < qmax:
            return qs
        else:
            return qmax * qs/mag