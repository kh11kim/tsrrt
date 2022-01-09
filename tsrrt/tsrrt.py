import numpy as np
from tsrrt.data_structure import Node, Tree
from spatial_math_mini import SO3, SE3
import datetime

class RRT:
    def __init__(self, start, goal):
        self._tree = Tree(start)
        self._start = start
        self._goal = goal
        self._last_node = None
        
        #config
        self._trans_step = 2/50
        self._rot_step = np.pi/50
        self._goal_bias = 0.05
    
    def interpolate(self, node1, node2):
        t1, t2 = node1.pos, node2.pos
        q1, q2 = node1.ori, node2.ori
        t_delta = t2 - t1
        lmda = q1@q2
        if lmda < 0:
            q2, lmda = -q2, -lmda
        if np.abs(1 - lmda) < 0.001:
            # the quaternions are nearly parallel, so use
            # linear interpolation
            num = int(np.linalg.norm(t_delta) / self._trans_step)
            rates = np.linspace(0,1,num)
            rr =  1 - rates
            ss = rates
        else:
            # slerp
            alpha = np.arccos(lmda)
            angle_delta = 2*alpha
            gamma = 1/np.sin(alpha)
            # calculate step number
            num_by_trans = int(np.linalg.norm(t_delta) / self._trans_step)
            num_by_angle = int(angle_delta / self._rot_step)
            num = max(num_by_trans, num_by_angle)
            rates = np.linspace(0,1,num)
            rr = np.sin((1-rates) * alpha) * gamma
            ss = np.sin(rates * alpha) * gamma
        
        # interpolate
        qq = q1*rr[:,None] + q2*ss[:,None]
        qq = [q/np.linalg.norm(q) for q in qq]
        tt = t1 + t_delta * rates[:,None]
        return [Node(SE3(q,t)) for t, q in zip(tt, qq)]
        
    def extend(self, node_new):
        node_near = self._tree.nearest(node_new)
        if np.allclose(node_near, node_new):
            return
        node_list = self.interpolate(node_near, node_new)
        parent = node_near
        for node in node_list:
            if not node.is_valid():
                break
            self._tree.add(parent, node)
            trans_delta, angle_delta = self.distances(self._goal, node)
            if (trans_delta <= self._trans_step) & \
                (angle_delta <= self._rot_step):
                self._last_node = node
                break
            parent = node
    
    def distances(self, node1, node2):
        """
        calculate translational, rotional distances
        """
        t1, t2 = node1.pos, node2.pos
        q1, q2 = node1.ori, node2.ori
        t_delta = t2 - t1
        lmda = q1@q2
        if lmda < 0:
            q2, lmda = -q2, -lmda
        if np.abs(1 - lmda) < 0.001:
            angle_delta = 0
        else:
            alpha = np.arccos(lmda)
            angle_delta = 2*alpha
        return np.linalg.norm(t_delta), angle_delta
    
    def is_goal_found(self):
        return self._last_node is not None
    
    def plan(self, max_time=5.0):
        start_time = datetime.datetime.now()
        while not self.is_goal_found():
            if np.random.random() > self._goal_bias:
                node_random = Node.random()
            else:
                node_random = self._goal
            self.extend(node_random)

            time_delta = (datetime.datetime.now() - start_time)
            elapsed = time_delta.seconds + 0.000001 * time_delta.microseconds
            if elapsed > max_time:
                break

        if self.is_goal_found():
            print("planning success")
        else:
            print("planning fail")
        print("elapsed time : {}".format(elapsed))
        print("nodes:{}".format(self._tree._num))
        return self.is_goal_found(), self.path()

    def path(self):
        if not self.is_goal_found():
            return []
        path = [self._last_node]
        child_idx = self._last_node.idx
        while True:
            parent_idx = self._tree._parent_of[child_idx]
            if parent_idx is None:
                break
            path.append(self._tree._data[parent_idx])
            child_idx = parent_idx
        return path[::-1]