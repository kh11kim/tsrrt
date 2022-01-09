import pybullet as p
import pybullet_data
import numpy as np
from itertools import combinations

class PandaCollision:
    def __init__(self, pos=None, ori=None, use_gui=False, ):
        if use_gui:
            self._uid = p.connect(p.GUI)
        else:
            self._uid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._robot_id = p.loadURDF("/urdf/panda.urdf", useFixedBase=True, physicsClientId=self._uid)
        p.resetBasePositionAndOrientation(self._robot_id, 
                                          [0, 0, 0.05], [0, 0, 0, 1], physicsClientId=self._uid)
        
        """table env setting"""
        #self._table_id = p.loadURDF("table/table.urdf", useFixedBase=True, physicsClientId=self._uid)
        #env setting
        
        # p.resetBasePositionAndOrientation(self._table_id, 
        #                                   [0.8, 0, -0.35], 
        #                                            p.getQuaternionFromEuler([0,0,3.14/2]), physicsClientId=self._uid)
        self._panda_link_list = range(11)

    def panda_self_collision_list(self):
        ignore_list = [(6,8), (8,10)]
        self_collision_check_list = list(combinations(self._panda_link_list, 2))
        self_collision_check_list = [pair for pair in self_collision_check_list if abs(pair[0]-pair[1]) != 1]
        self_collision_check_list = [pair for pair in self_collision_check_list if pair not in ignore_list]
        return self_collision_check_list
    
    def set_robot_joints(self, qs):
        for joint, joint_value in enumerate(qs):
            p.resetJointState(self._robot_id, joint, joint_value, targetVelocity=0, physicsClientId=self._uid)
            
    def is_self_collision(self):
        for link1, link2 in self.panda_self_collision_list():
            if abs(link1 - link2) == 1:
                continue
            dist_info = p.getClosestPoints(bodyA=self._robot_id, bodyB=self._robot_id, distance=0.0,
                                          linkIndexA=link1, linkIndexB=link2, physicsClientId=self._uid)
            if len(dist_info) != 0:
                return True
        return False