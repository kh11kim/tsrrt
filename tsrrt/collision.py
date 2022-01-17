import numpy as np
from tsrrt.kinematics import PandaKinematics
from itertools import combinations

class PandaCollision(PandaKinematics):
    """ TODO : collision check with another objects
    """
    def __init__(self, client, uid, pos=None, ori=None):
        self._robot = None # init by PandaKinematics
        super().__init__(client, uid, pos, ori)
        self._collision_ignore_list = [(6,8), (8,10)]

    def panda_self_collision_list(self):
        self_collision_check_list = list(combinations(self._all_joints, 2))
        self_collision_check_list = [pair for pair in self_collision_check_list if abs(pair[0]-pair[1]) != 1]
        self_collision_check_list = [pair for pair in self_collision_check_list if pair not in self._collision_ignore_list]
        return self_collision_check_list
            
    def is_self_collision(self, joint_positions=None):
        is_collision = False
        if joint_positions is not None:
            joint_positions_curr = self.get_arm_positions()
            self.set_joint_positions(joint_positions)

        for link1, link2 in self.panda_self_collision_list():
            if abs(link1 - link2) == 1:
                continue
            dist_info = self._client.getClosestPoints(bodyA=self._robot, bodyB=self._robot, distance=0.0,
                                          linkIndexA=link1, linkIndexB=link2, physicsClientId=self._uid)
            if len(dist_info) != 0:
                is_collision = True
        
        if joint_positions is not None:
            self.set_joint_positions(joint_positions_curr)
        return is_collision
    