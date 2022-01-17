import numpy as np
import pybullet as p
import os
from spatial_math_mini import SE3

class PandaConfig:
    def __init__(self):
        self._urdf_path = "../urdf/panda.urdf"
        self._urdf_path = os.path.join(os.path.dirname(__file__), self._urdf_path)
        self._all_joints = range(10)
        self._arm_joints = [i for i in range(7)]
        self._finger_joints = [9,10]
        self._movable_joints = self._arm_joints + self._finger_joints
        self._ee_idx = 11
        self._home_positions = [0,0,0,-np.pi/2,0,np.pi/2,np.pi/4]
        self._trans_eps = 0.05
        self._joint_lower_limits = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
        self._joint_upper_limits = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
        self._joint_mid_positions = [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0]
        self._joint_ranges = [5.9342, 3.6652, 5.9342, 3.1416, 5.9342, 3.9095999999999997, 5.9342]
    
    def get_joint_attribute_names(self):
        """ this string list used to parse p.getJointInfo()
        """
        return ["joint_index","joint_name","joint_type",
                "q_index", "u_index", "flags", 
                "joint_damping", "joint_friction","joint_lower_limit",
                "joint_upper_limit","joint_max_force","joint_max_velocity",
                "link_name","joint_axis","parent_frame_pos","parent_frame_orn","parent_index"]
    
    @staticmethod
    def xyzw_to_wxyz(ori):
        """ bullet: x,y,z,w  ->  w, x, y, z
        """
        return np.array([ori[-1], *ori[:3]])

    @staticmethod
    def wxyz_to_xyzw(ori):
        """  w, x, y, z -> bullet: x,y,z,w
        """
        return np.array([*ori[1:], ori[0]])

class PandaKinematics(PandaConfig):
    def __init__(self, client, uid, pos=None, ori=None):
        super().__init__()
        self._client = client
        self._uid = uid
        if pos is None:
            pos = [0, 0, 0.05]
        if ori is None:
            ori = [0, 0, 0, 1]
        self._robot = self._client.loadURDF(self._urdf_path, useFixedBase=True)
        self._client.resetBasePositionAndOrientation(self._robot, pos, ori)
        self._joint_info = self._get_joint_info()
        self.set_home_positions()
        self.eps = 0.01

    def _get_joint_info(self):
        result = {}
        attribute_names = self.get_joint_attribute_names()
        for i in self._all_joints:
            values = self._client.getJointInfo(self._robot, i)
            result[i] = {name:value for name, value in zip(attribute_names, values)}
        return result

    def get_states(self, target="arm"):
        if target == "arm":
            joint_indexes = self._arm_joints
        elif target == "finger":
            joint_indexes = self._finger_joints
        else: #all movable joints(arm+finger)
            joint_indexes = self._movable_joints
        result = {}
        state_names = ["position", "velocity", "wrench", "effort"]
        joint_states = self._client.getJointStates(self._robot, joint_indexes)
        for i, name in enumerate(state_names):
            result[name] = [states[i] for states in joint_states]
        return result
    
    def set_joint_positions(self, joint_positions):
        """ hard set of joint position (not control)
        """
        assert len(joint_positions) == 7
        for joint_idx, joint_value in enumerate(joint_positions):
            self._client.resetJointState(self._robot, joint_idx, joint_value, targetVelocity=0)

    def set_home_positions(self):
        self.set_joint_positions(self._home_positions)
    
    def get_arm_positions(self):
        result = self.get_states(target="arm")["position"]
        return np.array(result)
    
    def open(self):
        open_width = 0.08
        self._client.resetJointState(self._robot, 9, open_width/2, targetVelocity=0)
        self._client.resetJointState(self._robot, 10, open_width/2, targetVelocity=0)
    
    def get_link_pose(self, link_index):
        result = self._client.getLinkState(self._robot, link_index)
        pos, ori = np.array(result[0]), np.array(result[1])
        ori = self.xyzw_to_wxyz(ori)
        return pos, ori
    
    def get_ee_pose(self):
        return self.get_link_pose(self._ee_idx)
    
    def get_space_jacobian(self, joint_positions=None):
        """ input : [joint_positions] is len(7) iterable
        """
        if joint_positions is None:
            joint_positions = self.get_states(target="movable")["position"]
        else:
            assert len(joint_positions) == 7
            joint_positions = list(joint_positions) + [0, 0]
        n = len(self._movable_joints)
        trans, rot = self._client.calculateJacobian(bodyUniqueId=self._robot,
                                      linkIndex=11,
                                      localPosition=[0,0,0],
                                      objPositions=joint_positions,
                                      objVelocities=np.zeros(n).tolist(),
                                      objAccelerations=np.zeros(n).tolist())
        
        return np.vstack([rot, trans])[:,:-2] #remove finger joint part (2)

    def get_body_jacobian(self, joint_positions=None):
        """ joint_positions are len(7) iterable
        """
        space_jac = self.get_space_jacobian(joint_positions)
        if joint_positions is None:
            pos, ori = self.get_ee_pose()
        else:
            pos, ori = self.FK(joint_positions)
        return SE3(ori, pos).inv().to_adjoint() @ space_jac

    def FK(self, joint_positions):
        joint_positions_curr = self.get_arm_positions()
        self.set_joint_positions(joint_positions)
        pos, ori = self.get_ee_pose()
        self.set_joint_positions(joint_positions_curr)
        return pos, ori

    def IK(self, position, orientation=None):
        """ TODO : check the IK is success
        """
        success = False
        joint_positions_curr = self.get_arm_positions()
        self.set_home_positions()
        if orientation is None:
            result = p.calculateInverseKinematics(self._robot, 11, targetPosition=list(position))
            q = np.array(result)[:-2]
            self.set_joint_positions(q)
            pos, _ = self.get_ee_pose()
        else:
            result = p.calculateInverseKinematics(self._robot, 11, 
                                                  targetPosition=list(position),
                                                  targetOrientation=list(orientation))
            q = np.array(result)[:-2]
            self.set_joint_positions(q)
            pos, _ = self.get_ee_pose()
        if (np.linalg.norm(pos - position) < self._trans_eps):
                success = True
        #print(np.linalg.norm(pos - position),np.linalg.norm(ori - orientation) )
        self.set_joint_positions(joint_positions_curr)
        return success, q

if __name__ == "__main__":
    uid = p.connect(p.GUI)
    panda_kin = PandaKinematics(p, uid)
    panda_kin.set_joint_positions([0,0,0,3,0,0,0])
    a = panda_kin.get_body_jacobian()
    input()