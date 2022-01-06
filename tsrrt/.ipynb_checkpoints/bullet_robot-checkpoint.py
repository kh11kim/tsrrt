import pybullet as p
import pybullet_data
import numpy as np
import time

""" view_pose(), clear() functions are bullet utility function for debugging purpose.
"""
def view_pose(client, T):
    
    length = 0.1
    xaxis = np.array([length, 0, 0, 1])
    yaxis = np.array([0, length, 0, 1])
    zaxis = np.array([0, 0, length, 1])
    T_axis = np.array([xaxis, yaxis, zaxis]).T
    axes = T @ T_axis
    orig = T[:3,-1]
    xaxis = axes[:-1,0]
    yaxis = axes[:-1,1]
    zaxis = axes[:-1,2]
    x = client.addUserDebugLine(orig,xaxis, lineColorRGB=[1,0,0], lineWidth=5)
    y = client.addUserDebugLine(orig,yaxis, lineColorRGB=[0,1,0], lineWidth=5)
    z = client.addUserDebugLine(orig,zaxis, lineColorRGB=[0,0,1], lineWidth=5)
    pose_id = [x, y, z]
    return pose_id

def clear(client):
    for i in range(100):
        client.removeUserDebugItem(i)

class PandaBullet:
    def __init__(self, client):
        """ Init panda class
        """
        self.client = client
        self.robot = self.client.loadURDF("./urdf/panda.urdf", useFixedBase=True)
        # Simulation Configuration
        pos, ori = [0, 0, 0], [0, 0, 0, 1]
        self.client.resetBasePositionAndOrientation(self.robot, pos, ori)
        #create a constraint to keep the fingers centered
        c = self.client.createConstraint(self.robot,
                        9,
                        self.robot,
                        10,
                        jointType=self.client.JOINT_GEAR,
                        jointAxis=[1, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
        self.client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        # Set robot information
        self._all_joints = range(self.client.getNumJoints(self.robot))
        self._arm_joints = [i for i in range(7)]
        self._finger_joints = [9,10]
        
        self._movable_joints = self._arm_joints + self._finger_joints
        self._ee_idx = 11
        self._joint_info = self.get_joint_info()

    """ Configuration
    """
    def get_joint_attribute_names(self):
        return ["joint_index","joint_name","joint_type",
                "q_index", "u_index", "flags", 
                "joint_damping", "joint_friction","joint_lower_limit",
                "joint_upper_limit","joint_max_force","joint_max_velocity",
                "link_name","joint_axis","parent_frame_pos","parent_frame_orn","parent_index"]

    def get_joint_info(self):
        result = {}
        attribute_names = self.get_joint_attribute_names()
        for i in self._all_joints:
            values = self.client.getJointInfo(self.robot, i)
            result[i] = {name:value for name, value in zip(attribute_names, values)}
        return result

    def set_joint_positions(self, joint_positions):
        """ hard set of joint position (not control)
        """
        assert len(joint_positions) == 7
        for joint_idx, joint_value in enumerate(joint_positions):
            self.client.resetJointState(self.robot, joint_idx, joint_value, targetVelocity=0)

    """ Low-level controller
    """
    def set_control_mode(self, mode_str="position"):
        n = len(self._arm_joints)
        if mode_str == "position":
            joint_positions = self.get_arm_states()["position"]
            positionGains = np.ones(n) * 0.01
            self.client.setJointMotorControlArray(self.robot, self._arm_joints, self.client.POSITION_CONTROL,
                                                  targetPositions=joint_positions, positionGains=positionGains)
        else: #velocity or torque
            for joint in self._movable_joints:
                self.client.setJointMotorControl2(self.robot, joint, self.client.VELOCITY_CONTROL,
                                                  targetVelocity=0, force=0)

    def control_joint_positions(self, joint_positions, joint_indexes=None):
        if joint_indexes is None:
            joint_indexes = self._arm_joints
            assert len(joint_indexes) == len(joint_positions)
        
        positionGains = np.ones(7) * 0.01
        self.client.setJointMotorControlArray(self.robot, joint_indexes, controlMode=self.client.POSITION_CONTROL,
                                              targetPositions=joint_positions, positionGains=positionGains)
                                    
    def control_joint_torques(self, joint_torques, joint_indexes=None):
        if joint_indexes is None:
            joint_indexes = self._arm_joints
            assert len(joint_indexes) == len(joint_torques)
        joint_torques = list(joint_torques)
        self.client.setJointMotorControlArray(self.robot, joint_indexes, controlMode=self.client.TORQUE_CONTROL,
                                              forces=joint_torques)

    """ Robot states
    """
    def get_states(self, target="arm"):
        if target == "arm":
            joint_indexes = self._arm_joints
        elif target == "finger":
            joint_indexes = self._finger_joints
        else: #all movable joints(arm+finger)
            joint_indexes = self._movable_joints
        result = {}
        state_names = ["position", "velocity", "wrench", "effort"]
        joint_states = self.client.getJointStates(self.robot, joint_indexes)
        for i, name in enumerate(state_names):
            result[name] = [states[i] for states in joint_states]
        return result

    def get_arm_states(self):
        return self.get_states(target="arm")
    
    def get_finger_states(self):
        return self.get_states(target="finger")

    def get_all_states(self):
        return self.get_states(target="all")

    def get_movable_states(self):
        return self.get_states(target="movable")
    
    """ Robot Kinematics
    """
    def get_link_pose(self, link_index):
        result = self.client.getLinkState(self.robot, link_index)
        pos, ori = np.array(result[0]), np.array(result[1])
        R = np.array(self.client.getMatrixFromQuaternion(ori)).reshape((3,3))
        return np.block([[R,pos[:,None]],[np.zeros(3),1]])

    def get_link_velocity(self, link_index):
        result = self.client.getLinkState(self.robot, link_index, computeLinkVelocity=True)
        lin_vel, ang_vel = result[6], result[7]
        return np.array([*lin_vel, *ang_vel])

    def get_ee_pose(self):
        return self.get_link_pose(self._ee_idx)

    def get_ee_wrench(self):
        pass

    def get_ee_velocity(self):
        return self.get_link_velocity(self._ee_idx)
    
    def get_body_jacobian(self):
        states = self.get_movable_states()
        n = len(self._movable_joints)
        trans, rot = self.client.calculateJacobian(bodyUniqueId=self.robot,
                                      linkIndex=11,
                                      localPosition=[0,0,0],
                                      objPositions=states["position"],
                                      objVelocities=np.zeros(n).tolist(),
                                      objAccelerations=np.zeros(n).tolist())
        return np.vstack([trans, rot])[:,:-2] #remove finger joint part

    def IK(self):
        pass


if __name__ == "__main__":
    uid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # for loading plane
    
    # Simulation configuration
    rate = 240.
    p.setTimeStep(1/rate)
    p.resetSimulation() #init
    p.setGravity(0, 0, -9.8) #set gravity
    
    # Load
    plane_id = p.loadURDF("plane.urdf") # load plane
    panda = PandaBullet(p) # load robot
    
    time.sleep(3)
    
    #T = panda.get_link_pose(11)
    #jac = panda.get_body_jacobian()
    #view_pose(p, T)

    while(1):
        panda.set_control_mode(mode_str="position")
        panda.control_joint_positions([0,0,0,0,0,1,1])
        print(panda.get_body_jacobian())
        print()
        #panda.set_control_mode(mode_str="torque")
        #panda.control_joint_torques([0,0,0,0,0,1,100])
        
        p.stepSimulation()
        time.sleep(1/rate)
