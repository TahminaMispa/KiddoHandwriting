import vrep
import numpy as np

class VrepSim():
    def __init__(self):
        self.version='March, 9, 2020'
        print('vrep simulated robot, version=',self.version)
    def connect_and_start(self):
        vrep.simxFinish(-1)
        self.clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
        if self.clientID!=-1:
            print ('Connected to remote API server') 
            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
        else:
            print ('Failed connecting to remote API server')
            return False
        return True
    def get_obj_handle(self, name):
        e,h=vrep.simxGetObjectHandle(self.clientID, name, vrep.simx_opmode_oneshot_wait)
        return e,h
    def set_joint_vel(self, handle, v):
        er= vrep.simxSetJointTargetVelocity(self.clientID, h, v, vrep.simx_opmode_streaming)
        return er
    def set_joint_pos(self, handle, pos): 
        er=vrep.simxSetJointTargetPosition(self.clientID, handle, pos, vrep.simx_opmode_streaming)
        return er
    def set_joint_deg(self, handle, theta):
        theta=np.deg2rad(theta)
        e=vrep.simxSetJointTargetPosition(self.clientID, handle , theta, vrep.simx_opmode_oneshot_wait)
        return e
    def get_joint_deg(self, handle):
        e, theta=vrep.simxGetJointPosition(self.clientID, handle, vrep.simx_opmode_oneshot_wait)
        theta=np.rad2deg(theta)
        return theta
    def close(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
        vrep.simxFinish(self.clientID)
        print('connection closed...')
