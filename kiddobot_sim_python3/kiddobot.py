import vrep
import numpy as np
from myvrepsim import VrepSim

class Kiddobot():
    def __init__(self):
        print('Kiddobot class')
        self.joint_names=['mz', 'mz2', 'pmz']
    def start_sim(self):
        self.sim=VrepSim()
        self.sim.connect_and_start()
        e,self.h=self.sim.get_obj_handle(self.joint_names[0])
        e,self.h2=self.sim.get_obj_handle(self.joint_names[1])
        e,self.h3=self.sim.get_obj_handle(self.joint_names[2])
    def set_base(self, deg):
        e=self.sim.set_joint_deg(self.h, deg)
        return e
    def set_elbow(self, deg):
        e=self.sim.set_joint_deg(self.h2, deg)
        return e
    def set_base_elbow(self, deg, deg2):
        e=self.set_base(deg)
        e2=self.set_elbow(deg2)
        return e*e2
    def get_joint_pos(self):
        theta=self.sim.get_joint_deg(self.h)
        theta2=self.sim.get_joint_deg(self.h2)
        return theta, theta2
    def pen_down(self):
        print('pen down to do')
        e=self.sim.set_joint_pos(self.h3, -4.600e-02)
        return e
    def pen_up(self):
        print('pen up to do')
        e=self.sim.set_joint_pos(self.h3, 0)
        return e
    def go_def(self):
        e=self.set_base_elbow(30, 30)
        return e
    def close(self):
        self.sim.close()