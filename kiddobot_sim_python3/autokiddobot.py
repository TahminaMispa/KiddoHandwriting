import sys
import time
import numpy as np
from kin2d import Kin2D 
from kiddobot import Kiddobot

class AutoKiddobot():
    def __init__(self):
        print('Kiddobot sim and kin auto')
        self.l1=200
        self.l2=180
        self.robot=Kiddobot()
        self.kin=Kin2D(self.l1, self.l2, verbose=False)
        self.x_offset= 150
        self.y_offset=120
        self.x0= 0
        self.y0=0
        self.xmax= 300
        self.ymax=200
        self.x=0
        self.y=0
        print('max xy=',self.xmax, self.ymax)
    def print_info(self):
        print('link: ',self.l1, self.l2)
        print('max xy=',self.xmax, self.ymax)
        
    def start(self):
        self.robot.start_sim()
        self.go_to_xy(0, 0)
    def go_to_xy(self, x, y):
        if x<0 or y<0 or x>self.xmax or y>self.ymax:
            print('xy out of range. max xy=',self.xmax, self.ymax)
            return False
        
        s, d1, d2=self.kin.calcIK( self.x_offset-x , y+self.y_offset)
        if not s:
            return False
        else:
            self.robot.set_base_elbow(d1, d2)
        self.x=x
        self.y=y
        return True
    def get_current_xy(self):
        d1, d2=self.robot.get_joint_pos()
        x,y=self.kin.calcFK(d1, d2)
        x=self.x_offset-x
        y=y-self.y_offset
        self.x=x
        self.y=y
        return x,y
    def pen_up(self):
        self.robot.pen_up()
    def pen_down(self):
        self.robot.pen_down()
    def close(self):
        self.robot.close()