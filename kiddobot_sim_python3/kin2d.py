import numpy as np

class Kin2D():
    def __init__(self, l1=0.4, l2=0.5, verbose=True):
        self.version='March, 9, 2020'
        print('2D Kin. version=', self.version)
        self.verbose=verbose
        self.l1=l1
        self.l2=l2
        self.q1=0
        self.q2=0
        self.x=l1+l2
        self.y=0
        print('Link1=',l1,'mm', ' Link2=',l2,'mm')
    def calcFK(self, q1, q2):
        if self.verbose:
            print('calc fk, given thetas= ',q1, q2)
        q1=np.deg2rad(q1)
        q2=np.deg2rad(q2)
        self.q1=q1
        self.q2=q2
        self.x=self.l1* np.cos(q1) + self.l2* np.cos(q1+q2)
        self.y=self.l1* np.sin(q1) + self.l2* np.sin(q1+q2)
        return self.x, self.y
    def calcIK(self, x, y):
        if self.verbose:
            print('calc IK, given xy=',x,y)
        dd= x * x + y * y
        l1=self.l1
        l2=self.l2
        beta =np.arccos((l1 * l1 + l2 * l2 - dd) / (2 * l1 * l2))
        th2 = np.pi - beta;
        atan2 = np.arctan2(y, x)
        alpha = np.arccos((dd + l1 * l1 - l2 * l2) / (2 * np.sqrt(dd) * l1))
        th1 = atan2 - alpha;
        th1=np.rad2deg(th1)
        th2=np.rad2deg(th2)
        if np.isnan(th1) or np.isnan(th2):
            return False, th1, th2
        return True, th1,th2

def main():
	print('Kin2D test')
	kin=Kin2D(0.10, 0.11)
	print('x=',kin.x,' y=',kin.y)
	
if __name__=='__main__':
	main()
	