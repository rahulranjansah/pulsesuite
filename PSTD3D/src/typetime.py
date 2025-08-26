import numpy as np

class time:

    def __initi__(self):
        self.t           # Current Time
        self.dt          # Time Step 
        self.tf          # Final Time
        self.n           # Current Step


    def init(self, t, dt, tf, n):
        self.t = t
        self.dt = dt
        self.tf = tf
        self.n = n

    # --- Get Parameters --- #
    def GetTime(self):
        return self.t
    
    def Getdt(self):
        return self.dt
    
    def Gettf(self):
        return self.tf
    
    def Getn(self):
        return self.n
    
    # --- GetNt --- #
    def GetNt(self):
        return int(self.tf/self.dt) + 1
    

    # --- Update Time --- #
    def UpdateTime(self):
        self.t = self.t + self.dt
        self.n = self.n + 1
        return self.t, self.n
    
    def UpdateStep(self):
        self.n = self.n + 1
        self.t = self.n*self.dt
        return self.t, self.n4