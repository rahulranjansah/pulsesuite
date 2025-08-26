import numpy as np


class space:

    def __init__(self):
        self.dim
        self.Nx
        self.Ny
        self.Nz
        self.dx
        self.dy
        self.dz
        self.eps0

    def init(self, dim, Nx, Ny, Nz, dx, dy, dz, eps0):
        self.dim = dim
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.eps0 = eps0


# --- Get Parameters --- #
    def GetDim(self):
        return self.dim
    
    def GetNx(self):
        return self.Nx
    
    def GetNy(self):
        return self.Ny
    
    def GetNz(self):
        return self.Nz
    
    def Getdx(self):
        return self.dx
    
    def Getdy(self):
        return self.dy
    
    def Getdz(self):
        return self.dz
    
    def GetEps0(self):
        return self.eps0
    
# --- Set Parameters --- #
    def SetDim(self, dim):
        self.dim = dim
        return self.dim
    
    def SetNx(self, Nx):
        self.Nx = Nx
        return self.Nx
    
    def SetNy(self, Ny):
        self.Ny = Ny
        return self.Ny
    
    def SetNz(self, Nz):
        self.Nz = Nz
        return self.Nz
    
    def Setdx(self, dx):
        self.dx = dx
        return self.dx
    
    def Setdy(self, dy):
        self.dy = dy
        return self.dy
    
    def Setdz(self, dz):
        self.dz = dz
        return self.dz
    
    def SetEps0(self, eps0):
        self.eps0 = eps0
        return self.eps0

    
    # ------ Get Derived Parameters ------ #

    def GetVolume(self):
        return self.dx*self.dy*self.dz
    

    # --- Get Dimension Arrays --- #
    def GetXArray(self):
        return np.linspace(0, self.Nx*self.dx, self.Nx)
    
    def GetYArray(self):
        return np.linspace(0, self.Ny*self.dy, self.Ny)
    
    def GetZArray(self):
        return np.linspace(0, self.Nz*self.dz, self.Nz)
    

    # --- Get Widths --- #
    def GetXWidth(self):
        return self.Nx*self.dx
    
    def GetYWidth(self):
        return self.Ny*self.dy  
    
    def GetZWidth(self):
        return self.Nz*self.dz
    

    # --- Get k-space Arrays --- #
    def GetKxArray(self):
        dkx = 2*np.pi/(self.Nx*self.dx)
        if self.Nx % 2 == 0:
            return np.linspace(-self.Nx/2*dkx, (self.Nx/2-1)*dkx, self.Nx)
        else:
            return np.linspace(-(self.Nx-1)/2*dkx, (self.Nx-1)/2*dkx, self.Nx)
        
    def GetKyArray(self):
        dky = 2*np.pi/(self.Ny*self.dy)
        if self.Ny % 2 == 0:
            return np.linspace(-self.Ny/2*dky, (self.Ny/2-1)*dky, self.Ny)
        else:
            return np.linspace(-(self.Ny-1)/2*dky, (self.Ny-1)/2*dky, self.Ny)
        
    def GetKzArray(self):
        dkz = 2*np.pi/(self.Nz*self.dz)
        if self.Nz % 2 == 0:
            return np.linspace(-self.Nz/2*dkz, (self.Nz/2-1)*dkz, self.Nz)
        else:
            return np.linspace(-(self.Nz-1)/2*dkz, (self.Nz-1)/2*dkz, self.Nz)
        

    # --- Get dQ --- #
    def GetDQx(self):
        return 2* np.pi /self.GetXWidth()
    
    def GetDQy(self):
        return 2 * np.pi /self.GetYWidth()
    
    def GetDQz(self):
        return 2 * np.pi /self.GetZWidth()
    


    
