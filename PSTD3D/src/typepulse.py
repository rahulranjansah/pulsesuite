import numpy as np

class pulse:

    def __init__(self):

        self.lambda0    # Wavelength
        self.E00        # Peak Amplitude
        self.Tw         # Pulse Width
        self.Tp         # Time the pulse peak passes through the origin (s)
        self.chirp      # Chirp parameter
        self.pol        # Polarization angle (rad)

    def init(self, lambda0, E00, Tw, Tp, chirp, pol):
        self.lambda0 = lambda0
        self.E00 = E00
        self.Tw = Tw
        self.Tp = Tp
        self.chirp = chirp
        self.pol = pol

    # --- Read Parameters --- #
    def ReadPulseParams(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#') or line.strip() == '':
                    continue
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                if key == 'lambda0':
                    self.lambda0 = float(value)
                elif key == 'E00':
                    self.E00 = float(value)
                elif key == 'Tw':
                    self.Tw = float(value)
                elif key == 'Tp':
                    self.Tp = float(value)
                elif key == 'chirp':
                    self.chirp = float(value)
                elif key == 'pol':
                    self.pol = float(value)
        return (self.lambda0, self.E00, self.Tw, self.Tp, self.chirp, self.pol)


    # --- Get Parameters --- #
    def GetLambda0(self):
        return self.lambda0
    
    def GetE00(self):
        return self.E00
    
    def GetTw(self):
        return self.Tw
    
    def GetTp(self):
        return self.Tp
    
    def GetChirp(self):
        return self.chirp
    
    def GetPol(self):
        return self.pol
    
    # --- Set Parameters --- #
    def SetLambda0(self, lambda0):
        self.lambda0 = lambda0
        return self.lambda0
    
    def SetE00(self, E00):
        self.E00 = E00
        return self.E00
    
    def SetTw(self, Tw):
        self.Tw = Tw
        return self.Tw
    
    def SetTp(self, Tp):
        self.Tp = Tp
        return self.Tp
    
    def SetChirp(self, chirp):
        self.chirp = chirp
        return self.chirp
    
    def SetPol(self, pol):
        self.pol = pol
        return self.pol
    

    # --- Write Parameters --- #
    def WritePulseParams(self, filename):
        with open(filename, 'w') as f:
            f.write('# Pulse Parameters\n')
            f.write(f'lambda0 = {self.lambda0}\n')
            f.write(f'E00 = {self.E00}\n')
            f.write(f'Tw = {self.Tw}\n')
            f.write(f'Tp = {self.Tp}\n')
            f.write(f'chirp = {self.chirp}\n')
            f.write(f'pol = {self.pol}\n')

    
    # --- Calculate Frequency --- #
    def GetOmega0(self):
        c = 299792458
        omega0 = 2*np.pi*c/self.lambda0
        return omega0
    
    def Getk0(self):
        c = 299792458
        k0 = 2*np.pi/self.lambda0
        return k0
    
    def GetDOmega(self):
        domega = 2/self.Tw
        return domega
    
