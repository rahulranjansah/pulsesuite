import numpy as np
import matplotlib.pyplot as plt
import pulsesuite.PSTD3D.coulomb as coulomb
from scipy.constants import e as e0, hbar


L = 200e-9              # Wire length (m)
Delta0 = 5e-9           # Wire thickness (m)
N_y = 128               # Spatial grid points
N_k = 64                # Momentum grid points
m_e = 0.067 * 9.109e-31 # Electron effective mass (kg)
m_h = 0.45 * 9.109e-31  # Hole effective mass (kg)
epsr = 12.0             # Relative dielectric constant
ge = 1e12               # Electron dephasing rate (Hz)
gh = 1e12               # Hole dephasing rate (Hz)

# Confinement parameters (related to harmonic oscillator length)
HO = 0.1 * e0           # Energy level separation (J)
alphae = np.sqrt(m_e * HO) / hbar  # Electron confinement (1/m)
alphah = np.sqrt(m_h * HO) / hbar  # Hole confinement (1/m)

# Create spatial and momentum grids
y = np.linspace(-L/2, L/2, N_y)
ky = np.linspace(-2e8, 2e8, N_k)
dk = ky[1] - ky[0]

# Single-particle energies
E_e = hbar**2 * ky**2 / (2 * m_e)
E_h = hbar**2 * ky**2 / (2 * m_h)

print(f"Spatial grid: {y.min():.2e} to {y.max():.2e} m")
print(f"Momentum grid: {ky.min():.2e} to {ky.max():.2e} m^-1")
print(f"Electron confinement: {alphae:.2e} m^-1")
print(f"Hole confinement: {alphah:.2e} m^-1")