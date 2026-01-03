# #!/usr/bin/env python3
# """
# Test script for emission module to debug import issues.

# This script tests the basic import and initialization of the emission module.
# """

# import sys
# import os

# # Add src to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# print("="*60)
# print("Testing emission module import chain...")
# print("="*60)
# print(f"Python path (first 3): {sys.path[:3]}")

# # Test import chain step by step
# try:
#     print("\n1. Testing libpulsesuite.helpers import...")
#     from pulsesuite.libpulsesuite import helpers
#     print("   ✓ helpers import successful")
#     if hasattr(helpers, 'locator'):
#         print("   ✓ locator function found in helpers")
#     else:
#         print("   ✗ locator function NOT found in helpers")
# except Exception as e:
#     print(f"   ✗ helpers import failed: {type(e).__name__}: {e}")
#     import traceback
#     traceback.print_exc()
#     sys.exit(1)

# try:
#     print("\n2. Testing PSTD3D.usefulsubs import...")
#     from pulsesuite.PSTD3D import usefulsubs
#     print("   ✓ usefulsubs import successful")
# except Exception as e:
#     print(f"   ✗ usefulsubs import failed: {type(e).__name__}: {e}")
#     import traceback
#     traceback.print_exc()

# try:
#     print("\n3. Testing emission module import...")
#     import pulsesuite.PSTD3D.emission as emission
#     print("   ✓ Direct import successful")
# except Exception as e:
#     print(f"   ✗ Direct import failed: {type(e).__name__}: {e}")
#     print("\n   Full traceback:")
#     import traceback
#     traceback.print_exc()
#     sys.exit(1)

# try:
#     print("\n4. Testing import via PSTD3D (lazy import)...")
#     from pulsesuite.PSTD3D import emission as emission2
#     print("   ✓ Import via PSTD3D successful")
# except Exception as e:
#     print(f"   ✗ Import via PSTD3D failed: {type(e).__name__}: {e}")
#     import traceback
#     traceback.print_exc()

# try:
#     print("\n5. Checking available functions...")
#     funcs = [name for name in dir(emission) if not name.startswith('_') and callable(getattr(emission, name, None))]
#     print(f"   Available functions: {len(funcs)}")
#     print(f"   Functions: {', '.join(funcs[:15])}")
#     if len(funcs) > 15:
#         print(f"   ... and {len(funcs) - 15} more")
# except Exception as e:
#     print(f"   ✗ Failed to list functions: {type(e).__name__}: {e}")

# # Test if scipy is available before trying initialization
# try:
#     print("\n6. Checking dependencies...")
#     import numpy as np
#     print("   ✓ numpy available")
#     try:
#         from scipy.constants import e as e0, hbar
#         print("   ✓ scipy available")
#         scipy_available = True
#     except ImportError:
#         print("   ✗ scipy NOT available (this is expected in test environment)")
#         scipy_available = False

#     if scipy_available:
#         print("\n7. Testing basic initialization...")
#         # Basic parameters
#         N_k = 64
#         k_grid = np.linspace(-5e8, 5e8, N_k)
#         dcv = 1.0e-29
#         gap = 1.5 * e0
#         E_e = hbar**2 * k_grid**2 / (2 * 0.067 * 9.109e-31)
#         E_h = hbar**2 * k_grid**2 / (2 * 0.45 * 9.109e-31)
#         epsr = 12.0
#         geh = 1e12
#         ehint = 0.8

#         # Test InitializeEmission
#         if hasattr(emission, 'InitializeEmission'):
#             print("   InitializeEmission function found")
#             # Try calling it with proper parameters
#             try:
#                 emission.InitializeEmission(k_grid, E_e, E_h, dcv, epsr, geh, ehint)
#                 print("   ✓ InitializeEmission called successfully")
#                 if hasattr(emission, '_RScale'):
#                     print(f"   ✓ _RScale = {emission._RScale:.2e}")
#             except Exception as e:
#                 print(f"   ✗ InitializeEmission failed: {type(e).__name__}: {e}")
#                 import traceback
#                 traceback.print_exc()
#         else:
#             print("   ✗ InitializeEmission function not found")
#     else:
#         print("\n7. Skipping initialization test (scipy not available)")

# except Exception as e:
#     print(f"   ✗ Dependency check failed: {type(e).__name__}: {e}")
#     import traceback
#     traceback.print_exc()

# print("\n" + "="*60)
# print("Test complete!")
# print("="*60)

import numpy as np
import matplotlib.pyplot as plt
import pulsesuite.PSTD3D.emission as emission
from scipy.constants import e as e0, hbar, k as kB

# Physical Parameters for GaAs Quantum Wire
L_wire = 200e-9          # Wire length (m)
N_k = 128                # Momentum grid points
m_e = 0.067 * 9.109e-31  # Electron effective mass (kg)
m_h = 0.45 * 9.109e-31   # Hole effective mass (kg)
gap = 1.5 * e0           # Band gap energy (J)
dcv = 1.0e-29            # Dipole matrix element (C·m)
epsr = 12.0              # Relative dielectric constant
geh = 1e12               # Electron-hole dephasing rate (Hz)
ehint = 0.8              # Electron-hole interaction strength
Temp = 77.0              # Temperature (K)

# Initialize Momentum Grid
k_grid = np.linspace(-5e8, 5e8, N_k)  # Momentum grid (1/m)
dk = k_grid[1] - k_grid[0]

# Single-particle energies
E_e = hbar**2 * k_grid**2 / (2 * m_e)
E_h = hbar**2 * k_grid**2 / (2 * m_h)

print(f"Momentum grid: {k_grid.min():.2e} to {k_grid.max():.2e} m^-1")
print(f"Band gap: {gap/e0*1e3:.1f} meV")
print(f"Temperature: {Temp} K")