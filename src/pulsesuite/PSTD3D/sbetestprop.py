"""
SBETest program for testing Semiconductor Bloch Equations (SBE) calculations.

This program performs time-evolution of the SBEs for quantum wire systems,
calculating polarization responses to time-varying electric fields.

Converted from Fortran program SBETest.

Author: Rahul R. Sah
"""

import os

import numpy as np
from scipy.constants import c as c0_SI

from .SBEs import InitializeSBE, QWCalculator
from .typespace import GetKArray, GetSpaceArray

# Physical constants
c0 = c0_SI  # Speed of light (m/s)
twopi = 2.0 * np.pi  # 2π constant

# Simulation parameters
Nr = 100  # Number pixels along the quantum wire direction
drr = 10e-9  # Pixel size along the quantum wire direction (m)
n0 = 3.1  # Background refractive index
Nt = 10000  # Number pixels in time
dt = 10e-18  # Pixel size in time (s)
t = 0.0  # Time variable, starting at t=0 (s)

# Electric field parameters for X-direction
E0x = 1e7  # Peak Ex-field value (V/m)
twx = 10e-15  # Ex-field pulsewidth (s)
tpx = 50e-15  # Ex-field time of pulse peak value at origin (s)
lamX = 800e-9  # Field wavelength for Ex (m)

# Electric field parameters for Y-direction
# E0y = 0  # Peak Ey-field value (V/m)
# twy = 1  # Ey-field pulsewidth (s)
# tpy = 0  # Ey-field time of pulse peak value at origin (s)
# lamY = 0  # Field wavelength for Ey (m)

E0y = 2e7  # Peak Ey-field value (V/m)
twy = 10e-15  # Ey-field pulsewidth (s)
tpy = 50e-15  # Ey-field time of pulse peak value at origin (s)
lamY = 800e-9  # Field wavelength for Ey (m)

# Electric field parameters for Z-direction
E0z = 0  # Peak Ez-field value (V/m)
twz = 10e-15  # Ez-field pulsewidth (s)
tpz = 50e-15  # Ez-field time of pulse peak value at origin (s)
lamZ = 800e-9  # Field wavelength for Ez (m)

# Allocate all fields and initialize to zero
Exx = np.zeros(Nr, dtype=np.complex128)  # E-Field vector components along wire (V/m)
Eyy = np.zeros(Nr, dtype=np.complex128)
Ezz = np.zeros(Nr, dtype=np.complex128)

Pxx1 = np.zeros(Nr, dtype=np.complex128)  # Polarizations along wire 1 (C/m^2)
Pyy1 = np.zeros(Nr, dtype=np.complex128)
Pzz1 = np.zeros(Nr, dtype=np.complex128)

Pxx2 = np.zeros(Nr, dtype=np.complex128)  # Polarizations along wire 2 (C/m^2)
Pyy2 = np.zeros(Nr, dtype=np.complex128)
Pzz2 = np.zeros(Nr, dtype=np.complex128)

Pxx_mid = np.zeros(Nr, dtype=np.complex128)  # Polarizations along wire 1 and 2 mid_point avg (C/m^2)
Pyy_mid = np.zeros(Nr, dtype=np.complex128)
Pzz_mid = np.zeros(Nr, dtype=np.complex128)

Rho = np.zeros(Nr, dtype=np.complex128)  # Charge Density (C/m^2) along wire
Vrr = np.zeros(Nr, dtype=np.complex128)  # Potential (V) along wire
rr = np.zeros(Nr, dtype=np.float64)  # Spatial Array along the wire (m)
qrr = np.zeros(Nr, dtype=np.float64)  # Momentum arrays (rad/m)

# Angular frequencies and wave numbers (calculated later)
w0x = 0.0  # Angular time frequencies (rad/s)
w0y = 0.0
w0z = 0.0
k0x = 0.0  # Angular space frequencies (rad/m)
k0y = 0.0
k0z = 0.0
Tcx = 0.0  # Optical cycles (s)
Tcy = 0.0
Tcz = 0.0
Emax0 = 0.0  # Peak Electric field value (V/m)

# Dummy boolean values (as lists for QWCalculator which modifies them in-place)
boolF = [False]  # Dummy boolean value for false (list for in-place modification)
boolT = [True]  # Dummy boolean value for true (list for in-place modification)


def initializefields():
    """
    Initialize all field arrays to zero.

    Sets all electric field, polarization, charge density, and spatial arrays
    to zero as initial conditions.
    """
    global Exx, Eyy, Ezz, Pxx1, Pyy1, Pzz1, Pxx2, Pyy2, Pzz2
    global Pxx_mid, Pyy_mid, Pzz_mid, Rho, Vrr, rr, qrr

    Exx[:] = 0.0
    Eyy[:] = 0.0
    Ezz[:] = 0.0
    Pxx1[:] = 0.0
    Pyy1[:] = 0.0
    Pzz1[:] = 0.0
    Pxx2[:] = 0.0
    Pyy2[:] = 0.0
    Pzz2[:] = 0.0
    Pxx_mid[:] = 0.0
    Pyy_mid[:] = 0.0
    Pzz_mid[:] = 0.0
    Rho[:] = 0.0
    Vrr[:] = 0.0
    rr[:] = 0.0
    qrr[:] = 0.0


# Initialize fields
initializefields()

# Calculate angular frequencies, & optical cycle (for X-direction)
w0x = twopi * c0 / lamX
k0x = twopi / lamX * n0
Tcx = lamX / c0

# Calculate angular frequencies, & optical cycle (for Y-direction)
w0y = twopi * c0 / lamY
k0y = twopi / lamY * n0
Tcy = lamY / c0

# Calculate angular frequencies, & optical cycle (for Z-direction)
w0z = twopi * c0 / lamZ
k0z = twopi / lamZ * n0
Tcz = lamZ / c0

# Calculate the maximum field possible during the simulation
Emax0 = np.sqrt(E0x**2 + E0y**2 + E0z**2)

# Calculate the real-space & q-space array
rr = GetSpaceArray(Nr, (Nr - 1) * drr)
qrr = GetKArray(Nr, Nr * drr)

# Initialize the SBEs in SBEs.py
InitializeSBE(qrr, rr, 0.0, Emax0, lamX, 2, True)

# Create output directory
os.makedirs('fields', exist_ok=True)

# Open files to record data
file_Ex = open('fields/Ex.dat', 'w', encoding='utf-8')
file_Ey = open('fields/Ey.dat', 'w', encoding='utf-8')
file_Ez = open('fields/Ez.dat', 'w', encoding='utf-8')
file_Px1 = open('fields/Px1.dat', 'w', encoding='utf-8')
file_Py1 = open('fields/Py1.dat', 'w', encoding='utf-8')
file_Pz1 = open('fields/Pz1.dat', 'w', encoding='utf-8')
file_Px2 = open('fields/Px2.dat', 'w', encoding='utf-8')
file_Py2 = open('fields/Py2.dat', 'w', encoding='utf-8')
file_Pz2 = open('fields/Pz2.dat', 'w', encoding='utf-8')
file_Px_mid = open('fields/Px_mid.dat', 'w', encoding='utf-8')
file_Py_mid = open('fields/Py_mid.dat', 'w', encoding='utf-8')
file_Pz_mid = open('fields/Pz_mid.dat', 'w', encoding='utf-8')

# Begin Time-Evolving the SBEs
for n in range(1, Nt + 1):
    # Update the user on the command line
    print(n, Nt)

    # Calculate E-fields
    Exx[:] = (E0x * np.exp(-(w0x * (t - tpx))**2 / (w0x * twx)**2) *
              np.cos(w0x * (t - tpx)) *
              np.exp(-(w0x * (t - tpx))**20 / (2 * twx * w0x)**20))

    Eyy[:] = (E0y * np.exp(-(w0y * (t - tpy))**2 / (w0y * twy)**2) *
              np.cos(w0y * (t - tpy)) *
              np.exp(-(w0y * (t - tpy))**20 / (2 * twy * w0y)**20))

    # Eyy = E0y * exp(-(t-tpy)**2 / (twy)**2) * cos(w0y*(t-tpy)) * exp(-(t-tpy)**20 / (2*twy)**20)
    # Ezz = E0z * exp(-(w0z*(t-tpz))**2 / (w0z*twz)**2) * cos(w0z*(t-tpz)) * exp(-(w0z*(t-tpz))**20 / (2*twz*w0z)**20)

    # Time-Evolve the SBEs from t(n) to t(n+1)
    QWCalculator(Exx, Eyy, Ezz, Vrr, rr, qrr, dt, 1, Pxx1, Pyy1, Pzz1, Rho, boolT, boolF)
    QWCalculator(Exx, Eyy, Ezz, Vrr, rr, qrr, dt, 2, Pxx2, Pyy2, Pzz2, Rho, boolT, boolF)

    # Print*, "AAA"
    # stop

    Pxx_mid[:] = (Pxx1 + Pxx2) * 0.5
    Pyy_mid[:] = (Pyy1 + Pyy2) * 0.5
    Pzz_mid[:] = (Pzz1 + Pzz2) * 0.5

    # Compute mid-point average polarization at each pixel
    # for i in range(Nr):
    #     Pxx_mid[i] = (Pxx1[i] + Pxx2[i]) / 2.0
    #     Pyy_mid[i] = (Pyy1[i] + Pyy2[i]) / 2.0
    #     Pzz_mid[i] = (Pzz1[i] + Pzz2[i]) / 2.0

    # Print the electric field for the record
    file_Ex.write(f"{t} {np.real(Exx[Nr // 2])}\n")  # Record to file in 'fields/Ex.dat'
    file_Ey.write(f"{t} {np.real(Ezz[Nr // 2])}\n")  # Record to file in 'fields/Ey.dat' (matches Fortran unit=445)
    file_Ez.write(f"{t} {np.real(Eyy[Nr // 2])}\n")  # Record to file in 'fields/Ez.dat' (matches Fortran unit=446)
    file_Px1.write(f"{t} {np.real(Pxx1[Nr // 2])}\n")  # Record to file in 'fields/Px1.dat'
    file_Py1.write(f"{t} {np.real(Pyy1[Nr // 2])}\n")  # Record to file in 'fields/Py1.dat'
    file_Pz1.write(f"{t} {np.real(Pzz1[Nr // 2])}\n")  # Record to file in 'fields/Pz1.dat'
    file_Px2.write(f"{t} {np.real(Pxx2[Nr // 2])}\n")  # Record to file in 'fields/Px2.dat'
    file_Py2.write(f"{t} {np.real(Pyy2[Nr // 2])}\n")  # Record to file in 'fields/Py2.dat'
    file_Pz2.write(f"{t} {np.real(Pzz2[Nr // 2])}\n")  # Record to file in 'fields/Pz2.dat'
    file_Px_mid.write(f"{t} {np.real(Pxx_mid[Nr // 2])}\n")  # Record to file in 'fields/Px_mid.dat'
    file_Py_mid.write(f"{t} {np.real(Pyy_mid[Nr // 2])}\n")  # Record to file in 'fields/Py_mid.dat'
    file_Pz_mid.write(f"{t} {np.real(Pzz_mid[Nr // 2])}\n")  # Record to file in 'fields/Pz_mid.dat'

    t = t + dt

# Close files
file_Ex.close()
file_Ey.close()
file_Ez.close()
file_Px1.close()
file_Py1.close()
file_Pz1.close()
file_Px2.close()
file_Py2.close()
file_Pz2.close()
file_Px_mid.close()
file_Py_mid.close()
file_Pz_mid.close()

# Add after line 227 (after closing files):

# ============================================================================
# CUDA USAGE STATISTICS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("CUDA/JIT USAGE STATISTICS")
print("="*80)

try:
    from qwoptics import _HAS_CUDA as qwoptics_cuda, _cuda_stats
    from SBEs import _HAS_CUDA as sbes_cuda, _sbe_cuda_stats

    print("\nCUDA Status:")
    print(f"  qwoptics.py: {'✓ CUDA AVAILABLE' if qwoptics_cuda else '✗ CUDA NOT AVAILABLE'}")
    print(f"  SBEs.py:     {'✓ CUDA AVAILABLE' if sbes_cuda else '✗ CUDA NOT AVAILABLE'}")

    print("\nqwoptics.py Function Calls:")
    print("  QWPolarization3:")
    print(f"    CUDA:     {_cuda_stats.get('QWPolarization3_cuda', 0):8d} calls")
    print(f"    JIT:      {_cuda_stats.get('QWPolarization3_jit', 0):8d} calls")
    print(f"    Fallback: {_cuda_stats.get('QWPolarization3_fallback', 0):8d} calls")
    print("  QWRho5:")
    print(f"    CUDA:     {_cuda_stats.get('QWRho5_cuda', 0):8d} calls")
    print(f"    JIT:      {_cuda_stats.get('QWRho5_jit', 0):8d} calls")
    print(f"    Fallback: {_cuda_stats.get('QWRho5_fallback', 0):8d} calls")

    print("\nSBEs.py Function Calls:")
    print("  dpdt:")
    print(f"    CUDA:     {_sbe_cuda_stats.get('dpdt_cuda', 0):8d} calls")
    print(f"    JIT:      {_sbe_cuda_stats.get('dpdt_jit', 0):8d} calls")
    print(f"    Fallback: {_sbe_cuda_stats.get('dpdt_fallback', 0):8d} calls")
    print("  dCdt:")
    print(f"    CUDA:     {_sbe_cuda_stats.get('dCdt_cuda', 0):8d} calls")
    print(f"    JIT:      {_sbe_cuda_stats.get('dCdt_jit', 0):8d} calls")
    print(f"    Fallback: {_sbe_cuda_stats.get('dCdt_fallback', 0):8d} calls")
    print("  dDdt:")
    print(f"    CUDA:     {_sbe_cuda_stats.get('dDdt_cuda', 0):8d} calls")
    print(f"    JIT:      {_sbe_cuda_stats.get('dDdt_jit', 0):8d} calls")
    print(f"    Fallback: {_sbe_cuda_stats.get('dDdt_fallback', 0):8d} calls")

    # Summary
    total_cuda = (_cuda_stats.get('QWPolarization3_cuda', 0) +
                  _cuda_stats.get('QWRho5_cuda', 0) +
                  _sbe_cuda_stats.get('dpdt_cuda', 0) +
                  _sbe_cuda_stats.get('dCdt_cuda', 0) +
                  _sbe_cuda_stats.get('dDdt_cuda', 0))

    total_jit = (_cuda_stats.get('QWPolarization3_jit', 0) +
                 _cuda_stats.get('QWRho5_jit', 0) +
                 _sbe_cuda_stats.get('dpdt_jit', 0) +
                 _sbe_cuda_stats.get('dCdt_jit', 0) +
                 _sbe_cuda_stats.get('dDdt_jit', 0))

    total_fallback = (_cuda_stats.get('QWPolarization3_fallback', 0) +
                      _cuda_stats.get('QWRho5_fallback', 0) +
                      _sbe_cuda_stats.get('dpdt_fallback', 0) +
                      _sbe_cuda_stats.get('dCdt_fallback', 0) +
                      _sbe_cuda_stats.get('dDdt_fallback', 0))

    total = total_cuda + total_jit + total_fallback

    if total > 0:
        print("\nSummary:")
        print(f"  Total function calls: {total}")
        print(f"  CUDA:     {total_cuda:8d} ({100*total_cuda/total:.1f}%)")
        print(f"  JIT:      {total_jit:8d} ({100*total_jit/total:.1f}%)")
        print(f"  Fallback: {total_fallback:8d} ({100*total_fallback/total:.1f}%)")

        if total_cuda > 0:
            print(f"\n✓ CUDA IS BEING USED! ({total_cuda} calls)")
            print("  The 'Grid size' warnings are normal - they indicate CUDA is active.")
            print("  Small grid sizes mean your arrays are small, but CUDA still helps.")
        elif total_jit > 0:
            print("\nUsing JIT (CPU parallel) - CUDA not available or failed")
        else:
            print("\n✗ Using Python fallback - JIT/CUDA not working")

except ImportError as e:
    print(f"Could not import CUDA statistics: {e}")
    print("  Statistics may not be available if modules were modified.")
except Exception as e:
    print(f"Error reading CUDA statistics: {e}")
    import traceback
    traceback.print_exc()

print("="*80 + "\n")

# deallocate(Exx,Eyy,Ezz,Pxx,Pyy,Pzz,Rho,rr,qrr)
# Arrays are automatically deallocated when program exits



# """
# SBETest program for testing Semiconductor Bloch Equations (SBE) calculations.

# This program performs time-evolution of the SBEs for quantum wire systems,
# calculating polarization responses to time-varying electric fields.

# Converted from Fortran program SBETest.

# Author: Rahul R. Sah
# """

# import numpy as np
# import os
# import time
# from scipy.constants import c as c0_SI
# from typespace import GetSpaceArray, GetKArray
# from SBEs import InitializeSBE, QWCalculator

# # Physical constants
# c0 = c0_SI  # Speed of light (m/s)
# twopi = 2.0 * np.pi  # 2π constant

# # Simulation parameters
# Nr = 100  # Number pixels along the quantum wire direction
# drr = 10e-9  # Pixel size along the quantum wire direction (m)
# n0 = 3.1  # Background refractive index
# Nt = 10000  # Number pixels in time
# dt = 10e-18  # Pixel size in time (s)
# t = 0.0  # Time variable, starting at t=0 (s)

# # Electric field parameters for X-direction
# E0x = 1e7  # Peak Ex-field value (V/m)
# twx = 10e-15  # Ex-field pulsewidth (s)
# tpx = 50e-15  # Ex-field time of pulse peak value at origin (s)
# lamX = 800e-9  # Field wavelength for Ex (m)

# # Electric field parameters for Y-direction
# E0y = 2e7  # Peak Ey-field value (V/m)
# twy = 10e-15  # Ey-field pulsewidth (s)
# tpy = 50e-15  # Ey-field time of pulse peak value at origin (s)
# lamY = 800e-9  # Field wavelength for Ey (m)

# # Electric field parameters for Z-direction
# E0z = 0  # Peak Ez-field value (V/m)
# twz = 10e-15  # Ez-field pulsewidth (s)
# tpz = 50e-15  # Ez-field time of pulse peak value at origin (s)
# lamZ = 800e-9  # Field wavelength for Ez (m)

# # Allocate all fields and initialize to zero
# Exx = np.zeros(Nr, dtype=np.complex128)  # E-Field vector components along wire (V/m)
# Eyy = np.zeros(Nr, dtype=np.complex128)
# Ezz = np.zeros(Nr, dtype=np.complex128)

# Pxx1 = np.zeros(Nr, dtype=np.complex128)  # Polarizations along wire 1 (C/m^2)
# Pyy1 = np.zeros(Nr, dtype=np.complex128)
# Pzz1 = np.zeros(Nr, dtype=np.complex128)

# Pxx2 = np.zeros(Nr, dtype=np.complex128)  # Polarizations along wire 2 (C/m^2)
# Pyy2 = np.zeros(Nr, dtype=np.complex128)
# Pzz2 = np.zeros(Nr, dtype=np.complex128)

# Pxx_mid = np.zeros(Nr, dtype=np.complex128)  # Polarizations along wire 1 and 2 mid_point avg (C/m^2)
# Pyy_mid = np.zeros(Nr, dtype=np.complex128)
# Pzz_mid = np.zeros(Nr, dtype=np.complex128)

# Rho = np.zeros(Nr, dtype=np.complex128)  # Charge Density (C/m^2) along wire
# Vrr = np.zeros(Nr, dtype=np.complex128)  # Potential (V) along wire
# rr = np.zeros(Nr, dtype=np.float64)  # Spatial Array along the wire (m)
# qrr = np.zeros(Nr, dtype=np.float64)  # Momentum arrays (rad/m)

# # Angular frequencies and wave numbers (calculated later)
# w0x = 0.0  # Angular time frequencies (rad/s)
# w0y = 0.0
# w0z = 0.0
# k0x = 0.0  # Angular space frequencies (rad/m)
# k0y = 0.0
# k0z = 0.0
# Tcx = 0.0  # Optical cycles (s)
# Tcy = 0.0
# Tcz = 0.0
# Emax0 = 0.0  # Peak Electric field value (V/m)

# # Dummy boolean values (as lists for QWCalculator which modifies them in-place)
# boolF = [False]  # Dummy boolean value for false (list for in-place modification)
# boolT = [True]  # Dummy boolean value for true (list for in-place modification)


# def initializefields():
#     """
#     Initialize all field arrays to zero.

#     Sets all electric field, polarization, charge density, and spatial arrays
#     to zero as initial conditions.
#     """
#     global Exx, Eyy, Ezz, Pxx1, Pyy1, Pzz1, Pxx2, Pyy2, Pzz2
#     global Pxx_mid, Pyy_mid, Pzz_mid, Rho, Vrr, rr, qrr

#     Exx[:] = 0.0
#     Eyy[:] = 0.0
#     Ezz[:] = 0.0
#     Pxx1[:] = 0.0
#     Pyy1[:] = 0.0
#     Pzz1[:] = 0.0
#     Pxx2[:] = 0.0
#     Pyy2[:] = 0.0
#     Pzz2[:] = 0.0
#     Pxx_mid[:] = 0.0
#     Pyy_mid[:] = 0.0
#     Pzz_mid[:] = 0.0
#     Rho[:] = 0.0
#     Vrr[:] = 0.0
#     rr[:] = 0.0
#     qrr[:] = 0.0


# # Initialize fields
# initializefields()

# # Calculate angular frequencies, & optical cycle (for X-direction)
# w0x = twopi * c0 / lamX
# k0x = twopi / lamX * n0
# Tcx = lamX / c0

# # Calculate angular frequencies, & optical cycle (for Y-direction)
# w0y = twopi * c0 / lamY
# k0y = twopi / lamY * n0
# Tcy = lamY / c0

# # Calculate angular frequencies, & optical cycle (for Z-direction)
# w0z = twopi * c0 / lamZ
# k0z = twopi / lamZ * n0
# Tcz = lamZ / c0

# # Calculate the maximum field possible during the simulation
# Emax0 = np.sqrt(E0x**2 + E0y**2 + E0z**2)

# # Calculate the real-space & q-space array
# rr = GetSpaceArray(Nr, (Nr - 1) * drr)
# qrr = GetKArray(Nr, Nr * drr)

# # Initialize the SBEs in SBEs.py
# InitializeSBE(qrr, rr, 0.0, Emax0, lamX, 2, True)

# # ============================================================================
# # JIT AND PARALLEL VERIFICATION
# # ============================================================================
# print("\n" + "="*80)
# print("JIT COMPILATION AND PARALLEL EXECUTION VERIFICATION")
# print("="*80)

# try:
#     import numba
#     from numba.core import dispatcher
#     import numba.config

#     print(f"\n✓ Numba version: {numba.__version__}")
#     print(f"✓ Numba parallel threads: {numba.config.NUMBA_NUM_THREADS}")
#     print(f"✓ CPU count: {numba.config.NUMBA_DEFAULT_NUM_THREADS}")

#     # Check SBEs.py JIT functions
#     print("\nSBEs.py JIT functions:")
#     print("-" * 80)
#     jit_functions_sbes = {}

#     try:
#         from SBEs import _dpdt_jit, _dCdt_jit, _dDdt_jit
#         for name, func in [('_dpdt_jit', _dpdt_jit),
#                           ('_dCdt_jit', _dCdt_jit),
#                           ('_dDdt_jit', _dDdt_jit)]:
#             if isinstance(func, dispatcher.Dispatcher):
#                 sigs = func.signatures
#                 is_parallel = hasattr(func, 'parallel') and func.parallel
#                 jit_functions_sbes[name] = {
#                     'compiled': len(sigs) > 0,
#                     'signatures': len(sigs),
#                     'parallel': is_parallel
#                 }
#                 status = "✓ COMPILED" if len(sigs) > 0 else "✗ NOT COMPILED"
#                 parallel_status = " (PARALLEL)" if is_parallel else ""
#                 print(f"  {status:15s} {name:30s}: {len(sigs)} signature(s){parallel_status}")
#             else:
#                 jit_functions_sbes[name] = {'compiled': False, 'type': str(type(func))}
#                 print(f"  ✗ NOT JIT      {name:30s}: {type(func)}")
#     except Exception as e:
#         print(f"  ✗ Error checking SBEs: {e}")

#     # Check qwoptics.py JIT functions
#     print("\nqwoptics.py JIT functions:")
#     print("-" * 80)
#     jit_functions_qwoptics = {}

#     try:
#         from qwoptics import _QWPolarization3_jit, _QWRho5_jit
#         for name, func in [('_QWPolarization3_jit', _QWPolarization3_jit),
#                           ('_QWRho5_jit', _QWRho5_jit)]:
#             if isinstance(func, dispatcher.Dispatcher):
#                 sigs = func.signatures
#                 is_parallel = hasattr(func, 'parallel') and func.parallel
#                 jit_functions_qwoptics[name] = {
#                     'compiled': len(sigs) > 0,
#                     'signatures': len(sigs),
#                     'parallel': is_parallel
#                 }
#                 status = "✓ COMPILED" if len(sigs) > 0 else "✗ NOT COMPILED"
#                 parallel_status = " (PARALLEL)" if is_parallel else ""
#                 print(f"  {status:15s} {name:30s}: {len(sigs)} signature(s){parallel_status}")
#             else:
#                 jit_functions_qwoptics[name] = {'compiled': False, 'type': str(type(func))}
#                 print(f"  ✗ NOT JIT      {name:30s}: {type(func)}")
#     except Exception as e:
#         print(f"  ✗ Error checking qwoptics: {e}")

#     # Summary
#     all_functions = {**jit_functions_sbes, **jit_functions_qwoptics}
#     compiled = sum(1 for v in all_functions.values() if v.get('compiled', False))
#     parallel = sum(1 for v in all_functions.values() if v.get('parallel', False))
#     total = len(all_functions)

#     print(f"\nSummary:")
#     print(f"  Total JIT functions: {total}")
#     print(f"  Compiled: {compiled}/{total}")
#     print(f"  Parallel enabled: {parallel}/{total}")

#     if compiled == total:
#         print(f"\n✓ ALL JIT FUNCTIONS COMPILED")
#     if parallel > 0:
#         print(f"✓ {parallel} functions have parallel execution enabled")

# except ImportError:
#     print("✗ Numba NOT available - JIT will not work")
# except Exception as e:
#     print(f"✗ Error during verification: {e}")
#     import traceback
#     traceback.print_exc()

# print("="*80 + "\n")

# # Create output directory
# os.makedirs('fields', exist_ok=True)

# # Open files to record data
# file_Ex = open('fields/Ex.dat', 'w', encoding='utf-8')
# file_Ey = open('fields/Ey.dat', 'w', encoding='utf-8')
# file_Ez = open('fields/Ez.dat', 'w', encoding='utf-8')
# file_Px1 = open('fields/Px1.dat', 'w', encoding='utf-8')
# file_Py1 = open('fields/Py1.dat', 'w', encoding='utf-8')
# file_Pz1 = open('fields/Pz1.dat', 'w', encoding='utf-8')
# file_Px2 = open('fields/Px2.dat', 'w', encoding='utf-8')
# file_Py2 = open('fields/Py2.dat', 'w', encoding='utf-8')
# file_Pz2 = open('fields/Pz2.dat', 'w', encoding='utf-8')
# file_Px_mid = open('fields/Px_mid.dat', 'w', encoding='utf-8')
# file_Py_mid = open('fields/Py_mid.dat', 'w', encoding='utf-8')
# file_Pz_mid = open('fields/Pz_mid.dat', 'w', encoding='utf-8')

# # ============================================================================
# # PERFORMANCE TIMING
# # ============================================================================
# print("\n" + "="*80)
# print("PERFORMANCE BENCHMARKING")
# print("="*80)

# # Timing variables
# start_time = time.time()
# qwcalc_times = []
# fileio_times = []
# last_print_time = start_time
# print_interval = 1  # Print progress every N iterations

# print(f"Starting simulation: {Nt} iterations")
# print(f"Progress updates every {print_interval} iterations\n")

# # Begin Time-Evolving the SBEs
# for n in range(1, Nt + 1):
#     # Progress update
#     if n % print_interval == 0 or n == 1:
#         elapsed = time.time() - start_time
#         rate = n / elapsed if elapsed > 0 else 0
#         eta = (Nt - n) / rate if rate > 0 else 0
#         print(f"Progress: {n:6d}/{Nt} ({100*n/Nt:5.1f}%) | "
#               f"Elapsed: {elapsed:7.1f}s | Rate: {rate:6.1f} iter/s | "
#               f"ETA: {eta:7.1f}s")

#     # Calculate E-fields
#     Exx[:] = (E0x * np.exp(-(w0x * (t - tpx))**2 / (w0x * twx)**2) *
#               np.cos(w0x * (t - tpx)) *
#               np.exp(-(w0x * (t - tpx))**20 / (2 * twx * w0x)**20))

#     Eyy[:] = (E0y * np.exp(-(w0y * (t - tpy))**2 / (w0y * twy)**2) *
#               np.cos(w0y * (t - tpy)) *
#               np.exp(-(w0y * (t - tpy))**20 / (2 * twy * w0y)**20))

#     # Time-Evolve the SBEs from t(n) to t(n+1)
#     qwcalc_start = time.time()
#     QWCalculator(Exx, Eyy, Ezz, Vrr, rr, qrr, dt, 1, Pxx1, Pyy1, Pzz1, Rho, boolT, boolF)
#     QWCalculator(Exx, Eyy, Ezz, Vrr, rr, qrr, dt, 2, Pxx2, Pyy2, Pzz2, Rho, boolT, boolF)
#     qwcalc_times.append(time.time() - qwcalc_start)

#     Pxx_mid[:] = (Pxx1 + Pxx2) * 0.5
#     Pyy_mid[:] = (Pyy1 + Pyy2) * 0.5
#     Pzz_mid[:] = (Pzz1 + Pzz2) * 0.5

#     # File I/O timing
#     fileio_start = time.time()
#     file_Ex.write(f"{t} {np.real(Exx[Nr // 2])}\n")
#     file_Ey.write(f"{t} {np.real(Ezz[Nr // 2])}\n")
#     file_Ez.write(f"{t} {np.real(Eyy[Nr // 2])}\n")
#     file_Px1.write(f"{t} {np.real(Pxx1[Nr // 2])}\n")
#     file_Py1.write(f"{t} {np.real(Pyy1[Nr // 2])}\n")
#     file_Pz1.write(f"{t} {np.real(Pzz1[Nr // 2])}\n")
#     file_Px2.write(f"{t} {np.real(Pxx2[Nr // 2])}\n")
#     file_Py2.write(f"{t} {np.real(Pyy2[Nr // 2])}\n")
#     file_Pz2.write(f"{t} {np.real(Pzz2[Nr // 2])}\n")
#     file_Px_mid.write(f"{t} {np.real(Pxx_mid[Nr // 2])}\n")
#     file_Py_mid.write(f"{t} {np.real(Pyy_mid[Nr // 2])}\n")
#     file_Pz_mid.write(f"{t} {np.real(Pzz_mid[Nr // 2])}\n")
#     fileio_times.append(time.time() - fileio_start)

#     t = t + dt

# # Close files
# file_Ex.close()
# file_Ey.close()
# file_Ez.close()
# file_Px1.close()
# file_Py1.close()
# file_Pz1.close()
# file_Px2.close()
# file_Py2.close()
# file_Pz2.close()
# file_Px_mid.close()
# file_Py_mid.close()
# file_Pz_mid.close()

# # ============================================================================
# # PERFORMANCE SUMMARY
# # ============================================================================
# total_time = time.time() - start_time
# avg_qwcalc_time = np.mean(qwcalc_times) if qwcalc_times else 0
# avg_fileio_time = np.mean(fileio_times) if fileio_times else 0
# total_qwcalc_time = np.sum(qwcalc_times)
# total_fileio_time = np.sum(fileio_times)

# print("\n" + "="*80)
# print("PERFORMANCE SUMMARY")
# print("="*80)
# print(f"Total simulation time: {total_time:.2f} seconds")
# print(f"Total iterations: {Nt}")
# print(f"Average time per iteration: {total_time/Nt*1000:.2f} ms")
# print(f"Simulation rate: {Nt/total_time:.2f} iterations/second")
# print(f"\nBreakdown:")
# print(f"  QWCalculator time: {total_qwcalc_time:.2f}s ({100*total_qwcalc_time/total_time:.1f}%)")
# print(f"    Average per call: {avg_qwcalc_time*1000:.3f} ms")
# print(f"  File I/O time: {total_fileio_time:.2f}s ({100*total_fileio_time/total_time:.1f}%)")
# print(f"    Average per iteration: {avg_fileio_time*1000:.3f} ms")
# print(f"  Other overhead: {total_time - total_qwcalc_time - total_fileio_time:.2f}s "
#       f"({100*(total_time - total_qwcalc_time - total_fileio_time)/total_time:.1f}%)")

# # Check if parallel is actually being used
# print(f"\nParallel execution check:")
# try:
#     import numba
#     if hasattr(numba, 'config'):
#         threads = numba.config.NUMBA_NUM_THREADS
#         print(f"  Numba parallel threads configured: {threads}")
#         if threads > 1:
#             print(f"  ✓ Parallel execution should be active")
#         else:
#             print(f"  ⚠ Only 1 thread - parallel execution disabled")
#             print(f"    Set NUMBA_NUM_THREADS environment variable to enable")
# except:
#     pass

# print("="*80 + "\n")
