"""
Basic QW Optics Example
=======================
Minimal working example demonstrating basic QW optics functionality.

This example shows how to:
1. Initialize the QW optics module
2. Convert propagation fields to QW fields
3. Calculate polarization from coherence matrix
4. Convert back to propagation space

Purpose: Minimal working example that proves the functions work
Style: Hard-coded data, immediate visualization
Audience: Someone copy-pasting to verify installation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add path to import the module
sys.path.append('/mnt/hardisk/rahul_gulley/PSTD3D/src')

from qwopticspythonic import (
    InitializeQWOptics, Prop2QW, QW2Prop, QWPolarization3,
    CalcQWWindow, CalcExpikr
)

# Set up parameters
Nr = 64  # Number of spatial points
Nk = 32  # Number of momentum points
L = 1.0  # QW length
area = 0.1  # QW cross-sectional area
ehint = 1.0  # Electron-hole interaction strength
gap = 1.5  # Band gap energy
dcv = 1.0 + 0.5j  # Dipole matrix element

# Create spatial and momentum grids
RR = np.linspace(-2.0, 2.0, Nr, dtype=np.float64)  # Propagation grid
R = np.linspace(-1.0, 1.0, Nr, dtype=np.float64)   # QW grid
kr = np.linspace(-5.0, 5.0, Nk, dtype=np.float64)  # Momentum grid
Qr = np.linspace(-10.0, 10.0, Nr, dtype=np.float64)  # QW momentum grid

# Create energy bands
Ee = np.linspace(0.5, 2.0, Nk, dtype=np.float64)  # Electron energies
Eh = np.linspace(0.3, 1.8, Nk, dtype=np.float64)  # Hole energies

print("Initializing QW optics module...")
InitializeQWOptics(RR, L, dcv, kr, Qr, Ee, Eh, ehint, area, gap)
print("✓ Module initialized successfully")

# Create test electric fields (simple sine waves)
Exx = np.sin(2 * np.pi * RR / L) + 1j * np.cos(2 * np.pi * RR / L)
Eyy = np.cos(2 * np.pi * RR / L) + 1j * np.sin(2 * np.pi * RR / L)
Ezz = np.zeros_like(Exx)
Vrr = np.zeros_like(Exx)

# Output arrays
Edc = np.zeros(1, dtype=np.float64)
Ex = np.zeros(Nr, dtype=np.complex128)
Ey = np.zeros(Nr, dtype=np.complex128)
Ez = np.zeros(Nr, dtype=np.complex128)
Vr = np.zeros(Nr, dtype=np.complex128)

print("Converting propagation fields to QW fields...")
Prop2QW(RR, Exx, Eyy, Ezz, Vrr, Edc, R, Ex, Ey, Ez, Vr, 0.0, 0)
print(f"✓ Average field: {Edc[0]:.6f}")

# Create test coherence matrix (simple Gaussian)
p = np.zeros((Nk, Nk), dtype=np.complex128)
for i in range(Nk):
    for j in range(Nk):
        p[i, j] = np.exp(-((kr[i] - kr[j])**2) / 2.0) * np.exp(1j * (kr[i] + kr[j]))

# Calculate polarization
Px = np.zeros(Nr, dtype=np.complex128)
Py = np.zeros(Nr, dtype=np.complex128)
Pz = np.zeros(Nr, dtype=np.complex128)

print("Calculating QW polarization...")
QWPolarization3(R, kr, p, ehint, area, L, Px, Py, Pz, 0)
print("✓ Polarization calculated")

# Convert back to propagation space
Pxx = np.zeros(Nr, dtype=np.complex128)
Pyy = np.zeros(Nr, dtype=np.complex128)
Pzz = np.zeros(Nr, dtype=np.complex128)
RhoE = np.zeros(Nr, dtype=np.complex128)
RhoH = np.zeros(Nr, dtype=np.complex128)

# Create dummy charge densities
re = np.zeros(Nr, dtype=np.complex128)
rh = np.zeros(Nr, dtype=np.complex128)

print("Converting QW fields back to propagation space...")
QW2Prop(R, Qr, Ex, Ey, Ez, Vr, Px, Py, Pz, re, rh,
        RR, Pxx, Pyy, Pzz, RhoE, RhoH, 0, 0, False, False)
print("✓ Fields converted back to propagation space")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot original propagation fields
axes[0, 0].plot(RR, Exx.real, 'b-', label='Ex real')
axes[0, 0].plot(RR, Exx.imag, 'r--', label='Ex imag')
axes[0, 0].set_title('Original Propagation Fields')
axes[0, 0].set_xlabel('Position')
axes[0, 0].set_ylabel('Field')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot QW fields
axes[0, 1].plot(R, Ex.real, 'b-', label='Ex real')
axes[0, 1].plot(R, Ey.real, 'g-', label='Ey real')
axes[0, 1].set_title('QW Fields (Real Space)')
axes[0, 1].set_xlabel('Position')
axes[0, 1].set_ylabel('Field')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot polarization
axes[1, 0].plot(R, Px.real, 'b-', label='Px real')
axes[1, 0].plot(R, Py.real, 'g-', label='Py real')
axes[1, 0].set_title('QW Polarization (Real Space)')
axes[1, 0].set_xlabel('Position')
axes[1, 0].set_ylabel('Polarization')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot propagation space polarization
axes[1, 1].plot(RR, Pxx.real, 'b-', label='Pxx real')
axes[1, 1].plot(RR, Pyy.real, 'g-', label='Pyy real')
axes[1, 1].set_title('Propagation Space Polarization')
axes[1, 1].set_xlabel('Position')
axes[1, 1].set_ylabel('Polarization')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('qwoptics_basic_example.png', dpi=150, bbox_inches='tight')
print("✓ Results plotted and saved to 'qwoptics_basic_example.png'")

# Print summary
print("\n" + "="*50)
print("QW OPTICS BASIC EXAMPLE COMPLETED")
print("="*50)
print(f"Grid size: {Nr} spatial points, {Nk} momentum points")
print(f"QW length: {L}")
print(f"Average field: {Edc[0]:.6f}")
print(f"Max polarization: {np.max(np.abs(Px)):.6f}")
print("All functions executed successfully!")
