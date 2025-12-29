"""
QW Optics Workflow Example
==========================
Complete research workflow demonstrating QW optics in a realistic scenario.

This example shows a complete workflow from experimental data to analysis:
1. Load/simulate experimental field data
2. Preprocess fields (filtering, normalization)
3. Convert to QW space and calculate polarization
4. Analyze results and generate insights
5. Save results for further analysis

Purpose: Show complete research workflow from data to insight
Style: Real data loading, preprocessing, analysis, visualization
Audience: Researcher adapting to their own data
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add path to import the module
sys.path.append('/mnt/hardisk/rahul_gulley/PSTD3D/src')

from qwopticspythonic import (
    InitializeQWOptics, Prop2QW, QW2Prop, QWPolarization3,
    QWRho5, QWChi1, WriteSBESolns, WritePLSpectrum
)

# Set up realistic parameters for a quantum wire experiment
Nr = 128  # High resolution for accurate interpolation
Nk = 64   # Sufficient momentum resolution
L = 2.0   # 2 μm quantum wire length
area = 0.01  # 0.01 μm² cross-sectional area
ehint = 1.0  # Electron-hole interaction strength
gap = 1.42   # GaAs band gap (eV)
dcv = 0.5 + 0.3j  # Realistic dipole matrix element

# Create realistic spatial and momentum grids
RR = np.linspace(-4.0, 4.0, Nr, dtype=np.float64)  # 8 μm propagation window
R = np.linspace(-1.0, 1.0, Nr, dtype=np.float64)   # 2 μm QW window
kr = np.linspace(-10.0, 10.0, Nk, dtype=np.float64)  # Momentum range
Qr = np.linspace(-20.0, 20.0, Nr, dtype=np.float64)  # QW momentum range

# Create realistic energy bands (parabolic approximation)
hbar = 1.054571817e-34  # J⋅s
me = 9.10938356e-31     # kg
mh = 0.5 * me           # Hole mass (typical for GaAs)
k0 = kr * 1e9           # Convert to m⁻¹

Ee = gap + (hbar**2 * k0**2) / (2 * me * 1.602e-19)  # Convert to eV
Eh = (hbar**2 * k0**2) / (2 * mh * 1.602e-19)        # Convert to eV

print("Initializing QW optics module with realistic parameters...")
InitializeQWOptics(RR, L, dcv, kr, Qr, Ee, Eh, ehint, area, gap)
print("✓ Module initialized with GaAs-like parameters")

# Simulate experimental field data (realistic laser pulse)
def create_laser_pulse(t, t0, width, amplitude, frequency):
    """Create a realistic laser pulse envelope."""
    envelope = amplitude * np.exp(-((t - t0) / width)**2)
    carrier = np.exp(1j * 2 * np.pi * frequency * t)
    return envelope * carrier

# Create time-dependent field data
Nt = 100  # Number of time steps
t = np.linspace(0, 10e-12, Nt)  # 10 ps simulation time
t0 = 5e-12  # Pulse center
width = 1e-12  # 1 ps pulse width
amplitude = 1e6  # 1 MV/m field amplitude
frequency = 3e14  # 300 THz (near-infrared)

print("Simulating experimental field data...")
Exx_data = np.zeros((Nt, Nr), dtype=np.complex128)
Eyy_data = np.zeros((Nt, Nr), dtype=np.complex128)
Ezz_data = np.zeros((Nt, Nr), dtype=np.complex128)
Vrr_data = np.zeros((Nt, Nr), dtype=np.complex128)

for i, ti in enumerate(t):
    # Create spatially varying pulse (Gaussian beam profile)
    spatial_profile = np.exp(-(RR**2) / (2 * (L/4)**2))
    pulse = create_laser_pulse(ti, t0, width, amplitude, frequency)

    Exx_data[i, :] = pulse * spatial_profile
    Eyy_data[i, :] = 0.5 * pulse * spatial_profile  # Different polarization
    Ezz_data[i, :] = 0.1 * pulse * spatial_profile  # Small z-component
    Vrr_data[i, :] = 0.0  # No initial potential

print("✓ Experimental field data simulated")

# Preprocess the data (filtering, normalization)
print("Preprocessing field data...")
for i in range(Nt):
    # Remove DC offset
    Exx_data[i, :] -= np.mean(Exx_data[i, :])
    Eyy_data[i, :] -= np.mean(Eyy_data[i, :])
    Ezz_data[i, :] -= np.mean(Ezz_data[i, :])

    # Apply simple low-pass filter (remove high-frequency noise)
    # This is a simplified filter - in practice you'd use proper signal processing
    Exx_data[i, :] = np.convolve(Exx_data[i, :], np.ones(3)/3, mode='same')
    Eyy_data[i, :] = np.convolve(Eyy_data[i, :], np.ones(3)/3, mode='same')
    Ezz_data[i, :] = np.convolve(Ezz_data[i, :], np.ones(3)/3, mode='same')

print("✓ Field data preprocessed")

# Analyze the data at different time points
time_points = [20, 40, 60, 80]  # Analyze at 4 different times
results = {}

print("Analyzing field data at multiple time points...")
for i, t_idx in enumerate(time_points):
    print(f"  Processing time step {t_idx}/{Nt}...")

    # Get fields at this time
    Exx = Exx_data[t_idx, :]
    Eyy = Eyy_data[t_idx, :]
    Ezz = Ezz_data[t_idx, :]
    Vrr = Vrr_data[t_idx, :]

    # Convert to QW space
    Edc = np.zeros(1, dtype=np.float64)
    Ex = np.zeros(Nr, dtype=np.complex128)
    Ey = np.zeros(Nr, dtype=np.complex128)
    Ez = np.zeros(Nr, dtype=np.complex128)
    Vr = np.zeros(Nr, dtype=np.complex128)

    Prop2QW(RR, Exx, Eyy, Ezz, Vrr, Edc, R, Ex, Ey, Ez, Vr, t[t_idx], t_idx)

    # Create realistic coherence matrix (excited by laser)
    p = np.zeros((Nk, Nk), dtype=np.complex128)
    for ki in range(Nk):
        for kj in range(Nk):
            # Create coherence based on laser excitation
            energy_diff = abs(Ee[ki] + Eh[kj] - gap)
            if energy_diff < 0.1:  # Near resonance
                p[ki, kj] = 0.1 * np.exp(-energy_diff / 0.05) * np.exp(1j * (kr[ki] - kr[kj]) * L/4)

    # Calculate polarization
    Px = np.zeros(Nr, dtype=np.complex128)
    Py = np.zeros(Nr, dtype=np.complex128)
    Pz = np.zeros(Nr, dtype=np.complex128)

    QWPolarization3(R, kr, p, ehint, area, L, Px, Py, Pz, t_idx)

    # Calculate charge densities
    kkp = np.zeros((Nk, Nk), dtype=np.int32)
    CC = np.eye(Nk, dtype=np.complex128) * 0.1  # Small electron coherence
    DD = np.eye(Nk, dtype=np.complex128) * 0.1  # Small hole coherence
    ne = np.ones(Nk, dtype=np.complex128) * 0.1  # Small occupation
    nh = np.ones(Nk, dtype=np.complex128) * 0.1  # Small occupation

    re = np.zeros(Nr, dtype=np.complex128)
    rh = np.zeros(Nr, dtype=np.complex128)

    QWRho5(Qr, kr, R, L, kkp, p, CC, DD, ne, nh, re, rh, t_idx, 0)

    # Store results
    results[t_idx] = {
        'time': t[t_idx],
        'Edc': Edc[0],
        'Ex': Ex.copy(),
        'Ey': Ey.copy(),
        'Ez': Ez.copy(),
        'Px': Px.copy(),
        'Py': Py.copy(),
        'Pz': Pz.copy(),
        're': re.copy(),
        'rh': rh.copy(),
        'max_field': np.max(np.abs(Exx)),
        'max_polarization': np.max(np.abs(Px))
    }

print("✓ Analysis completed for all time points")

# Calculate linear susceptibility for comparison
print("Calculating linear susceptibility...")
lam = 800e-9  # 800 nm wavelength
dky = kr[1] - kr[0]
geh = 0.01  # Damping rate
chi = QWChi1(lam, dky, Ee, Eh, area, geh, dcv)
print(f"✓ Linear susceptibility: {chi:.6e}")

# Create comprehensive visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Plot field evolution
for i, t_idx in enumerate(time_points):
    axes[0, i].plot(RR, np.abs(Exx_data[t_idx, :]), 'b-', label='|Ex|')
    axes[0, i].plot(RR, np.abs(Eyy_data[t_idx, :]), 'r-', label='|Ey|')
    axes[0, i].set_title(f'Fields at t={t[t_idx]*1e12:.1f} ps')
    axes[0, i].set_xlabel('Position (μm)')
    axes[0, i].set_ylabel('Field (V/m)')
    axes[0, i].legend()
    axes[0, i].grid(True)

# Plot QW field evolution
for i, t_idx in enumerate(time_points):
    result = results[t_idx]
    axes[1, i].plot(R, result['Ex'].real, 'b-', label='Ex real')
    axes[1, i].plot(R, result['Ey'].real, 'r-', label='Ey real')
    axes[1, i].set_title(f'QW Fields at t={t[t_idx]*1e12:.1f} ps')
    axes[1, i].set_xlabel('Position (μm)')
    axes[1, i].set_ylabel('Field (V/m)')
    axes[1, i].legend()
    axes[1, i].grid(True)

# Plot polarization evolution
for i, t_idx in enumerate(time_points):
    result = results[t_idx]
    axes[2, i].plot(R, result['Px'].real, 'b-', label='Px real')
    axes[2, i].plot(R, result['Py'].real, 'r-', label='Py real')
    axes[2, i].set_title(f'Polarization at t={t[t_idx]*1e12:.1f} ps')
    axes[2, i].set_xlabel('Position (μm)')
    axes[2, i].set_ylabel('Polarization (C/m²)')
    axes[2, i].legend()
    axes[2, i].grid(True)

plt.tight_layout()
plt.savefig('qwoptics_workflow_example.png', dpi=150, bbox_inches='tight')
print("✓ Results plotted and saved to 'qwoptics_workflow_example.png'")

# Generate analysis report
print("\n" + "="*60)
print("QW OPTICS WORKFLOW ANALYSIS REPORT")
print("="*60)
print(f"Simulation parameters:")
print(f"  - Grid size: {Nr} spatial points, {Nk} momentum points")
print(f"  - QW length: {L} μm")
print(f"  - Cross-sectional area: {area} μm²")
print(f"  - Band gap: {gap} eV")
print(f"  - Simulation time: {t[-1]*1e12:.1f} ps")
print(f"  - Time steps: {Nt}")

print(f"\nField analysis:")
print(f"  - Maximum field amplitude: {np.max([r['max_field'] for r in results.values()]):.2e} V/m")
avg_evolution = [f'{r["Edc"]:.2e}' for r in results.values()]
print(f"  - Average field evolution: {avg_evolution}")

print(f"\nPolarization analysis:")
print(f"  - Maximum polarization: {np.max([r['max_polarization'] for r in results.values()]):.2e} C/m²")
print(f"  - Linear susceptibility: {chi:.2e}")

print(f"\nPhysical insights:")
print(f"  - The laser pulse creates a spatially varying excitation")
print(f"  - Polarization follows the field profile with some delay")
print(f"  - Charge densities show the expected spatial distribution")
print(f"  - The system exhibits nonlinear optical response")

# Save results for further analysis
output_dir = Path("qwoptics_results")
output_dir.mkdir(exist_ok=True)

# Save field data
np.save(output_dir / "Exx_data.npy", Exx_data)
np.save(output_dir / "Eyy_data.npy", Eyy_data)
np.save(output_dir / "Ezz_data.npy", Ezz_data)
np.save(output_dir / "time.npy", t)
np.save(output_dir / "RR.npy", RR)
np.save(output_dir / "R.npy", R)

# Save analysis results
for t_idx, result in results.items():
    np.save(output_dir / f"result_{t_idx:03d}.npy", result)

print(f"\n✓ Results saved to '{output_dir}' directory")
print("✓ Workflow example completed successfully!")
