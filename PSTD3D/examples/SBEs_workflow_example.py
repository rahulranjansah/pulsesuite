"""
SBE Workflow Example - Complete research workflow using SBEspythonic module

This example demonstrates a complete workflow for semiconductor Bloch equation
simulations in quantum wires, including initialization, time evolution, and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tempfile
import shutil

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from SBEspythonic import *
from constants import *


def create_test_environment():
    """Create temporary test environment with parameter files."""
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    # Create params directory
    os.makedirs('params', exist_ok=True)

    # Create qw.params file
    with open('params/qw.params', 'w') as f:
        f.write("100e-9\n")  # L
        f.write("5e-9\n")    # Delta0
        f.write("1.5\n")     # gap (eV)
        f.write("0.07\n")    # me (me0)
        f.write("0.45\n")    # mh (me0)
        f.write("0.1\n")     # HO (eV)
        f.write("1e12\n")    # gam_e
        f.write("1e12\n")    # gam_h
        f.write("1e12\n")    # gam_eh
        f.write("9.1\n")     # epsr
        f.write("36e-3\n")   # Oph (eV)
        f.write("3e-3\n")    # Gph (eV)
        f.write("0.0\n")     # Edc
        f.write("100\n")     # jmax
        f.write("1000\n")    # ntmax

    # Create mb.params file
    with open('params/mb.params', 'w') as f:
        f.write("1\n")  # Optics
        f.write("1\n")  # Excitons
        f.write("1\n")  # EHs
        f.write("1\n")  # Screened
        f.write("1\n")  # Phonon
        f.write("0\n")  # DCTrans
        f.write("1\n")  # LF
        f.write("0\n")  # FreePot
        f.write("1\n")  # DiagDph
        f.write("1\n")  # OffDiagDph
        f.write("0\n")  # Recomb
        f.write("0\n")  # PLSpec
        f.write("0\n")  # ignorewire
        f.write("0\n")  # Xqwparams
        f.write("0\n")  # LorentzDelta

    return temp_dir, original_dir


def cleanup_test_environment(temp_dir, original_dir):
    """Clean up temporary test environment."""
    os.chdir(original_dir)
    shutil.rmtree(temp_dir)


def simulate_quantum_wire_response():
    """Simulate quantum wire response to external fields."""
    print("Quantum Wire Response Simulation")
    print("===============================")

    # Set up simulation parameters
    Nk = 20
    Nr = 40
    Nw = 1
    dt = 1e-15  # 1 fs time step
    Nt = 100    # Number of time steps

    # Create spatial and momentum arrays
    q = np.linspace(-2e6, 2e6, Nr, dtype=np.float64)
    rr = np.linspace(-200e-9, 200e-9, Nr, dtype=np.float64)

    # Initialize SBE module
    print("Initializing SBE module...")
    InitializeSBE(q, rr, 0.0, 1e6, 800e-9, Nw, True)
    print(f"✓ Initialized with Nk={Nk}, Nr={Nr}")

    # Set up external fields (Gaussian pulse)
    t0 = 50e-15  # Pulse center time
    tau = 20e-15  # Pulse width
    E0 = 1e6      # Peak field strength

    # Initialize field arrays
    Exx = np.zeros(Nr, dtype=np.complex128)
    Eyy = np.zeros(Nr, dtype=np.complex128)
    Ezz = np.zeros(Nr, dtype=np.complex128)
    Vrr = np.zeros(Nr, dtype=np.complex128)

    # Initialize output arrays
    Pxx = np.zeros(Nr, dtype=np.complex128)
    Pyy = np.zeros(Nr, dtype=np.complex128)
    Pzz = np.zeros(Nr, dtype=np.complex128)
    Rho = np.zeros(Nr, dtype=np.complex128)

    # Storage for time evolution
    times = []
    field_amplitudes = []
    polarization_amplitudes = []
    charge_densities = []

    print("Running time evolution...")

    for step in range(Nt):
        t = step * dt

        # Create Gaussian pulse
        pulse = E0 * np.exp(-((t - t0) / tau)**2)

        # Apply pulse to fields
        Exx[:] = pulse * np.exp(1j * 2 * np.pi * c0 * t / 800e-9)
        Eyy[:] = 0.5 * pulse * np.exp(1j * 2 * np.pi * c0 * t / 800e-9)
        Ezz[:] = 0.0
        Vrr[:] = 0.0

        # Initialize output arrays
        Pxx[:] = 0.0
        Pyy[:] = 0.0
        Pzz[:] = 0.0
        Rho[:] = 0.0

        # Calculate QW response
        DoQWP = False
        DoQWDl = False

        try:
            QWCalculator(Exx, Eyy, Ezz, Vrr, rr, q, dt, 0, Pxx, Pyy, Pzz, Rho, DoQWP, DoQWDl)
        except Exception as e:
            print(f"Warning: QWCalculator failed at step {step}: {e}")
            continue

        # Store results
        if step % 10 == 0:  # Store every 10th step
            times.append(t)
            field_amplitudes.append(np.max(np.abs(Exx)))
            polarization_amplitudes.append(np.max(np.abs(Pxx)))
            charge_densities.append(np.max(np.abs(Rho)))

    print(f"✓ Completed {Nt} time steps")

    return times, field_amplitudes, polarization_amplitudes, charge_densities


def analyze_susceptibility():
    """Analyze linear susceptibility of the quantum wire."""
    print("\nSusceptibility Analysis")
    print("======================")

    # Get linear susceptibility
    chi = chiqw()
    print(f"Linear susceptibility: {chi:.2e}")

    # Calculate refractive index
    n = np.sqrt(1 + chi.real)
    k = chi.imag / (2 * n)

    print(f"Refractive index: {n:.3f}")
    print(f"Extinction coefficient: {k:.3f}")

    return chi, n, k


def analyze_band_structure():
    """Analyze the band structure of the quantum wire."""
    print("\nBand Structure Analysis")
    print("=======================")

    if Ee is not None and Eh is not None and kr is not None:
        # Calculate effective masses
        me_eff = hbar**2 / (2 * Ee[1] / kr[1]**2) if len(Ee) > 1 else me
        mh_eff = hbar**2 / (2 * Eh[1] / kr[1]**2) if len(Eh) > 1 else mh

        print(f"Electron effective mass: {me_eff/me0:.3f} me")
        print(f"Hole effective mass: {mh_eff/me0:.3f} me")

        # Calculate band gap
        band_gap = gap / eV
        print(f"Band gap: {band_gap:.3f} eV")

        # Calculate exciton binding energy (approximate)
        a0 = 4 * np.pi * eps0 * hbar**2 / (e0**2 * me_eff)
        E_bind = e0**2 / (8 * np.pi * eps0 * a0)
        print(f"Exciton binding energy: {E_bind/eV:.3f} eV")

        return me_eff, mh_eff, band_gap, E_bind
    else:
        print("Band structure not available")
        return None, None, None, None


def plot_results(times, field_amplitudes, polarization_amplitudes, charge_densities):
    """Plot simulation results."""
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Convert times to femtoseconds
    times_fs = [t * 1e15 for t in times]

    # Plot field amplitude
    axes[0, 0].plot(times_fs, field_amplitudes)
    axes[0, 0].set_xlabel('Time (fs)')
    axes[0, 0].set_ylabel('Field Amplitude (V/m)')
    axes[0, 0].set_title('External Field')
    axes[0, 0].grid(True)

    # Plot polarization amplitude
    axes[0, 1].plot(times_fs, polarization_amplitudes)
    axes[0, 1].set_xlabel('Time (fs)')
    axes[0, 1].set_ylabel('Polarization (C/m²)')
    axes[0, 1].set_title('Quantum Wire Polarization')
    axes[0, 1].grid(True)

    # Plot charge density
    axes[1, 0].plot(times_fs, charge_densities)
    axes[1, 0].set_xlabel('Time (fs)')
    axes[1, 0].set_ylabel('Charge Density (C/m³)')
    axes[1, 0].set_title('Charge Density')
    axes[1, 0].grid(True)

    # Plot band structure if available
    if Ee is not None and Eh is not None and kr is not None:
        axes[1, 1].plot(kr * 1e-9, Ee / eV, 'b-', label='Electrons')
        axes[1, 1].plot(kr * 1e-9, Eh / eV, 'r-', label='Holes')
        axes[1, 1].set_xlabel('Momentum (nm⁻¹)')
        axes[1, 1].set_ylabel('Energy (eV)')
        axes[1, 1].set_title('Band Structure')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'Band structure\nnot available',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Band Structure')

    plt.tight_layout()
    plt.savefig('sbe_workflow_results.png', dpi=300, bbox_inches='tight')
    print("✓ Plots saved to sbe_workflow_results.png")

    return fig


def main():
    """Run complete SBE workflow example."""
    print("SBE Workflow Example")
    print("===================")
    print("This example demonstrates a complete workflow for semiconductor")
    print("Bloch equation simulations in quantum wires.")

    # Create test environment
    temp_dir, original_dir = create_test_environment()

    try:
        # Run simulation
        times, field_amplitudes, polarization_amplitudes, charge_densities = simulate_quantum_wire_response()

        # Analyze results
        chi, n, k = analyze_susceptibility()
        me_eff, mh_eff, band_gap, E_bind = analyze_band_structure()

        # Generate plots
        fig = plot_results(times, field_amplitudes, polarization_amplitudes, charge_densities)

        # Print summary
        print("\nSimulation Summary")
        print("==================")
        print(f"Total time steps: {len(times)}")
        print(f"Time range: {times[0]*1e15:.1f} - {times[-1]*1e15:.1f} fs")
        print(f"Peak field amplitude: {max(field_amplitudes):.2e} V/m")
        print(f"Peak polarization: {max(polarization_amplitudes):.2e} C/m²")
        print(f"Peak charge density: {max(charge_densities):.2e} C/m³")

        if chi is not None:
            print(f"Linear susceptibility: {chi:.2e}")
            print(f"Refractive index: {n:.3f}")

        if me_eff is not None:
            print(f"Electron effective mass: {me_eff/me0:.3f} me")
            print(f"Hole effective mass: {mh_eff/me0:.3f} me")
            print(f"Band gap: {band_gap:.3f} eV")
            print(f"Exciton binding energy: {E_bind/eV:.3f} eV")

        print("\n✓ Workflow completed successfully!")
        print("\nThis example demonstrates:")
        print("- Complete SBE module initialization")
        print("- Time evolution of quantum wire response")
        print("- Analysis of optical properties")
        print("- Band structure analysis")
        print("- Visualization of results")

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        cleanup_test_environment(temp_dir, original_dir)


if __name__ == "__main__":
    main()
