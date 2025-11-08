"""
SBE Domain Example - Physics simulation with GaAs quantum wire parameters

This example demonstrates semiconductor Bloch equation simulations for a
GaAs quantum wire with realistic material parameters and physical scenarios.
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


def create_gaas_parameters():
    """Create parameter files for GaAs quantum wire."""
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    # Create params directory
    os.makedirs('params', exist_ok=True)

    # GaAs parameters
    L = 50e-9        # Wire length: 50 nm
    Delta0 = 10e-9   # Wire thickness: 10 nm
    gap = 1.42       # Band gap: 1.42 eV
    me = 0.067       # Electron effective mass: 0.067 me
    mh = 0.45        # Hole effective mass: 0.45 me
    HO = 0.1         # Confinement energy: 100 meV
    gam_e = 1e12     # Electron lifetime: 1 ps
    gam_h = 1e12     # Hole lifetime: 1 ps
    gam_eh = 1e12    # Dephasing rate: 1 ps
    epsr = 12.9      # GaAs dielectric constant
    Oph = 36e-3      # LO phonon energy: 36 meV
    Gph = 3e-3       # Phonon lifetime: 3 meV
    Edc = 0.0        # No DC field
    jmax = 100       # Print every 100 steps
    ntmax = 1000     # Maximum time steps

    # Create qw.params file
    with open('params/qw.params', 'w') as f:
        f.write(f"{L}\n")
        f.write(f"{Delta0}\n")
        f.write(f"{gap}\n")
        f.write(f"{me}\n")
        f.write(f"{mh}\n")
        f.write(f"{HO}\n")
        f.write(f"{gam_e}\n")
        f.write(f"{gam_h}\n")
        f.write(f"{gam_eh}\n")
        f.write(f"{epsr}\n")
        f.write(f"{Oph}\n")
        f.write(f"{Gph}\n")
        f.write(f"{Edc}\n")
        f.write(f"{jmax}\n")
        f.write(f"{ntmax}\n")

    # Create mb.params file
    with open('params/mb.params', 'w') as f:
        f.write("1\n")  # Optics: Include optical coupling
        f.write("1\n")  # Excitons: Include excitonic effects
        f.write("1\n")  # EHs: Include electron-hole interactions
        f.write("1\n")  # Screened: Include screening
        f.write("1\n")  # Phonon: Include phonon interactions
        f.write("0\n")  # DCTrans: No DC transport
        f.write("1\n")  # LF: Include longitudinal field
        f.write("0\n")  # FreePot: No free charge potential
        f.write("1\n")  # DiagDph: Include diagonal dephasing
        f.write("1\n")  # OffDiagDph: Include off-diagonal dephasing
        f.write("0\n")  # Recomb: No recombination
        f.write("0\n")  # PLSpec: No PL spectrum
        f.write("0\n")  # ignorewire: Don't ignore wire
        f.write("0\n")  # Xqwparams: Don't write Xqw params
        f.write("0\n")  # LorentzDelta: Use Gaussian delta

    return temp_dir, original_dir


def cleanup_environment(temp_dir, original_dir):
    """Clean up temporary environment."""
    os.chdir(original_dir)
    shutil.rmtree(temp_dir)


def simulate_optical_response():
    """Simulate optical response of GaAs quantum wire."""
    print("GaAs Quantum Wire Optical Response")
    print("=================================")

    # Set up simulation parameters
    Nk = 30
    Nr = 60
    Nw = 1
    dt = 0.5e-15  # 0.5 fs time step
    Nt = 200      # Number of time steps

    # Create arrays
    q = np.linspace(-3e6, 3e6, Nr, dtype=np.float64)
    rr = np.linspace(-100e-9, 100e-9, Nr, dtype=np.float64)

    # Initialize SBE module
    print("Initializing SBE module with GaAs parameters...")
    InitializeSBE(q, rr, 0.0, 1e6, 800e-9, Nw, True)
    print(f"✓ Initialized with Nk={Nk}, Nr={Nr}")

    # Set up optical pulse parameters
    t0 = 100e-15  # Pulse center time
    tau = 30e-15  # Pulse width (FWHM)
    E0 = 5e5      # Peak field strength (500 kV/m)
    wavelength = 800e-9  # 800 nm wavelength

    # Initialize arrays
    Exx = np.zeros(Nr, dtype=np.complex128)
    Eyy = np.zeros(Nr, dtype=np.complex128)
    Ezz = np.zeros(Nr, dtype=np.complex128)
    Vrr = np.zeros(Nr, dtype=np.complex128)
    Pxx = np.zeros(Nr, dtype=np.complex128)
    Pyy = np.zeros(Nr, dtype=np.complex128)
    Pzz = np.zeros(Nr, dtype=np.complex128)
    Rho = np.zeros(Nr, dtype=np.complex128)

    # Storage for analysis
    times = []
    field_data = []
    polarization_data = []
    charge_data = []
    energy_data = []

    print("Running optical response simulation...")

    for step in range(Nt):
        t = step * dt

        # Create Gaussian pulse
        pulse = E0 * np.exp(-((t - t0) / tau)**2)

        # Apply pulse to fields
        Exx[:] = pulse * np.exp(1j * 2 * np.pi * c0 * t / wavelength)
        Eyy[:] = 0.5 * pulse * np.exp(1j * 2 * np.pi * c0 * t / wavelength)
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
        if step % 20 == 0:  # Store every 20th step
            times.append(t)
            field_data.append({
                'amplitude': np.max(np.abs(Exx)),
                'phase': np.angle(Exx[np.argmax(np.abs(Exx))])
            })
            polarization_data.append({
                'amplitude': np.max(np.abs(Pxx)),
                'phase': np.angle(Pxx[np.argmax(np.abs(Pxx))])
            })
            charge_data.append(np.max(np.abs(Rho)))

            # Calculate energy
            if Ee is not None and Eh is not None:
                energy_data.append({
                    'field': np.sum(np.abs(Exx)**2) * (rr[1] - rr[0]),
                    'polarization': np.sum(np.abs(Pxx)**2) * (rr[1] - rr[0])
                })

    print(f"✓ Completed {Nt} time steps")

    return times, field_data, polarization_data, charge_data, energy_data


def analyze_optical_properties():
    """Analyze optical properties of the quantum wire."""
    print("\nOptical Properties Analysis")
    print("==========================")

    # Get linear susceptibility
    chi = chiqw()
    print(f"Linear susceptibility: {chi:.2e}")

    # Calculate optical constants
    n = np.sqrt(1 + chi.real)
    k = chi.imag / (2 * n)
    alpha = 4 * np.pi * k / (800e-9)  # Absorption coefficient

    print(f"Refractive index: {n:.3f}")
    print(f"Extinction coefficient: {k:.3f}")
    print(f"Absorption coefficient: {alpha:.2e} m⁻¹")

    # Calculate oscillator strength
    if dcv is not None:
        f_osc = 2 * me * abs(dcv)**2 / (hbar * e0**2)
        print(f"Oscillator strength: {f_osc:.3f}")

    return chi, n, k, alpha


def analyze_band_structure():
    """Analyze band structure and material properties."""
    print("\nBand Structure Analysis")
    print("=======================")

    if Ee is not None and Eh is not None and kr is not None:
        # Calculate effective masses
        if len(Ee) > 1:
            me_eff = hbar**2 / (2 * Ee[1] / kr[1]**2)
            mh_eff = hbar**2 / (2 * Eh[1] / kr[1]**2)
        else:
            me_eff = me
            mh_eff = mh

        print(f"Electron effective mass: {me_eff/me0:.3f} me")
        print(f"Hole effective mass: {mh_eff/me0:.3f} me")

        # Calculate band gap
        band_gap = gap / eV
        print(f"Band gap: {band_gap:.3f} eV")

        # Calculate exciton binding energy
        a0 = 4 * np.pi * eps0 * hbar**2 / (e0**2 * me_eff)
        E_bind = e0**2 / (8 * np.pi * eps0 * a0)
        print(f"Exciton binding energy: {E_bind/eV:.3f} eV")

        # Calculate quantum wire dimensions
        wire_area = QWArea()
        wire_radius = np.sqrt(wire_area / np.pi)
        print(f"Wire area: {wire_area:.2e} m²")
        print(f"Wire radius: {wire_radius:.2e} m")

        # Calculate confinement energies
        E_conf_e = hbar**2 * np.pi**2 / (2 * me_eff * L**2)
        E_conf_h = hbar**2 * np.pi**2 / (2 * mh_eff * L**2)
        print(f"Electron confinement energy: {E_conf_e/eV:.3f} eV")
        print(f"Hole confinement energy: {E_conf_h/eV:.3f} eV")

        return me_eff, mh_eff, band_gap, E_bind, wire_area, E_conf_e, E_conf_h
    else:
        print("Band structure not available")
        return None, None, None, None, None, None, None


def plot_optical_response(times, field_data, polarization_data, charge_data, energy_data):
    """Plot optical response results."""
    print("\nGenerating optical response plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert times to femtoseconds
    times_fs = [t * 1e15 for t in times]

    # Plot field amplitude and phase
    field_amps = [d['amplitude'] for d in field_data]
    field_phases = [d['phase'] for d in field_data]

    axes[0, 0].plot(times_fs, field_amps, 'b-', label='Amplitude')
    axes[0, 0].set_xlabel('Time (fs)')
    axes[0, 0].set_ylabel('Field Amplitude (V/m)')
    axes[0, 0].set_title('External Field')
    axes[0, 0].grid(True)

    axes[0, 1].plot(times_fs, field_phases, 'r-', label='Phase')
    axes[0, 1].set_xlabel('Time (fs)')
    axes[0, 1].set_ylabel('Field Phase (rad)')
    axes[0, 1].set_title('Field Phase')
    axes[0, 1].grid(True)

    # Plot polarization amplitude and phase
    pol_amps = [d['amplitude'] for d in polarization_data]
    pol_phases = [d['phase'] for d in polarization_data]

    axes[0, 2].plot(times_fs, pol_amps, 'g-', label='Amplitude')
    axes[0, 2].set_xlabel('Time (fs)')
    axes[0, 2].set_ylabel('Polarization (C/m²)')
    axes[0, 2].set_title('Quantum Wire Polarization')
    axes[0, 2].grid(True)

    # Plot charge density
    axes[1, 0].plot(times_fs, charge_data, 'm-', label='Charge Density')
    axes[1, 0].set_xlabel('Time (fs)')
    axes[1, 0].set_ylabel('Charge Density (C/m³)')
    axes[1, 0].set_title('Charge Density')
    axes[1, 0].grid(True)

    # Plot energy
    if energy_data:
        field_energy = [d['field'] for d in energy_data]
        pol_energy = [d['polarization'] for d in energy_data]

        axes[1, 1].plot(times_fs, field_energy, 'b-', label='Field Energy')
        axes[1, 1].plot(times_fs, pol_energy, 'g-', label='Polarization Energy')
        axes[1, 1].set_xlabel('Time (fs)')
        axes[1, 1].set_ylabel('Energy (arb. units)')
        axes[1, 1].set_title('Energy Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    # Plot band structure
    if Ee is not None and Eh is not None and kr is not None:
        axes[1, 2].plot(kr * 1e-9, Ee / eV, 'b-', label='Electrons')
        axes[1, 2].plot(kr * 1e-9, Eh / eV, 'r-', label='Holes')
        axes[1, 2].set_xlabel('Momentum (nm⁻¹)')
        axes[1, 2].set_ylabel('Energy (eV)')
        axes[1, 2].set_title('GaAs Band Structure')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    else:
        axes[1, 2].text(0.5, 0.5, 'Band structure\nnot available',
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Band Structure')

    plt.tight_layout()
    plt.savefig('gaas_optical_response.png', dpi=300, bbox_inches='tight')
    print("✓ Plots saved to gaas_optical_response.png")

    return fig


def main():
    """Run GaAs quantum wire domain example."""
    print("GaAs Quantum Wire Domain Example")
    print("===============================")
    print("This example simulates the optical response of a GaAs quantum wire")
    print("with realistic material parameters and physical scenarios.")

    # Create GaAs parameter environment
    temp_dir, original_dir = create_gaas_parameters()

    try:
        # Run optical response simulation
        times, field_data, polarization_data, charge_data, energy_data = simulate_optical_response()

        # Analyze optical properties
        chi, n, k, alpha = analyze_optical_properties()

        # Analyze band structure
        me_eff, mh_eff, band_gap, E_bind, wire_area, E_conf_e, E_conf_h = analyze_band_structure()

        # Generate plots
        fig = plot_optical_response(times, field_data, polarization_data, charge_data, energy_data)

        # Print summary
        print("\nGaAs Quantum Wire Summary")
        print("========================")
        print(f"Wire length: {L:.1e} m")
        print(f"Wire thickness: {Delta0:.1e} m")
        print(f"Band gap: {band_gap:.3f} eV")
        print(f"Electron effective mass: {me_eff/me0:.3f} me")
        print(f"Hole effective mass: {mh_eff/me0:.3f} me")
        print(f"Exciton binding energy: {E_bind/eV:.3f} eV")
        print(f"Wire area: {wire_area:.2e} m²")
        print(f"Electron confinement energy: {E_conf_e/eV:.3f} eV")
        print(f"Hole confinement energy: {E_conf_h/eV:.3f} eV")

        print(f"\nOptical Properties:")
        print(f"Refractive index: {n:.3f}")
        print(f"Extinction coefficient: {k:.3f}")
        print(f"Absorption coefficient: {alpha:.2e} m⁻¹")

        print(f"\nSimulation Results:")
        print(f"Total time steps: {len(times)}")
        print(f"Time range: {times[0]*1e15:.1f} - {times[-1]*1e15:.1f} fs")
        print(f"Peak field amplitude: {max([d['amplitude'] for d in field_data]):.2e} V/m")
        print(f"Peak polarization: {max([d['amplitude'] for d in polarization_data]):.2e} C/m²")
        print(f"Peak charge density: {max(charge_data):.2e} C/m³")

        print("\n✓ GaAs quantum wire simulation completed successfully!")
        print("\nThis example demonstrates:")
        print("- Realistic GaAs material parameters")
        print("- Optical pulse propagation and response")
        print("- Quantum wire band structure analysis")
        print("- Exciton binding energy calculations")
        print("- Optical property analysis")
        print("- Time-resolved response visualization")

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        cleanup_environment(temp_dir, original_dir)


if __name__ == "__main__":
    main()
