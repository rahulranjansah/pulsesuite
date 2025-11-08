#!/usr/bin/env python3
"""
DC Field Workflow Example
=========================

Workflow Example: Show complete research workflow from data to insight
Style: Real data loading, preprocessing, analysis, visualization
Audience: Researcher adapting to their own data

This example demonstrates a complete workflow for analyzing DC field effects
in quantum wire semiconductor devices with realistic parameters.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.dcfieldpythonic as dcf
from src.constants import hbar, e0, eps0

def create_realistic_quantum_wire_data():
    """Create realistic quantum wire parameters and data."""
    print("Creating realistic quantum wire data...")

    # Physical parameters for GaAs quantum wire
    me = 0.067 * 9.109e-31  # Electron effective mass (kg)
    mh = 0.45 * 9.109e-31   # Hole effective mass (kg)
    L = 100e-9              # Wire length (m)
    W = 10e-9               # Wire width (m)

    # Momentum grid (realistic for 100nm wire)
    N = 64
    k_max = np.pi / W  # Maximum momentum for wire width
    ky = np.linspace(-k_max, k_max, N)

    # Realistic energy dispersions with confinement
    Ee = (hbar**2 * ky**2) / (2 * me) + 0.1 * 1.602e-19  # Band gap + kinetic
    Eh = (hbar**2 * ky**2) / (2 * mh) + 0.1 * 1.602e-19

    # Realistic population distributions (Fermi-like)
    kT = 0.025 * 1.602e-19  # Thermal energy at 300K (J)
    mu_e = 0.05 * 1.602e-19  # Chemical potential (J)
    mu_h = -0.05 * 1.602e-19

    ne = 1.0 / (1.0 + np.exp((Ee - mu_e) / kT))
    nh = 1.0 / (1.0 + np.exp((Eh - mu_h) / kT))

    # Convert to complex arrays
    ne = ne.astype(np.complex128)
    nh = nh.astype(np.complex128)

    # Realistic Coulomb potential matrices
    Vee = np.zeros((N, N), dtype=np.float64)
    Vhh = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            q = abs(ky[i] - ky[j])
            if q > 1e6:  # Avoid singularity
                Vee[i, j] = e0**2 / (4 * np.pi * eps0 * 12.0 * q)  # GaAs dielectric
                Vhh[i, j] = Vee[i, j]
            else:
                Vee[i, j] = 1e-20
                Vhh[i, j] = 1e-20

    # Phonon coupling (realistic for GaAs)
    Cq2 = np.ones(N, dtype=np.float64) * 1e-18

    return {
        'me': me, 'mh': mh, 'ky': ky, 'Ee': Ee, 'Eh': Eh,
        'ne': ne, 'nh': nh, 'Vee': Vee, 'Vhh': Vhh, 'Cq2': Cq2
    }

def analyze_dc_field_response(data, field_range):
    """Analyze DC field response over a range of field strengths."""
    print("Analyzing DC field response...")

    # Initialize solver
    solver = dcf.InitializeDC(data['ky'], data['me'], data['mh'])

    results = {
        'fields': [],
        'currents': [],
        'e_drift_rates': [],
        'h_drift_rates': [],
        'e_drift_velocities': [],
        'h_drift_velocities': []
    }

    for Edc in field_range:
        # Calculate DC contributions
        DC_e = dcf.CalcDCE2(
            DCTrans=True, ky=data['ky'], Cq2=data['Cq2'], Edc=Edc,
            me=data['me'], ge=1e12, Ephn=1e13, N0=0.0,
            ne=data['ne'], Ee=data['Ee'], Vee=data['Vee'],
            n=1, j=1, solver=solver
        )

        DC_h = dcf.CalcDCH2(
            DCTrans=True, ky=data['ky'], Cq2=data['Cq2'], Edc=Edc,
            mh=data['mh'], gh=1e12, Ephn=1e13, N0=0.0,
            nh=data['nh'], Eh=data['Eh'], Vhh=data['Vhh'],
            n=1, j=1, solver=solver
        )

        # Calculate current
        VC = np.stack([np.zeros_like(data['Vee']), data['Vee'], data['Vhh']], axis=2)
        dk = data['ky'][1] - data['ky'][0]
        I0 = dcf.CalcI0(data['ne'], data['nh'], data['Ee'], data['Eh'],
                       VC, dk, data['ky'], solver)

        # Store results
        results['fields'].append(Edc)
        results['currents'].append(I0)
        results['e_drift_rates'].append(dcf.GetEDrift(solver))
        results['h_drift_rates'].append(dcf.GetHDrift(solver))
        results['e_drift_velocities'].append(dcf.GetVEDrift(solver))
        results['h_drift_velocities'].append(dcf.GetVHDrift(solver))

    return results

def analyze_momentum_distribution_evolution(data, Edc, time_steps):
    """Analyze how momentum distribution evolves under DC field."""
    print("Analyzing momentum distribution evolution...")

    solver = dcf.InitializeDC(data['ky'], data['me'], data['mh'])

    # Initial distributions
    ne_evolved = data['ne'].copy()
    nh_evolved = data['nh'].copy()

    evolution = {
        'times': [],
        'ne_distributions': [],
        'nh_distributions': [],
        'total_currents': []
    }

    dt = 1e-15  # Time step (1 fs)

    for step in range(time_steps):
        t = step * dt

        # Calculate DC contributions
        DC_e = dcf.CalcDCE2(
            DCTrans=True, ky=data['ky'], Cq2=data['Cq2'], Edc=Edc,
            me=data['me'], ge=1e12, Ephn=1e13, N0=0.0,
            ne=ne_evolved, Ee=data['Ee'], Vee=data['Vee'],
            n=step, j=1, solver=solver
        )

        DC_h = dcf.CalcDCH2(
            DCTrans=True, ky=data['ky'], Cq2=data['Cq2'], Edc=Edc,
            mh=data['mh'], gh=1e12, Ephn=1e13, N0=0.0,
            nh=nh_evolved, Eh=data['Eh'], Vhh=data['Vhh'],
            n=step, j=1, solver=solver
        )

        # Simple evolution (Euler method)
        ne_evolved += DC_e * dt
        nh_evolved += DC_h * dt

        # Ensure physical bounds
        ne_evolved = np.clip(ne_evolved, 0, 1)
        nh_evolved = np.clip(nh_evolved, 0, 1)

        # Calculate current
        VC = np.stack([np.zeros_like(data['Vee']), data['Vee'], data['Vhh']], axis=2)
        dk = data['ky'][1] - data['ky'][0]
        I0 = dcf.CalcI0(ne_evolved, nh_evolved, data['Ee'], data['Eh'],
                       VC, dk, data['ky'], solver)

        # Store evolution
        if step % 10 == 0:  # Store every 10 steps
            evolution['times'].append(t)
            evolution['ne_distributions'].append(ne_evolved.copy())
            evolution['nh_distributions'].append(nh_evolved.copy())
            evolution['total_currents'].append(I0)

    return evolution

def create_comprehensive_visualization(data, field_results, evolution):
    """Create comprehensive visualization of results."""
    if not HAS_MATPLOTLIB:
        print("Note: matplotlib not available for visualization")
        return

    print("Creating comprehensive visualization...")

    fig = plt.figure(figsize=(16, 12))

    # 1. Initial momentum distributions
    plt.subplot(3, 4, 1)
    plt.plot(data['ky']/1e8, np.real(data['ne']), 'b-', label='Electrons', linewidth=2)
    plt.plot(data['ky']/1e8, np.real(data['nh']), 'r-', label='Holes', linewidth=2)
    plt.xlabel('Momentum (10^8 1/m)')
    plt.ylabel('Population')
    plt.title('Initial Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Energy dispersions
    plt.subplot(3, 4, 2)
    plt.plot(data['ky']/1e8, data['Ee']/1.602e-19, 'b-', label='Electrons', linewidth=2)
    plt.plot(data['ky']/1e8, data['Eh']/1.602e-19, 'r-', label='Holes', linewidth=2)
    plt.xlabel('Momentum (10^8 1/m)')
    plt.ylabel('Energy (eV)')
    plt.title('Energy Dispersions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Current vs Field
    plt.subplot(3, 4, 3)
    plt.plot(np.array(field_results['fields'])/1e5, np.array(field_results['currents'])/1e-6, 'g-', linewidth=2)
    plt.xlabel('DC Field (10^5 V/m)')
    plt.ylabel('Current (μA)')
    plt.title('I-V Characteristic')
    plt.grid(True, alpha=0.3)

    # 4. Drift velocities vs Field
    plt.subplot(3, 4, 4)
    plt.plot(np.array(field_results['fields'])/1e5, np.array(field_results['e_drift_velocities'])/1e3, 'b-', label='Electrons', linewidth=2)
    plt.plot(np.array(field_results['fields'])/1e5, np.array(field_results['h_drift_velocities'])/1e3, 'r-', label='Holes', linewidth=2)
    plt.xlabel('DC Field (10^5 V/m)')
    plt.ylabel('Drift Velocity (10^3 m/s)')
    plt.title('Drift Velocities')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Drift rates vs Field
    plt.subplot(3, 4, 5)
    plt.plot(np.array(field_results['fields'])/1e5, np.array(field_results['e_drift_rates']), 'b-', label='Electrons', linewidth=2)
    plt.plot(np.array(field_results['fields'])/1e5, np.array(field_results['h_drift_rates']), 'r-', label='Holes', linewidth=2)
    plt.xlabel('DC Field (10^5 V/m)')
    plt.ylabel('Drift Rate (1/s)')
    plt.title('Drift Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Evolution of electron distribution
    plt.subplot(3, 4, 6)
    for i, (t, ne_dist) in enumerate(zip(evolution['times'], evolution['ne_distributions'])):
        alpha = 0.3 + 0.7 * i / len(evolution['times'])
        plt.plot(data['ky']/1e8, np.real(ne_dist), 'b-', alpha=alpha, linewidth=1)
    plt.xlabel('Momentum (10^8 1/m)')
    plt.ylabel('Electron Population')
    plt.title('Electron Evolution')
    plt.grid(True, alpha=0.3)

    # 7. Evolution of hole distribution
    plt.subplot(3, 4, 7)
    for i, (t, nh_dist) in enumerate(zip(evolution['times'], evolution['nh_distributions'])):
        alpha = 0.3 + 0.7 * i / len(evolution['times'])
        plt.plot(data['ky']/1e8, np.real(nh_dist), 'r-', alpha=alpha, linewidth=1)
    plt.xlabel('Momentum (10^8 1/m)')
    plt.ylabel('Hole Population')
    plt.title('Hole Evolution')
    plt.grid(True, alpha=0.3)

    # 8. Current evolution
    plt.subplot(3, 4, 8)
    plt.plot(np.array(evolution['times'])/1e-12, np.array(evolution['total_currents'])/1e-6, 'g-', linewidth=2)
    plt.xlabel('Time (ps)')
    plt.ylabel('Current (μA)')
    plt.title('Current Evolution')
    plt.grid(True, alpha=0.3)

    # 9. Coulomb potential matrix (electron-electron)
    plt.subplot(3, 4, 9)
    im1 = plt.imshow(data['Vee']/1e-20, cmap='viridis', aspect='auto')
    plt.colorbar(im1, label='Vee (10^-20 J)')
    plt.title('Electron-Electron Potential')
    plt.xlabel('Momentum Index')
    plt.ylabel('Momentum Index')

    # 10. Coulomb potential matrix (hole-hole)
    plt.subplot(3, 4, 10)
    im2 = plt.imshow(data['Vhh']/1e-20, cmap='viridis', aspect='auto')
    plt.colorbar(im2, label='Vhh (10^-20 J)')
    plt.title('Hole-Hole Potential')
    plt.xlabel('Momentum Index')
    plt.ylabel('Momentum Index')

    # 11. Phonon coupling
    plt.subplot(3, 4, 11)
    plt.plot(data['ky']/1e8, data['Cq2']/1e-18, 'm-', linewidth=2)
    plt.xlabel('Momentum (10^8 1/m)')
    plt.ylabel('Cq2 (10^-18)')
    plt.title('Phonon Coupling')
    plt.grid(True, alpha=0.3)

    # 12. Summary statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')

    # Calculate summary statistics
    max_current = np.max(np.abs(field_results['currents']))
    max_e_velocity = np.max(np.abs(field_results['e_drift_velocities']))
    max_h_velocity = np.max(np.abs(field_results['h_drift_velocities']))

    summary_text = f"""
    Summary Statistics:

    Max Current: {max_current/1e-6:.2f} μA
    Max e- Velocity: {max_e_velocity/1e3:.2f} km/s
    Max h+ Velocity: {max_h_velocity/1e3:.2f} km/s

    Grid Points: {len(data['ky'])}
    Wire Width: {10e-9/1e-9:.1f} nm
    Temperature: 300 K

    Material: GaAs
    e- Mass: {data['me']/9.109e-31:.3f} me
    h+ Mass: {data['mh']/9.109e-31:.3f} me
    """

    plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.savefig('dcfield_workflow_results.png', dpi=150, bbox_inches='tight')
    print("✓ Comprehensive results saved to dcfield_workflow_results.png")

def main():
    """Main workflow example."""
    print("DC Field Workflow Example")
    print("=" * 50)

    # 1. Create realistic quantum wire data
    data = create_realistic_quantum_wire_data()
    print(f"✓ Created realistic data for {len(data['ky'])} momentum points")

    # 2. Analyze DC field response
    field_range = np.linspace(1e4, 1e6, 20)  # 0.01 to 1 MV/m
    field_results = analyze_dc_field_response(data, field_range)
    print(f"✓ Analyzed DC field response for {len(field_range)} field values")

    # 3. Analyze momentum distribution evolution
    Edc_test = 5e5  # 0.5 MV/m
    time_steps = 100
    evolution = analyze_momentum_distribution_evolution(data, Edc_test, time_steps)
    print(f"✓ Analyzed evolution over {time_steps} time steps")

    # 4. Create comprehensive visualization
    create_comprehensive_visualization(data, field_results, evolution)

    # 5. Print key insights
    print("\nKey Insights:")
    print("-" * 30)

    max_current_idx = np.argmax(np.abs(field_results['currents']))
    max_field = field_results['fields'][max_current_idx]
    max_current = field_results['currents'][max_current_idx]

    print(f"• Maximum current: {max_current/1e-6:.2f} μA at {max_field/1e5:.1f} × 10^5 V/m")
    print(f"• Current shows {'linear' if np.corrcoef(field_results['fields'], field_results['currents'])[0,1] > 0.9 else 'non-linear'} behavior")
    print(f"• Electron drift velocity: {np.max(np.abs(field_results['e_drift_velocities']))/1e3:.1f} km/s")
    print(f"• Hole drift velocity: {np.max(np.abs(field_results['h_drift_velocities']))/1e3:.1f} km/s")

    # Check for saturation
    current_ratio = np.max(np.abs(field_results['currents'])) / np.min(np.abs(field_results['currents']))
    if current_ratio > 10:
        print("• Current shows significant field dependence (no saturation)")
    else:
        print("• Current shows saturation behavior at high fields")

    print("\n✓ Workflow analysis completed successfully!")

if __name__ == "__main__":
    main()
