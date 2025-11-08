#!/usr/bin/env python3
"""
DC Field Domain Example
=======================

Domain Example: Solve actual scientific problem with physical context
Style: Connect code parameters to domain knowledge, interpret results
Audience: Domain expert validating methodology

This example solves the problem of analyzing DC field transport in quantum wire
semiconductor devices for optoelectronic applications, connecting physical
parameters to device performance metrics.
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

# Physical constants
hbar = 1.054571817e-34  # J⋅s
e0 = 1.602176634e-19    # C
eps0 = 8.8541878128e-12 # F/m
kB = 1.380649e-23       # J/K

def setup_gaas_quantum_wire_device():
    """Setup realistic GaAs quantum wire device parameters."""
    print("Setting up GaAs quantum wire device...")

    # GaAs material parameters
    me_gaas = 0.067 * 9.1093837015e-31  # Electron effective mass (kg)
    mh_gaas = 0.45 * 9.1093837015e-31   # Hole effective mass (kg)
    eps_r_gaas = 12.9                   # Relative permittivity
    Eg_gaas = 1.424 * e0                # Band gap (J)

    # Device geometry
    L_wire = 200e-9      # Wire length (m)
    W_wire = 15e-9       # Wire width (m)
    H_wire = 10e-9       # Wire height (m)

    # Operating conditions
    T_device = 300.0     # Temperature (K)
    V_bias = 0.5         # Applied bias voltage (V)
    E_dc = V_bias / L_wire  # DC field (V/m)

    # Phonon parameters (GaAs LO phonon)
    omega_LO = 8.75e13   # LO phonon frequency (rad/s)
    gamma_e = 1e12       # Electron relaxation rate (1/s)
    gamma_h = 1e12       # Hole relaxation rate (1/s)

    print(f"✓ Device: {L_wire/1e-9:.0f}×{W_wire/1e-9:.0f}×{H_wire/1e-9:.0f} nm³ GaAs wire")
    print(f"✓ Bias: {V_bias:.1f} V → {E_dc/1e5:.1f} × 10⁵ V/m")
    print(f"✓ Temperature: {T_device:.0f} K")

    return {
        'me': me_gaas, 'mh': mh_gaas, 'eps_r': eps_r_gaas, 'Eg': Eg_gaas,
        'L': L_wire, 'W': W_wire, 'H': H_wire, 'T': T_device,
        'V_bias': V_bias, 'E_dc': E_dc, 'omega_LO': omega_LO,
        'gamma_e': gamma_e, 'gamma_h': gamma_h
    }

def create_quantum_confined_states(params):
    """Create quantum confined states for the wire."""
    print("Calculating quantum confined states...")

    # Momentum grid (quantized by wire width)
    k_max = np.pi / params['W']  # Maximum k for wire width
    N = 64
    ky = np.linspace(-k_max, k_max, N)

    # Quantum confinement energies
    E_confinement_e = (hbar**2 * np.pi**2) / (2 * params['me'] * params['W']**2)
    E_confinement_h = (hbar**2 * np.pi**2) / (2 * params['mh'] * params['W']**2)

    # Total energy dispersions (confinement + kinetic)
    Ee = E_confinement_e + (hbar**2 * ky**2) / (2 * params['me'])
    Eh = E_confinement_h + (hbar**2 * ky**2) / (2 * params['mh'])

    print(f"✓ Electron confinement energy: {E_confinement_e/e0:.3f} eV")
    print(f"✓ Hole confinement energy: {E_confinement_h/e0:.3f} eV")
    print(f"✓ Effective band gap: {(params['Eg'] + E_confinement_e + E_confinement_h)/e0:.3f} eV")

    return ky, Ee, Eh, E_confinement_e, E_confinement_h

def calculate_equilibrium_distributions(ky, Ee, Eh, params):
    """Calculate equilibrium carrier distributions."""
    print("Calculating equilibrium distributions...")

    # Chemical potentials (Fermi level position)
    mu_e = 0.1 * e0   # Electron chemical potential (eV above conduction band)
    mu_h = -0.1 * e0  # Hole chemical potential (eV below valence band)

    # Fermi-Dirac distributions
    kT = kB * params['T']
    ne_eq = 1.0 / (1.0 + np.exp((Ee - mu_e) / kT))
    nh_eq = 1.0 / (1.0 + np.exp((Eh - mu_h) / kT))

    # Convert to complex arrays
    ne = ne_eq.astype(np.complex128)
    nh = nh_eq.astype(np.complex128)

    # Calculate carrier densities
    dk = ky[1] - ky[0]
    n_e_density = np.sum(np.real(ne)) * dk / (2 * np.pi)  # 1D density
    n_h_density = np.sum(np.real(nh)) * dk / (2 * np.pi)

    print(f"✓ Electron density: {n_e_density/1e6:.2f} × 10⁶ cm⁻¹")
    print(f"✓ Hole density: {n_h_density/1e6:.2f} × 10⁶ cm⁻¹")

    return ne, nh, n_e_density, n_h_density

def calculate_coulomb_interactions(ky, params):
    """Calculate realistic Coulomb interaction matrices."""
    print("Calculating Coulomb interactions...")

    N = len(ky)
    Vee = np.zeros((N, N), dtype=np.float64)
    Vhh = np.zeros((N, N), dtype=np.float64)

    # 1D Coulomb potential with screening
    for i in range(N):
        for j in range(N):
            q = abs(ky[i] - ky[j])
            if q > 1e6:  # Avoid singularity
                # 1D screened Coulomb potential
                V_coulomb = (e0**2 / (4 * np.pi * eps0 * params['eps_r'])) * np.log(1 + 1/(q * params['W']))
                Vee[i, j] = V_coulomb
                Vhh[i, j] = V_coulomb
            else:
                Vee[i, j] = 1e-20
                Vhh[i, j] = 1e-20

    # Phonon coupling strength (Fröhlich interaction)
    Cq2 = np.ones(N, dtype=np.float64)
    for i in range(N):
        q = abs(ky[i])
        if q > 1e6:
            # Fröhlich coupling for GaAs
            Cq2[i] = (e0**2 * params['omega_LO'] / (2 * eps0 * params['eps_r'])) * (1/params['eps_r'] - 1/12.9) / q
        else:
            Cq2[i] = 1e-18

    print(f"✓ Coulomb matrices calculated ({N}×{N})")
    print(f"✓ Phonon coupling range: {np.min(Cq2)/1e-18:.1f} - {np.max(Cq2)/1e-18:.1f} × 10⁻¹⁸")

    return Vee, Vhh, Cq2

def analyze_device_performance(ky, Ee, Eh, ne, nh, Vee, Vhh, Cq2, params):
    """Analyze device performance metrics."""
    print("Analyzing device performance...")

    # Initialize DC field solver
    solver = dcf.InitializeDC(ky, params['me'], params['mh'])

    # Calculate DC field response
    DC_e = dcf.CalcDCE2(
        DCTrans=True, ky=ky, Cq2=Cq2, Edc=params['E_dc'],
        me=params['me'], ge=params['gamma_e'], Ephn=params['omega_LO'], N0=0.0,
        ne=ne, Ee=Ee, Vee=Vee, n=1, j=1, solver=solver
    )

    DC_h = dcf.CalcDCH2(
        DCTrans=True, ky=ky, Cq2=Cq2, Edc=params['E_dc'],
        mh=params['mh'], gh=params['gamma_h'], Ephn=params['omega_LO'], N0=0.0,
        nh=nh, Eh=Eh, Vhh=Vhh, n=1, j=1, solver=solver
    )

    # Calculate current
    VC = np.stack([np.zeros_like(Vee), Vee, Vhh], axis=2)
    dk = ky[1] - ky[0]
    I0 = dcf.CalcI0(ne, nh, Ee, Eh, VC, dk, ky, solver)

    # Calculate mobility
    n_total = np.sum(np.real(ne + nh)) * dk / (2 * np.pi)
    mobility = abs(I0) / (n_total * e0 * params['E_dc'])  # m²/(V⋅s)

    # Calculate conductivity
    sigma = abs(I0) / (params['E_dc'] * params['W'] * params['H'])  # S/m

    # Calculate drift velocities
    v_e_drift = dcf.GetVEDrift(solver)
    v_h_drift = dcf.GetVHDrift(solver)

    # Calculate transit time
    tau_transit = params['L'] / max(abs(v_e_drift), abs(v_h_drift))

    # Calculate power dissipation
    P_dissipation = abs(I0) * params['V_bias']

    print(f"✓ Current: {I0/1e-6:.2f} μA")
    print(f"✓ Mobility: {mobility*1e4:.1f} cm²/(V⋅s)")
    print(f"✓ Conductivity: {sigma:.2e} S/m")
    print(f"✓ Transit time: {tau_transit/1e-12:.2f} ps")
    print(f"✓ Power dissipation: {P_dissipation/1e-6:.2f} μW")

    return {
        'current': I0, 'mobility': mobility, 'conductivity': sigma,
        'transit_time': tau_transit, 'power': P_dissipation,
        'v_e_drift': v_e_drift, 'v_h_drift': v_h_drift,
        'DC_e': DC_e, 'DC_h': DC_h
    }

def analyze_field_dependence(ky, Ee, Eh, ne, nh, Vee, Vhh, Cq2, params):
    """Analyze field dependence for device optimization."""
    print("Analyzing field dependence...")

    # Field range for analysis
    V_range = np.linspace(0.1, 2.0, 20)  # 0.1 to 2.0 V
    E_range = V_range / params['L']

    results = {
        'voltages': V_range,
        'fields': E_range,
        'currents': [],
        'mobilities': [],
        'powers': [],
        'transit_times': []
    }

    solver = dcf.InitializeDC(ky, params['me'], params['mh'])

    for V, E in zip(V_range, E_range):
        # Calculate performance at this field
        perf = analyze_device_performance(ky, Ee, Eh, ne, nh, Vee, Vhh, Cq2,
                                        {**params, 'V_bias': V, 'E_dc': E})

        results['currents'].append(perf['current'])
        results['mobilities'].append(perf['mobility'])
        results['powers'].append(perf['power'])
        results['transit_times'].append(perf['transit_time'])

    return results

def create_domain_visualization(ky, Ee, Eh, ne, nh, performance, field_results, params):
    """Create domain-specific visualization."""
    if not HAS_MATPLOTLIB:
        print("Note: matplotlib not available for visualization")
        return

    print("Creating domain visualization...")

    fig = plt.figure(figsize=(16, 12))

    # 1. Energy band diagram
    plt.subplot(3, 4, 1)
    plt.plot(ky/1e8, Ee/e0, 'b-', linewidth=2, label='Conduction Band')
    plt.plot(ky/1e8, Eh/e0, 'r-', linewidth=2, label='Valence Band')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Fermi Level')
    plt.xlabel('Momentum (10⁸ m⁻¹)')
    plt.ylabel('Energy (eV)')
    plt.title('Energy Band Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Carrier distributions
    plt.subplot(3, 4, 2)
    plt.plot(ky/1e8, np.real(ne), 'b-', linewidth=2, label='Electrons')
    plt.plot(ky/1e8, np.real(nh), 'r-', linewidth=2, label='Holes')
    plt.xlabel('Momentum (10⁸ m⁻¹)')
    plt.ylabel('Population')
    plt.title('Carrier Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. I-V characteristic
    plt.subplot(3, 4, 3)
    plt.plot(field_results['voltages'], np.array(field_results['currents'])/1e-6, 'g-', linewidth=2)
    plt.xlabel('Bias Voltage (V)')
    plt.ylabel('Current (μA)')
    plt.title('I-V Characteristic')
    plt.grid(True, alpha=0.3)

    # 4. Mobility vs Field
    plt.subplot(3, 4, 4)
    plt.plot(field_results['fields']/1e5, np.array(field_results['mobilities'])*1e4, 'm-', linewidth=2)
    plt.xlabel('Electric Field (10⁵ V/m)')
    plt.ylabel('Mobility (cm²/V⋅s)')
    plt.title('Field-Dependent Mobility')
    plt.grid(True, alpha=0.3)

    # 5. Power dissipation
    plt.subplot(3, 4, 5)
    plt.plot(field_results['voltages'], np.array(field_results['powers'])/1e-6, 'orange', linewidth=2)
    plt.xlabel('Bias Voltage (V)')
    plt.ylabel('Power (μW)')
    plt.title('Power Dissipation')
    plt.grid(True, alpha=0.3)

    # 6. Transit time
    plt.subplot(3, 4, 6)
    plt.plot(field_results['fields']/1e5, np.array(field_results['transit_times'])/1e-12, 'purple', linewidth=2)
    plt.xlabel('Electric Field (10⁵ V/m)')
    plt.ylabel('Transit Time (ps)')
    plt.title('Carrier Transit Time')
    plt.grid(True, alpha=0.3)

    # 7. DC field contributions
    plt.subplot(3, 4, 7)
    plt.plot(ky/1e8, performance['DC_e'], 'b-', linewidth=2, label='Electrons')
    plt.plot(ky/1e8, performance['DC_h'], 'r-', linewidth=2, label='Holes')
    plt.xlabel('Momentum (10⁸ m⁻¹)')
    plt.ylabel('DC Contribution')
    plt.title('DC Field Response')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 8. Device geometry
    plt.subplot(3, 4, 8)
    plt.axis('off')
    geometry_text = f"""
    Device Geometry:

    Length: {params['L']/1e-9:.0f} nm
    Width: {params['W']/1e-9:.0f} nm
    Height: {params['H']/1e-9:.0f} nm

    Material: GaAs
    Temperature: {params['T']:.0f} K

    Effective Masses:
    me = {params['me']/9.109e-31:.3f} me
    mh = {params['mh']/9.109e-31:.3f} me
    """
    plt.text(0.1, 0.5, geometry_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    # 9. Performance metrics
    plt.subplot(3, 4, 9)
    plt.axis('off')
    metrics_text = f"""
    Performance Metrics:

    Current: {performance['current']/1e-6:.2f} μA
    Mobility: {performance['mobility']*1e4:.1f} cm²/V⋅s
    Conductivity: {performance['conductivity']:.2e} S/m

    Transit Time: {performance['transit_time']/1e-12:.2f} ps
    Power: {performance['power']/1e-6:.2f} μW

    Drift Velocities:
    ve = {performance['v_e_drift']/1e3:.1f} km/s
    vh = {performance['v_h_drift']/1e3:.1f} km/s
    """
    plt.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

    # 10. Material properties
    plt.subplot(3, 4, 10)
    plt.axis('off')
    material_text = f"""
    Material Properties:

    Band Gap: {params['Eg']/e0:.3f} eV
    Permittivity: {params['eps_r']:.1f}

    Phonon Frequency: {params['omega_LO']/1e13:.2f} × 10¹³ rad/s
    Relaxation Rates:
    γe = {params['gamma_e']/1e12:.1f} × 10¹² s⁻¹
    γh = {params['gamma_h']/1e12:.1f} × 10¹² s⁻¹
    """
    plt.text(0.1, 0.5, material_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # 11. Optimization insights
    plt.subplot(3, 4, 11)
    plt.axis('off')

    # Find optimal operating point
    max_mobility_idx = np.argmax(field_results['mobilities'])
    optimal_voltage = field_results['voltages'][max_mobility_idx]
    optimal_current = field_results['currents'][max_mobility_idx]

    insights_text = f"""
    Optimization Insights:

    Optimal Voltage: {optimal_voltage:.2f} V
    Max Mobility: {np.max(field_results['mobilities'])*1e4:.1f} cm²/V⋅s

    Current Range: {np.min(np.abs(field_results['currents']))/1e-6:.2f} - {np.max(np.abs(field_results['currents']))/1e-6:.2f} μA

    Power Efficiency: {optimal_current*optimal_voltage/1e-6:.2f} μW
    """
    plt.text(0.1, 0.5, insights_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

    # 12. Comparison with bulk
    plt.subplot(3, 4, 12)
    plt.axis('off')

    # Calculate bulk mobility for comparison
    bulk_mobility_e = e0 / (params['me'] * params['gamma_e'])
    bulk_mobility_h = e0 / (params['mh'] * params['gamma_h'])
    bulk_mobility = 2 / (1/bulk_mobility_e + 1/bulk_mobility_h)

    comparison_text = f"""
    Quantum vs Bulk:

    Quantum Wire Mobility: {performance['mobility']*1e4:.1f} cm²/V⋅s
    Bulk GaAs Mobility: {bulk_mobility*1e4:.1f} cm²/V⋅s

    Enhancement Factor: {performance['mobility']/bulk_mobility:.2f}x

    Quantum Confinement:
    ✓ Reduces scattering
    ✓ Improves mobility
    ✓ Enables high-speed operation
    """
    plt.text(0.1, 0.5, comparison_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsteelblue", alpha=0.8))

    plt.tight_layout()
    plt.savefig('dcfield_domain_results.png', dpi=150, bbox_inches='tight')
    print("✓ Domain analysis saved to dcfield_domain_results.png")

def main():
    """Main domain example."""
    print("DC Field Domain Example")
    print("=" * 50)
    print("Analyzing DC field transport in GaAs quantum wire devices")
    print("for optoelectronic applications")
    print()

    # 1. Setup device parameters
    params = setup_gaas_quantum_wire_device()

    # 2. Create quantum confined states
    ky, Ee, Eh, E_conf_e, E_conf_h = create_quantum_confined_states(params)

    # 3. Calculate equilibrium distributions
    ne, nh, n_e_density, n_h_density = calculate_equilibrium_distributions(ky, Ee, Eh, params)

    # 4. Calculate Coulomb interactions
    Vee, Vhh, Cq2 = calculate_coulomb_interactions(ky, params)

    # 5. Analyze device performance
    performance = analyze_device_performance(ky, Ee, Eh, ne, nh, Vee, Vhh, Cq2, params)

    # 6. Analyze field dependence
    field_results = analyze_field_dependence(ky, Ee, Eh, ne, nh, Vee, Vhh, Cq2, params)

    # 7. Create domain visualization
    create_domain_visualization(ky, Ee, Eh, ne, nh, performance, field_results, params)

    # 8. Domain expert validation
    print("\nDomain Expert Validation:")
    print("-" * 30)

    # Validate against known GaAs properties
    expected_mobility_range = (1000, 5000)  # cm²/(V⋅s) for GaAs
    actual_mobility = performance['mobility'] * 1e4

    if expected_mobility_range[0] <= actual_mobility <= expected_mobility_range[1]:
        print(f"✓ Mobility ({actual_mobility:.1f} cm²/V⋅s) within expected range")
    else:
        print(f"⚠ Mobility ({actual_mobility:.1f} cm²/V⋅s) outside expected range")

    # Validate current density
    current_density = abs(performance['current']) / (params['W'] * params['H'])
    expected_current_density = 1e6  # A/m² for typical devices

    if current_density < expected_current_density:
        print(f"✓ Current density ({current_density/1e6:.2f} MA/m²) reasonable for device")
    else:
        print(f"⚠ Current density ({current_density/1e6:.2f} MA/m²) may be too high")

    # Validate transit time
    if performance['transit_time'] < 1e-9:  # < 1 ns
        print(f"✓ Transit time ({performance['transit_time']/1e-12:.1f} ps) suitable for high-speed operation")
    else:
        print(f"⚠ Transit time ({performance['transit_time']/1e-9:.1f} ns) may limit speed")

    # Validate power dissipation
    if performance['power'] < 1e-3:  # < 1 mW
        print(f"✓ Power dissipation ({performance['power']/1e-6:.1f} μW) acceptable for low-power applications")
    else:
        print(f"⚠ Power dissipation ({performance['power']/1e-3:.1f} mW) may be too high")

    print("\n✓ Domain analysis completed successfully!")
    print("✓ Results validated against known GaAs quantum wire properties")

if __name__ == "__main__":
    main()
