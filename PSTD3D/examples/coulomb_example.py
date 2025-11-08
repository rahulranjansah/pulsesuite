#!/usr/bin/env python3
"""
Example script demonstrating the alternative Coulomb module.

This script shows how to use the alt_coulomb module to calculate Coulomb
interactions in a quantum wire system. It includes examples of:
- Setting up system parameters
- Creating momentum grids
- Calculating screened potentials
- Computing collision rates
- Analyzing results

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alt_coulomb import (
    create_quantum_wire_system, create_momentum_grid, CoulombSolver,
    DeltaFunctionType
)


def create_gaas_quantum_wire():
    """Create a GaAs quantum wire system with realistic parameters."""
    return create_quantum_wire_system(
        length=1e-6,                    # 1 micron wire length
        thickness=1e-8,                 # 10 nm wire thickness
        dielectric_constant=12.9,       # GaAs dielectric constant
        electron_mass=0.067 * 9.109e-31, # GaAs electron mass
        hole_mass=0.45 * 9.109e-31,     # GaAs hole mass
        electron_confinement=1e8,       # Confinement parameter (1/m)
        hole_confinement=1e8,           # Confinement parameter (1/m)
        electron_relaxation=1e12,       # 1 THz relaxation rate
        hole_relaxation=1e12            # 1 THz relaxation rate
    )


def create_momentum_grid_1d(N=32, k_max=1e8):
    """Create a 1D momentum grid for quantum wire calculations."""
    ky = np.linspace(-k_max, k_max, N)
    y = np.linspace(-5e-8, 5e-8, N)  # Real space grid
    qy = np.linspace(0, 2*k_max, N)  # Momentum difference grid

    # Create a simple lookup table (in practice, this would be more sophisticated)
    kkp = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            q_idx = int(abs(i - j))
            if q_idx < N:
                kkp[i, j] = q_idx
            else:
                kkp[i, j] = -1  # Invalid state

    return create_momentum_grid(ky, y, qy, kkp)


def calculate_energy_dispersions(ky, me, mh):
    """Calculate parabolic energy dispersions for electrons and holes."""
    hbar = 1.05457159e-34  # Reduced Planck constant

    # Parabolic dispersions
    Ee = hbar**2 * ky**2 / (2 * me)
    Eh = hbar**2 * ky**2 / (2 * mh)

    return Ee, Eh


def create_thermal_populations(ky, Ee, Eh, T=300, mu_e=0, mu_h=0):
    """Create thermal carrier populations using Fermi-Dirac statistics."""
    kB = 1.38064852e-23  # Boltzmann constant

    # Fermi-Dirac distribution
    ne = 1.0 / (1.0 + np.exp((Ee - mu_e) / (kB * T)))
    nh = 1.0 / (1.0 + np.exp((Eh - mu_h) / (kB * T)))

    return ne.astype(np.complex128), nh.astype(np.complex128)


def main():
    """Main example function."""
    print("Alternative Coulomb Module Example")
    print("=" * 40)

    # 1. Create system parameters
    print("1. Creating GaAs quantum wire system...")
    params = create_gaas_quantum_wire()
    print(f"   Wire length: {params.length*1e6:.1f} μm")
    print(f"   Wire thickness: {params.thickness*1e9:.1f} nm")
    print(f"   Dielectric constant: {params.dielectric_constant}")

    # 2. Create momentum grid
    print("\n2. Creating momentum grid...")
    N = 32
    grid = create_momentum_grid_1d(N)
    print(f"   Grid size: {grid.size}")
    print(f"   Momentum range: {grid.ky[0]*1e-8:.1f} to {grid.ky[-1]*1e-8:.1f} ×10⁸ m⁻¹")

    # 3. Create Coulomb solver
    print("\n3. Creating Coulomb solver...")
    solver = CoulombSolver(params, grid, DeltaFunctionType.GAUSSIAN)

    # 4. Calculate energy dispersions
    print("\n4. Calculating energy dispersions...")
    Ee, Eh = calculate_energy_dispersions(grid.ky, params.electron_mass, params.hole_mass)
    print(f"   Electron mass: {params.electron_mass/9.109e-31:.3f} m₀")
    print(f"   Hole mass: {params.hole_mass/9.109e-31:.3f} m₀")

    # 5. Initialize solver
    print("\n5. Initializing Coulomb solver...")
    solver.initialize(Ee, Eh)
    print("   Solver initialized successfully!")

    # 6. Create carrier populations
    print("\n6. Creating carrier populations...")
    ne, nh = create_thermal_populations(grid.ky, Ee, Eh, T=300)
    print(f"   Electron density: {np.sum(ne.real)/params.length*1e-6:.2f} ×10⁶ cm⁻¹")
    print(f"   Hole density: {np.sum(nh.real)/params.length*1e-6:.2f} ×10⁶ cm⁻¹")

    # 7. Calculate screened potentials
    print("\n7. Calculating screened Coulomb potentials...")
    Veh, Vee, Vhh = solver.get_screened_potentials(ne, nh)
    print(f"   Veh matrix shape: {Veh.shape}")
    print(f"   Vee matrix shape: {Vee.shape}")
    print(f"   Vhh matrix shape: {Vhh.shape}")
    print(f"   Veh range: {Veh.min()*1e21:.2f} to {Veh.max()*1e21:.2f} ×10⁻²¹ J")

    # 8. Calculate collision rates
    print("\n8. Calculating many-body collision rates...")
    Win, Wout = solver.calculate_collision_rates(ne.real, nh.real, Ee, Eh)
    print(f"   Win range: {Win.min()*1e-12:.2f} to {Win.max()*1e-12:.2f} ×10¹² s⁻¹")
    print(f"   Wout range: {Wout.min()*1e-12:.2f} to {Wout.max()*1e-12:.2f} ×10¹² s⁻¹")

    # 9. Calculate band gap renormalization
    print("\n9. Calculating band gap renormalization...")
    BGR = solver.calculate_band_gap_renormalization(ne, nh)
    print(f"   BGR matrix shape: {BGR.shape}")
    print(f"   BGR range: {BGR.real.min()*1e21:.2f} to {BGR.real.max()*1e21:.2f} ×10⁻²¹ J")

    # 10. Create visualizations
    print("\n10. Creating visualizations...")
    create_plots(grid, Ee, Eh, ne, nh, Veh, Vee, Vhh, Win, Wout, BGR)

    print("\nExample completed successfully!")
    print("Check the generated plots for visualization of results.")


def create_plots(grid, Ee, Eh, ne, nh, Veh, Vee, Vhh, Win, Wout, BGR):
    """Create visualization plots of the results."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Coulomb Interactions in Quantum Wire System', fontsize=16)

    # 1. Energy dispersions
    ax = axes[0, 0]
    ax.plot(grid.ky*1e-8, Ee*1e21, 'b-', label='Electrons', linewidth=2)
    ax.plot(grid.ky*1e-8, Eh*1e21, 'r-', label='Holes', linewidth=2)
    ax.set_xlabel('Momentum (×10⁸ m⁻¹)')
    ax.set_ylabel('Energy (×10⁻²¹ J)')
    ax.set_title('Energy Dispersions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Carrier populations
    ax = axes[0, 1]
    ax.plot(grid.ky*1e-8, ne.real, 'b-', label='Electrons', linewidth=2)
    ax.plot(grid.ky*1e-8, nh.real, 'r-', label='Holes', linewidth=2)
    ax.set_xlabel('Momentum (×10⁸ m⁻¹)')
    ax.set_ylabel('Population')
    ax.set_title('Carrier Populations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Screened potentials (diagonal elements)
    ax = axes[0, 2]
    ax.plot(grid.ky*1e-8, np.diag(Veh)*1e21, 'g-', label='Veh', linewidth=2)
    ax.plot(grid.ky*1e-8, np.diag(Vee)*1e21, 'b-', label='Vee', linewidth=2)
    ax.plot(grid.ky*1e-8, np.diag(Vhh)*1e21, 'r-', label='Vhh', linewidth=2)
    ax.set_xlabel('Momentum (×10⁸ m⁻¹)')
    ax.set_ylabel('Potential (×10⁻²¹ J)')
    ax.set_title('Screened Coulomb Potentials')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Collision rates
    ax = axes[1, 0]
    ax.plot(grid.ky*1e-8, Win*1e-12, 'b-', label='Win', linewidth=2)
    ax.plot(grid.ky*1e-8, Wout*1e-12, 'r-', label='Wout', linewidth=2)
    ax.set_xlabel('Momentum (×10⁸ m⁻¹)')
    ax.set_ylabel('Rate (×10¹² s⁻¹)')
    ax.set_title('Collision Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Band gap renormalization (diagonal elements)
    ax = axes[1, 1]
    ax.plot(grid.ky*1e-8, np.diag(BGR.real)*1e21, 'g-', linewidth=2)
    ax.set_xlabel('Momentum (×10⁸ m⁻¹)')
    ax.set_ylabel('BGR (×10⁻²¹ J)')
    ax.set_title('Band Gap Renormalization')
    ax.grid(True, alpha=0.3)

    # 6. Potential matrix heatmap
    ax = axes[1, 2]
    im = ax.imshow(Veh*1e21, cmap='viridis', aspect='auto')
    ax.set_xlabel('Momentum Index')
    ax.set_ylabel('Momentum Index')
    ax.set_title('Veh Matrix')
    plt.colorbar(im, ax=ax, label='Potential (×10⁻²¹ J)')

    plt.tight_layout()
    plt.savefig('coulomb_example_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("   Plots saved as 'coulomb_example_results.png'")


def performance_benchmark():
    """Benchmark the performance of Coulomb calculations."""
    print("\nPerformance Benchmark")
    print("=" * 20)

    import time

    # Test different grid sizes
    grid_sizes = [16, 32, 64]

    for N in grid_sizes:
        print(f"\nTesting grid size N = {N}")

        # Setup
        params = create_gaas_quantum_wire()
        grid = create_momentum_grid_1d(N)
        solver = CoulombSolver(params, grid)

        Ee, Eh = calculate_energy_dispersions(grid.ky, params.electron_mass, params.hole_mass)
        ne, nh = create_thermal_populations(grid.ky, Ee, Eh)

        # Benchmark initialization
        start = time.time()
        solver.initialize(Ee, Eh)
        init_time = time.time() - start

        # Benchmark screened potentials
        start = time.time()
        for _ in range(10):
            Veh, Vee, Vhh = solver.get_screened_potentials(ne, nh)
        pot_time = (time.time() - start) / 10

        # Benchmark collision rates
        start = time.time()
        for _ in range(10):
            Win, Wout = solver.calculate_collision_rates(ne.real, nh.real, Ee, Eh)
        coll_time = (time.time() - start) / 10

        print(f"   Initialization: {init_time:.3f} s")
        print(f"   Screened potentials: {pot_time:.3f} s")
        print(f"   Collision rates: {coll_time:.3f} s")


if __name__ == "__main__":
    # Run main example
    main()

    # Run performance benchmark
    performance_benchmark()
