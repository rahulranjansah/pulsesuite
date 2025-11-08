#!/usr/bin/env python3
"""
CoulombPythonic Example Script

This script demonstrates the usage of the coulombpythonic module for quantum wire
Coulomb calculations. It shows both the modern Pythonic interface and the
Fortran-compatible interface.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coulombpythonic import (
    CoulombParameters, MomentumGrid, CoulombSolver,
    InitializeCoulomb, CalcCoulombArrays, Vint
)


def create_test_system(N=32):
    """Create a test quantum wire system."""
    print(f"Creating test system with N={N} grid points...")

    # Physical parameters for a typical quantum wire
    params = CoulombParameters(
        length=1e-6,                    # 1 μm length
        thickness=1e-8,                 # 10 nm thickness
        dielectric_constant=12.0,       # GaAs-like material
        electron_mass=0.067 * 9.109e-31,  # GaAs electron mass
        hole_mass=0.45 * 9.109e-31,     # GaAs hole mass
        electron_confinement=1e8,       # Confinement parameter
        hole_confinement=1e8,           # Confinement parameter
        electron_relaxation=1e12,       # 1 ps relaxation time
        hole_relaxation=1e12            # 1 ps relaxation time
    )

    # Create momentum grid
    ky = np.linspace(-1e8, 1e8, N)      # Momentum range
    y = np.linspace(-5e-8, 5e-8, N)     # Spatial range
    qy = np.linspace(0, 2e8, N)         # Momentum transfer range
    kkp = np.random.randint(-1, N, (N, N))  # Momentum conservation lookup

    grid = MomentumGrid(ky=ky, y=y, qy=qy, kkp=kkp)

    # Create energy dispersions (parabolic bands)
    Ee = 1e-20 * ky**2  # Electron energy
    Eh = 1e-20 * ky**2  # Hole energy

    return params, grid, Ee, Eh


def demonstrate_pythonic_interface():
    """Demonstrate the modern Pythonic interface."""
    print("\n" + "="*60)
    print("DEMONSTRATING PYTHONIC INTERFACE")
    print("="*60)

    # Create test system
    params, grid, Ee, Eh = create_test_system(N=32)

    # Create solver
    print("Creating Coulomb solver...")
    solver = CoulombSolver(params, grid, delta_type="lorentzian")

    # Initialize solver
    print("Initializing solver...")
    start_time = time.time()
    solver.initialize(Ee, Eh)
    init_time = time.time() - start_time
    print(f"Initialization completed in {init_time:.3f} seconds")

    # Create test populations
    N = grid.size
    ne = np.ones(N, dtype=np.complex128) * 0.1  # 10% electron population
    nh = np.ones(N, dtype=np.complex128) * 0.1  # 10% hole population

    # Calculate screened potentials
    print("Calculating screened potentials...")
    start_time = time.time()
    Veh, Vee, Vhh = solver.get_screened_potentials(ne, nh)
    pot_time = time.time() - start_time
    print(f"Potential calculation completed in {pot_time:.3f} seconds")

    # Calculate collision rates
    print("Calculating collision rates...")
    start_time = time.time()
    Win, Wout = solver.calculate_collision_rates(ne.real, nh.real, Ee, Eh)
    coll_time = time.time() - start_time
    print(f"Collision rate calculation completed in {coll_time:.3f} seconds")

    # Calculate band gap renormalization
    print("Calculating band gap renormalization...")
    start_time = time.time()
    BGR = solver.calculate_band_gap_renormalization(ne, nh)
    bgr_time = time.time() - start_time
    print(f"Band gap renormalization completed in {bgr_time:.3f} seconds")

    # Display results
    print(f"\nResults for N={N}:")
    print(f"  Veh matrix shape: {Veh.shape}")
    print(f"  Vee matrix shape: {Vee.shape}")
    print(f"  Vhh matrix shape: {Vhh.shape}")
    print(f"  Win array shape: {Win.shape}")
    print(f"  Wout array shape: {Wout.shape}")
    print(f"  BGR matrix shape: {BGR.shape}")

    print(f"\nPerformance summary:")
    print(f"  Initialization: {init_time:.3f}s")
    print(f"  Potentials: {pot_time:.3f}s")
    print(f"  Collision rates: {coll_time:.3f}s")
    print(f"  Band gap renormalization: {bgr_time:.3f}s")
    print(f"  Total time: {init_time + pot_time + coll_time + bgr_time:.3f}s")

    return Veh, Vee, Vhh, Win, Wout, BGR


def demonstrate_fortran_interface():
    """Demonstrate the Fortran-compatible interface."""
    print("\n" + "="*60)
    print("DEMONSTRATING FORTRAN-COMPATIBLE INTERFACE")
    print("="*60)

    # Create test system
    params, grid, Ee, Eh = create_test_system(N=16)

    # Extract arrays for Fortran interface
    y = grid.y
    ky = grid.ky
    Qy = grid.qy
    kkp = grid.kkp

    L = params.length
    Delta0 = params.thickness
    me = params.electron_mass
    mh = params.hole_mass
    ge = params.electron_relaxation
    gh = params.hole_relaxation
    alphae = params.electron_confinement
    alphah = params.hole_confinement
    er = params.dielectric_constant
    screened = True

    # Test InitializeCoulomb function
    print("Testing InitializeCoulomb function...")
    start_time = time.time()
    solver = InitializeCoulomb(
        y, ky, L, Delta0, me, mh, Ee, Eh, ge, gh,
        alphae, alphah, er, Qy, kkp, screened
    )
    init_time = time.time() - start_time
    print(f"InitializeCoulomb completed in {init_time:.3f} seconds")

    # Test CalcCoulombArrays function
    print("Testing CalcCoulombArrays function...")
    start_time = time.time()
    Veh, Vee, Vhh = CalcCoulombArrays(
        y, ky, er, alphae, alphah, L, Delta0, Qy, kkp
    )
    arrays_time = time.time() - start_time
    print(f"CalcCoulombArrays completed in {arrays_time:.3f} seconds")

    # Test Vint function
    print("Testing Vint function...")
    start_time = time.time()
    Qyk = 1e7
    integral = Vint(Qyk, y, alphae, alphah, Delta0)
    vint_time = time.time() - start_time
    print(f"Vint completed in {vint_time:.3f} seconds")

    print(f"\nFortran interface results:")
    print(f"  Veh matrix shape: {Veh.shape}")
    print(f"  Vee matrix shape: {Vee.shape}")
    print(f"  Vhh matrix shape: {Vhh.shape}")
    print(f"  Vint result: {integral:.2e}")

    print(f"\nFortran interface performance:")
    print(f"  InitializeCoulomb: {init_time:.3f}s")
    print(f"  CalcCoulombArrays: {arrays_time:.3f}s")
    print(f"  Vint: {vint_time:.3f}s")

    return Veh, Vee, Vhh, integral


def demonstrate_performance_scaling():
    """Demonstrate performance scaling with system size."""
    print("\n" + "="*60)
    print("PERFORMANCE SCALING ANALYSIS")
    print("="*60)

    grid_sizes = [16, 32, 64]
    results = []

    for N in grid_sizes:
        print(f"\nTesting N={N}...")

        # Create test system
        params, grid, Ee, Eh = create_test_system(N)

        # Create solver
        solver = CoulombSolver(params, grid)

        # Time initialization
        start_time = time.time()
        solver.initialize(Ee, Eh)
        init_time = time.time() - start_time

        # Create populations
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1

        # Time potential calculation
        start_time = time.time()
        Veh, Vee, Vhh = solver.get_screened_potentials(ne, nh)
        pot_time = time.time() - start_time

        # Time collision rate calculation
        start_time = time.time()
        Win, Wout = solver.calculate_collision_rates(ne.real, nh.real, Ee, Eh)
        coll_time = time.time() - start_time

        results.append({
            'N': N,
            'init_time': init_time,
            'pot_time': pot_time,
            'coll_time': coll_time,
            'total_time': init_time + pot_time + coll_time
        })

        print(f"  Initialization: {init_time:.3f}s")
        print(f"  Potentials: {pot_time:.3f}s")
        print(f"  Collision rates: {coll_time:.3f}s")
        print(f"  Total: {init_time + pot_time + coll_time:.3f}s")

    # Analyze scaling
    print(f"\nScaling Analysis:")
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        N_ratio = curr['N'] / prev['N']
        time_ratio = curr['total_time'] / prev['total_time']
        scaling = np.log(time_ratio) / np.log(N_ratio)
        print(f"  N={prev['N']} → N={curr['N']}: {time_ratio:.1f}x time, {scaling:.1f} scaling exponent")


def create_visualization(Veh, Vee, Vhh, Win, Wout, BGR):
    """Create visualization of the results."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Coulomb Calculation Results', fontsize=16)

    # Plot potential matrices
    im1 = axes[0, 0].imshow(Veh, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Electron-Hole Potential (Veh)')
    axes[0, 0].set_xlabel('Momentum Index')
    axes[0, 0].set_ylabel('Momentum Index')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(Vee, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Electron-Electron Potential (Vee)')
    axes[0, 1].set_xlabel('Momentum Index')
    axes[0, 1].set_ylabel('Momentum Index')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(Vhh, cmap='viridis', aspect='auto')
    axes[0, 2].set_title('Hole-Hole Potential (Vhh)')
    axes[0, 2].set_xlabel('Momentum Index')
    axes[0, 2].set_ylabel('Momentum Index')
    plt.colorbar(im3, ax=axes[0, 2])

    # Plot collision rates
    axes[1, 0].plot(Win, 'b-', label='Win', linewidth=2)
    axes[1, 0].plot(Wout, 'r-', label='Wout', linewidth=2)
    axes[1, 0].set_title('Collision Rates')
    axes[1, 0].set_xlabel('Momentum Index')
    axes[1, 0].set_ylabel('Rate (1/s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot band gap renormalization (real part)
    im4 = axes[1, 1].imshow(BGR.real, cmap='RdBu_r', aspect='auto')
    axes[1, 1].set_title('Band Gap Renormalization (Real)')
    axes[1, 1].set_xlabel('Momentum Index')
    axes[1, 1].set_ylabel('Momentum Index')
    plt.colorbar(im4, ax=axes[1, 1])

    # Plot band gap renormalization (imaginary part)
    im5 = axes[1, 2].imshow(BGR.imag, cmap='RdBu_r', aspect='auto')
    axes[1, 2].set_title('Band Gap Renormalization (Imag)')
    axes[1, 2].set_xlabel('Momentum Index')
    axes[1, 2].set_ylabel('Momentum Index')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig('coulomb_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'coulomb_results.png'")
    plt.show()


def main():
    """Main example function."""
    print("CoulombPythonic Example Script")
    print("="*60)

    try:
        # Demonstrate Pythonic interface
        Veh, Vee, Vhh, Win, Wout, BGR = demonstrate_pythonic_interface()

        # Demonstrate Fortran interface
        demonstrate_fortran_interface()

        # Demonstrate performance scaling
        demonstrate_performance_scaling()

        # Create visualizations
        create_visualization(Veh, Vee, Vhh, Win, Wout, BGR)

        print("\n" + "="*60)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The coulombpythonic module is working correctly.")
        print("Check 'coulomb_results.png' for visualization of the results.")

    except Exception as e:
        print(f"\nError: {e}")
        print("Please check that all dependencies are installed:")
        print("  pip install numpy matplotlib numba")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
