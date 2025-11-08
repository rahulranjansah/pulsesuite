#!/usr/bin/env python3
"""
PhononsPythonic Example Script
==============================

This script demonstrates the usage of the phononspythonic module for quantum wire
semiconductor simulations. It shows both the modern Pythonic interface and the
Fortran-compatible interface.

Author: AI Assistant
Date: 2024
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from phononspythonic import (
    PhononParameters, MomentumGrid, PhononSolver,
    InitializePhonons, MBPE, MBPH, Cq2, FermiDistr, BoseDistr, N00
)


def create_test_system(N=32):
    """Create a test quantum wire system."""
    print(f"Creating test system with N={N} momentum points")

    # Physical parameters
    params = PhononParameters(
        temperature=77.0,  # Liquid nitrogen temperature
        phonon_frequency=1e13,  # 10 THz phonon frequency
        phonon_relaxation=1e12  # 1 THz relaxation rate
    )

    # Momentum grid
    ky = np.linspace(-1e8, 1e8, N)  # Momentum range: ±1e8 m⁻¹
    grid = MomentumGrid(ky=ky)

    # Energy dispersions (parabolic bands)
    Ee = 1e-20 * ky**2  # Electron energy (J)
    Eh = 1e-20 * ky**2  # Hole energy (J)

    # System parameters
    length = 1e-6  # 1 μm quantum wire length
    dielectric_constant = 12.0  # GaAs-like dielectric constant

    return params, grid, Ee, Eh, length, dielectric_constant


def modern_pythonic_interface():
    """Demonstrate the modern Pythonic interface."""
    print("\n" + "="*60)
    print("MODERN PYTHONIC INTERFACE")
    print("="*60)

    # Create test system
    params, grid, Ee, Eh, length, dielectric_constant = create_test_system(N=32)

    # Create solver
    solver = PhononSolver(params, grid)

    # Initialize solver
    print("Initializing phonon solver...")
    solver.initialize(Ee, Eh, length, dielectric_constant,
                     params.phonon_frequency, params.phonon_relaxation)

    # Create test populations
    N = grid.size
    ne = np.ones(N, dtype=np.float64) * 0.1  # 10% electron population
    nh = np.ones(N, dtype=np.float64) * 0.1  # 10% hole population

    # Create test potentials
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20  # Coulomb potentials
    E1D = np.ones((N, N), dtype=np.float64)  # 1D dielectric function

    # Calculate phonon rates
    print("Calculating electron-phonon rates...")
    Win_e, Wout_e = solver.calculate_electron_phonon_rates(ne, VC, E1D)

    print("Calculating hole-phonon rates...")
    Win_h, Wout_h = solver.calculate_hole_phonon_rates(nh, VC, E1D)

    # Get Bose distribution
    bose_dist = solver.get_bose_distribution()

    # Display results
    print(f"\nResults for N={N} system:")
    print(f"  Electron phonon rates:")
    print(f"    Win range: [{Win_e.min():.2e}, {Win_e.max():.2e}] Hz")
    print(f"    Wout range: [{Wout_e.min():.2e}, {Wout_e.max():.2e}] Hz")
    print(f"  Hole phonon rates:")
    print(f"    Win range: [{Win_h.min():.2e}, {Win_h.max():.2e}] Hz")
    print(f"    Wout range: [{Wout_h.min():.2e}, {Wout_h.max():.2e}] Hz")
    print(f"  Bose distribution: {bose_dist:.6f}")

    return solver, Win_e, Wout_e, Win_h, Wout_h


def fortran_compatible_interface():
    """Demonstrate the Fortran-compatible interface."""
    print("\n" + "="*60)
    print("FORTRAN-COMPATIBLE INTERFACE")
    print("="*60)

    # Create test system
    params, grid, Ee, Eh, length, dielectric_constant = create_test_system(N=32)

    # Initialize using Fortran-compatible function
    print("Initializing phonons using Fortran-compatible interface...")
    solver = InitializePhonons(
        grid.ky, Ee, Eh, length, dielectric_constant,
        params.phonon_frequency, params.phonon_relaxation
    )

    # Create test data
    N = grid.size
    ne = np.ones(N, dtype=np.float64) * 0.1
    nh = np.ones(N, dtype=np.float64) * 0.1
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
    E1D = np.ones((N, N), dtype=np.float64)

    # Initialize rate arrays
    Win_e = np.zeros(N, dtype=np.float64)
    Wout_e = np.zeros(N, dtype=np.float64)
    Win_h = np.zeros(N, dtype=np.float64)
    Wout_h = np.zeros(N, dtype=np.float64)

    # Calculate rates using Fortran-compatible functions
    print("Calculating electron-phonon rates using MBPE...")
    MBPE(ne, VC, E1D, Win_e, Wout_e, solver)

    print("Calculating hole-phonon rates using MBPH...")
    MBPH(nh, VC, E1D, Win_h, Wout_h, solver)

    # Test distribution functions
    print("Testing distribution functions...")
    fermi_result = FermiDistr(0.1)  # 0.1 eV
    bose_result = BoseDistr(0.1)    # 0.1 eV
    n00_result = N00(solver)

    # Display results
    print(f"\nResults for N={N} system:")
    print(f"  Electron phonon rates:")
    print(f"    Win range: [{Win_e.min():.2e}, {Win_e.max():.2e}] Hz")
    print(f"    Wout range: [{Wout_e.min():.2e}, {Wout_e.max():.2e}] Hz")
    print(f"  Hole phonon rates:")
    print(f"    Win range: [{Win_h.min():.2e}, {Win_h.max():.2e}] Hz")
    print(f"    Wout range: [{Wout_h.min():.2e}, {Wout_h.max():.2e}] Hz")
    print(f"  Distribution functions:")
    print(f"    Fermi(0.1 eV): {fermi_result:.6f}")
    print(f"    Bose(0.1 eV): {bose_result:.6f}")
    print(f"    N00 (thermal): {n00_result:.6f}")

    return solver, Win_e, Wout_e, Win_h, Wout_h


def test_cq2_function():
    """Test the Cq2 function for DC field calculations."""
    print("\n" + "="*60)
    print("TESTING CQ2 FUNCTION")
    print("="*60)

    # Create test system
    params, grid, Ee, Eh, length, dielectric_constant = create_test_system(N=32)

    # Initialize solver
    solver = InitializePhonons(
        grid.ky, Ee, Eh, length, dielectric_constant,
        params.phonon_frequency, params.phonon_relaxation
    )

    # Create test data
    N = grid.size
    q = np.linspace(0, 1e8, 20)  # Test momentum range
    V = np.ones((N, N), dtype=np.float64) * 1e-20
    E1D = np.ones((N, N), dtype=np.float64)

    # Calculate Cq2
    print("Calculating Cq2 for DC field module...")
    Cq2_result = Cq2(q, V, E1D, solver)

    # Display results
    print(f"\nCq2 results for {len(q)} momentum points:")
    print(f"  q range: [{q.min():.2e}, {q.max():.2e}] m⁻¹")
    print(f"  Cq2 range: [{Cq2_result.min():.2e}, {Cq2_result.max():.2e}]")
    print(f"  Cq2 shape: {Cq2_result.shape}")

    return q, Cq2_result


def performance_comparison():
    """Compare performance between different system sizes."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    import time

    system_sizes = [16, 32, 64]
    results = {}

    for N in system_sizes:
        print(f"\nTesting N={N} system...")

        # Create test system
        params, grid, Ee, Eh, length, dielectric_constant = create_test_system(N=N)

        # Create solver
        solver = PhononSolver(params, grid)

        # Create test data
        ne = np.ones(N, dtype=np.float64) * 0.1
        nh = np.ones(N, dtype=np.float64) * 0.1
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
        E1D = np.ones((N, N), dtype=np.float64)

        # Time initialization
        start = time.time()
        solver.initialize(Ee, Eh, length, dielectric_constant,
                         params.phonon_frequency, params.phonon_relaxation)
        init_time = time.time() - start

        # Time electron-phonon rates
        start = time.time()
        Win_e, Wout_e = solver.calculate_electron_phonon_rates(ne, VC, E1D)
        elec_time = time.time() - start

        # Time hole-phonon rates
        start = time.time()
        Win_h, Wout_h = solver.calculate_hole_phonon_rates(nh, VC, E1D)
        hole_time = time.time() - start

        results[N] = {
            'init_time': init_time,
            'elec_time': elec_time,
            'hole_time': hole_time,
            'total_time': init_time + elec_time + hole_time
        }

        print(f"  Initialization: {init_time:.3f}s")
        print(f"  Electron rates: {elec_time:.3f}s")
        print(f"  Hole rates: {hole_time:.3f}s")
        print(f"  Total time: {results[N]['total_time']:.3f}s")

    # Display performance summary
    print(f"\nPerformance Summary:")
    print(f"{'N':<4} {'Init (s)':<10} {'Elec (s)':<10} {'Hole (s)':<10} {'Total (s)':<10}")
    print("-" * 50)
    for N in system_sizes:
        r = results[N]
        print(f"{N:<4} {r['init_time']:<10.3f} {r['elec_time']:<10.3f} {r['hole_time']:<10.3f} {r['total_time']:<10.3f}")

    return results


def create_visualization(solver, Win_e, Wout_e, Win_h, Wout_h):
    """Create visualization of phonon rates."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)

    try:
        import matplotlib.pyplot as plt

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Phonon Collision Rates', fontsize=16)

        # Plot electron rates
        axes[0, 0].semilogy(Win_e, 'b-', label='Win', linewidth=2)
        axes[0, 0].semilogy(Wout_e, 'r-', label='Wout', linewidth=2)
        axes[0, 0].set_title('Electron-Phonon Rates')
        axes[0, 0].set_xlabel('Momentum Index')
        axes[0, 0].set_ylabel('Rate (Hz)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot hole rates
        axes[0, 1].semilogy(Win_h, 'b-', label='Win', linewidth=2)
        axes[0, 1].semilogy(Wout_h, 'r-', label='Wout', linewidth=2)
        axes[0, 1].set_title('Hole-Phonon Rates')
        axes[0, 1].set_xlabel('Momentum Index')
        axes[0, 1].set_ylabel('Rate (Hz)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot rate ratios
        axes[1, 0].plot(Win_e / (Wout_e + 1e-20), 'g-', linewidth=2)
        axes[1, 0].set_title('Electron Rate Ratio (Win/Wout)')
        axes[1, 0].set_xlabel('Momentum Index')
        axes[1, 0].set_ylabel('Ratio')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(Win_h / (Wout_h + 1e-20), 'g-', linewidth=2)
        axes[1, 1].set_title('Hole Rate Ratio (Win/Wout)')
        axes[1, 1].set_xlabel('Momentum Index')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_file = Path(__file__).parent / "phonon_rates_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")

        # Show plot if in interactive mode
        if hasattr(plt, 'show'):
            plt.show()

    except ImportError:
        print("Matplotlib not available. Skipping visualization.")
    except Exception as e:
        print(f"Error creating visualization: {e}")


def main():
    """Main example function."""
    print("PhononsPythonic Example Script")
    print("=" * 60)

    try:
        # Test modern Pythonic interface
        solver, Win_e, Wout_e, Win_h, Wout_h = modern_pythonic_interface()

        # Test Fortran-compatible interface
        fortran_compatible_interface()

        # Test Cq2 function
        test_cq2_function()

        # Performance comparison
        performance_comparison()

        # Create visualization
        create_visualization(solver, Win_e, Wout_e, Win_h, Wout_h)

        print("\n" + "="*60)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Modern Pythonic interface")
        print("✓ Fortran-compatible interface")
        print("✓ Performance optimization with JIT")
        print("✓ Comprehensive error handling")
        print("✓ Type hints and validation")
        print("✓ Modular architecture")
        print("✓ Memory-efficient calculations")

    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
