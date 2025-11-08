#!/usr/bin/env python3
"""
DC Field Basic Example
======================

Basic Example: Minimal working example that proves the DC field module works
Style: Hard-coded data, immediate visualization
Audience: Someone copy-pasting to verify installation

This example demonstrates basic DC field calculations for quantum wire carriers.
"""

import sys
import os
import numpy as np

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.dcfieldpythonic as dcf

def main():
    """Basic DC field calculation example."""
    print("DC Field Basic Example")
    print("=" * 40)

    # Create simple test parameters
    me = 0.067 * 9.109e-31  # Electron mass (kg)
    mh = 0.45 * 9.109e-31   # Hole mass (kg)

    # Create momentum grid
    N = 32
    ky = np.linspace(-1e8, 1e8, N)  # Momentum grid (1/m)

    # Initialize DC field solver
    solver = dcf.InitializeDC(ky, me, mh)
    print(f"✓ DC field solver initialized with {N} momentum points")

    # Create test populations
    ne = np.ones(N, dtype=np.complex128) * 0.1  # Electron population
    nh = np.ones(N, dtype=np.complex128) * 0.1  # Hole population

    # Create energy dispersions (parabolic)
    Ee = 1e-20 * ky**2  # Electron energy (J)
    Eh = 1e-20 * ky**2  # Hole energy (J)

    # Create Coulomb potential matrices
    Vee = np.ones((N, N), dtype=np.float64) * 1e-20  # Electron-electron
    Vhh = np.ones((N, N), dtype=np.float64) * 1e-20  # Hole-hole

    # Create phonon coupling array
    Cq2 = np.ones(N, dtype=np.float64) * 1e-20

    # Calculate DC field contributions
    Edc = 1e5  # DC field (V/m)
    DC_e = dcf.CalcDCE2(
        DCTrans=True, ky=ky, Cq2=Cq2, Edc=Edc, me=me, ge=1e12,
        Ephn=1e13, N0=0.0, ne=ne, Ee=Ee, Vee=Vee, n=1, j=1, solver=solver
    )

    DC_h = dcf.CalcDCH2(
        DCTrans=True, ky=ky, Cq2=Cq2, Edc=Edc, mh=mh, gh=1e12,
        Ephn=1e13, N0=0.0, nh=nh, Eh=Eh, Vhh=Vhh, n=1, j=1, solver=solver
    )

    print(f"✓ DC electron contribution calculated: max = {np.max(DC_e):.2e}")
    print(f"✓ DC hole contribution calculated: max = {np.max(DC_h):.2e}")

    # Calculate current
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
    dk = ky[1] - ky[0]
    I0 = dcf.CalcI0(ne, nh, Ee, Eh, VC, dk, ky, solver)
    print(f"✓ Current calculated: I0 = {I0:.2e} A")

    # Test individual functions
    n_real = np.ones(N, dtype=np.float64) * 0.1
    Ec = dcf.EkReNorm(n_real, Ee, Vee)
    v_drift = dcf.DriftVt(n_real, Ec)
    print(f"✓ Drift velocity: v = {v_drift:.2e} m/s")

    # Get drift rates
    e_rate = dcf.GetEDrift(solver)
    h_rate = dcf.GetHDrift(solver)
    print(f"✓ Electron drift rate: {e_rate:.2e}")
    print(f"✓ Hole drift rate: {h_rate:.2e}")

    print("\n✓ Basic DC field calculations completed successfully!")

    # Simple visualization
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(ky/1e8, DC_e, 'b-', label='Electrons')
        plt.xlabel('Momentum (10^8 1/m)')
        plt.ylabel('DC Contribution')
        plt.title('Electron DC Field')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(ky/1e8, DC_h, 'r-', label='Holes')
        plt.xlabel('Momentum (10^8 1/m)')
        plt.ylabel('DC Contribution')
        plt.title('Hole DC Field')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(ky/1e8, Ee/1e-20, 'g-', label='Electron Energy')
        plt.plot(ky/1e8, Eh/1e-20, 'm-', label='Hole Energy')
        plt.xlabel('Momentum (10^8 1/m)')
        plt.ylabel('Energy (10^-20 J)')
        plt.title('Energy Dispersions')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('dcfield_basic_results.png', dpi=150, bbox_inches='tight')
        print("✓ Results saved to dcfield_basic_results.png")
    else:
        print("Note: matplotlib not available for visualization")

if __name__ == "__main__":
    main()
