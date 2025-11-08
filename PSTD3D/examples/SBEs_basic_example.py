"""
Basic SBE Example - Minimal working example for SBEspythonic module

This example demonstrates basic usage of the semiconductor Bloch equations
module for quantum wire simulations.
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from SBEspythonic import *
from constants import *


def main():
    """Run basic SBE example."""
    print("Basic SBE Example")
    print("================")

    # Set up basic parameters
    Nk = 10
    Nr = 20
    Nw = 1

    # Create test arrays
    q = np.linspace(-1e6, 1e6, Nr, dtype=np.float64)
    rr = np.linspace(-100e-9, 100e-9, Nr, dtype=np.float64)

    # Initialize SBE module
    print("Initializing SBE module...")
    try:
        InitializeSBE(q, rr, 0.0, 1e6, 800e-9, Nw, True)
        print("✓ SBE module initialized successfully")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return

    # Test basic functions
    print("\nTesting basic functions...")

    # Test QWArea
    area = QWArea()
    print(f"Quantum wire area: {area:.2e} m²")

    # Test chiqw
    chi = chiqw()
    print(f"Linear susceptibility: {chi:.2e}")

    # Test getqc
    qc = getqc()
    print(f"Characteristic momentum: {qc:.2e} m⁻¹")

    # Test parameter reading
    print("\nTesting parameter reading...")
    ReadQWParams()
    ReadMBParams()
    print(f"Wire length: {L:.2e} m")
    print(f"Band gap: {gap/eV:.2f} eV")
    print(f"Electron mass: {me/me0:.2f} me")
    print(f"Hole mass: {mh/me0:.2f} me")

    # Test array functions
    print("\nTesting array functions...")
    x = np.zeros(Nr, dtype=np.float64)
    qx = np.zeros(Nr, dtype=np.float64)
    kx = np.zeros(Nk, dtype=np.float64)

    GetArrays(x, qx, kx)
    print(f"Spatial array range: [{x[0]:.2e}, {x[-1]:.2e}] m")
    print(f"Momentum array range: [{kx[0]:.2e}, {kx[-1]:.2e}] m⁻¹")

    # Test kkp calculation
    print("\nTesting kkp calculation...")
    MakeKKP()
    print(f"kkp array shape: {kkp.shape}")
    print(f"kkp[0,0]: {kkp[0,0]}")

    # Test SBE derivative functions
    print("\nTesting SBE derivative functions...")

    # Set up test matrices
    C = np.eye(Nk, dtype=np.complex128) * 0.1
    D = np.eye(Nk, dtype=np.complex128) * 0.1
    P = np.ones((Nk, Nk), dtype=np.complex128) * 0.01
    Heh = np.ones((Nk, Nk), dtype=np.complex128) * 1e-20
    Hee = np.eye(Nk, dtype=np.complex128) * 1e-19
    Hhh = np.eye(Nk, dtype=np.complex128) * 1e-19
    GamE = np.ones(Nk, dtype=np.float64) * 1e12
    GamH = np.ones(Nk, dtype=np.float64) * 1e12
    OffP = np.zeros((Nk, Nk), dtype=np.complex128)

    # Test dpdt
    dpdt_result = dpdt(C, D, P, Heh, Hee, Hhh, GamE, GamH, OffP)
    print(f"dpdt result shape: {dpdt_result.shape}")
    print(f"dpdt max magnitude: {np.max(np.abs(dpdt_result)):.2e}")

    # Test dCdt
    dCdt_result = dCdt(C, D, P, Heh, Hee, Hhh, GamE, GamH, OffP)
    print(f"dCdt result shape: {dCdt_result.shape}")
    print(f"dCdt max magnitude: {np.max(np.abs(dCdt_result)):.2e}")

    # Test dDdt
    dDdt_result = dDdt(C, D, P, Heh, Hee, Hhh, GamE, GamH, OffP)
    print(f"dDdt result shape: {dDdt_result.shape}")
    print(f"dDdt max magnitude: {np.max(np.abs(dDdt_result)):.2e}")

    # Test matrix element calculations
    print("\nTesting matrix element calculations...")

    Ex = np.zeros(Nr, dtype=np.complex128)
    Ey = np.zeros(Nr, dtype=np.complex128)
    Ez = np.zeros(Nr, dtype=np.complex128)
    Meh = np.zeros((Nk, Nk), dtype=np.complex128)

    CalcMeh(Ex, Ey, Ez, Meh)
    print(f"Meh matrix shape: {Meh.shape}")
    print(f"Meh max magnitude: {np.max(np.abs(Meh)):.2e}")

    # Test Wnn calculation
    Vr = np.zeros(Nr, dtype=np.complex128)
    Wnn = np.zeros((Nk, Nk), dtype=np.complex128)

    CalcWnn(e0, Vr, Wnn)
    print(f"Wnn matrix shape: {Wnn.shape}")
    print(f"Wnn max magnitude: {np.max(np.abs(Wnn)):.2e}")

    # Test utility functions
    print("\nTesting utility functions...")

    # Test ShutOffOptics
    original_optics = Optics
    ShutOffOptics()
    print(f"Optics after ShutOffOptics: {Optics}")
    Optics = original_optics  # Restore

    print("\n✓ All basic tests completed successfully!")
    print("\nThis example demonstrates:")
    print("- Module initialization")
    print("- Parameter reading")
    print("- Array setup and manipulation")
    print("- SBE derivative calculations")
    print("- Matrix element calculations")
    print("- Utility function usage")


if __name__ == "__main__":
    main()
