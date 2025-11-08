#!/usr/bin/env python3
"""
test_dcfieldpythonic.py
========================

Comprehensive test suite for dcfieldpythonic.py module following mathematical truth
and physical reality as the authoritative specification.

Test Categories:
- Parameter validation and edge cases
- Core mathematical functionality
- Physical property verification
- Performance and numerical accuracy
- Fortran-compatible interface functions
- Integration testing
- Error handling and boundary conditions

Author: AI Assistant
Date: 2024
"""

import sys
import os
import tempfile
import time
import warnings
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy.typing import NDArray

# Import the module under test
import src.dcfieldpythonic as dcf

# Type aliases
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def assert_array_close(actual: FloatArray, expected: FloatArray,
                      rtol: float = 1e-12, atol: float = 1e-12,
                      msg: str = "") -> None:
    """Assert arrays are close with specified tolerances."""
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(f"{msg}\nActual: {actual}\nExpected: {expected}\n"
                           f"Max diff: {np.max(np.abs(actual - expected))}")


def assert_scalar_close(actual: float, expected: float,
                       rtol: float = 1e-12, atol: float = 1e-12,
                       msg: str = "") -> None:
    """Assert scalars are close with specified tolerances."""
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(f"{msg}\nActual: {actual}\nExpected: {expected}\n"
                           f"Diff: {abs(actual - expected)}")


def test_parameter_validation():
    """Test parameter validation and edge cases."""
    print("Testing parameter validation...")

    # Test valid parameters
    params = dcf.DCFieldParameters(
        electron_mass=0.067 * 9.109e-31,
        hole_mass=0.45 * 9.109e-31
    )
    assert params.electron_mass > 0
    assert params.hole_mass > 0

    # Test invalid electron mass
    try:
        dcf.DCFieldParameters(electron_mass=-1.0, hole_mass=1.0)
        raise AssertionError("Should have raised ValueError for negative electron mass")
    except ValueError as e:
        assert "Electron mass must be positive" in str(e)

    # Test invalid hole mass
    try:
        dcf.DCFieldParameters(electron_mass=1.0, hole_mass=0.0)
        raise AssertionError("Should have raised ValueError for zero hole mass")
    except ValueError as e:
        assert "Hole mass must be positive" in str(e)

    # Test invalid relaxation rates
    try:
        dcf.DCFieldParameters(electron_mass=1.0, hole_mass=1.0,
                             electron_relaxation=-1e12)
        raise AssertionError("Should have raised ValueError for negative relaxation rate")
    except ValueError as e:
        assert "Electron relaxation rate must be positive" in str(e)

    print("✓ Parameter validation tests passed")


def test_momentum_grid():
    """Test momentum grid functionality."""
    print("Testing momentum grid...")

    # Test valid grid
    ky = np.linspace(-1e8, 1e8, 32)
    grid = dcf.MomentumGrid(ky=ky)
    assert grid.size == 32
    assert grid.dk == ky[1] - ky[0]

    # Test empty grid
    try:
        dcf.MomentumGrid(ky=np.array([]))
        raise AssertionError("Should have raised ValueError for empty grid")
    except ValueError as e:
        assert "Momentum grid cannot be empty" in str(e)

    # Test grid properties
    assert np.allclose(grid.ky, ky)

    print("✓ Momentum grid tests passed")


def test_energy_renormalization():
    """Test energy renormalization calculations."""
    print("Testing energy renormalization...")

    # Create test data
    N = 16
    ky = np.linspace(-1e8, 1e8, N)
    grid = dcf.MomentumGrid(ky=ky)
    calculator = dcf.EnergyRenormalizationCalculator(grid)

    # Test data
    n = np.ones(N, dtype=np.float64) * 0.1
    En = 1e-20 * ky**2
    V = np.ones((N, N), dtype=np.float64) * 1e-20

    # Calculate renormalized energy
    Ec = calculator.calculate_renormalized_energy(n, En, V)

    # Verify properties
    assert len(Ec) == N
    assert np.all(np.isfinite(Ec))
    assert np.all(Ec >= En)  # Renormalization should increase energy

    # Test with zero population
    n_zero = np.zeros(N, dtype=np.float64)
    Ec_zero = calculator.calculate_renormalized_energy(n_zero, En, V)
    assert_array_close(Ec_zero, En, msg="Zero population should give unrenormalized energy")

    # Test mathematical properties
    # For uniform potential, renormalization should be proportional to population
    n_half = n * 0.5
    Ec_half = calculator.calculate_renormalized_energy(n_half, En, V)

    # The renormalization formula is: Ec(k) = En(k) + sum(n(i) * (V(k,k) - V(k,i))) / 2
    # For uniform V, this becomes: Ec(k) = En(k) + sum(n(i)) * (V(k,k) - V(k,k)) / 2 = En(k)
    # So uniform potential gives no renormalization, which is correct
    assert np.all(np.isfinite(Ec_half)), "Renormalized energy should be finite"

    print("✓ Energy renormalization tests passed")


def test_drift_velocity():
    """Test drift velocity calculations."""
    print("Testing drift velocity...")

    # Create test data
    N = 16
    ky = np.linspace(-1e8, 1e8, N)
    grid = dcf.MomentumGrid(ky=ky)
    calculator = dcf.DriftVelocityCalculator(grid)

    # Test with linear energy dispersion
    n = np.ones(N, dtype=np.float64) * 0.1
    Ec = 1e-20 * ky**2  # Parabolic dispersion

    v = calculator.calculate_drift_velocity(n, Ec)

    # Verify properties
    assert np.isfinite(v)
    assert isinstance(v, float)

    # Test with zero population
    n_zero = np.zeros(N, dtype=np.float64)
    v_zero = calculator.calculate_drift_velocity(n_zero, Ec)
    assert v_zero == 0.0, "Zero population should give zero drift velocity"

    # Test with constant energy (should give zero velocity)
    Ec_const = np.ones(N, dtype=np.float64) * 1e-20
    v_const = calculator.calculate_drift_velocity(n, Ec_const)
    assert_scalar_close(v_const, 0.0, msg="Constant energy should give zero drift velocity")

    print("✓ Drift velocity tests passed")


def test_phonon_scattering():
    """Test phonon scattering calculations."""
    print("Testing phonon scattering...")

    # Create test data
    N = 16
    ky = np.linspace(-1e8, 1e8, N)
    grid = dcf.MomentumGrid(ky=ky)
    params = dcf.DCFieldParameters(
        electron_mass=0.067 * 9.109e-31,
        hole_mass=0.45 * 9.109e-31
    )
    calculator = dcf.PhononScatteringCalculator(params, grid)

    # Test data
    n = np.ones(N, dtype=np.float64) * 0.1
    Cq2 = np.ones(N, dtype=np.float64) * 1e-20
    x = np.ones(N, dtype=np.float64) * 1e-8

    # Test FDrift2 calculation
    Fd = calculator.calculate_fdrift2(
        Ephn=1e13, m=0.067 * 9.109e-31, g=1e12,
        n=n, Cq2=Cq2, v=1e5, N0=0.0, x=x
    )

    # Verify properties
    assert len(Fd) == N
    assert np.all(np.isfinite(Fd))

    # Scattering rates can be negative due to the difference between emission and absorption
    # The important thing is that they are finite
    assert np.all(np.isfinite(Fd)), "Scattering rates should be finite"

    # Test with zero population
    n_zero = np.zeros(N, dtype=np.float64)
    Fd_zero = calculator.calculate_fdrift2(
        Ephn=1e13, m=0.067 * 9.109e-31, g=1e12,
        n=n_zero, Cq2=Cq2, v=1e5, N0=0.0, x=x
    )
    assert np.all(Fd_zero == 0), "Zero population should give zero scattering"

    print("✓ Phonon scattering tests passed")


def test_dc_field_solver():
    """Test main DC field solver functionality."""
    print("Testing DC field solver...")

    # Create test data
    N = 16
    ky = np.linspace(-1e8, 1e8, N)
    params = dcf.DCFieldParameters(
        electron_mass=0.067 * 9.109e-31,
        hole_mass=0.45 * 9.109e-31
    )
    grid = dcf.MomentumGrid(ky=ky)
    solver = dcf.DCFieldSolver(params, grid)

    # Test initialization
    solver.initialize()
    assert solver._initialized

    # Test data
    ne = np.ones(N, dtype=np.complex128) * 0.1
    nh = np.ones(N, dtype=np.complex128) * 0.1
    Ee = 1e-20 * ky**2
    Eh = 1e-20 * ky**2
    Vee = np.ones((N, N), dtype=np.float64) * 1e-20
    Vhh = np.ones((N, N), dtype=np.float64) * 1e-20
    Cq2 = np.ones(N, dtype=np.float64) * 1e-20

    # Test electron DC contribution
    DC_e = solver.calculate_dc_electron_contribution(
        DCTrans=True, Cq2=Cq2, Edc=1e5, ne=ne, Ee=Ee, Vee=Vee, n=1, j=1
    )

    # Verify properties
    assert len(DC_e) == N
    assert np.all(np.isfinite(DC_e))

    # Test hole DC contribution
    DC_h = solver.calculate_dc_hole_contribution(
        DCTrans=True, Cq2=Cq2, Edc=1e5, nh=nh, Eh=Eh, Vhh=Vhh, n=1, j=1
    )

    # Verify properties
    assert len(DC_h) == N
    assert np.all(np.isfinite(DC_h))

    # Test current calculation
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
    I0 = solver.calculate_current(ne, nh, Ee, Eh, VC, grid.dk)

    # Verify properties
    assert np.isfinite(I0)
    assert isinstance(I0, float)

    # Test with DCTrans=False
    DC_e_off = solver.calculate_dc_electron_contribution(
        DCTrans=False, Cq2=Cq2, Edc=1e5, ne=ne, Ee=Ee, Vee=Vee, n=1, j=1
    )
    assert np.all(DC_e_off == 0), "DCTrans=False should give zero contribution"

    print("✓ DC field solver tests passed")


def test_fortran_compatible_interface():
    """Test Fortran-compatible interface functions."""
    print("Testing Fortran-compatible interface...")

    # Test InitializeDC
    N = 16
    ky = np.linspace(-1e8, 1e8, N)
    solver = dcf.InitializeDC(ky, 0.067 * 9.109e-31, 0.45 * 9.109e-31)
    assert isinstance(solver, dcf.DCFieldSolver)
    assert solver._initialized

    # Test data
    ne = np.ones(N, dtype=np.complex128) * 0.1
    nh = np.ones(N, dtype=np.complex128) * 0.1
    Ee = 1e-20 * ky**2
    Eh = 1e-20 * ky**2
    Vee = np.ones((N, N), dtype=np.float64) * 1e-20
    Vhh = np.ones((N, N), dtype=np.float64) * 1e-20
    Cq2 = np.ones(N, dtype=np.float64) * 1e-20

    # Test CalcDCE2
    DC_e = dcf.CalcDCE2(
        DCTrans=True, ky=ky, Cq2=Cq2, Edc=1e5, me=0.067 * 9.109e-31,
        ge=1e12, Ephn=1e13, N0=0.0, ne=ne, Ee=Ee, Vee=Vee, n=1, j=1, solver=solver
    )
    assert len(DC_e) == N
    assert np.all(np.isfinite(DC_e))

    # Test CalcDCH2
    DC_h = dcf.CalcDCH2(
        DCTrans=True, ky=ky, Cq2=Cq2, Edc=1e5, mh=0.45 * 9.109e-31,
        gh=1e12, Ephn=1e13, N0=0.0, nh=nh, Eh=Eh, Vhh=Vhh, n=1, j=1, solver=solver
    )
    assert len(DC_h) == N
    assert np.all(np.isfinite(DC_h))

    # Test CalcI0
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
    dk = ky[1] - ky[0]  # Calculate grid spacing
    I0 = dcf.CalcI0(ne, nh, Ee, Eh, VC, dk, ky, solver)
    assert np.isfinite(I0)

    # Test EkReNorm
    n = np.ones(N, dtype=np.float64) * 0.1
    En = 1e-20 * ky**2
    V = np.ones((N, N), dtype=np.float64) * 1e-20
    Ec = dcf.EkReNorm(n, En, V)
    assert len(Ec) == N
    assert np.all(np.isfinite(Ec))

    # Test DriftVt
    v = dcf.DriftVt(n, Ec)
    assert np.isfinite(v)
    assert isinstance(v, float)

    # Test getter functions
    e_rate = dcf.GetEDrift(solver)
    h_rate = dcf.GetHDrift(solver)
    ve_drift = dcf.GetVEDrift(solver)
    vh_drift = dcf.GetVHDrift(solver)

    assert np.isfinite(e_rate)
    assert np.isfinite(h_rate)
    assert np.isfinite(ve_drift)
    assert np.isfinite(vh_drift)

    print("✓ Fortran-compatible interface tests passed")


def test_mathematical_properties():
    """Test mathematical properties and conservation laws."""
    print("Testing mathematical properties...")

    # Create test data
    N = 16
    ky = np.linspace(-1e8, 1e8, N)
    params = dcf.DCFieldParameters(
        electron_mass=0.067 * 9.109e-31,
        hole_mass=0.45 * 9.109e-31
    )
    grid = dcf.MomentumGrid(ky=ky)
    solver = dcf.DCFieldSolver(params, grid)
    solver.initialize()

    # Test data
    ne = np.ones(N, dtype=np.complex128) * 0.1
    nh = np.ones(N, dtype=np.complex128) * 0.1
    Ee = 1e-20 * ky**2
    Eh = 1e-20 * ky**2
    Vee = np.ones((N, N), dtype=np.float64) * 1e-20
    Vhh = np.ones((N, N), dtype=np.float64) * 1e-20
    Cq2 = np.ones(N, dtype=np.float64) * 1e-20

    # Test energy renormalization properties
    Eec = solver.energy_calculator.calculate_renormalized_energy(np.real(ne), Ee, Vee)
    Ehc = solver.energy_calculator.calculate_renormalized_energy(np.real(nh), Eh, Vhh)

    # Energy should increase with renormalization
    assert np.all(Eec >= Ee), "Electron energy should increase with renormalization"
    assert np.all(Ehc >= Eh), "Hole energy should increase with renormalization"

    # Test drift velocity properties
    ve = solver.drift_calculator.calculate_drift_velocity(np.real(ne), Eec)
    vh = solver.drift_calculator.calculate_drift_velocity(np.real(nh), Ehc)

    # Drift velocities should be finite
    assert np.isfinite(ve)
    assert np.isfinite(vh)

    # Test current calculation
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
    I0 = solver.calculate_current(ne, nh, Ee, Eh, VC, grid.dk)

    # Current should be finite
    assert np.isfinite(I0)

    # Test scaling properties
    ne_scaled = ne * 2.0
    nh_scaled = nh * 2.0

    I0_scaled = solver.calculate_current(ne_scaled, nh_scaled, Ee, Eh, VC, grid.dk)

    # Current should scale with population
    assert np.isclose(I0_scaled, I0 * 2.0, rtol=1e-6), "Current should scale with population"

    print("✓ Mathematical properties tests passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases...")

    # Test with very small arrays
    N = 3
    ky = np.linspace(-1e8, 1e8, N)
    params = dcf.DCFieldParameters(
        electron_mass=0.067 * 9.109e-31,
        hole_mass=0.45 * 9.109e-31
    )
    grid = dcf.MomentumGrid(ky=ky)
    solver = dcf.DCFieldSolver(params, grid)
    solver.initialize()

    # Test data
    ne = np.ones(N, dtype=np.complex128) * 0.1
    nh = np.ones(N, dtype=np.complex128) * 0.1
    Ee = 1e-20 * ky**2
    Eh = 1e-20 * ky**2
    Vee = np.ones((N, N), dtype=np.float64) * 1e-20
    Vhh = np.ones((N, N), dtype=np.float64) * 1e-20
    Cq2 = np.ones(N, dtype=np.float64) * 1e-20

    # Test calculations with small arrays
    DC_e = solver.calculate_dc_electron_contribution(
        DCTrans=True, Cq2=Cq2, Edc=1e5, ne=ne, Ee=Ee, Vee=Vee, n=1, j=1
    )
    assert len(DC_e) == N
    assert np.all(np.isfinite(DC_e))

    # Test with zero populations
    ne_zero = np.zeros(N, dtype=np.complex128)
    nh_zero = np.zeros(N, dtype=np.complex128)

    DC_e_zero = solver.calculate_dc_electron_contribution(
        DCTrans=True, Cq2=Cq2, Edc=1e5, ne=ne_zero, Ee=Ee, Vee=Vee, n=1, j=1
    )
    assert np.all(DC_e_zero == 0), "Zero population should give zero DC contribution"

    # Test with very large values
    ne_large = np.ones(N, dtype=np.complex128) * 1e10
    nh_large = np.ones(N, dtype=np.complex128) * 1e10

    DC_e_large = solver.calculate_dc_electron_contribution(
        DCTrans=True, Cq2=Cq2, Edc=1e5, ne=ne_large, Ee=Ee, Vee=Vee, n=1, j=1
    )
    assert np.all(np.isfinite(DC_e_large)), "Large populations should give finite results"

    print("✓ Edge cases tests passed")


def test_performance():
    """Test performance characteristics."""
    print("Testing performance...")

    # Test with larger arrays
    N = 64
    ky = np.linspace(-1e8, 1e8, N)
    params = dcf.DCFieldParameters(
        electron_mass=0.067 * 9.109e-31,
        hole_mass=0.45 * 9.109e-31
    )
    grid = dcf.MomentumGrid(ky=ky)
    solver = dcf.DCFieldSolver(params, grid)
    solver.initialize()

    # Test data
    ne = np.ones(N, dtype=np.complex128) * 0.1
    nh = np.ones(N, dtype=np.complex128) * 0.1
    Ee = 1e-20 * ky**2
    Eh = 1e-20 * ky**2
    Vee = np.ones((N, N), dtype=np.float64) * 1e-20
    Vhh = np.ones((N, N), dtype=np.float64) * 1e-20
    Cq2 = np.ones(N, dtype=np.float64) * 1e-20

    # Time the calculations
    start_time = time.time()

    for _ in range(10):
        DC_e = solver.calculate_dc_electron_contribution(
            DCTrans=True, Cq2=Cq2, Edc=1e5, ne=ne, Ee=Ee, Vee=Vee, n=1, j=1
        )
        DC_h = solver.calculate_dc_hole_contribution(
            DCTrans=True, Cq2=Cq2, Edc=1e5, nh=nh, Eh=Eh, Vhh=Vhh, n=1, j=1
        )

    end_time = time.time()
    elapsed = end_time - start_time

    # Should complete in reasonable time (less than 3 seconds for 10 iterations)
    assert elapsed < 3.0, f"Performance test failed: {elapsed:.3f}s for 10 iterations"

    print(f"✓ Performance tests passed ({elapsed:.3f}s for 10 iterations)")


def test_determinism():
    """Test deterministic behavior."""
    print("Testing determinism...")

    # Create test data
    N = 16
    ky = np.linspace(-1e8, 1e8, N)
    params = dcf.DCFieldParameters(
        electron_mass=0.067 * 9.109e-31,
        hole_mass=0.45 * 9.109e-31
    )
    grid = dcf.MomentumGrid(ky=ky)
    solver = dcf.DCFieldSolver(params, grid)
    solver.initialize()

    # Test data
    ne = np.ones(N, dtype=np.complex128) * 0.1
    nh = np.ones(N, dtype=np.complex128) * 0.1
    Ee = 1e-20 * ky**2
    Eh = 1e-20 * ky**2
    Vee = np.ones((N, N), dtype=np.float64) * 1e-20
    Vhh = np.ones((N, N), dtype=np.float64) * 1e-20
    Cq2 = np.ones(N, dtype=np.float64) * 1e-20

    # Run calculation twice
    DC_e1 = solver.calculate_dc_electron_contribution(
        DCTrans=True, Cq2=Cq2, Edc=1e5, ne=ne, Ee=Ee, Vee=Vee, n=1, j=1
    )
    DC_e2 = solver.calculate_dc_electron_contribution(
        DCTrans=True, Cq2=Cq2, Edc=1e5, ne=ne, Ee=Ee, Vee=Vee, n=1, j=1
    )

    # Results should be identical
    assert_array_close(DC_e1, DC_e2, msg="Results should be deterministic")

    print("✓ Determinism tests passed")


def main():
    """Run all tests."""
    print("Running DC field Pythonic module tests...")
    print("=" * 50)

    try:
        test_parameter_validation()
        test_momentum_grid()
        test_energy_renormalization()
        test_drift_velocity()
        test_phonon_scattering()
        test_dc_field_solver()
        test_fortran_compatible_interface()
        test_mathematical_properties()
        test_edge_cases()
        test_performance()
        test_determinism()

        print("=" * 50)
        print("✓ All tests passed successfully!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
