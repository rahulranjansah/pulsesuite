"""
test_emissionpythonic_simple.py
===============================

Simple test suite for emissionpythonic.py without pytest dependencies.
Tests define the intended behavior based on mathematical truth and physical reality.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import emissionpythonic as ep
from emissionpythonic import (
    EmissionParameters, MomentumGrid, EmissionSolver,
    PhotonGridCalculator, CoulombEnergyCalculator,
    SpontaneousEmissionCalculator, PhotoluminescenceCalculator
)

# Test configuration
np.random.seed(42)  # Fixed seed for deterministic tests
RTOL = 1e-12  # Relative tolerance for float64 operations
ATOL = 1e-12  # Absolute tolerance for float64 operations


def assert_close(actual, expected, rtol=RTOL, atol=ATOL, msg=""):
    """Assert two values are close with custom message."""
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(f"{msg}: expected {expected}, got {actual}")


def assert_array_close(actual, expected, rtol=RTOL, atol=ATOL, msg=""):
    """Assert two arrays are close with custom message."""
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(f"{msg}: arrays not close")


def test_emission_parameters():
    """Test EmissionParameters dataclass validation."""
    print("Testing EmissionParameters...")

    # Test valid parameters
    params = EmissionParameters(
        temperature=100.0,
        boltzmann_constant=1.38e-23,
        dipole_matrix_element=1e-28,
        dielectric_constant=12.0,
        dephasing_rate=1e12,
        eh_overlap_integral=0.8
    )
    assert params.temperature == 100.0
    assert params.boltzmann_constant == 1.38e-23

    # Test invalid temperature
    try:
        EmissionParameters(temperature=0.0)
        raise AssertionError("Should have raised ValueError for zero temperature")
    except ValueError as e:
        assert "Temperature must be positive" in str(e)

    # Test invalid Boltzmann constant
    try:
        EmissionParameters(boltzmann_constant=0.0)
        raise AssertionError("Should have raised ValueError for zero Boltzmann constant")
    except ValueError as e:
        assert "Boltzmann constant must be positive" in str(e)

    print("✓ EmissionParameters tests passed")


def test_momentum_grid():
    """Test MomentumGrid validation and properties."""
    print("Testing MomentumGrid...")

    # Test valid grid
    ky = np.linspace(-1e8, 1e8, 32)
    grid = MomentumGrid(ky=ky)
    assert grid.size == 32
    assert np.array_equal(grid.ky, ky)

    # Test empty grid
    try:
        MomentumGrid(ky=np.array([]))
        raise AssertionError("Should have raised ValueError for empty grid")
    except ValueError as e:
        assert "Momentum grid cannot be empty" in str(e)

    print("✓ MomentumGrid tests passed")


def test_photon_grid_calculator():
    """Test PhotonGridCalculator functionality."""
    print("Testing PhotonGridCalculator...")

    params = EmissionParameters()
    calculator = PhotonGridCalculator(params)

    # Test rscale calculation
    dcv = 1e-28
    epsr = 12.0
    ehint = 1.0
    expected = 3.0 * dcv**2 / ep.eps0 / np.sqrt(epsr) * ehint**2
    result = calculator._calculate_rscale(dcv, epsr, ehint)
    assert_close(result, expected, msg="Rscale calculation")

    # Test photon grid calculation
    kBT = 1e-20
    hg = 1e-21
    grid = calculator._calculate_photon_grid(kBT, hg)

    # Check grid properties
    assert len(grid) > 10, "Grid should have more than 10 points"
    assert np.all(grid > 0), "All energies should be positive"
    assert np.all(np.diff(grid) > 0), "Grid should be monotonically increasing"

    # Check grid spacing
    dhw = grid[1] - grid[0]
    expected_dhw = min(kBT, hg) / 20.0
    assert_close(dhw, expected_dhw, msg="Grid spacing")

    # Test low temperature error - use zero values to trigger division by zero
    try:
        calculator._calculate_photon_grid(0.0, 0.0)
        raise AssertionError("Should have raised error for zero values")
    except (ValueError, ZeroDivisionError) as e:
        # Either ValueError for low temperature or ZeroDivisionError for zero values
        assert "Temperature is too low" in str(e) or "division by zero" in str(e)

    # Test initialization
    ky = np.linspace(-1e8, 1e8, 16)
    calculator.initialize(ky, 1e-28, 12.0, 1e12, 1.0)

    # Check properties are accessible
    grid = calculator.photon_grid
    weights = calculator.spectral_weights
    rscale = calculator.rscale

    assert len(grid) > 0, "Grid should not be empty"
    assert len(weights) == len(grid), "Weights should match grid length"
    assert rscale > 0, "Rscale should be positive"

    print("✓ PhotonGridCalculator tests passed")


def test_coulomb_energy_calculator():
    """Test CoulombEnergyCalculator functionality."""
    print("Testing CoulombEnergyCalculator...")

    grid = MomentumGrid(ky=np.linspace(-1e8, 1e8, 8))
    calculator = CoulombEnergyCalculator(grid)

    # Test identity delta matrix
    idel = calculator._build_identity_delta_matrix()
    N = calculator.grid.size

    # Check shape
    assert idel.shape == (N, N), "Identity delta should be square"

    # Check diagonal is zero
    assert np.all(np.diag(idel) == 0), "Diagonal should be zero"

    # Check off-diagonal is one
    mask = ~np.eye(N, dtype=bool)
    assert np.all(idel[mask] == 1), "Off-diagonal should be one"

    # Test Coulomb energy calculation
    ne = np.ones(N, dtype=np.float64) * 0.1
    nh = np.ones(N, dtype=np.float64) * 0.2
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

    Ec = calculator.calculate_coulomb_energy(ne, nh, VC)

    # Check output properties
    assert Ec.shape == (N,), "Ec should have correct shape"
    assert Ec.dtype == np.float64, "Ec should be float64"
    assert np.all(np.isfinite(Ec)), "Ec should be finite"

    print("✓ CoulombEnergyCalculator tests passed")


def test_spontaneous_emission_calculator():
    """Test SpontaneousEmissionCalculator functionality."""
    print("Testing SpontaneousEmissionCalculator...")

    params = EmissionParameters()
    photon_calc = PhotonGridCalculator(params)
    ky = np.linspace(-1e8, 1e8, 8)
    photon_calc.initialize(ky, 1e-28, 12.0, 1e12, 1.0)
    calculator = SpontaneousEmissionCalculator(params, photon_calc)

    # Test photon density of states
    hw = 1.5  # eV
    rho = calculator.calculate_photon_density_of_states(hw)

    # Check formula: rho0 = hw^2 / (c0^3 * pi^2 * hbar^3)
    expected = (hw**2) / (ep.c0**3 * ep.pi**2 * ep.hbar**3)
    assert_close(rho, expected, msg="Photon density of states")

    # Test vectorized photon density of states
    hw = np.array([1.0, 1.5, 2.0])
    rho = calculator.calculate_photon_density_of_states(hw)

    # Check shape and values
    assert rho.shape == hw.shape, "Rho should have correct shape"
    assert np.all(rho > 0), "Rho should be positive"

    # Test spontaneous emission integral
    Ek = 1.5
    result = calculator.calculate_spontaneous_emission_integral(Ek)

    # Check output properties
    assert isinstance(result, np.float64), "Result should be float64"
    assert result >= 0, "Result should be non-negative"
    assert np.isfinite(result), "Result should be finite"

    # Test vectorized integral
    Ek = np.array([1.0, 1.5, 2.0])
    result = calculator.calculate_spontaneous_emission_integral(Ek)

    assert result.shape == Ek.shape, "Result should have correct shape"
    assert np.all(result >= 0), "All results should be non-negative"
    assert np.all(np.isfinite(result)), "All results should be finite"

    print("✓ SpontaneousEmissionCalculator tests passed")


def test_photoluminescence_calculator():
    """Test PhotoluminescenceCalculator functionality."""
    print("Testing PhotoluminescenceCalculator...")

    params = EmissionParameters()
    photon_calc = PhotonGridCalculator(params)
    ky = np.linspace(-1e8, 1e8, 8)
    photon_calc.initialize(ky, 1e-28, 12.0, 1e12, 1.0)
    calculator = PhotoluminescenceCalculator(params, photon_calc)

    # Test linear interpolation
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 4.0])
    xq = np.array([0.5, 1.5])

    result = calculator._linear_interpolate_real(y, x, xq)
    expected = np.array([0.5, 2.5])
    assert_array_close(result, expected, msg="Real linear interpolation")

    # Test complex linear interpolation
    y = np.array([0.0+0.0j, 1.0+1.0j, 4.0+2.0j])
    result = calculator._linear_interpolate_complex(y, x, xq)
    expected = np.array([0.5+0.5j, 2.5+1.5j])
    assert_array_close(result, expected, msg="Complex linear interpolation")

    print("✓ PhotoluminescenceCalculator tests passed")


def test_emission_solver():
    """Test main EmissionSolver class."""
    print("Testing EmissionSolver...")

    params = EmissionParameters()
    grid = MomentumGrid(ky=np.linspace(-1e8, 1e8, 8))
    solver = EmissionSolver(params, grid)

    # Test initialization
    assert not solver._initialized, "Solver should not be initialized initially"

    solver.initialize(1e-28, 12.0, 1e12, 1.0)
    assert solver._initialized, "Solver should be initialized after initialize()"

    # Test uninitialized error
    N = 8
    ne = np.ones(N, dtype=np.complex128) * 0.1
    nh = np.ones(N, dtype=np.complex128) * 0.1
    Ee = np.linspace(0.1, 0.2, N)
    Eh = np.linspace(0.1, 0.2, N)
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

    # Reset solver to test error
    solver._initialized = False
    try:
        solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)
        raise AssertionError("Should have raised RuntimeError for uninitialized solver")
    except RuntimeError as e:
        assert "not initialized" in str(e)

    # Re-initialize and test calculations
    solver.initialize(1e-28, 12.0, 1e12, 1.0)

    rates = solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)
    assert rates.shape == (N,), "Rates should have correct shape"
    assert rates.dtype == np.float64, "Rates should be float64"
    assert np.all(rates >= 0), "Rates should be non-negative"

    # Test photoluminescence spectrum
    hw = np.linspace(1.0, 2.0, 10)
    spectrum = solver.calculate_photoluminescence_spectrum(
        ne, nh, Ee, Eh, 1.5, 1e12, VC, hw, 1e-12
    )

    assert spectrum.shape == hw.shape, "Spectrum should have correct shape"
    assert spectrum.dtype == np.float64, "Spectrum should be float64"
    assert np.all(spectrum >= 0), "Spectrum should be non-negative"

    print("✓ EmissionSolver tests passed")


def test_fortran_compatible_interface():
    """Test Fortran-compatible interface functions."""
    print("Testing Fortran-compatible interface...")

    # Test InitializeEmission
    ky = np.linspace(-1e8, 1e8, 8)
    Ee = np.linspace(0.1, 0.2, 8)
    Eh = np.linspace(0.1, 0.2, 8)

    solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)
    assert isinstance(solver, EmissionSolver), "Should return EmissionSolver"
    assert solver._initialized, "Solver should be initialized"

    # Test rho0 function
    hw = 1.5
    result = ep.rho0(hw)
    expected = (hw**2) / (ep.c0**3 * ep.pi**2 * ep.hbar**3)
    assert_close(result, expected, msg="rho0 function")

    # Test CalcHOmega function
    kBT = 1e-20
    hg = 1e-21
    result = ep.CalcHOmega(kBT, hg)

    assert len(result) > 10, "HOmega should have more than 10 points"
    assert np.all(result > 0), "HOmega should be positive"
    assert np.all(np.diff(result) > 0), "HOmega should be increasing"

    # Test Calchw function
    hw = np.zeros(10)
    PLS = np.zeros(10)
    Estart = 1.0
    Emax = 2.0

    ep.Calchw(hw, PLS, Estart, Emax)

    # Check hw array - Calchw uses Estart + w * dhw, not np.linspace
    dhw = (Emax - Estart) / 10.0
    w = np.arange(10, dtype=np.float64)
    expected_hw = Estart + w * dhw
    assert_array_close(hw, expected_hw, msg="Calchw hw array")

    # Check PLS array is zeroed
    assert np.all(PLS == 0.0), "PLS should be zeroed"

    # Test SpontEmission function
    N = 8
    ne = np.ones(N, dtype=np.complex128) * 0.1
    nh = np.ones(N, dtype=np.complex128) * 0.1
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
    RSP = np.zeros(N)

    ep.SpontEmission(ne, nh, Ee, Eh, 1.5, 1e12, VC, RSP, solver)

    assert np.all(RSP >= 0), "RSP should be non-negative"
    assert np.all(np.isfinite(RSP)), "RSP should be finite"

    # Test PLSpectrum function
    hw = np.linspace(1.0, 2.0, 10)
    PLS = np.zeros(10)

    ep.PLSpectrum(ne, nh, Ee, Eh, 1.5, 1e12, VC, hw, 1e-12, PLS, solver)

    assert np.all(PLS >= 0), "PLS should be non-negative"
    assert np.all(np.isfinite(PLS)), "PLS should be finite"

    print("✓ Fortran-compatible interface tests passed")


def test_mathematical_properties():
    """Test mathematical properties and invariants."""
    print("Testing mathematical properties...")

    # Test photon density of states scaling
    hw1 = 1.0
    hw2 = 2.0

    rho1 = ep.rho0(hw1)
    rho2 = ep.rho0(hw2)

    # Should scale as (hw2/hw1)²
    expected_ratio = (hw2/hw1)**2
    actual_ratio = rho2/rho1

    assert_close(actual_ratio, expected_ratio, msg="Photon density scaling")

    # Test Coulomb energy linearity
    N = 8
    grid = MomentumGrid(ky=np.linspace(-1e8, 1e8, N))
    calculator = CoulombEnergyCalculator(grid)

    # Create test data
    ne1 = np.ones(N, dtype=np.float64) * 0.1
    nh1 = np.ones(N, dtype=np.float64) * 0.1
    ne2 = ne1 * 2.0
    nh2 = nh1 * 2.0
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

    Ec1 = calculator.calculate_coulomb_energy(ne1, nh1, VC)
    Ec2 = calculator.calculate_coulomb_energy(ne2, nh2, VC)

    # Should be linear: Ec2 ≈ 2 * Ec1
    assert_array_close(Ec2, 2.0 * Ec1, rtol=1e-6, atol=1e-6, msg="Coulomb energy linearity")

    # Test spontaneous emission rates are always positive
    ky = np.linspace(-1e8, 1e8, 8)
    Ee = np.linspace(0.1, 0.2, 8)
    Eh = np.linspace(0.1, 0.2, 8)
    solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

    N = 8
    ne = np.ones(N, dtype=np.complex128) * 0.1
    nh = np.ones(N, dtype=np.complex128) * 0.1
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

    rates = solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)

    # Physical requirement: rates must be non-negative
    assert np.all(rates >= 0), "Spontaneous emission rates should be non-negative"

    print("✓ Mathematical properties tests passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases...")

    # Test single momentum point
    ky = np.array([0.0])
    Ee = np.array([0.1])
    Eh = np.array([0.1])

    solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

    ne = np.ones(1, dtype=np.complex128) * 0.1
    nh = np.ones(1, dtype=np.complex128) * 0.1
    VC = np.ones((1, 1, 3), dtype=np.float64) * 1e-20

    rates = solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)

    assert rates.shape == (1,), "Single point should have shape (1,)"
    assert rates[0] >= 0, "Rate should be non-negative"

    # Test zero populations
    ky = np.linspace(-1e8, 1e8, 8)
    Ee = np.linspace(0.1, 0.2, 8)
    Eh = np.linspace(0.1, 0.2, 8)
    solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

    N = 8
    ne = np.zeros(N, dtype=np.complex128)
    nh = np.zeros(N, dtype=np.complex128)
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

    rates = solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)

    # With zero populations, rates should still be finite due to band gap energy
    # but should be non-negative and finite
    assert np.all(rates >= 0), "Rates should be non-negative"
    assert np.all(np.isfinite(rates)), "Rates should be finite"

    print("✓ Edge cases tests passed")


def test_performance():
    """Test performance with moderate system size."""
    print("Testing performance...")

    N = 32  # Moderate system size
    ky = np.linspace(-1e8, 1e8, N)
    Ee = np.linspace(0.1, 0.2, N)
    Eh = np.linspace(0.1, 0.2, N)

    solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

    ne = np.ones(N, dtype=np.complex128) * 0.1
    nh = np.ones(N, dtype=np.complex128) * 0.1
    VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

    # Time the calculation
    import time
    start = time.time()
    rates = solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)
    end = time.time()

    # Should complete in reasonable time (< 5 seconds)
    assert (end - start) < 5.0, f"Calculation took too long: {end - start:.2f} seconds"
    assert rates.shape == (N,), "Rates should have correct shape"

    print(f"✓ Performance test passed (took {end - start:.3f} seconds)")


def test_documentation():
    """Test documentation requirements and docstring quality."""
    print("Testing documentation...")

    # Test that key functions have docstrings
    functions_to_check = [
        ep.InitializeEmission,
        ep.SpontEmission,
        ep.Ec,
        ep.SpontIntegral,
        ep.rho0,
        ep.CalcHOmega,
        ep.Calchw,
        ep.PLSpectrum
    ]

    for func in functions_to_check:
        assert func.__doc__ is not None, f"{func.__name__} should have a docstring"
        assert len(func.__doc__.strip()) > 10, f"{func.__name__} docstring should be substantial"

    # Test that classes have docstrings
    classes_to_check = [
        ep.EmissionParameters,
        ep.MomentumGrid,
        ep.EmissionSolver,
        ep.PhotonGridCalculator,
        ep.CoulombEnergyCalculator,
        ep.SpontaneousEmissionCalculator,
        ep.PhotoluminescenceCalculator
    ]

    for cls in classes_to_check:
        assert cls.__doc__ is not None, f"{cls.__name__} should have a docstring"
        assert len(cls.__doc__.strip()) > 10, f"{cls.__name__} docstring should be substantial"

    # Test that key methods have docstrings with mathematical content
    solver = ep.EmissionSolver(ep.EmissionParameters(), ep.MomentumGrid(ky=np.array([0.0])))

    methods_to_check = [
        solver.calculate_spontaneous_emission_rates,
        solver.calculate_photoluminescence_spectrum,
        solver.calculate_coulomb_energy
    ]

    for method in methods_to_check:
        assert method.__doc__ is not None, f"{method.__name__} should have a docstring"
        doc = method.__doc__.lower()
        # Check for mathematical/physical content
        has_math = any(word in doc for word in ['energy', 'photon', 'emission', 'coulomb', 'spectrum'])
        assert has_math, f"{method.__name__} docstring should contain physical/mathematical content"

    # Test that rho0 function has a docstring (may be minimal for interface functions)
    rho0_doc = ep.rho0.__doc__
    assert rho0_doc is not None, "rho0 should have a docstring"
    assert len(rho0_doc.strip()) > 5, "rho0 docstring should not be empty"

    print("✓ Documentation tests passed")


def main():
    """Run all tests."""
    print("Running emissionpythonic.py test suite...")
    print("=" * 50)

    try:
        test_emission_parameters()
        test_momentum_grid()
        test_photon_grid_calculator()
        test_coulomb_energy_calculator()
        test_spontaneous_emission_calculator()
        test_photoluminescence_calculator()
        test_emission_solver()
        test_fortran_compatible_interface()
        test_mathematical_properties()
        test_edge_cases()
        test_performance()
        test_documentation()

        print("=" * 50)
        print("🎉 All tests passed successfully!")
        print("The emissionpythonic.py module is working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
