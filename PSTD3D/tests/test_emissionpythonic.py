"""
test_emissionpythonic.py
========================

Comprehensive test suite for emissionpythonic.py following the testing directives.
Tests define the intended behavior based on mathematical truth and physical reality.
"""

import pytest
import numpy as np
from numpy.typing import NDArray
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module under test
import sys
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


class TestEmissionParameters:
    """Test EmissionParameters dataclass validation."""

    def test_valid_parameters(self):
        """Test valid parameter initialization."""
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

    @pytest.mark.parametrize("temp", [0.0, -1.0, -100.0])
    def test_invalid_temperature(self, temp):
        """Test temperature validation."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            EmissionParameters(temperature=temp)

    @pytest.mark.parametrize("kb", [0.0, -1.0])
    def test_invalid_boltzmann_constant(self, kb):
        """Test Boltzmann constant validation."""
        with pytest.raises(ValueError, match="Boltzmann constant must be positive"):
            EmissionParameters(boltzmann_constant=kb)

    @pytest.mark.parametrize("dcv", [0.0, -1e-28])
    def test_invalid_dipole_matrix_element(self, dcv):
        """Test dipole matrix element validation."""
        with pytest.raises(ValueError, match="Dipole matrix element must be positive"):
            EmissionParameters(dipole_matrix_element=dcv)


class TestMomentumGrid:
    """Test MomentumGrid validation and properties."""

    def test_valid_grid(self):
        """Test valid grid initialization."""
        ky = np.linspace(-1e8, 1e8, 32)
        grid = MomentumGrid(ky=ky)
        assert grid.size == 32
        assert np.array_equal(grid.ky, ky)

    def test_empty_grid(self):
        """Test empty grid validation."""
        with pytest.raises(ValueError, match="Momentum grid cannot be empty"):
            MomentumGrid(ky=np.array([]))


class TestPhotonGridCalculator:
    """Test PhotonGridCalculator functionality."""

    @pytest.fixture
    def params(self):
        return EmissionParameters()

    @pytest.fixture
    def calculator(self, params):
        return PhotonGridCalculator(params)

    def test_rscale_calculation(self, calculator):
        """Test scaling constant calculation."""
        dcv = 1e-28
        epsr = 12.0
        ehint = 1.0
        expected = 3.0 * dcv**2 / ep.eps0 / np.sqrt(epsr) * ehint**2
        result = calculator._calculate_rscale(dcv, epsr, ehint)
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_photon_grid_calculation(self, calculator):
        """Test photon energy grid calculation."""
        kBT = 1e-20
        hg = 1e-21
        grid = calculator._calculate_photon_grid(kBT, hg)

        # Check grid properties
        assert len(grid) > 10  # Nw >= 10 requirement
        assert np.all(grid > 0)  # All energies positive
        assert np.all(np.diff(grid) > 0)  # Monotonically increasing

        # Check grid spacing
        dhw = grid[1] - grid[0]
        expected_dhw = min(kBT, hg) / 20.0
        assert np.isclose(dhw, expected_dhw, rtol=RTOL, atol=ATOL)

    def test_low_temperature_error(self, calculator):
        """Test error for too low temperature."""
        kBT = 1e-30  # Very low temperature
        hg = 1e-31
        with pytest.raises(ValueError, match="Temperature is too low"):
            calculator._calculate_photon_grid(kBT, hg)

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        ky = np.linspace(-1e8, 1e8, 16)
        dcv = 1e-28
        epsr = 12.0
        geh = 1e12
        ehint = 1.0

        calculator.initialize(ky, dcv, epsr, geh, ehint)

        # Check properties are accessible
        grid = calculator.photon_grid
        weights = calculator.spectral_weights
        rscale = calculator.rscale

        assert len(grid) > 0
        assert len(weights) == len(grid)
        assert rscale > 0


class TestCoulombEnergyCalculator:
    """Test CoulombEnergyCalculator functionality."""

    @pytest.fixture
    def grid(self):
        return MomentumGrid(ky=np.linspace(-1e8, 1e8, 8))

    @pytest.fixture
    def calculator(self, grid):
        return CoulombEnergyCalculator(grid)

    def test_identity_delta_matrix(self, calculator):
        """Test identity delta matrix construction."""
        idel = calculator._build_identity_delta_matrix()
        N = calculator.grid.size

        # Check shape
        assert idel.shape == (N, N)

        # Check diagonal is zero
        assert np.all(np.diag(idel) == 0)

        # Check off-diagonal is one
        mask = ~np.eye(N, dtype=bool)
        assert np.all(idel[mask] == 1)

    def test_coulomb_energy_calculation(self, calculator):
        """Test Coulomb energy calculation."""
        N = calculator.grid.size

        # Create test data
        ne = np.ones(N, dtype=np.float64) * 0.1
        nh = np.ones(N, dtype=np.float64) * 0.2
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

        # Calculate Coulomb energy
        Ec = calculator.calculate_coulomb_energy(ne, nh, VC)

        # Check output properties
        assert Ec.shape == (N,)
        assert Ec.dtype == np.float64
        assert np.all(np.isfinite(Ec))

    def test_coulomb_energy_symmetry(self, calculator):
        """Test Coulomb energy calculation symmetry properties."""
        N = calculator.grid.size

        # Create symmetric test data
        ne = np.ones(N, dtype=np.float64) * 0.1
        nh = np.ones(N, dtype=np.float64) * 0.1
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

        # Make VC symmetric
        VC[:, :, 1] = VC[:, :, 1] + VC[:, :, 1].T  # Vee
        VC[:, :, 2] = VC[:, :, 2] + VC[:, :, 2].T  # Vhh

        Ec = calculator.calculate_coulomb_energy(ne, nh, VC)

        # For symmetric case, all Ec should be equal
        assert np.allclose(Ec, Ec[0], rtol=RTOL, atol=ATOL)


class TestSpontaneousEmissionCalculator:
    """Test SpontaneousEmissionCalculator functionality."""

    @pytest.fixture
    def params(self):
        return EmissionParameters()

    @pytest.fixture
    def photon_calc(self, params):
        calc = PhotonGridCalculator(params)
        ky = np.linspace(-1e8, 1e8, 8)
        calc.initialize(ky, 1e-28, 12.0, 1e12, 1.0)
        return calc

    @pytest.fixture
    def calculator(self, params, photon_calc):
        return SpontaneousEmissionCalculator(params, photon_calc)

    def test_photon_density_of_states(self, calculator):
        """Test photon density of states calculation."""
        hw = 1.5  # eV
        rho = calculator.calculate_photon_density_of_states(hw)

        # Check formula: rho0 = hw^2 / (c0^3 * pi^2 * hbar^3)
        expected = (hw**2) / (ep.c0**3 * ep.pi**2 * ep.hbar**3)
        assert np.isclose(rho, expected, rtol=RTOL, atol=ATOL)

    def test_photon_density_of_states_vectorized(self, calculator):
        """Test vectorized photon density of states calculation."""
        hw = np.array([1.0, 1.5, 2.0])
        rho = calculator.calculate_photon_density_of_states(hw)

        # Check shape and values
        assert rho.shape == hw.shape
        assert np.all(rho > 0)

        # Check individual values
        for i, hw_i in enumerate(hw):
            expected = (hw_i**2) / (ep.c0**3 * ep.pi**2 * ep.hbar**3)
            assert np.isclose(rho[i], expected, rtol=RTOL, atol=ATOL)

    def test_spontaneous_emission_integral_scalar(self, calculator):
        """Test spontaneous emission integral for scalar input."""
        Ek = 1.5
        result = calculator.calculate_spontaneous_emission_integral(Ek)

        # Check output properties
        assert isinstance(result, np.float64)
        assert result >= 0  # Should be non-negative
        assert np.isfinite(result)

    def test_spontaneous_emission_integral_vectorized(self, calculator):
        """Test vectorized spontaneous emission integral."""
        Ek = np.array([1.0, 1.5, 2.0])
        result = calculator.calculate_spontaneous_emission_integral(Ek)

        # Check output properties
        assert result.shape == Ek.shape
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))

    def test_spontaneous_emission_rates(self, calculator):
        """Test spontaneous emission rates calculation."""
        N = 8
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1
        Ee = np.linspace(0.1, 0.2, N)
        Eh = np.linspace(0.1, 0.2, N)
        gap = 1.5
        geh = 1e12
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

        rates = calculator.calculate_spontaneous_emission_rates(
            ne, nh, Ee, Eh, gap, geh, VC
        )

        # Check output properties
        assert rates.shape == (N,)
        assert rates.dtype == np.float64
        assert np.all(rates >= 0)
        assert np.all(np.isfinite(rates))


class TestPhotoluminescenceCalculator:
    """Test PhotoluminescenceCalculator functionality."""

    @pytest.fixture
    def params(self):
        return EmissionParameters()

    @pytest.fixture
    def photon_calc(self, params):
        calc = PhotonGridCalculator(params)
        ky = np.linspace(-1e8, 1e8, 8)
        calc.initialize(ky, 1e-28, 12.0, 1e12, 1.0)
        return calc

    @pytest.fixture
    def calculator(self, params, photon_calc):
        return PhotoluminescenceCalculator(params, photon_calc)

    def test_linear_interpolation_real(self, calculator):
        """Test real linear interpolation."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])
        xq = np.array([0.5, 1.5])

        result = calculator._linear_interpolate_real(y, x, xq)
        expected = np.array([0.5, 2.5])

        assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_linear_interpolation_complex(self, calculator):
        """Test complex linear interpolation."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0+0.0j, 1.0+1.0j, 4.0+2.0j])
        xq = np.array([0.5, 1.5])

        result = calculator._linear_interpolate_complex(y, x, xq)
        expected = np.array([0.5+0.5j, 2.5+1.5j])

        assert np.allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_photoluminescence_spectrum(self, calculator):
        """Test photoluminescence spectrum calculation."""
        N = 8
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1
        Ee = np.linspace(0.1, 0.2, N)
        Eh = np.linspace(0.1, 0.2, N)
        gap = 1.5
        geh = 1e12
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
        hw = np.linspace(1.0, 2.0, 10)
        t = 1e-12

        spectrum = calculator.calculate_photoluminescence_spectrum(
            ne, nh, Ee, Eh, gap, geh, VC, hw, t
        )

        # Check output properties
        assert spectrum.shape == hw.shape
        assert spectrum.dtype == np.float64
        assert np.all(spectrum >= 0)
        assert np.all(np.isfinite(spectrum))


class TestEmissionSolver:
    """Test main EmissionSolver class."""

    @pytest.fixture
    def params(self):
        return EmissionParameters()

    @pytest.fixture
    def grid(self):
        return MomentumGrid(ky=np.linspace(-1e8, 1e8, 8))

    @pytest.fixture
    def solver(self, params, grid):
        return EmissionSolver(params, grid)

    def test_initialization(self, solver):
        """Test solver initialization."""
        assert not solver._initialized

        solver.initialize(1e-28, 12.0, 1e12, 1.0)
        assert solver._initialized

    def test_uninitialized_error(self, solver):
        """Test error when using uninitialized solver."""
        N = 8
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1
        Ee = np.linspace(0.1, 0.2, N)
        Eh = np.linspace(0.1, 0.2, N)
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

        with pytest.raises(RuntimeError, match="not initialized"):
            solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)

    def test_spontaneous_emission_rates(self, solver):
        """Test spontaneous emission rates calculation."""
        solver.initialize(1e-28, 12.0, 1e12, 1.0)

        N = 8
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1
        Ee = np.linspace(0.1, 0.2, N)
        Eh = np.linspace(0.1, 0.2, N)
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

        rates = solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)

        assert rates.shape == (N,)
        assert rates.dtype == np.float64
        assert np.all(rates >= 0)

    def test_photoluminescence_spectrum(self, solver):
        """Test photoluminescence spectrum calculation."""
        solver.initialize(1e-28, 12.0, 1e12, 1.0)

        N = 8
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1
        Ee = np.linspace(0.1, 0.2, N)
        Eh = np.linspace(0.1, 0.2, N)
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
        hw = np.linspace(1.0, 2.0, 10)

        spectrum = solver.calculate_photoluminescence_spectrum(
            ne, nh, Ee, Eh, 1.5, 1e12, VC, hw, 1e-12
        )

        assert spectrum.shape == hw.shape
        assert spectrum.dtype == np.float64
        assert np.all(spectrum >= 0)


class TestFortranCompatibleInterface:
    """Test Fortran-compatible interface functions."""

    def test_initialize_emission(self):
        """Test InitializeEmission function."""
        ky = np.linspace(-1e8, 1e8, 8)
        Ee = np.linspace(0.1, 0.2, 8)
        Eh = np.linspace(0.1, 0.2, 8)

        solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

        assert isinstance(solver, EmissionSolver)
        assert solver._initialized

    def test_rho0_function(self):
        """Test rho0 function."""
        hw = 1.5
        result = ep.rho0(hw)
        expected = (hw**2) / (ep.c0**3 * ep.pi**2 * ep.hbar**3)

        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_calc_homega_function(self):
        """Test CalcHOmega function."""
        kBT = 1e-20
        hg = 1e-21
        result = ep.CalcHOmega(kBT, hg)

        assert len(result) > 10
        assert np.all(result > 0)
        assert np.all(np.diff(result) > 0)

    def test_calc_hw_function(self):
        """Test Calchw function."""
        hw = np.zeros(10)
        PLS = np.zeros(10)
        Estart = 1.0
        Emax = 2.0

        ep.Calchw(hw, PLS, Estart, Emax)

        # Check hw array
        expected_hw = np.linspace(Estart, Emax, 10)
        assert np.allclose(hw, expected_hw, rtol=RTOL, atol=ATOL)

        # Check PLS array is zeroed
        assert np.all(PLS == 0.0)

    def test_spont_emission_function(self):
        """Test SpontEmission function."""
        ky = np.linspace(-1e8, 1e8, 8)
        Ee = np.linspace(0.1, 0.2, 8)
        Eh = np.linspace(0.1, 0.2, 8)
        solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

        N = 8
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
        RSP = np.zeros(N)

        ep.SpontEmission(ne, nh, Ee, Eh, 1.5, 1e12, VC, RSP, solver)

        assert np.all(RSP >= 0)
        assert np.all(np.isfinite(RSP))

    def test_pl_spectrum_function(self):
        """Test PLSpectrum function."""
        ky = np.linspace(-1e8, 1e8, 8)
        Ee = np.linspace(0.1, 0.2, 8)
        Eh = np.linspace(0.1, 0.2, 8)
        solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

        N = 8
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
        hw = np.linspace(1.0, 2.0, 10)
        PLS = np.zeros(10)

        ep.PLSpectrum(ne, nh, Ee, Eh, 1.5, 1e12, VC, hw, 1e-12, PLS, solver)

        assert np.all(PLS >= 0)
        assert np.all(np.isfinite(PLS))


class TestMathematicalProperties:
    """Test mathematical properties and invariants."""

    def test_photon_density_of_states_scaling(self):
        """Test photon density of states scales as ω²."""
        hw1 = 1.0
        hw2 = 2.0

        rho1 = ep.rho0(hw1)
        rho2 = ep.rho0(hw2)

        # Should scale as (hw2/hw1)²
        expected_ratio = (hw2/hw1)**2
        actual_ratio = rho2/rho1

        assert np.isclose(actual_ratio, expected_ratio, rtol=RTOL, atol=ATOL)

    def test_coulomb_energy_linearity(self):
        """Test Coulomb energy is linear in populations."""
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
        assert np.allclose(Ec2, 2.0 * Ec1, rtol=1e-6, atol=1e-6)

    def test_spontaneous_emission_positive(self):
        """Test spontaneous emission rates are always positive."""
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
        assert np.all(rates >= 0)

    def test_photoluminescence_spectrum_positive(self):
        """Test photoluminescence spectrum is always positive."""
        ky = np.linspace(-1e8, 1e8, 8)
        Ee = np.linspace(0.1, 0.2, 8)
        Eh = np.linspace(0.1, 0.2, 8)
        solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

        N = 8
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
        hw = np.linspace(1.0, 2.0, 10)

        spectrum = solver.calculate_photoluminescence_spectrum(
            ne, nh, Ee, Eh, 1.5, 1e12, VC, hw, 1e-12
        )

        # Physical requirement: spectrum must be non-negative
        assert np.all(spectrum >= 0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_momentum_point(self):
        """Test with single momentum point."""
        ky = np.array([0.0])
        Ee = np.array([0.1])
        Eh = np.array([0.1])

        solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

        ne = np.ones(1, dtype=np.complex128) * 0.1
        nh = np.ones(1, dtype=np.complex128) * 0.1
        VC = np.ones((1, 1, 3), dtype=np.float64) * 1e-20

        rates = solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)

        assert rates.shape == (1,)
        assert rates[0] >= 0

    def test_zero_populations(self):
        """Test with zero populations."""
        ky = np.linspace(-1e8, 1e8, 8)
        Ee = np.linspace(0.1, 0.2, 8)
        Eh = np.linspace(0.1, 0.2, 8)
        solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

        N = 8
        ne = np.zeros(N, dtype=np.complex128)
        nh = np.zeros(N, dtype=np.complex128)
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

        rates = solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)

        # With zero populations, rates should be zero
        assert np.allclose(rates, 0.0, rtol=RTOL, atol=ATOL)

    def test_extreme_energies(self):
        """Test with extreme energy values."""
        ky = np.linspace(-1e8, 1e8, 8)
        Ee = np.array([1e-30, 1e-20, 1e-10, 1e-5, 1e-3, 1e-1, 1e1, 1e3])
        Eh = Ee.copy()
        solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

        N = 8
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

        rates = solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)

        assert np.all(np.isfinite(rates))
        assert np.all(rates >= 0)


class TestPerformance:
    """Test performance and memory usage."""

    @pytest.mark.slow
    def test_large_system_performance(self):
        """Test performance with large system."""
        N = 128  # Large system
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

        # Should complete in reasonable time (< 10 seconds)
        assert (end - start) < 10.0
        assert rates.shape == (N,)

    def test_memory_efficiency(self):
        """Test memory efficiency with large arrays."""
        N = 64
        ky = np.linspace(-1e8, 1e8, N)
        Ee = np.linspace(0.1, 0.2, N)
        Eh = np.linspace(0.1, 0.2, N)

        solver = ep.InitializeEmission(ky, Ee, Eh, 1e-28, 12.0, 1e12, 1.0)

        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20

        # Multiple calculations should not cause memory issues
        for _ in range(10):
            rates = solver.calculate_spontaneous_emission_rates(ne, nh, Ee, Eh, 1.5, 1e12, VC)
            assert rates.shape == (N,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
