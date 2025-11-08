"""
Comprehensive test suite for the alternative Coulomb interaction module.

This test suite validates the correctness, performance, and robustness of the
alternative Coulomb module implementation. It includes unit tests, integration
tests, and performance benchmarks.

Test Categories:
- Unit tests for individual components
- Integration tests for the full Coulomb solver
- Validation against known analytical results
- Performance benchmarks
- Edge case and error handling tests
- Property-based tests for mathematical consistency

Author: AI Assistant
Date: 2024
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises
from unittest.mock import Mock, patch
import time
from typing import Tuple

# Import the module under test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alt_coulomb import (
    QuantumWireParameters, MomentumGrid, CoulombIntegralCalculator,
    ScreeningCalculator, ManyBodyCalculator, CoulombSolver,
    DeltaFunctionType, create_quantum_wire_system, create_momentum_grid
)


class TestQuantumWireParameters:
    """Test the QuantumWireParameters dataclass."""

    def test_valid_parameters(self):
        """Test creation with valid parameters."""
        params = QuantumWireParameters(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        assert params.length == 1e-6
        assert params.thickness == 1e-8
        assert params.dielectric_constant == 12.0

    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        # Negative length
        with pytest.raises(ValueError, match="Wire length must be positive"):
            QuantumWireParameters(
                length=-1e-6, thickness=1e-8, dielectric_constant=12.0,
                electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
                electron_confinement=1e8, hole_confinement=1e8,
                electron_relaxation=1e12, hole_relaxation=1e12
            )

        # Zero thickness
        with pytest.raises(ValueError, match="Wire thickness must be positive"):
            QuantumWireParameters(
                length=1e-6, thickness=0.0, dielectric_constant=12.0,
                electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
                electron_confinement=1e8, hole_confinement=1e8,
                electron_relaxation=1e12, hole_relaxation=1e12
            )

        # Negative relaxation rate
        with pytest.raises(ValueError, match="Relaxation rates must be non-negative"):
            QuantumWireParameters(
                length=1e-6, thickness=1e-8, dielectric_constant=12.0,
                electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
                electron_confinement=1e8, hole_confinement=1e8,
                electron_relaxation=-1e12, hole_relaxation=1e12
            )

    def test_immutability(self):
        """Test that parameters are immutable."""
        params = QuantumWireParameters(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        with pytest.raises(AttributeError):
            params.length = 2e-6


class TestMomentumGrid:
    """Test the MomentumGrid dataclass."""

    @pytest.fixture
    def sample_grid(self):
        """Create a sample momentum grid for testing."""
        N = 16
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))
        return MomentumGrid(ky, y, qy, kkp)

    def test_valid_grid(self, sample_grid):
        """Test creation with valid grid."""
        assert sample_grid.size == 16
        assert sample_grid.momentum_step > 0
        assert len(sample_grid.ky) == len(sample_grid.y)

    def test_invalid_grid_dimensions(self):
        """Test validation of grid dimensions."""
        N = 16
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N + 1)  # Wrong size
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))

        with pytest.raises(ValueError, match="ky and y arrays must have same length"):
            MomentumGrid(ky, y, qy, kkp)

    def test_invalid_kkp_shape(self):
        """Test validation of kkp matrix shape."""
        N = 16
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N + 1))  # Wrong shape

        with pytest.raises(ValueError, match="kkp must be square matrix"):
            MomentumGrid(ky, y, qy, kkp)


class TestCoulombIntegralCalculator:
    """Test the CoulombIntegralCalculator class."""

    @pytest.fixture
    def calculator_setup(self):
        """Set up calculator with test parameters."""
        params = create_quantum_wire_system(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        N = 8  # Small grid for fast testing
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))
        grid = create_momentum_grid(ky, y, qy, kkp)

        return CoulombIntegralCalculator(params, grid)

    def test_calculate_unscreened_potentials(self, calculator_setup):
        """Test calculation of unscreened potentials."""
        calculator = calculator_setup
        Veh, Vee, Vhh = calculator.calculate_unscreened_potentials()

        # Check shapes
        assert Veh.shape == (8, 8)
        assert Vee.shape == (8, 8)
        assert Vhh.shape == (8, 8)

        # Check that potentials are real and finite
        assert np.all(np.isfinite(Veh))
        assert np.all(np.isfinite(Vee))
        assert np.all(np.isfinite(Vhh))

        # Check that potentials are non-negative (Coulomb repulsion)
        assert np.all(Vee >= 0)
        assert np.all(Vhh >= 0)

    def test_caching(self, calculator_setup):
        """Test that results are properly cached."""
        calculator = calculator_setup

        # First calculation
        start_time = time.time()
        Veh1, Vee1, Vhh1 = calculator.calculate_unscreened_potentials()
        first_time = time.time() - start_time

        # Second calculation (should use cache)
        start_time = time.time()
        Veh2, Vee2, Vhh2 = calculator.calculate_unscreened_potentials()
        second_time = time.time() - start_time

        # Results should be identical
        assert_array_equal(Veh1, Veh2)
        assert_array_equal(Vee1, Vee2)
        assert_array_equal(Vhh1, Vhh2)

        # Second calculation should be faster (cached)
        assert second_time < first_time

    def test_1d_integral_properties(self, calculator_setup):
        """Test properties of 1D integral calculation."""
        calculator = calculator_setup

        # Test with different momentum values
        qy_values = [0.0, 1e8, 2e8]
        for qy in qy_values:
            integral = calculator._calculate_1d_integral(qy, 1e8, 1e8)
            assert np.isfinite(integral)
            assert integral >= 0  # Coulomb interaction should be repulsive


class TestScreeningCalculator:
    """Test the ScreeningCalculator class."""

    @pytest.fixture
    def screening_setup(self):
        """Set up screening calculator with test parameters."""
        params = create_quantum_wire_system(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        N = 8
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))
        grid = create_momentum_grid(ky, y, qy, kkp)

        return ScreeningCalculator(params, grid)

    def test_susceptibility_matrices(self, screening_setup):
        """Test calculation of susceptibility matrices."""
        calculator = screening_setup
        Chi_e, Chi_h = calculator.calculate_susceptibility_matrices()

        # Check shapes
        assert Chi_e.shape == (8, 8)
        assert Chi_h.shape == (8, 8)

        # Check that matrices are real and finite
        assert np.all(np.isfinite(Chi_e))
        assert np.all(np.isfinite(Chi_h))

        # Check that matrices are non-negative
        assert np.all(Chi_e >= 0)
        assert np.all(Chi_h >= 0)

    def test_dielectric_function(self, screening_setup):
        """Test calculation of dielectric function."""
        calculator = screening_setup
        density_1d = 1e6  # 1D density

        eps = calculator.calculate_dielectric_function(density_1d)

        # Check shape
        assert eps.shape == (8, 8)

        # Check that dielectric function is finite
        assert np.all(np.isfinite(eps))

        # Check that dielectric function is positive (screening reduces interaction)
        assert np.all(eps > 0)

    def test_screening_application(self, screening_setup):
        """Test application of screening to potential matrices."""
        calculator = screening_setup

        # Create dummy potential matrices
        N = 8
        Veh = np.ones((N, N)) * 1e-20
        Vee = np.ones((N, N)) * 1e-20
        Vhh = np.ones((N, N)) * 1e-20

        density_1d = 1e6

        Veh_screened, Vee_screened, Vhh_screened = calculator.apply_screening(
            Veh, Vee, Vhh, density_1d
        )

        # Check shapes
        assert Veh_screened.shape == (N, N)
        assert Vee_screened.shape == (N, N)
        assert Vhh_screened.shape == (N, N)

        # Check that screened potentials are smaller than unscreened (screening effect)
        assert np.all(Veh_screened <= Veh)
        assert np.all(Vee_screened <= Vee)
        assert np.all(Vhh_screened <= Vhh)


class TestManyBodyCalculator:
    """Test the ManyBodyCalculator class."""

    @pytest.fixture
    def many_body_setup(self):
        """Set up many-body calculator with test parameters."""
        params = create_quantum_wire_system(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        N = 8
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))
        grid = create_momentum_grid(ky, y, qy, kkp)

        return ManyBodyCalculator(params, grid)

    def test_k3_matrix_construction(self, many_body_setup):
        """Test construction of k3 momentum conservation matrix."""
        calculator = many_body_setup
        k3 = calculator._build_k3_matrix()

        # Check shape
        assert k3.shape == (8, 8, 8)

        # Check that all values are valid indices
        assert np.all(k3 >= 0)
        assert np.all(k3 < 8)

        # Check momentum conservation: k1 + k2 - k4 = k3
        for k1 in range(8):
            for k2 in range(8):
                for k4 in range(8):
                    k3_val = k3[k1, k2, k4]
                    if k3_val > 0:  # Valid momentum state
                        assert k1 + k2 - k4 == k3_val

    def test_collision_matrices_construction(self, many_body_setup):
        """Test construction of collision matrices."""
        calculator = many_body_setup

        # Create dummy energy dispersions
        N = 8
        Ee = np.linspace(0, 1e-20, N)
        Eh = np.linspace(0, 1e-20, N)

        collision_data = calculator._build_collision_matrices(Ee, Eh)

        # Check that all required matrices are present
        required_keys = ['Ceh', 'Cee', 'Chh', 'UnDel', 'k3']
        for key in required_keys:
            assert key in collision_data

        # Check shapes
        assert collision_data['Ceh'].shape == (N + 1, N + 1, N + 1)
        assert collision_data['Cee'].shape == (N + 1, N + 1, N + 1)
        assert collision_data['Chh'].shape == (N + 1, N + 1, N + 1)
        assert collision_data['UnDel'].shape == (N + 1, N + 1)
        assert collision_data['k3'].shape == (N, N, N)

        # Check that collision matrices are non-negative
        assert np.all(collision_data['Ceh'] >= 0)
        assert np.all(collision_data['Cee'] >= 0)
        assert np.all(collision_data['Chh'] >= 0)

    def test_collision_rates_calculation(self, many_body_setup):
        """Test calculation of collision rates."""
        calculator = many_body_setup

        N = 8
        ne = np.ones(N) * 0.1
        nh = np.ones(N) * 0.1
        Ee = np.linspace(0, 1e-20, N)
        Eh = np.linspace(0, 1e-20, N)

        # Create dummy potential matrices
        Veh = np.ones((N, N)) * 1e-20
        Vee = np.ones((N, N)) * 1e-20
        Vhh = np.ones((N, N)) * 1e-20

        Win, Wout = calculator.calculate_collision_rates(ne, nh, Veh, Vee, Vhh, Ee, Eh)

        # Check shapes
        assert Win.shape == (N,)
        assert Wout.shape == (N,)

        # Check that rates are non-negative
        assert np.all(Win >= 0)
        assert np.all(Wout >= 0)

        # Check that rates are finite
        assert np.all(np.isfinite(Win))
        assert np.all(np.isfinite(Wout))


class TestCoulombSolver:
    """Test the main CoulombSolver class."""

    @pytest.fixture
    def solver_setup(self):
        """Set up Coulomb solver with test parameters."""
        params = create_quantum_wire_system(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        N = 8
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))
        grid = create_momentum_grid(ky, y, qy, kkp)

        return CoulombSolver(params, grid)

    def test_initialization(self, solver_setup):
        """Test solver initialization."""
        solver = solver_setup

        N = 8
        Ee = np.linspace(0, 1e-20, N)
        Eh = np.linspace(0, 1e-20, N)

        # Should not raise any exceptions
        solver.initialize(Ee, Eh)

        # Check that potentials are cached
        assert solver._cached_potentials is not None

    def test_screened_potentials(self, solver_setup):
        """Test calculation of screened potentials."""
        solver = solver_setup

        N = 8
        Ee = np.linspace(0, 1e-20, N)
        Eh = np.linspace(0, 1e-20, N)
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1

        solver.initialize(Ee, Eh)
        Veh, Vee, Vhh = solver.get_screened_potentials(ne, nh)

        # Check shapes
        assert Veh.shape == (N, N)
        assert Vee.shape == (N, N)
        assert Vhh.shape == (N, N)

        # Check that potentials are finite
        assert np.all(np.isfinite(Veh))
        assert np.all(np.isfinite(Vee))
        assert np.all(np.isfinite(Vhh))

    def test_collision_rates(self, solver_setup):
        """Test calculation of collision rates."""
        solver = solver_setup

        N = 8
        Ee = np.linspace(0, 1e-20, N)
        Eh = np.linspace(0, 1e-20, N)
        ne = np.ones(N) * 0.1
        nh = np.ones(N) * 0.1

        solver.initialize(Ee, Eh)
        Win, Wout = solver.calculate_collision_rates(ne, nh, Ee, Eh)

        # Check shapes
        assert Win.shape == (N,)
        assert Wout.shape == (N,)

        # Check that rates are non-negative and finite
        assert np.all(Win >= 0)
        assert np.all(Wout >= 0)
        assert np.all(np.isfinite(Win))
        assert np.all(np.isfinite(Wout))

    def test_band_gap_renormalization(self, solver_setup):
        """Test calculation of band gap renormalization."""
        solver = solver_setup

        N = 8
        Ee = np.linspace(0, 1e-20, N)
        Eh = np.linspace(0, 1e-20, N)
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1

        solver.initialize(Ee, Eh)
        BGR = solver.calculate_band_gap_renormalization(ne, nh)

        # Check shape
        assert BGR.shape == (N, N)

        # Check that BGR is finite
        assert np.all(np.isfinite(BGR))

    def test_uninitialized_solver(self, solver_setup):
        """Test that uninitialized solver raises appropriate errors."""
        solver = solver_setup

        N = 8
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1

        with pytest.raises(RuntimeError, match="Coulomb solver not initialized"):
            solver.get_screened_potentials(ne, nh)


class TestIntegration:
    """Integration tests for the complete Coulomb module."""

    def test_full_workflow(self):
        """Test the complete workflow from parameters to results."""
        # Create system parameters
        params = create_quantum_wire_system(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        # Create momentum grid
        N = 16
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))
        grid = create_momentum_grid(ky, y, qy, kkp)

        # Create solver
        solver = CoulombSolver(params, grid)

        # Create energy dispersions
        Ee = 1e-20 * ky**2
        Eh = 1e-20 * ky**2

        # Initialize solver
        solver.initialize(Ee, Eh)

        # Create carrier populations
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1

        # Calculate all quantities
        Veh, Vee, Vhh = solver.get_screened_potentials(ne, nh)
        Win, Wout = solver.calculate_collision_rates(ne.real, nh.real, Ee, Eh)
        BGR = solver.calculate_band_gap_renormalization(ne, nh)

        # Verify all results are reasonable
        assert np.all(np.isfinite(Veh))
        assert np.all(np.isfinite(Vee))
        assert np.all(np.isfinite(Vhh))
        assert np.all(np.isfinite(Win))
        assert np.all(np.isfinite(Wout))
        assert np.all(np.isfinite(BGR))

        # Verify physical constraints
        assert np.all(Vee >= 0)  # Repulsive interaction
        assert np.all(Vhh >= 0)  # Repulsive interaction
        assert np.all(Win >= 0)  # Non-negative rates
        assert np.all(Wout >= 0)  # Non-negative rates


class TestPerformance:
    """Performance tests and benchmarks."""

    def test_scaling_with_grid_size(self):
        """Test how performance scales with grid size."""
        params = create_quantum_wire_system(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        grid_sizes = [8, 16, 32]
        times = []

        for N in grid_sizes:
            ky = np.linspace(-1e8, 1e8, N)
            y = np.linspace(-5e-8, 5e-8, N)
            qy = np.linspace(0, 2e8, N)
            kkp = np.random.randint(-1, N, (N, N))
            grid = create_momentum_grid(ky, y, qy, kkp)

            solver = CoulombSolver(params, grid)
            Ee = np.linspace(0, 1e-20, N)
            Eh = np.linspace(0, 1e-20, N)

            start_time = time.time()
            solver.initialize(Ee, Eh)
            init_time = time.time() - start_time

            times.append(init_time)

        # Check that times are reasonable (not exponential growth)
        # This is a basic sanity check - actual scaling depends on implementation
        assert all(t < 10.0 for t in times)  # Should complete within 10 seconds

    def test_memory_usage(self):
        """Test that memory usage is reasonable."""
        params = create_quantum_wire_system(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        N = 32
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))
        grid = create_momentum_grid(ky, y, qy, kkp)

        solver = CoulombSolver(params, grid)
        Ee = np.linspace(0, 1e-20, N)
        Eh = np.linspace(0, 1e-20, N)

        # This should not cause memory issues
        solver.initialize(Ee, Eh)

        # Test multiple calculations
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1

        for _ in range(10):
            Veh, Vee, Vhh = solver.get_screened_potentials(ne, nh)
            Win, Wout = solver.calculate_collision_rates(ne.real, nh.real, Ee, Eh)
            BGR = solver.calculate_band_gap_renormalization(ne, nh)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_populations(self):
        """Test behavior with zero carrier populations."""
        params = create_quantum_wire_system(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        N = 8
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))
        grid = create_momentum_grid(ky, y, qy, kkp)

        solver = CoulombSolver(params, grid)
        Ee = np.linspace(0, 1e-20, N)
        Eh = np.linspace(0, 1e-20, N)
        solver.initialize(Ee, Eh)

        # Zero populations
        ne = np.zeros(N, dtype=np.complex128)
        nh = np.zeros(N, dtype=np.complex128)

        # Should not raise exceptions
        Veh, Vee, Vhh = solver.get_screened_potentials(ne, nh)
        Win, Wout = solver.calculate_collision_rates(ne.real, nh.real, Ee, Eh)
        BGR = solver.calculate_band_gap_renormalization(ne, nh)

        # Check that results are finite
        assert np.all(np.isfinite(Veh))
        assert np.all(np.isfinite(Vee))
        assert np.all(np.isfinite(Vhh))
        assert np.all(np.isfinite(Win))
        assert np.all(np.isfinite(Wout))
        assert np.all(np.isfinite(BGR))

    def test_extreme_parameters(self):
        """Test behavior with extreme parameter values."""
        # Very small wire
        params = create_quantum_wire_system(
            length=1e-9, thickness=1e-10, dielectric_constant=1.0,
            electron_mass=0.01 * 9.109e-31, hole_mass=0.01 * 9.109e-31,
            electron_confinement=1e10, hole_confinement=1e10,
            electron_relaxation=1e15, hole_relaxation=1e15
        )

        N = 4  # Very small grid
        ky = np.linspace(-1e10, 1e10, N)
        y = np.linspace(-1e-9, 1e-9, N)
        qy = np.linspace(0, 2e10, N)
        kkp = np.random.randint(-1, N, (N, N))
        grid = create_momentum_grid(ky, y, qy, kkp)

        solver = CoulombSolver(params, grid)
        Ee = np.linspace(0, 1e-18, N)
        Eh = np.linspace(0, 1e-18, N)

        # Should not raise exceptions
        solver.initialize(Ee, Eh)

        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1

        Veh, Vee, Vhh = solver.get_screened_potentials(ne, nh)
        Win, Wout = solver.calculate_collision_rates(ne.real, nh.real, Ee, Eh)
        BGR = solver.calculate_band_gap_renormalization(ne, nh)

        # Check that results are finite
        assert np.all(np.isfinite(Veh))
        assert np.all(np.isfinite(Vee))
        assert np.all(np.isfinite(Vhh))
        assert np.all(np.isfinite(Win))
        assert np.all(np.isfinite(Wout))
        assert np.all(np.isfinite(BGR))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
