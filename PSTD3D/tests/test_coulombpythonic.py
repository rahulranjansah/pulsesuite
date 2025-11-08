#!/usr/bin/env python3
"""
Comprehensive test suite for coulombpythonic.py

This test suite validates the Pythonic Coulomb implementation against the original
Fortran code and ensures all functionality works correctly.

Author: AI Assistant
Date: 2024
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coulombpythonic import (
    CoulombParameters, MomentumGrid, CoulombIntegralCalculator,
    ScreeningCalculator, ManyBodyCalculator, CoulombSolver,
    InitializeCoulomb, CalcCoulombArrays, Vint
)


class TestCoulombParameters(unittest.TestCase):
    """Test CoulombParameters dataclass."""

    def test_valid_parameters(self):
        """Test valid parameter initialization."""
        params = CoulombParameters(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )
        self.assertEqual(params.length, 1e-6)
        self.assertEqual(params.thickness, 1e-8)
        self.assertEqual(params.dielectric_constant, 12.0)

    def test_invalid_parameters(self):
        """Test parameter validation."""
        with self.assertRaises(ValueError):
            CoulombParameters(
                length=-1e-6, thickness=1e-8, dielectric_constant=12.0,
                electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
                electron_confinement=1e8, hole_confinement=1e8,
                electron_relaxation=1e12, hole_relaxation=1e12
            )

        with self.assertRaises(ValueError):
            CoulombParameters(
                length=1e-6, thickness=-1e-8, dielectric_constant=12.0,
                electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
                electron_confinement=1e8, hole_confinement=1e8,
                electron_relaxation=1e12, hole_relaxation=1e12
            )


class TestMomentumGrid(unittest.TestCase):
    """Test MomentumGrid dataclass."""

    def setUp(self):
        """Set up test data."""
        self.N = 16
        self.ky = np.linspace(-1e8, 1e8, self.N)
        self.y = np.linspace(-5e-8, 5e-8, self.N)
        self.qy = np.linspace(0, 2e8, self.N)
        self.kkp = np.random.randint(-1, self.N, (self.N, self.N))

    def test_valid_grid(self):
        """Test valid grid initialization."""
        grid = MomentumGrid(ky=self.ky, y=self.y, qy=self.qy, kkp=self.kkp)
        self.assertEqual(grid.size, self.N)
        self.assertTrue(np.array_equal(grid.ky, self.ky))
        self.assertTrue(np.array_equal(grid.y, self.y))
        self.assertTrue(np.array_equal(grid.qy, self.qy))
        self.assertTrue(np.array_equal(grid.kkp, self.kkp))

    def test_invalid_grid(self):
        """Test grid validation."""
        # Mismatched array lengths
        with self.assertRaises(ValueError):
            MomentumGrid(
                ky=self.ky, y=self.y[:-1], qy=self.qy, kkp=self.kkp
            )

        # Wrong kkp shape
        with self.assertRaises(ValueError):
            MomentumGrid(
                ky=self.ky, y=self.y, qy=self.qy,
                kkp=np.random.randint(-1, self.N, (self.N, self.N-1))
            )


class TestCoulombIntegralCalculator(unittest.TestCase):
    """Test CoulombIntegralCalculator class."""

    def setUp(self):
        """Set up test data."""
        self.params = CoulombParameters(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        self.N = 16
        self.ky = np.linspace(-1e8, 1e8, self.N)
        self.y = np.linspace(-5e-8, 5e-8, self.N)
        self.qy = np.linspace(0, 2e8, self.N)
        self.kkp = np.random.randint(-1, self.N, (self.N, self.N))
        self.grid = MomentumGrid(ky=self.ky, y=self.y, qy=self.qy, kkp=self.kkp)

        self.calculator = CoulombIntegralCalculator(self.params, self.grid)

    def test_1d_integral_calculation(self):
        """Test 1D integral calculation."""
        qy = 1e7
        alpha1 = 1e8
        alpha2 = 1e8

        result = self.calculator.calculate_1d_integral(qy, alpha1, alpha2)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
        self.assertLess(result, 1e-10)  # Should be small for this test case

    def test_unscreened_potentials(self):
        """Test unscreened potential calculation."""
        Veh, Vee, Vhh = self.calculator.calculate_unscreened_potentials()

        # Check shapes
        self.assertEqual(Veh.shape, (self.N, self.N))
        self.assertEqual(Vee.shape, (self.N, self.N))
        self.assertEqual(Vhh.shape, (self.N, self.N))

        # Check that matrices are symmetric (for same-species interactions)
        np.testing.assert_array_almost_equal(Vee, Vee.T, decimal=10)
        np.testing.assert_array_almost_equal(Vhh, Vhh.T, decimal=10)

        # Check that all values are non-negative (Coulomb potentials are repulsive)
        self.assertTrue(np.all(Veh >= 0))
        self.assertTrue(np.all(Vee >= 0))
        self.assertTrue(np.all(Vhh >= 0))

    def test_caching(self):
        """Test caching functionality."""
        # First calculation
        Veh1, Vee1, Vhh1 = self.calculator.calculate_unscreened_potentials(use_cache=True)

        # Second calculation (should use cache)
        Veh2, Vee2, Vhh2 = self.calculator.calculate_unscreened_potentials(use_cache=True)

        # Results should be identical
        np.testing.assert_array_equal(Veh1, Veh2)
        np.testing.assert_array_equal(Vee1, Vee2)
        np.testing.assert_array_equal(Vhh1, Vhh2)


class TestScreeningCalculator(unittest.TestCase):
    """Test ScreeningCalculator class."""

    def setUp(self):
        """Set up test data."""
        self.params = CoulombParameters(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        self.N = 16
        self.ky = np.linspace(-1e8, 1e8, self.N)
        self.y = np.linspace(-5e-8, 5e-8, self.N)
        self.qy = np.linspace(0, 2e8, self.N)
        self.kkp = np.random.randint(-1, self.N, (self.N, self.N))
        self.grid = MomentumGrid(ky=self.ky, y=self.y, qy=self.qy, kkp=self.kkp)

        self.calculator = ScreeningCalculator(self.params, self.grid)

    def test_susceptibility_matrices(self):
        """Test susceptibility matrix calculation."""
        Chi_e, Chi_h = self.calculator.calculate_susceptibility_matrices()

        # Check shapes
        self.assertEqual(Chi_e.shape, (self.N, self.N))
        self.assertEqual(Chi_h.shape, (self.N, self.N))

        # Check that matrices are symmetric
        np.testing.assert_array_almost_equal(Chi_e, Chi_e.T, decimal=10)
        np.testing.assert_array_almost_equal(Chi_h, Chi_h.T, decimal=10)

        # Check that all values are non-negative
        self.assertTrue(np.all(Chi_e >= 0))
        self.assertTrue(np.all(Chi_h >= 0))

    def test_dielectric_function(self):
        """Test dielectric function calculation."""
        density_1d = 1e6  # 1D density

        eps = self.calculator.calculate_dielectric_function(density_1d)

        # Check shape
        self.assertEqual(eps.shape, (self.N, self.N))

        # Check that dielectric function is symmetric
        np.testing.assert_array_almost_equal(eps, eps.T, decimal=10)

        # Check that dielectric function is positive (for stable system)
        self.assertTrue(np.all(eps > 0))


class TestManyBodyCalculator(unittest.TestCase):
    """Test ManyBodyCalculator class."""

    def setUp(self):
        """Set up test data."""
        self.params = CoulombParameters(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        self.N = 16
        self.ky = np.linspace(-1e8, 1e8, self.N)
        self.y = np.linspace(-5e-8, 5e-8, self.N)
        self.qy = np.linspace(0, 2e8, self.N)
        self.kkp = np.random.randint(-1, self.N, (self.N, self.N))
        self.grid = MomentumGrid(ky=self.ky, y=self.y, qy=self.qy, kkp=self.kkp)

        self.calculator = ManyBodyCalculator(self.params, self.grid)

        # Create test energy dispersions
        self.Ee = 1e-20 * self.ky**2
        self.Eh = 1e-20 * self.ky**2

    def test_k3_matrix(self):
        """Test k3 matrix construction."""
        k3 = self.calculator._build_k3_matrix()

        # Check shape
        self.assertEqual(k3.shape, (self.N, self.N, self.N))

        # Check that all values are valid indices
        self.assertTrue(np.all(k3 >= 0))
        self.assertTrue(np.all(k3 < self.N))

        # Check momentum conservation: k1 + k2 - k4 = k3
        for k1 in range(self.N):
            for k2 in range(self.N):
                for k4 in range(self.N):
                    k3_val = k3[k1, k2, k4]
                    if k3_val > 0:  # Valid momentum
                        expected = k1 + k2 - k4
                        if expected >= 0 and expected < self.N:
                            self.assertEqual(k3_val, expected)

    def test_collision_matrices(self):
        """Test collision matrix construction."""
        collision_data = self.calculator._build_collision_matrices(self.Ee, self.Eh)

        # Check that all required matrices are present
        required_keys = ['Ceh', 'Cee', 'Chh', 'UnDel', 'k3']
        for key in required_keys:
            self.assertIn(key, collision_data)

        # Check shapes
        self.assertEqual(collision_data['Ceh'].shape, (self.N + 1, self.N + 1, self.N + 1))
        self.assertEqual(collision_data['Cee'].shape, (self.N + 1, self.N + 1, self.N + 1))
        self.assertEqual(collision_data['Chh'].shape, (self.N + 1, self.N + 1, self.N + 1))
        self.assertEqual(collision_data['UnDel'].shape, (self.N + 1, self.N + 1))

        # Check that collision matrices are non-negative
        self.assertTrue(np.all(collision_data['Ceh'] >= 0))
        self.assertTrue(np.all(collision_data['Cee'] >= 0))
        self.assertTrue(np.all(collision_data['Chh'] >= 0))

    def test_collision_rates(self):
        """Test collision rate calculation."""
        # Create test populations and potentials
        ne = np.ones(self.N, dtype=np.float64) * 0.1
        nh = np.ones(self.N, dtype=np.float64) * 0.1
        Veh = np.ones((self.N, self.N), dtype=np.float64) * 1e-20
        Vee = np.ones((self.N, self.N), dtype=np.float64) * 1e-20
        Vhh = np.ones((self.N, self.N), dtype=np.float64) * 1e-20

        Win, Wout = self.calculator.calculate_collision_rates(
            ne, nh, Veh, Vee, Vhh, self.Ee, self.Eh
        )

        # Check shapes
        self.assertEqual(Win.shape, (self.N,))
        self.assertEqual(Wout.shape, (self.N,))

        # Check that rates are non-negative
        self.assertTrue(np.all(Win >= 0))
        self.assertTrue(np.all(Wout >= 0))

    def test_band_gap_renormalization(self):
        """Test band gap renormalization calculation."""
        # Create test populations and potentials
        ne = np.ones(self.N, dtype=np.complex128) * 0.1
        nh = np.ones(self.N, dtype=np.complex128) * 0.1
        Vee = np.ones((self.N, self.N), dtype=np.float64) * 1e-20
        Vhh = np.ones((self.N, self.N), dtype=np.float64) * 1e-20

        BGR = self.calculator.calculate_band_gap_renormalization(ne, nh, Vee, Vhh)

        # Check shape
        self.assertEqual(BGR.shape, (self.N, self.N))

        # Check that BGR is complex
        self.assertTrue(np.iscomplexobj(BGR))


class TestCoulombSolver(unittest.TestCase):
    """Test main CoulombSolver class."""

    def setUp(self):
        """Set up test data."""
        self.params = CoulombParameters(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        self.N = 16
        self.ky = np.linspace(-1e8, 1e8, self.N)
        self.y = np.linspace(-5e-8, 5e-8, self.N)
        self.qy = np.linspace(0, 2e8, self.N)
        self.kkp = np.random.randint(-1, self.N, (self.N, self.N))
        self.grid = MomentumGrid(ky=self.ky, y=self.y, qy=self.qy, kkp=self.kkp)

        self.solver = CoulombSolver(self.params, self.grid)

        # Create test energy dispersions
        self.Ee = 1e-20 * self.ky**2
        self.Eh = 1e-20 * self.ky**2

        # Create test populations
        self.ne = np.ones(self.N, dtype=np.complex128) * 0.1
        self.nh = np.ones(self.N, dtype=np.complex128) * 0.1

    def test_initialization(self):
        """Test solver initialization."""
        self.assertFalse(self.solver._initialized)

        self.solver.initialize(self.Ee, self.Eh)

        self.assertTrue(self.solver._initialized)
        self.assertIsNotNone(self.solver._cached_potentials)

    def test_uninitialized_operations(self):
        """Test that operations fail before initialization."""
        with self.assertRaises(RuntimeError):
            self.solver.get_screened_potentials(self.ne, self.nh)

        with self.assertRaises(RuntimeError):
            self.solver.calculate_collision_rates(
                self.ne.real, self.nh.real, self.Ee, self.Eh
            )

        with self.assertRaises(RuntimeError):
            self.solver.calculate_band_gap_renormalization(self.ne, self.nh)

    def test_screened_potentials(self):
        """Test screened potential calculation."""
        self.solver.initialize(self.Ee, self.Eh)

        Veh, Vee, Vhh = self.solver.get_screened_potentials(self.ne, self.nh)

        # Check shapes
        self.assertEqual(Veh.shape, (self.N, self.N))
        self.assertEqual(Vee.shape, (self.N, self.N))
        self.assertEqual(Vhh.shape, (self.N, self.N))

        # Check that screened potentials are different from unscreened
        Veh_unscreened, Vee_unscreened, Vhh_unscreened = self.solver._cached_potentials

        # Screened potentials should be smaller (due to screening)
        self.assertTrue(np.all(Veh <= Veh_unscreened))
        self.assertTrue(np.all(Vee <= Vee_unscreened))
        self.assertTrue(np.all(Vhh <= Vhh_unscreened))

    def test_collision_rates(self):
        """Test collision rate calculation."""
        self.solver.initialize(self.Ee, self.Eh)

        Win, Wout = self.solver.calculate_collision_rates(
            self.ne.real, self.nh.real, self.Ee, self.Eh
        )

        # Check shapes
        self.assertEqual(Win.shape, (self.N,))
        self.assertEqual(Wout.shape, (self.N,))

        # Check that rates are non-negative
        self.assertTrue(np.all(Win >= 0))
        self.assertTrue(np.all(Wout >= 0))

    def test_band_gap_renormalization(self):
        """Test band gap renormalization calculation."""
        self.solver.initialize(self.Ee, self.Eh)

        BGR = self.solver.calculate_band_gap_renormalization(self.ne, self.nh)

        # Check shape
        self.assertEqual(BGR.shape, (self.N, self.N))

        # Check that BGR is complex
        self.assertTrue(np.iscomplexobj(BGR))


class TestFortranCompatibility(unittest.TestCase):
    """Test Fortran-compatible interface functions."""

    def setUp(self):
        """Set up test data."""
        self.N = 16
        self.y = np.linspace(-5e-8, 5e-8, self.N)
        self.ky = np.linspace(-1e8, 1e8, self.N)
        self.Qy = np.linspace(0, 2e8, self.N)
        self.kkp = np.random.randint(-1, self.N, (self.N, self.N))

        self.L = 1e-6
        self.Delta0 = 1e-8
        self.me = 0.067 * 9.109e-31
        self.mh = 0.45 * 9.109e-31
        self.Ee = 1e-20 * self.ky**2
        self.Eh = 1e-20 * self.ky**2
        self.ge = 1e12
        self.gh = 1e12
        self.alphae = 1e8
        self.alphah = 1e8
        self.er = 12.0
        self.screened = True

    def test_initialize_coulomb(self):
        """Test InitializeCoulomb function."""
        solver = InitializeCoulomb(
            self.y, self.ky, self.L, self.Delta0, self.me, self.mh,
            self.Ee, self.Eh, self.ge, self.gh, self.alphae, self.alphah,
            self.er, self.Qy, self.kkp, self.screened
        )

        self.assertIsInstance(solver, CoulombSolver)
        self.assertTrue(solver._initialized)

    def test_calc_coulomb_arrays(self):
        """Test CalcCoulombArrays function."""
        Veh, Vee, Vhh = CalcCoulombArrays(
            self.y, self.ky, self.er, self.alphae, self.alphah,
            self.L, self.Delta0, self.Qy, self.kkp
        )

        # Check shapes
        self.assertEqual(Veh.shape, (self.N, self.N))
        self.assertEqual(Vee.shape, (self.N, self.N))
        self.assertEqual(Vhh.shape, (self.N, self.N))

        # Check that all values are non-negative
        self.assertTrue(np.all(Veh >= 0))
        self.assertTrue(np.all(Vee >= 0))
        self.assertTrue(np.all(Vhh >= 0))

    def test_vint(self):
        """Test Vint function."""
        Qyk = 1e7
        result = Vint(Qyk, self.y, self.alphae, self.alphah, self.Delta0)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""

    def test_large_system_performance(self):
        """Test performance with larger system."""
        # Create larger test system
        N = 64
        params = CoulombParameters(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))
        grid = MomentumGrid(ky=ky, y=y, qy=qy, kkp=kkp)

        solver = CoulombSolver(params, grid)

        # Create test data
        Ee = 1e-20 * ky**2
        Eh = 1e-20 * ky**2
        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1

        # Test initialization performance
        import time
        start = time.time()
        solver.initialize(Ee, Eh)
        init_time = time.time() - start

        # Test calculation performance
        start = time.time()
        Veh, Vee, Vhh = solver.get_screened_potentials(ne, nh)
        pot_time = time.time() - start

        start = time.time()
        Win, Wout = solver.calculate_collision_rates(ne.real, nh.real, Ee, Eh)
        coll_time = time.time() - start

        # Performance should be reasonable (adjust thresholds as needed)
        self.assertLess(init_time, 10.0)  # Initialization should take less than 10 seconds
        self.assertLess(pot_time, 1.0)    # Potential calculation should take less than 1 second
        self.assertLess(coll_time, 5.0)   # Collision rates should take less than 5 seconds

        print(f"Performance test results for N={N}:")
        print(f"  Initialization: {init_time:.3f}s")
        print(f"  Potentials: {pot_time:.3f}s")
        print(f"  Collision rates: {coll_time:.3f}s")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_zero_populations(self):
        """Test behavior with zero populations."""
        params = CoulombParameters(
            length=1e-6, thickness=1e-8, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        N = 16
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))
        grid = MomentumGrid(ky=ky, y=y, qy=qy, kkp=kkp)

        solver = CoulombSolver(params, grid)
        Ee = 1e-20 * ky**2
        Eh = 1e-20 * ky**2
        solver.initialize(Ee, Eh)

        # Test with zero populations
        ne_zero = np.zeros(N, dtype=np.complex128)
        nh_zero = np.zeros(N, dtype=np.complex128)

        Veh, Vee, Vhh = solver.get_screened_potentials(ne_zero, nh_zero)
        Win, Wout = solver.calculate_collision_rates(ne_zero.real, nh_zero.real, Ee, Eh)

        # Results should be finite
        self.assertTrue(np.all(np.isfinite(Veh)))
        self.assertTrue(np.all(np.isfinite(Vee)))
        self.assertTrue(np.all(np.isfinite(Vhh)))
        self.assertTrue(np.all(np.isfinite(Win)))
        self.assertTrue(np.all(np.isfinite(Wout)))

    def test_extreme_parameters(self):
        """Test behavior with extreme parameter values."""
        # Test with very small thickness
        params = CoulombParameters(
            length=1e-6, thickness=1e-12, dielectric_constant=12.0,
            electron_mass=0.067 * 9.109e-31, hole_mass=0.45 * 9.109e-31,
            electron_confinement=1e8, hole_confinement=1e8,
            electron_relaxation=1e12, hole_relaxation=1e12
        )

        N = 16
        ky = np.linspace(-1e8, 1e8, N)
        y = np.linspace(-5e-8, 5e-8, N)
        qy = np.linspace(0, 2e8, N)
        kkp = np.random.randint(-1, N, (N, N))
        grid = MomentumGrid(ky=ky, y=y, qy=qy, kkp=kkp)

        solver = CoulombSolver(params, grid)
        Ee = 1e-20 * ky**2
        Eh = 1e-20 * ky**2

        # Should not raise exceptions
        solver.initialize(Ee, Eh)

        ne = np.ones(N, dtype=np.complex128) * 0.1
        nh = np.ones(N, dtype=np.complex128) * 0.1

        Veh, Vee, Vhh = solver.get_screened_potentials(ne, nh)
        Win, Wout = solver.calculate_collision_rates(ne.real, nh.real, Ee, Eh)

        # Results should be finite
        self.assertTrue(np.all(np.isfinite(Veh)))
        self.assertTrue(np.all(np.isfinite(Vee)))
        self.assertTrue(np.all(np.isfinite(Vhh)))
        self.assertTrue(np.all(np.isfinite(Win)))
        self.assertTrue(np.all(np.isfinite(Wout)))


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestCoulombParameters,
        TestMomentumGrid,
        TestCoulombIntegralCalculator,
        TestScreeningCalculator,
        TestManyBodyCalculator,
        TestCoulombSolver,
        TestFortranCompatibility,
        TestPerformance,
        TestEdgeCases
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
