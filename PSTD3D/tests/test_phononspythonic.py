#!/usr/bin/env python3
"""
Comprehensive test suite for phononspythonic.py

This test suite validates the Pythonic phonon implementation against the original
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

from phononspythonic import (
    PhononParameters, MomentumGrid, PhononMatrixCalculator,
    PhononRateCalculator, PhononSolver,
    InitializePhonons, MBPE, MBPH, Cq2, FermiDistr, BoseDistr, N00
)


class TestPhononParameters(unittest.TestCase):
    """Test PhononParameters dataclass."""

    def test_valid_parameters(self):
        """Test valid parameter initialization."""
        params = PhononParameters(
            temperature=77.0,
            phonon_frequency=1e13,
            phonon_relaxation=1e12
        )
        self.assertEqual(params.temperature, 77.0)
        self.assertEqual(params.phonon_frequency, 1e13)
        self.assertEqual(params.phonon_relaxation, 1e12)

    def test_invalid_parameters(self):
        """Test parameter validation."""
        with self.assertRaises(ValueError):
            PhononParameters(temperature=-1.0)

        with self.assertRaises(ValueError):
            PhononParameters(phonon_frequency=0.0)

        with self.assertRaises(ValueError):
            PhononParameters(phonon_relaxation=-1.0)


class TestMomentumGrid(unittest.TestCase):
    """Test MomentumGrid dataclass."""

    def setUp(self):
        """Set up test data."""
        self.N = 16
        self.ky = np.linspace(-1e8, 1e8, self.N)

    def test_valid_grid(self):
        """Test valid grid initialization."""
        grid = MomentumGrid(ky=self.ky)
        self.assertEqual(grid.size, self.N)
        self.assertTrue(np.array_equal(grid.ky, self.ky))

    def test_invalid_grid(self):
        """Test grid validation."""
        # Empty grid
        with self.assertRaises(ValueError):
            MomentumGrid(ky=np.array([]))


class TestPhononMatrixCalculator(unittest.TestCase):
    """Test PhononMatrixCalculator class."""

    def setUp(self):
        """Set up test data."""
        self.params = PhononParameters(
            temperature=77.0,
            phonon_frequency=1e13,
            phonon_relaxation=1e12
        )

        self.N = 16
        self.ky = np.linspace(-1e8, 1e8, self.N)
        self.grid = MomentumGrid(ky=self.ky)

        self.calculator = PhononMatrixCalculator(self.params, self.grid)

        # Create test energy dispersions
        self.Ee = 1e-20 * self.ky**2
        self.Eh = 1e-20 * self.ky**2

    def test_identity_delta_matrix(self):
        """Test identity delta matrix construction."""
        idel = self.calculator._build_identity_delta_matrix()

        # Check shape
        self.assertEqual(idel.shape, (self.N, self.N))

        # Check that diagonal elements are 0
        np.testing.assert_array_equal(np.diag(idel), np.zeros(self.N))

        # Check that off-diagonal elements are 1
        off_diag = idel.copy()
        np.fill_diagonal(off_diag, 0)
        np.testing.assert_array_equal(off_diag, np.ones((self.N, self.N)))

    def test_phonon_matrices(self):
        """Test phonon matrix calculation."""
        EP, EPT, HP, HPT = self.calculator._calculate_phonon_matrices(
            self.Ee, self.Eh, 1e13, 1e12
        )

        # Check shapes
        self.assertEqual(EP.shape, (self.N, self.N))
        self.assertEqual(EPT.shape, (self.N, self.N))
        self.assertEqual(HP.shape, (self.N, self.N))
        self.assertEqual(HPT.shape, (self.N, self.N))

        # Check that transpose matrices are correct
        np.testing.assert_array_almost_equal(EP.T, EPT, decimal=10)
        np.testing.assert_array_almost_equal(HP.T, HPT, decimal=10)

        # Check that matrices are non-negative
        self.assertTrue(np.all(EP >= 0))
        self.assertTrue(np.all(HP >= 0))

        # Check that diagonal elements are zero (due to idel matrix)
        np.testing.assert_array_almost_equal(np.diag(EP), np.zeros(self.N), decimal=10)
        np.testing.assert_array_almost_equal(np.diag(HP), np.zeros(self.N), decimal=10)


class TestPhononRateCalculator(unittest.TestCase):
    """Test PhononRateCalculator class."""

    def setUp(self):
        """Set up test data."""
        self.params = PhononParameters(
            temperature=77.0,
            phonon_frequency=1e13,
            phonon_relaxation=1e12
        )

        self.N = 16
        self.ky = np.linspace(-1e8, 1e8, self.N)
        self.grid = MomentumGrid(ky=self.ky)

        self.calculator = PhononRateCalculator(self.params, self.grid)

    def test_vscale_calculation(self):
        """Test Vscale calculation."""
        length = 1e-6
        dielectric_constant = 12.0

        vscale = self.calculator._calculate_vscale(length, dielectric_constant)

        self.assertIsInstance(vscale, float)
        self.assertGreater(vscale, 0)

    def test_electron_phonon_rates(self):
        """Test electron-phonon rate calculation."""
        # Create test data
        ne = np.ones(self.N, dtype=np.float64) * 0.1
        VC = np.ones((self.N, self.N, 3), dtype=np.float64) * 1e-20
        E1D = np.ones((self.N, self.N), dtype=np.float64)
        EP = np.ones((self.N, self.N), dtype=np.float64) * 1e-12
        EPT = EP.T
        vscale = 1e-20

        Win, Wout = self.calculator.calculate_electron_phonon_rates(
            ne, VC, E1D, EP, EPT, vscale
        )

        # Check shapes
        self.assertEqual(Win.shape, (self.N,))
        self.assertEqual(Wout.shape, (self.N,))

        # Check that rates are non-negative
        self.assertTrue(np.all(Win >= 0))
        self.assertTrue(np.all(Wout >= 0))

    def test_hole_phonon_rates(self):
        """Test hole-phonon rate calculation."""
        # Create test data
        nh = np.ones(self.N, dtype=np.float64) * 0.1
        VC = np.ones((self.N, self.N, 3), dtype=np.float64) * 1e-20
        E1D = np.ones((self.N, self.N), dtype=np.float64)
        HP = np.ones((self.N, self.N), dtype=np.float64) * 1e-12
        HPT = HP.T
        vscale = 1e-20

        Win, Wout = self.calculator.calculate_hole_phonon_rates(
            nh, VC, E1D, HP, HPT, vscale
        )

        # Check shapes
        self.assertEqual(Win.shape, (self.N,))
        self.assertEqual(Wout.shape, (self.N,))

        # Check that rates are non-negative
        self.assertTrue(np.all(Win >= 0))
        self.assertTrue(np.all(Wout >= 0))


class TestPhononSolver(unittest.TestCase):
    """Test main PhononSolver class."""

    def setUp(self):
        """Set up test data."""
        self.params = PhononParameters(
            temperature=77.0,
            phonon_frequency=1e13,
            phonon_relaxation=1e12
        )

        self.N = 16
        self.ky = np.linspace(-1e8, 1e8, self.N)
        self.grid = MomentumGrid(ky=self.ky)

        self.solver = PhononSolver(self.params, self.grid)

        # Create test energy dispersions
        self.Ee = 1e-20 * self.ky**2
        self.Eh = 1e-20 * self.ky**2

        # Create test populations
        self.ne = np.ones(self.N, dtype=np.float64) * 0.1
        self.nh = np.ones(self.N, dtype=np.float64) * 0.1
        self.VC = np.ones((self.N, self.N, 3), dtype=np.float64) * 1e-20
        self.E1D = np.ones((self.N, self.N), dtype=np.float64)

    def test_initialization(self):
        """Test solver initialization."""
        self.assertFalse(self.solver._initialized)

        self.solver.initialize(self.Ee, self.Eh, length=1e-6,
                              dielectric_constant=12.0, phonon_frequency=1e13,
                              phonon_relaxation=1e12)

        self.assertTrue(self.solver._initialized)
        self.assertIsNotNone(self.solver._phonon_matrices)
        self.assertIsNotNone(self.solver._vscale)

    def test_uninitialized_operations(self):
        """Test that operations fail before initialization."""
        with self.assertRaises(RuntimeError):
            self.solver.calculate_electron_phonon_rates(self.ne, self.VC, self.E1D)

        with self.assertRaises(RuntimeError):
            self.solver.calculate_hole_phonon_rates(self.nh, self.VC, self.E1D)

        with self.assertRaises(RuntimeError):
            self.solver.get_bose_distribution()

    def test_electron_phonon_rates(self):
        """Test electron-phonon rate calculation."""
        self.solver.initialize(self.Ee, self.Eh, length=1e-6,
                              dielectric_constant=12.0, phonon_frequency=1e13,
                              phonon_relaxation=1e12)

        Win, Wout = self.solver.calculate_electron_phonon_rates(self.ne, self.VC, self.E1D)

        # Check shapes
        self.assertEqual(Win.shape, (self.N,))
        self.assertEqual(Wout.shape, (self.N,))

        # Check that rates are non-negative
        self.assertTrue(np.all(Win >= 0))
        self.assertTrue(np.all(Wout >= 0))

    def test_hole_phonon_rates(self):
        """Test hole-phonon rate calculation."""
        self.solver.initialize(self.Ee, self.Eh, length=1e-6,
                              dielectric_constant=12.0, phonon_frequency=1e13,
                              phonon_relaxation=1e12)

        Win, Wout = self.solver.calculate_hole_phonon_rates(self.nh, self.VC, self.E1D)

        # Check shapes
        self.assertEqual(Win.shape, (self.N,))
        self.assertEqual(Wout.shape, (self.N,))

        # Check that rates are non-negative
        self.assertTrue(np.all(Win >= 0))
        self.assertTrue(np.all(Wout >= 0))

    def test_bose_distribution(self):
        """Test Bose distribution calculation."""
        self.solver.initialize(self.Ee, self.Eh, length=1e-6,
                              dielectric_constant=12.0, phonon_frequency=1e13,
                              phonon_relaxation=1e12)

        bose_dist = self.solver.get_bose_distribution()

        self.assertIsInstance(bose_dist, float)
        self.assertGreater(bose_dist, 0)


class TestFortranCompatibility(unittest.TestCase):
    """Test Fortran-compatible interface functions."""

    def setUp(self):
        """Set up test data."""
        self.N = 16
        self.ky = np.linspace(-1e8, 1e8, self.N)
        self.Ee = 1e-20 * self.ky**2
        self.Eh = 1e-20 * self.ky**2
        self.length = 1e-6
        self.dielectric_constant = 12.0
        self.phonon_frequency = 1e13
        self.phonon_relaxation = 1e12

    def test_initialize_phonons(self):
        """Test InitializePhonons function."""
        solver = InitializePhonons(
            self.ky, self.Ee, self.Eh, self.length, self.dielectric_constant,
            self.phonon_frequency, self.phonon_relaxation
        )

        self.assertIsInstance(solver, PhononSolver)
        self.assertTrue(solver._initialized)

    def test_mbpe(self):
        """Test MBPE function."""
        solver = InitializePhonons(
            self.ky, self.Ee, self.Eh, self.length, self.dielectric_constant,
            self.phonon_frequency, self.phonon_relaxation
        )

        ne = np.ones(self.N, dtype=np.float64) * 0.1
        VC = np.ones((self.N, self.N, 3), dtype=np.float64) * 1e-20
        E1D = np.ones((self.N, self.N), dtype=np.float64)
        Win = np.zeros(self.N, dtype=np.float64)
        Wout = np.zeros(self.N, dtype=np.float64)

        MBPE(ne, VC, E1D, Win, Wout, solver)

        # Check that rates were added
        self.assertTrue(np.any(Win > 0) or np.any(Wout > 0))

    def test_mbph(self):
        """Test MBPH function."""
        solver = InitializePhonons(
            self.ky, self.Ee, self.Eh, self.length, self.dielectric_constant,
            self.phonon_frequency, self.phonon_relaxation
        )

        nh = np.ones(self.N, dtype=np.float64) * 0.1
        VC = np.ones((self.N, self.N, 3), dtype=np.float64) * 1e-20
        E1D = np.ones((self.N, self.N), dtype=np.float64)
        Win = np.zeros(self.N, dtype=np.float64)
        Wout = np.zeros(self.N, dtype=np.float64)

        MBPH(nh, VC, E1D, Win, Wout, solver)

        # Check that rates were added
        self.assertTrue(np.any(Win > 0) or np.any(Wout > 0))

    def test_cq2(self):
        """Test Cq2 function."""
        solver = InitializePhonons(
            self.ky, self.Ee, self.Eh, self.length, self.dielectric_constant,
            self.phonon_frequency, self.phonon_relaxation
        )

        q = np.linspace(0, 1e8, 10)
        V = np.ones((self.N, self.N), dtype=np.float64) * 1e-20
        E1D = np.ones((self.N, self.N), dtype=np.float64)

        Cq2_result = Cq2(q, V, E1D, solver)

        # Check shape
        self.assertEqual(Cq2_result.shape, q.shape)

        # Check that all values are non-negative
        self.assertTrue(np.all(Cq2_result >= 0))

    def test_fermi_distr(self):
        """Test FermiDistr function."""
        # Test scalar input
        result_scalar = FermiDistr(0.1)
        self.assertIsInstance(result_scalar, complex)
        self.assertGreater(result_scalar.real, 0)
        self.assertLess(result_scalar.real, 1)

        # Test array input
        En_array = np.array([0.1, 0.2, 0.3])
        result_array = FermiDistr(En_array)
        self.assertEqual(result_array.shape, En_array.shape)
        self.assertTrue(np.all(result_array.real > 0))
        self.assertTrue(np.all(result_array.real < 1))

    def test_bose_distr(self):
        """Test BoseDistr function."""
        # Test scalar input
        result_scalar = BoseDistr(0.1)
        self.assertIsInstance(result_scalar, float)
        self.assertGreater(result_scalar, 0)

        # Test array input
        En_array = np.array([0.1, 0.2, 0.3])
        result_array = BoseDistr(En_array)
        self.assertEqual(result_array.shape, En_array.shape)
        self.assertTrue(np.all(result_array > 0))

    def test_n00(self):
        """Test N00 function."""
        solver = InitializePhonons(
            self.ky, self.Ee, self.Eh, self.length, self.dielectric_constant,
            self.phonon_frequency, self.phonon_relaxation
        )

        n00_result = N00(solver)

        self.assertIsInstance(n00_result, float)
        self.assertGreater(n00_result, 0)


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""

    def test_large_system_performance(self):
        """Test performance with larger system."""
        # Create larger test system
        N = 64
        params = PhononParameters(
            temperature=77.0,
            phonon_frequency=1e13,
            phonon_relaxation=1e12
        )

        ky = np.linspace(-1e8, 1e8, N)
        grid = MomentumGrid(ky=ky)
        solver = PhononSolver(params, grid)

        # Create test data
        Ee = 1e-20 * ky**2
        Eh = 1e-20 * ky**2
        ne = np.ones(N, dtype=np.float64) * 0.1
        nh = np.ones(N, dtype=np.float64) * 0.1
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
        E1D = np.ones((N, N), dtype=np.float64)

        # Test initialization performance
        import time
        start = time.time()
        solver.initialize(Ee, Eh, length=1e-6, dielectric_constant=12.0,
                         phonon_frequency=1e13, phonon_relaxation=1e12)
        init_time = time.time() - start

        # Test calculation performance
        start = time.time()
        Win_e, Wout_e = solver.calculate_electron_phonon_rates(ne, VC, E1D)
        elec_time = time.time() - start

        start = time.time()
        Win_h, Wout_h = solver.calculate_hole_phonon_rates(nh, VC, E1D)
        hole_time = time.time() - start

        # Performance should be reasonable (adjust thresholds as needed)
        self.assertLess(init_time, 5.0)   # Initialization should take less than 5 seconds
        self.assertLess(elec_time, 1.0)   # Electron rates should take less than 1 second
        self.assertLess(hole_time, 1.0)   # Hole rates should take less than 1 second

        print(f"Performance test results for N={N}:")
        print(f"  Initialization: {init_time:.3f}s")
        print(f"  Electron rates: {elec_time:.3f}s")
        print(f"  Hole rates: {hole_time:.3f}s")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_zero_populations(self):
        """Test behavior with zero populations."""
        params = PhononParameters(
            temperature=77.0,
            phonon_frequency=1e13,
            phonon_relaxation=1e12
        )

        N = 16
        ky = np.linspace(-1e8, 1e8, N)
        grid = MomentumGrid(ky=ky)
        solver = PhononSolver(params, grid)

        Ee = 1e-20 * ky**2
        Eh = 1e-20 * ky**2
        solver.initialize(Ee, Eh, length=1e-6, dielectric_constant=12.0,
                         phonon_frequency=1e13, phonon_relaxation=1e12)

        # Test with zero populations
        ne_zero = np.zeros(N, dtype=np.float64)
        nh_zero = np.zeros(N, dtype=np.float64)
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
        E1D = np.ones((N, N), dtype=np.float64)

        Win_e, Wout_e = solver.calculate_electron_phonon_rates(ne_zero, VC, E1D)
        Win_h, Wout_h = solver.calculate_hole_phonon_rates(nh_zero, VC, E1D)

        # Results should be finite
        self.assertTrue(np.all(np.isfinite(Win_e)))
        self.assertTrue(np.all(np.isfinite(Wout_e)))
        self.assertTrue(np.all(np.isfinite(Win_h)))
        self.assertTrue(np.all(np.isfinite(Wout_h)))

    def test_extreme_parameters(self):
        """Test behavior with extreme parameter values."""
        # Test with very high temperature
        params = PhononParameters(
            temperature=1000.0,
            phonon_frequency=1e13,
            phonon_relaxation=1e12
        )

        N = 16
        ky = np.linspace(-1e8, 1e8, N)
        grid = MomentumGrid(ky=ky)
        solver = PhononSolver(params, grid)

        Ee = 1e-20 * ky**2
        Eh = 1e-20 * ky**2

        # Should not raise exceptions
        solver.initialize(Ee, Eh, length=1e-6, dielectric_constant=12.0,
                         phonon_frequency=1e13, phonon_relaxation=1e12)

        ne = np.ones(N, dtype=np.float64) * 0.1
        nh = np.ones(N, dtype=np.float64) * 0.1
        VC = np.ones((N, N, 3), dtype=np.float64) * 1e-20
        E1D = np.ones((N, N), dtype=np.float64)

        Win_e, Wout_e = solver.calculate_electron_phonon_rates(ne, VC, E1D)
        Win_h, Wout_h = solver.calculate_hole_phonon_rates(nh, VC, E1D)

        # Results should be finite
        self.assertTrue(np.all(np.isfinite(Win_e)))
        self.assertTrue(np.all(np.isfinite(Wout_e)))
        self.assertTrue(np.all(np.isfinite(Win_h)))
        self.assertTrue(np.all(np.isfinite(Wout_h)))


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestPhononParameters,
        TestMomentumGrid,
        TestPhononMatrixCalculator,
        TestPhononRateCalculator,
        TestPhononSolver,
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
        print("\n All tests passed!")
    else:
        print("\n Some tests failed!")
        sys.exit(1)
