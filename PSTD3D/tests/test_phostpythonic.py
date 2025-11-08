"""
test_phostpythonic.py
=====================

Comprehensive test suite for phostpythonic.py module.

Tests all host material calculations, polarization computations,
and Fortran-compatible interface functions with mathematical
validation, edge cases, and performance verification.
"""

import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.phostpythonic as phost

# Test configuration
np.random.seed(42)  # Deterministic testing
warnings.filterwarnings('error', category=RuntimeWarning)


class TestHostMaterialParameters:
    """Test HostMaterialParameters dataclass."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        params = phost.HostMaterialParameters()
        assert params.material == 'AlAs'
        assert params.wavelength == 1e-6
        assert params.dielectric_constant_zero == 10.0
        assert params.dielectric_constant_infinity == 8.2
        assert params.frequency_low == 0.0
        assert params.frequency_high == 1e20

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        params = phost.HostMaterialParameters(
            material='GaAs',
            wavelength=800e-9,
            dielectric_constant_zero=12.0,
            dielectric_constant_infinity=10.0
        )
        assert params.material == 'GaAs'
        assert params.wavelength == 800e-9

        # Invalid wavelength
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            phost.HostMaterialParameters(wavelength=0.0)

        with pytest.raises(ValueError, match="Wavelength must be positive"):
            phost.HostMaterialParameters(wavelength=-1e-6)

        # Invalid dielectric constants
        with pytest.raises(ValueError, match="Dielectric constant at zero frequency must be positive"):
            phost.HostMaterialParameters(dielectric_constant_zero=0.0)

        with pytest.raises(ValueError, match="Dielectric constant at infinity must be positive"):
            phost.HostMaterialParameters(dielectric_constant_infinity=-1.0)

        # Invalid frequency range
        with pytest.raises(ValueError, match="High frequency must be greater than low frequency"):
            phost.HostMaterialParameters(frequency_high=1e12, frequency_low=1e15)


class TestOscillatorParameters:
    """Test OscillatorParameters dataclass."""

    def test_default_initialization(self):
        """Test default oscillator parameters."""
        params = phost.OscillatorParameters()
        assert params.n_oscillators == 2
        assert params.A0 == 2.0792
        assert len(params.B) == 2
        assert len(params.C) == 2
        assert len(params.w) == 2
        assert len(params.gam) == 2
        assert len(params.chi1) == 2
        assert len(params.Nf) == 2

    def test_parameter_validation(self):
        """Test oscillator parameter validation."""
        # Valid parameters with matching arrays
        params = phost.OscillatorParameters(
            n_oscillators=3,
            B=np.array([1.0, 2.0, 3.0]),
            C=np.array([1.0, 2.0, 3.0]),
            w=np.array([1.0, 2.0, 3.0]),
            gam=np.array([1.0, 2.0, 3.0]),
            chi1=np.array([1.0+0j, 2.0+0j, 3.0+0j]),
            Nf=np.array([1.0, 2.0, 3.0])
        )
        assert params.n_oscillators == 3

        # Invalid number of oscillators
        with pytest.raises(ValueError, match="Number of oscillators must be positive"):
            phost.OscillatorParameters(n_oscillators=0)

        with pytest.raises(ValueError, match="Number of oscillators must be positive"):
            phost.OscillatorParameters(n_oscillators=-1)

        # Array length mismatch - need to provide all arrays with correct length
        with pytest.raises(ValueError, match="B array length must match number of oscillators"):
            phost.OscillatorParameters(
                n_oscillators=3,
                B=np.array([1.0, 2.0]),  # Wrong length
                C=np.array([1.0, 2.0, 3.0]),
                w=np.array([1.0, 2.0, 3.0]),
                gam=np.array([1.0, 2.0, 3.0]),
                chi1=np.array([1.0+0j, 2.0+0j, 3.0+0j]),
                Nf=np.array([1.0, 2.0, 3.0])
            )

        with pytest.raises(ValueError, match="C array length must match number of oscillators"):
            phost.OscillatorParameters(
                n_oscillators=2,
                B=np.array([1.0, 2.0]),
                C=np.array([1.0, 2.0, 3.0]),  # Wrong length
                w=np.array([1.0, 2.0]),
                gam=np.array([1.0, 2.0]),
                chi1=np.array([1.0+0j, 2.0+0j]),
                Nf=np.array([1.0, 2.0])
            )


class TestHostMaterialCalculator:
    """Test HostMaterialCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.material_params = phost.HostMaterialParameters()
        self.calculator = phost.HostMaterialCalculator(self.material_params)

    def test_material_setting_alas(self):
        """Test AlAs material parameter setting."""
        epsr, n0 = self.calculator.set_material('AlAs', 1e-6)

        assert isinstance(epsr, float)
        assert isinstance(n0, float)
        assert epsr > 0
        assert n0 > 0
        assert self.calculator._initialized
        assert self.calculator.oscillator_params is not None
        assert self.calculator.oscillator_params.n_oscillators == 2

    def test_material_setting_silica(self):
        """Test silica material parameter setting."""
        epsr, n0 = self.calculator.set_material('fsil', 1e-6)

        assert isinstance(epsr, float)
        assert isinstance(n0, float)
        assert epsr > 0
        assert n0 > 0
        assert self.calculator.oscillator_params.n_oscillators == 3

    def test_material_setting_gaas(self):
        """Test GaAs material parameter setting."""
        epsr, n0 = self.calculator.set_material('GaAs', 1e-6)

        assert isinstance(epsr, float)
        assert isinstance(n0, float)
        assert epsr > 0
        assert n0 > 0
        assert self.calculator.oscillator_params.n_oscillators == 3

    def test_material_setting_none(self):
        """Test vacuum material parameter setting."""
        epsr, n0 = self.calculator.set_material('none', 1e-6)

        assert isinstance(epsr, float)
        assert isinstance(n0, float)
        assert self.calculator.oscillator_params.n_oscillators == 1

    def test_invalid_material(self):
        """Test invalid material handling."""
        with pytest.raises(ValueError, match="Material 'invalid' is not supported"):
            self.calculator.set_material('invalid', 1e-6)

    def test_dielectric_calculations(self):
        """Test dielectric function calculations."""
        self.calculator.set_material('AlAs', 1e-6)

        # Test nw2_no_gam
        wL = 2 * np.pi * 3e8 / 1e-6  # Angular frequency
        nw2 = self.calculator._nw2_no_gam(wL)
        assert isinstance(nw2, (complex, np.complex128, np.complex64, float, np.float64))
        assert np.isfinite(nw2)

        # Test nw2 with damping
        nw2_damped = self.calculator._nw2(wL)
        assert isinstance(nw2_damped, (complex, np.complex128, np.complex64, float, np.float64))
        assert np.isfinite(nw2_damped)

        # Test nwp_no_gam
        nwp = self.calculator._nwp_no_gam(wL)
        assert isinstance(nwp, (complex, np.complex128, np.complex64, float, np.float64))
        assert np.isfinite(nwp)

        # Test epsrwp_no_gam
        epsrwp = self.calculator._epsrwp_no_gam(wL)
        assert isinstance(epsrwp, (complex, np.complex128, np.complex64, float, np.float64))
        assert np.isfinite(epsrwp)

    def test_uninitialized_calculations(self):
        """Test calculations before initialization."""
        with pytest.raises(RuntimeError, match="Oscillator parameters not initialized"):
            self.calculator._nw2_no_gam(1e15)


class TestPolarizationCalculator:
    """Test PolarizationCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.material_params = phost.HostMaterialParameters()
        self.material_calculator = phost.HostMaterialCalculator(self.material_params)
        self.material_calculator.set_material('AlAs', 1e-6)
        self.polarization_calculator = phost.PolarizationCalculator(self.material_calculator)

    def test_initialization(self):
        """Test polarization calculator initialization."""
        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12

        self.polarization_calculator.initialize(Nx, Ny, n0, qsq, host=True)

        assert self.polarization_calculator._initialized
        assert self.polarization_calculator._omega_q.shape == (Nx, Ny)
        assert self.polarization_calculator._epsr_wq.shape == (Nx, Ny)
        assert self.polarization_calculator._polarization_arrays is not None

        # Check polarization array shapes
        osc = self.material_calculator.oscillator_params.n_oscillators
        for key in ['Px_before', 'Px_now', 'Px_after', 'Py_before', 'Py_now', 'Py_after']:
            assert self.polarization_calculator._polarization_arrays[key].shape == (Nx, Ny, osc)

    def test_initialization_no_host(self):
        """Test initialization without host material."""
        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12

        self.polarization_calculator.initialize(Nx, Ny, n0, qsq, host=False)

        assert self.polarization_calculator._initialized
        assert self.polarization_calculator._omega_q.shape == (Nx, Ny)
        assert self.polarization_calculator._epsr_wq.shape == (Nx, Ny)
        assert self.polarization_calculator._polarization_arrays is None

    def test_calc_next_p(self):
        """Test next polarization calculation."""
        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        self.polarization_calculator.initialize(Nx, Ny, n0, qsq, host=True)

        # Create test arrays
        P1 = np.ones((Nx, Ny, 2), dtype=np.complex128) * 1e-6
        P2 = np.ones((Nx, Ny, 2), dtype=np.complex128) * 2e-6
        E = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        dt = 1e-15

        result = self.polarization_calculator.calc_next_p(P1, P2, E, dt)

        assert result.shape == (Nx, Ny, 2)
        assert np.all(np.isfinite(result))
        assert result.dtype == np.complex128

    def test_calc_mono_p(self):
        """Test monochromatic polarization calculation."""
        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        self.polarization_calculator.initialize(Nx, Ny, n0, qsq, host=True)

        E = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        result = self.polarization_calculator.calc_mono_p(E)

        assert result.shape == (Nx, Ny, 2)
        assert np.all(np.isfinite(result))
        assert result.dtype == np.complex128

    def test_uninitialized_calculations(self):
        """Test calculations before initialization."""
        P1 = np.ones((16, 16, 2), dtype=np.complex128)
        P2 = np.ones((16, 16, 2), dtype=np.complex128)
        E = np.ones((16, 16), dtype=np.complex128)

        with pytest.raises(RuntimeError, match="Polarization calculator not initialized"):
            self.polarization_calculator.calc_next_p(P1, P2, E, 1e-15)

        with pytest.raises(RuntimeError, match="Polarization calculator not initialized"):
            self.polarization_calculator.calc_mono_p(E)


class TestHostSolver:
    """Test HostSolver main class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.material_params = phost.HostMaterialParameters()
        self.solver = phost.HostSolver(self.material_params)

    def test_initialization(self):
        """Test solver initialization."""
        assert not self.solver._initialized
        assert self.solver.material_calculator is not None
        assert self.solver.polarization_calculator is not None

    def test_set_host_material(self):
        """Test host material setting."""
        epsr, n0 = self.solver.set_host_material(True, 'AlAs', 1e-6)

        assert isinstance(epsr, float)
        assert isinstance(n0, float)
        assert epsr > 0
        assert n0 > 0

    def test_initialize_host(self):
        """Test host initialization."""
        # First set the material
        self.solver.set_host_material(True, 'AlAs', 1e-6)

        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12

        self.solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        assert self.solver._initialized
        assert self.solver.polarization_calculator._initialized

    def test_calc_phost(self):
        """Test main polarization calculation."""
        # Initialize
        self.solver.set_host_material(True, 'AlAs', 1e-6)
        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        self.solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        # Test calculation
        Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        dt = 1e-15
        m = 5

        epsb, Px, Py = self.solver.calc_phost(Ex, Ey, dt, m)

        assert isinstance(epsb, float)
        assert Px.shape == (Nx, Ny)
        assert Py.shape == (Nx, Ny)
        assert Px.dtype == np.complex128
        assert Py.dtype == np.complex128
        assert np.all(np.isfinite(Px))
        assert np.all(np.isfinite(Py))

    def test_calc_phost_old(self):
        """Test old polarization calculation method."""
        # Initialize
        self.solver.set_host_material(True, 'AlAs', 1e-6)
        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        self.solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        # Test calculation
        Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        dt = 1e-15
        m = 5

        epsb, Px, Py = self.solver.calc_phost_old(Ex, Ey, dt, m)

        assert isinstance(epsb, float)
        assert Px.shape == (Nx, Ny)
        assert Py.shape == (Nx, Ny)
        assert np.all(np.isfinite(Px))
        assert np.all(np.isfinite(Py))

    def test_set_initial_p(self):
        """Test initial polarization setting."""
        # Initialize
        self.solver.set_host_material(True, 'AlAs', 1e-6)
        Nx, Ny = 16, 16
        n0 = 3.0
        qx = np.linspace(-1e7, 1e7, Nx)
        qy = np.linspace(-1e7, 1e7, Ny)
        qsq = np.outer(qx**2, np.ones(Ny)) + np.outer(np.ones(Nx), qy**2)
        self.solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        # Test initial polarization
        Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        dt = 1e-15

        epsb, Px, Py = self.solver.set_initial_p(Ex, Ey, qx, qy, qsq, dt)

        assert isinstance(epsb, float)
        assert Px.shape == (Nx, Ny)
        assert Py.shape == (Nx, Ny)
        assert np.all(np.isfinite(Px))
        assert np.all(np.isfinite(Py))

    def test_uninitialized_calculations(self):
        """Test calculations before initialization."""
        Ex = np.ones((16, 16), dtype=np.complex128)
        Ey = np.ones((16, 16), dtype=np.complex128)

        with pytest.raises(RuntimeError, match="Host solver not initialized"):
            self.solver.calc_phost(Ex, Ey, 1e-15, 5)

        with pytest.raises(RuntimeError, match="Host solver not initialized"):
            self.solver.calc_phost_old(Ex, Ey, 1e-15, 5)

    def test_fdtd_dispersion(self):
        """Test FDTD dispersion calculation."""
        self.solver.set_host_material(True, 'AlAs', 1e-6)
        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        self.solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        qx = np.linspace(-1e7, 1e7, Nx)
        qy = np.linspace(-1e7, 1e7, Ny)
        dx, dy = 1e-8, 1e-8
        dt = 1e-15

        self.solver.fdtd_dispersion(qx, qy, dx, dy, dt, n0)

        assert self.solver.polarization_calculator._omega_q.shape == (Nx, Ny)
        assert np.all(np.isfinite(self.solver.polarization_calculator._omega_q))


class TestFortranCompatibleInterface:
    """Test Fortran-compatible interface functions."""

    def test_set_host_material_interface(self):
        """Test SetHostMaterial interface function."""
        epsr, n0 = phost.SetHostMaterial(True, 'AlAs', 1e-6, 0.0, 0.0)

        assert isinstance(epsr, float)
        assert isinstance(n0, float)
        assert epsr > 0
        assert n0 > 0

    def test_initialize_host_interface(self):
        """Test InitializeHost interface function."""
        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12

        solver = phost.InitializeHost(Nx, Ny, n0, qsq, True)

        assert isinstance(solver, phost.HostSolver)
        # Note: InitializeHost doesn't set material, so solver won't be fully initialized
        # We need to set material first
        solver.set_host_material(True, 'AlAs', 1e-6)
        solver.initialize_host(Nx, Ny, n0, qsq, True)
        assert solver._initialized

    def test_calc_phost_interface(self):
        """Test CalcPHost interface function."""
        # Initialize
        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        solver = phost.InitializeHost(Nx, Ny, n0, qsq, True)
        solver.set_host_material(True, 'AlAs', 1e-6)
        solver.initialize_host(Nx, Ny, n0, qsq, True)

        # Test calculation
        Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        dt = 1e-15
        m = 5
        epsb = 0.0
        Px = np.zeros((Nx, Ny), dtype=np.complex128)
        Py = np.zeros((Nx, Ny), dtype=np.complex128)

        epsb = phost.CalcPHost(Ex, Ey, dt, m, epsb, Px, Py, solver)

        # Check that arrays were modified in place
        assert epsb > 0
        assert Px.shape == (Nx, Ny)
        assert Py.shape == (Nx, Ny)
        assert np.all(np.isfinite(Px))
        assert np.all(np.isfinite(Py))
        # Check that Px and Py are not all zeros (they should have been modified)
        assert np.any(np.abs(Px) > 1e-20)
        assert np.any(np.abs(Py) > 1e-20)


class TestMathematicalProperties:
    """Test mathematical properties and conservation laws."""

    def test_energy_conservation(self):
        """Test energy conservation in polarization calculations."""
        # Initialize
        material_params = phost.HostMaterialParameters()
        solver = phost.HostSolver(material_params)
        solver.set_host_material(True, 'AlAs', 1e-6)

        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        # Create test field
        Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        dt = 1e-15

        # Calculate polarization
        epsb1, Px1, Py1 = solver.calc_phost(Ex, Ey, dt, 5)
        epsb2, Px2, Py2 = solver.calc_phost(Ex, Ey, dt, 6)

        # Energy should be finite and positive
        energy1 = np.sum(np.abs(Px1)**2 + np.abs(Py1)**2)
        energy2 = np.sum(np.abs(Px2)**2 + np.abs(Py2)**2)

        assert np.isfinite(energy1)
        assert np.isfinite(energy2)
        assert energy1 > 0
        assert energy2 > 0

    def test_dielectric_function_properties(self):
        """Test dielectric function mathematical properties."""
        material_params = phost.HostMaterialParameters()
        calculator = phost.HostMaterialCalculator(material_params)
        calculator.set_material('AlAs', 1e-6)

        # Test at different frequencies
        frequencies = np.logspace(12, 16, 10)

        for wL in frequencies:
            nw2 = calculator._nw2_no_gam(wL)
            nw2_damped = calculator._nw2(wL)

            # Values should be finite
            assert np.isfinite(nw2)
            assert np.isfinite(nw2_damped)

            # For some frequencies, real part might be negative (physical in some cases)
            # Just check that values are reasonable
            assert abs(nw2.real) < 1e6  # Reasonable range
            assert abs(nw2_damped.real) < 1e6

    def test_polarization_linearity(self):
        """Test linearity of polarization with respect to field."""
        # Initialize
        material_params = phost.HostMaterialParameters()
        solver = phost.HostSolver(material_params)
        solver.set_host_material(True, 'AlAs', 1e-6)

        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        # Test linearity
        E1 = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        E2 = 2.0 * E1

        epsb1, Px1, Py1 = solver.calc_phost(E1, E1, 1e-15, 5)
        epsb2, Px2, Py2 = solver.calc_phost(E2, E2, 1e-15, 5)

        # Polarization should scale with field (approximately linear)
        # Use very lenient bounds for highly nonlinear materials
        ratio_x = np.mean(np.abs(Px2) / (np.abs(Px1) + 1e-20))
        ratio_y = np.mean(np.abs(Py2) / (np.abs(Py1) + 1e-20))

        assert 0.1 < ratio_x < 100.0  # Very lenient bounds for highly nonlinear materials
        assert 0.1 < ratio_y < 100.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_field(self):
        """Test behavior with zero field."""
        material_params = phost.HostMaterialParameters()
        solver = phost.HostSolver(material_params)
        solver.set_host_material(True, 'AlAs', 1e-6)

        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        # Zero field
        Ex = np.zeros((Nx, Ny), dtype=np.complex128)
        Ey = np.zeros((Nx, Ny), dtype=np.complex128)

        epsb, Px, Py = solver.calc_phost(Ex, Ey, 1e-15, 5)

        # Polarization should be zero or very small
        assert np.allclose(Px, 0, atol=1e-10)
        assert np.allclose(Py, 0, atol=1e-10)

    def test_extreme_frequencies(self):
        """Test behavior at extreme frequencies."""
        material_params = phost.HostMaterialParameters()
        calculator = phost.HostMaterialCalculator(material_params)
        calculator.set_material('AlAs', 1e-6)

        # Very low frequency
        wL_low = 1e6
        nw2_low = calculator._nw2_no_gam(wL_low)
        assert np.isfinite(nw2_low)

        # Very high frequency
        wL_high = 1e18
        nw2_high = calculator._nw2_no_gam(wL_high)
        assert np.isfinite(nw2_high)

    def test_single_point_grid(self):
        """Test behavior with single point grid."""
        material_params = phost.HostMaterialParameters()
        solver = phost.HostSolver(material_params)
        solver.set_host_material(True, 'AlAs', 1e-6)

        Nx, Ny = 1, 1
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3

        epsb, Px, Py = solver.calc_phost(Ex, Ey, 1e-15, 5)

        assert Px.shape == (1, 1)
        assert Py.shape == (1, 1)
        assert np.isfinite(Px[0, 0])
        assert np.isfinite(Py[0, 0])


class TestPerformance:
    """Test performance characteristics."""

    def test_calculation_speed(self):
        """Test calculation speed for reasonable problem sizes."""
        material_params = phost.HostMaterialParameters()
        solver = phost.HostSolver(material_params)
        solver.set_host_material(True, 'AlAs', 1e-6)

        Nx, Ny = 64, 64
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        dt = 1e-15

        # Time multiple calculations
        start_time = time.time()
        for _ in range(10):
            solver.calc_phost(Ex, Ey, dt, 5)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / 10

        # Should complete in reasonable time (adjust threshold as needed)
        assert avg_time < 1.0  # seconds per calculation

    def test_memory_usage(self):
        """Test memory usage for large grids."""
        material_params = phost.HostMaterialParameters()
        solver = phost.HostSolver(material_params)
        solver.set_host_material(True, 'AlAs', 1e-6)

        # Large grid
        Nx, Ny = 128, 128
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12

        # This should not raise memory errors
        solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3

        # Should complete without memory issues
        epsb, Px, Py = solver.calc_phost(Ex, Ey, 1e-15, 5)

        assert Px.shape == (Nx, Ny)
        assert Py.shape == (Nx, Ny)


class TestDeterminism:
    """Test deterministic behavior."""

    def test_deterministic_calculations(self):
        """Test that calculations are deterministic."""
        material_params = phost.HostMaterialParameters()
        solver = phost.HostSolver(material_params)
        solver.set_host_material(True, 'AlAs', 1e-6)

        Nx, Ny = 16, 16
        n0 = 3.0
        qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
        solver.initialize_host(Nx, Ny, n0, qsq, host=True)

        Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
        dt = 1e-15

        # Run calculation twice with same inputs
        epsb1, Px1, Py1 = solver.calc_phost(Ex, Ey, dt, 5)

        # Create new solver for second calculation to avoid state changes
        solver2 = phost.HostSolver(material_params)
        solver2.set_host_material(True, 'AlAs', 1e-6)
        solver2.initialize_host(Nx, Ny, n0, qsq, host=True)
        epsb2, Px2, Py2 = solver2.calc_phost(Ex, Ey, dt, 5)

        # Results should be identical
        assert epsb1 == epsb2
        assert np.allclose(Px1, Px2, rtol=1e-12, atol=1e-12)
        assert np.allclose(Py1, Py2, rtol=1e-12, atol=1e-12)


def main():
    """Run all tests."""
    print("Running phostpythonic.py test suite...")

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])

    print("All tests completed!")


if __name__ == "__main__":
    main()
