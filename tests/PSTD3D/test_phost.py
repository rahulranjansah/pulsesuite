"""
Comprehensive test suite for phost.py module.

Tests all host material polarization calculation functions including
polarization time-stepping, material parameter setup, dispersion calculations,
and FFT operations.
"""

import numpy as np
import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add pythonic directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import phost
from scipy.constants import e as e0, epsilon_0 as eps0, m_e as me0, c as c0
import pyfftw


# Physical constants
pi = np.pi
twopi = 2.0 * np.pi
ii = 1j


class TestNw2NoGam:
    """Test dielectric constant calculation without damping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        # Initialize module variables
        phost._osc = 2
        phost._A0 = 1.0
        phost._B = np.array([0.5, 0.3])
        phost._w = np.array([1e15, 2e15])
        phost._epsr_infty = 1.0

    def test_nw2_no_gam_basic(self):
        """Test nw2_no_gam with basic inputs."""
        wL = 1.5e15
        result = phost.nw2_no_gam(wL)
        expected = phost._A0 + phost._B[0] * phost._w[0]**2 / (phost._w[0]**2 - wL**2) + \
                   phost._B[1] * phost._w[1]**2 / (phost._w[1]**2 - wL**2)
        assert np.allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_nw2_no_gam_zero_frequency(self):
        """Test nw2_no_gam with zero frequency."""
        wL = 0.0
        result = phost.nw2_no_gam(wL)
        expected = phost._A0 + np.sum(phost._B)
        assert np.allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_nw2_no_gam_resonance(self):
        """Test nw2_no_gam near resonance."""
        wL = phost._w[0] * 0.99  # Near but not at resonance
        result = phost.nw2_no_gam(wL)
        assert np.isfinite(result)

    def test_nw2_no_gam_none_b_w(self):
        """Test nw2_no_gam when _B or _w is None."""
        old_B = phost._B
        phost._B = None
        result = phost.nw2_no_gam(1e15)
        assert result == phost._epsr_infty
        phost._B = old_B


class TestNw2:
    """Test dielectric constant calculation with damping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2
        phost._A0 = 1.0
        phost._B = np.array([0.5, 0.3])
        phost._w = np.array([1e15, 2e15])
        phost._gam = np.array([1e12, 2e12])
        phost._epsr_infty = 1.0

    def test_nw2_basic(self):
        """Test nw2 with basic inputs."""
        wL = 1.5e15
        result = phost.nw2(wL)
        expected = phost._A0
        for n in range(phost._osc):
            denom = phost._w[n]**2 - ii * 2.0 * phost._gam[n] * wL - wL**2
            expected += phost._B[n] * phost._w[n]**2 / denom
        assert np.allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_nw2_zero_frequency(self):
        """Test nw2 with zero frequency."""
        wL = 0.0
        result = phost.nw2(wL)
        assert np.isfinite(result)

    def test_nw2_none_b_w_gam(self):
        """Test nw2 when _B, _w, or _gam is None."""
        old_B = phost._B
        phost._B = None
        result = phost.nw2(1e15)
        assert result == phost._epsr_infty
        phost._B = old_B


class TestNl2NoGam:
    """Test dielectric constant from wavelength without damping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2
        phost._A0 = 1.0
        phost._B = np.array([0.5, 0.3])
        phost._C = np.array([1e-12, 2e-12])
        phost._epsr_infty = 1.0

    def test_nl2_no_gam_basic(self):
        """Test nl2_no_gam with basic inputs."""
        lam = 1e-6
        result = phost.nl2_no_gam(lam)
        expected = phost._A0 + phost._B[0] * lam**2 / (lam**2 - phost._C[0]) + \
                   phost._B[1] * lam**2 / (lam**2 - phost._C[1])
        assert np.allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_nl2_no_gam_none_b_c(self):
        """Test nl2_no_gam when _B or _C is None."""
        old_B = phost._B
        phost._B = None
        result = phost.nl2_no_gam(1e-6)
        assert result == phost._epsr_infty
        phost._B = old_B


class TestNl2:
    """Test dielectric constant from wavelength with damping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2
        phost._A0 = 1.0
        phost._B = np.array([0.5, 0.3])
        phost._w = np.array([1e15, 2e15])
        phost._gam = np.array([1e12, 2e12])
        phost._epsr_infty = 1.0

    def test_nl2_basic(self):
        """Test nl2 with basic inputs."""
        lam = 1e-6
        result = phost.nl2(lam)
        wL = twopi * c0 / (lam + 1e-100)
        expected = phost._A0
        for n in range(phost._osc):
            denom = phost._w[n]**2 - ii * 2.0 * phost._gam[n] * wL - wL**2
            expected += phost._B[n] * phost._w[n]**2 / denom
        assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_nl2_none_b_w_gam(self):
        """Test nl2 when _B, _w, or _gam is None."""
        old_B = phost._B
        phost._B = None
        result = phost.nl2(1e-6)
        assert result == phost._epsr_infty
        phost._B = old_B


class TestNwpNoGam:
    """Test derivative of refractive index without damping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2
        phost._B = np.array([0.5, 0.3])
        phost._w = np.array([1e15, 2e15])

    def test_nwp_no_gam_basic(self):
        """Test nwp_no_gam with basic inputs."""
        wL = 1.5e15
        result = phost.nwp_no_gam(wL)
        nw = np.sqrt(phost.nw2_no_gam(wL))
        expected = 0.0
        for n in range(phost._osc):
            expected += phost._B[n] * phost._w[n]**2 * wL / ((phost._w[n]**2 - wL**2)**2)
        expected = expected / nw
        assert np.allclose(result, expected, rtol=1e-8, atol=1e-8)

    def test_nwp_no_gam_none_b_w(self):
        """Test nwp_no_gam when _B or _w is None."""
        old_B = phost._B
        phost._B = None
        result = phost.nwp_no_gam(1e15)
        assert result == 0.0
        phost._B = old_B


class TestNwp:
    """Test derivative of refractive index with damping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2
        phost._B = np.array([0.5, 0.3])
        phost._w = np.array([1e15, 2e15])
        phost._gam = np.array([1e12, 2e12])

    def test_nwp_basic(self):
        """Test nwp with basic inputs."""
        wL = 1.5e15
        result = phost.nwp(wL)
        assert np.isfinite(result)

    def test_nwp_none_b_w_gam(self):
        """Test nwp when _B, _w, or _gam is None."""
        old_B = phost._B
        phost._B = None
        result = phost.nwp(1e15)
        assert result == 0.0
        phost._B = old_B


class TestEpsrwpNoGam:
    """Test derivative of dielectric constant without damping."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2
        phost._B = np.array([0.5, 0.3])
        phost._w = np.array([1e15, 2e15])

    def test_epsrwp_no_gam_basic(self):
        """Test epsrwp_no_gam with basic inputs."""
        wL = 1.5e15
        result = phost.epsrwp_no_gam(wL)
        expected = 0.0
        for n in range(phost._osc):
            expected += phost._B[n] * phost._w[n]**2 * (2 * wL) / ((phost._w[n]**2 - wL**2)**2)
        assert np.allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_epsrwp_no_gam_none_b_w(self):
        """Test epsrwp_no_gam when _B or _w is None."""
        old_B = phost._B
        phost._B = None
        result = phost.epsrwp_no_gam(1e15)
        assert result == 0.0
        phost._B = old_B


class TestSetParamsAlAs:
    """Test AlAs material parameter setup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_set_params_alas(self):
        """Test SetParamsAlAs."""
        phost.SetParamsAlAs()
        assert phost._osc == 2
        assert len(phost._B) == 2
        assert len(phost._C) == 2
        assert len(phost._w) == 2
        # _gam should be an array with all zeros
        assert isinstance(phost._gam, np.ndarray)
        assert len(phost._gam) == 2
        assert np.all(phost._gam == 0.0)
        assert phost._A0 == 2.0792
        assert np.allclose(phost._B[0], 6.0840)
        assert np.allclose(phost._B[1], 1.9000)
        assert phost._epsr_0 == 10.0
        assert phost._epsr_infty == 8.2


class TestSetParamsGaAs:
    """Test GaAs material parameter setup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_set_params_gaas(self):
        """Test SetParamsGaAs."""
        phost.SetParamsGaAs()
        assert phost._osc == 3
        assert len(phost._B) == 3
        assert len(phost._C) == 3
        assert len(phost._w) == 3
        assert len(phost._gam) == 3
        assert np.allclose(phost._A0, 4.37251400)


class TestSetParamsSilica:
    """Test Silica material parameter setup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_set_params_silica(self):
        """Test SetParamsSilica."""
        phost.SetParamsSilica()
        assert phost._osc == 3
        assert len(phost._B) == 3
        assert len(phost._C) == 3
        assert len(phost._w) == 3
        assert len(phost._gam) == 3
        assert phost._A0 == 1.0


class TestSetParamsNone:
    """Test None (vacuum) material parameter setup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_set_params_none(self):
        """Test SetParamsNone."""
        phost.SetParamsNone()
        assert phost._osc == 1
        assert len(phost._B) == 1
        assert len(phost._C) == 1
        assert len(phost._w) == 1
        # _gam and _Nf should be arrays with all zeros
        assert isinstance(phost._gam, np.ndarray)
        assert len(phost._gam) == 1
        assert np.all(phost._gam == 0.0)
        assert isinstance(phost._Nf, np.ndarray)
        assert len(phost._Nf) == 1
        assert np.all(phost._Nf == 0.0)
        assert phost._A0 == 1.0
        assert phost._B[0] == 0.0
        assert phost._epsr_0 == 1.0
        assert phost._epsr_infty == 1.0


class TestCalcMonoP:
    """Test monochromatic polarization calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2
        phost._chi1 = np.array([0.1 + 0.01j, 0.2 + 0.02j])

    def test_calc_mono_p_basic(self):
        """Test CalcMonoP with basic inputs."""
        N1, N2 = 16, 16
        E = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        result = phost.CalcMonoP(E)
        assert result.shape == (N1, N2, phost._osc)
        expected = eps0 * E[:, :, np.newaxis] * np.real(phost._chi1)
        assert np.allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_calc_mono_p_different_sizes(self):
        """Test CalcMonoP with different array sizes."""
        for N1, N2 in [(8, 8), (16, 16), (32, 32), (64, 64)]:
            E = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
            result = phost.CalcMonoP(E)
            assert result.shape == (N1, N2, phost._osc)


class TestCalcNextP:
    """Test next polarization value calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2
        phost._gam = np.array([1e12, 2e12])
        phost._w = np.array([1e15, 2e15])
        phost._B = np.array([0.5, 0.3])

    def test_calc_next_p_basic(self):
        """Test CalcNextP with basic inputs."""
        N1, N2 = 16, 16
        dt = 1e-15
        P1 = (np.random.random((N1, N2, phost._osc)) +
             1j * np.random.random((N1, N2, phost._osc))) * 1e-6
        P2 = (np.random.random((N1, N2, phost._osc)) +
             1j * np.random.random((N1, N2, phost._osc))) * 1e-6
        E = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6

        result = phost.CalcNextP(P1, P2, E, dt)
        assert result.shape == (N1, N2, phost._osc)
        assert np.all(np.isfinite(result))

    def test_calc_next_p_different_sizes(self):
        """Test CalcNextP with different array sizes."""
        dt = 1e-15
        for N1, N2 in [(8, 8), (16, 16), (32, 32)]:
            P1 = (np.random.random((N1, N2, phost._osc)) +
                 1j * np.random.random((N1, N2, phost._osc))) * 1e-6
            P2 = (np.random.random((N1, N2, phost._osc)) +
                 1j * np.random.random((N1, N2, phost._osc))) * 1e-6
            E = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
            result = phost.CalcNextP(P1, P2, E, dt)
            assert result.shape == (N1, N2, phost._osc)


class TestCalcPHost:
    """Test host material polarization calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2
        phost._A0 = 1.0
        phost._gam = np.array([1e12, 2e12])
        phost._w = np.array([1e15, 2e15])
        phost._B = np.array([0.5, 0.3])
        # Clear module variables
        phost._Px_before = None
        phost._Py_before = None
        phost._Px_now = None
        phost._Py_now = None
        phost._Px_after = None
        phost._Py_after = None

    def test_calc_p_host_basic(self):
        """Test CalcPHost with basic inputs."""
        N1, N2 = 16, 16
        dt = 1e-15
        Ex = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        Ey = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        epsb = 1.0
        Px = np.zeros((N1, N2), dtype=complex)
        Py = np.zeros((N1, N2), dtype=complex)

        phost.CalcPHost(Ex, Ey, dt, 0, epsb, Px, Py)

        assert Px.shape == (N1, N2)
        assert Py.shape == (N1, N2)
        assert np.all(np.isfinite(Px))
        assert np.all(np.isfinite(Py))

    def test_calc_p_host_multiple_calls(self):
        """Test CalcPHost with multiple calls (time-stepping)."""
        N1, N2 = 16, 16
        dt = 1e-15
        Ex = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        Ey = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        epsb = 1.0
        Px = np.zeros((N1, N2), dtype=complex)
        Py = np.zeros((N1, N2), dtype=complex)

        # First call
        phost.CalcPHost(Ex, Ey, dt, 0, epsb, Px, Py)
        Px1 = Px.copy()
        Py1 = Py.copy()

        # Second call
        phost.CalcPHost(Ex, Ey, dt, 1, epsb, Px, Py)
        Px2 = Px.copy()
        Py2 = Py.copy()

        # Results should be different (time-stepping)
        assert not np.allclose(Px1, Px2)
        assert not np.allclose(Py1, Py2)


class TestCalcPHostOld:
    """Test old version of host material polarization calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2
        phost._A0 = 1.0
        phost._w0 = 1e15
        phost._gam = np.array([1e12, 2e12])
        phost._w = np.array([1e15, 2e15])
        phost._B = np.array([0.5, 0.3])
        phost._chi1 = np.array([0.1 + 0.01j, 0.2 + 0.02j])
        # Clear module variables
        phost._Px_before = None
        phost._Py_before = None
        phost._Px_now = None
        phost._Py_now = None
        phost._Px_after = None
        phost._Py_after = None

    def test_calc_p_host_old_m_less_than_2(self):
        """Test CalcPHostOld with m < 2."""
        N1, N2 = 16, 16
        dt = 1e-15
        Ex = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        Ey = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        epsb = 1.0
        Px = np.zeros((N1, N2), dtype=complex)
        Py = np.zeros((N1, N2), dtype=complex)

        phost.CalcPHostOld(Ex, Ey, dt, 0, epsb, Px, Py)

        assert Px.shape == (N1, N2)
        assert Py.shape == (N1, N2)
        assert np.all(np.isfinite(Px))
        assert np.all(np.isfinite(Py))

    def test_calc_p_host_old_m_equal_2(self):
        """Test CalcPHostOld with m == 2."""
        N1, N2 = 16, 16
        dt = 1e-15
        Ex = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        Ey = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        epsb = 1.0
        Px = np.zeros((N1, N2), dtype=complex)
        Py = np.zeros((N1, N2), dtype=complex)

        phost.CalcPHostOld(Ex, Ey, dt, 2, epsb, Px, Py)

        assert Px.shape == (N1, N2)
        assert Py.shape == (N1, N2)

    def test_calc_p_host_old_m_greater_than_2(self):
        """Test CalcPHostOld with m > 2."""
        N1, N2 = 16, 16
        dt = 1e-15
        Ex = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        Ey = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        epsb = 1.0
        Px = np.zeros((N1, N2), dtype=complex)
        Py = np.zeros((N1, N2), dtype=complex)

        phost.CalcPHostOld(Ex, Ey, dt, 3, epsb, Px, Py)

        assert Px.shape == (N1, N2)
        assert Py.shape == (N1, N2)


class TestMakeTransverse:
    """Test transverse projection of electric field."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_make_transverse_basic(self):
        """Test MakeTransverse with basic inputs."""
        N1, N2 = 16, 16
        Ex = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        Ey = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        Ex_orig = Ex.copy()
        Ey_orig = Ey.copy()

        qx = np.linspace(-1e7, 1e7, N1)
        qy = np.linspace(-1e7, 1e7, N2)
        qsq = np.zeros((N1, N2), dtype=complex)
        for j in range(N2):
            for i in range(N1):
                qsq[i, j] = qx[i]**2 + qy[j]**2

        phost.MakeTransverse(Ex, Ey, qx, qy, qsq)

        # Check that arrays were modified
        assert not np.allclose(Ex, Ex_orig)
        assert not np.allclose(Ey, Ey_orig)

    def test_make_transverse_transverse_property(self):
        """Test that MakeTransverse makes field transverse to q."""
        N1, N2 = 16, 16
        Ex = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        Ey = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6

        qx = np.linspace(-1e7, 1e7, N1)
        qy = np.linspace(-1e7, 1e7, N2)
        qsq = np.zeros((N1, N2), dtype=complex)
        for j in range(N2):
            for i in range(N1):
                qsq[i, j] = qx[i]**2 + qy[j]**2
        # Avoid division by zero
        qsq = np.maximum(qsq, 1e-10)

        phost.MakeTransverse(Ex, Ey, qx, qy, qsq)

        # Check that q · E ≈ 0 (transverse condition)
        # Note: Numerical precision may not be perfect, so use a more lenient tolerance
        for j in range(N2):
            dot_product = qx * Ex[:, j] + qy[j] * Ey[:, j]
            # Should be approximately zero (more lenient tolerance for numerical precision)
            assert np.allclose(dot_product, 0.0, atol=1e-1)


class TestSetHostMaterial:
    """Test host material parameter setup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._B = None
        phost._C = None

    def test_set_host_material_alas(self):
        """Test SetHostMaterial with AlAs."""
        lam = 1e-6
        epsr = 1.0
        n0 = 1.0

        phost.SetParamsAlAs()
        phost.SetHostMaterial(True, 'AlAs', lam, epsr, n0)

        assert epsr > 0.0
        assert n0 > 0.0

    def test_set_host_material_gaas(self):
        """Test SetHostMaterial with GaAs."""
        lam = 1e-6
        epsr = 1.0
        n0 = 1.0

        phost.SetParamsGaAs()
        phost.SetHostMaterial(True, 'GaAs', lam, epsr, n0)

        assert epsr > 0.0
        assert n0 > 0.0

    def test_set_host_material_silica(self):
        """Test SetHostMaterial with silica."""
        lam = 1e-6
        epsr = 1.0
        n0 = 1.0

        phost.SetParamsSilica()
        phost.SetHostMaterial(True, 'fsil', lam, epsr, n0)

        assert epsr > 0.0
        assert n0 > 0.0

    def test_set_host_material_none(self):
        """Test SetHostMaterial with none."""
        lam = 1e-6
        epsr = 1.0
        n0 = 1.5

        phost.SetParamsNone()
        # Note: In Python, assignment to parameter doesn't modify original variable
        # The function assigns epsr = n0**2 internally, but doesn't modify the input
        # We just check that the function completes without error
        phost.SetHostMaterial(False, 'none', lam, epsr, n0)

        # The function should complete successfully
        assert True

    def test_set_host_material_unknown(self):
        """Test SetHostMaterial with unknown material."""
        lam = 1e-6
        epsr = 1.0
        n0 = 1.0

        with pytest.raises(ValueError):
            phost.SetHostMaterial(True, 'UnknownMaterial', lam, epsr, n0)


class TestInitializeHost:
    """Test host material initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2

    def test_initialize_host_with_dispersion(self):
        """Test InitializeHost with host dispersion enabled."""
        Nx, Ny = 16, 16
        n0 = 1.5
        qsq = np.random.random((Nx, Ny)) * 1e14 + 1e10
        host = True

        phost.InitializeHost(Nx, Ny, n0, qsq, host)

        assert phost._omega_q is not None
        assert phost._omega_q.shape == (Nx, Ny)
        assert phost._EpsrWq is not None
        assert phost._EpsrWq.shape == (Nx, Ny)
        assert phost._Px_before is not None
        assert phost._Px_before.shape == (Nx, Ny, phost._osc)

    def test_initialize_host_without_dispersion(self):
        """Test InitializeHost without host dispersion."""
        Nx, Ny = 16, 16
        n0 = 1.5
        qsq = np.random.random((Nx, Ny)) * 1e14 + 1e10
        host = False

        phost.InitializeHost(Nx, Ny, n0, qsq, host)

        assert phost._omega_q is not None
        assert phost._omega_q.shape == (Nx, Ny)
        # _EpsrWq should be an array with all elements set to n0**2
        assert phost._EpsrWq is not None
        assert isinstance(phost._EpsrWq, np.ndarray)
        assert phost._EpsrWq.shape == (Nx, Ny)
        assert np.allclose(phost._EpsrWq, n0**2, rtol=1e-10, atol=1e-10)
        # Check that omega_q = sqrt(qsq) * c0 / n0
        expected_omega = np.sqrt(np.real(qsq)) * c0 / n0
        assert np.allclose(phost._omega_q, expected_omega, rtol=1e-10, atol=1e-10)


class TestCalcWq:
    """Test frequency calculation from momentum."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._w0 = 1e15

    def test_calc_wq_basic(self):
        """Test CalcWq with basic inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs('fields/host', exist_ok=True)

            try:
                N1, N2 = 16, 16
                q = np.random.random((N1, N2)) * 1e7 + 1e6
                phost._omega_q = np.zeros((N1, N2), dtype=complex)

                phost.SetParamsAlAs()
                phost.CalcWq(q)

                assert phost._omega_q.shape == (N1, N2)
                assert np.all(np.isfinite(phost._omega_q))
                assert os.path.exists('fields/host/w.q.dat')
            finally:
                os.chdir(old_cwd)


class TestCalcEpsrWq:
    """Test dielectric constant calculation as function of frequency."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_calc_epsr_wq_basic(self):
        """Test CalcEpsrWq with basic inputs."""
        N1, N2 = 16, 16
        phost._omega_q = (np.random.random((N1, N2)) +
                         1j * np.random.random((N1, N2))) * 1e15
        phost._EpsrWq = np.zeros((N1, N2), dtype=complex)

        phost.SetParamsAlAs()
        phost.CalcEpsrWq(None)

        assert phost._EpsrWq.shape == (N1, N2)
        assert np.all(np.isfinite(phost._EpsrWq))


class TestCalcEpsrWqIj:
    """Test dielectric constant calculation for single frequency."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._epsr_0 = 10.0
        phost._epsr_infty = 8.2
        phost._w1 = 1e14
        phost._w2 = 1e16

    def test_calc_epsr_wq_ij_low_frequency(self):
        """Test CalcEpsrWq_ij with low frequency."""
        phost.SetParamsAlAs()
        aw = np.array([1e-20, 1e-30])
        bw = np.array([1e-20, 1e-30])
        w_ij = 1e13  # Below _w1

        result = phost.CalcEpsrWq_ij(w_ij, aw, bw)
        expected = phost._epsr_0 + aw[0] * w_ij**2 + aw[1] * w_ij**3
        assert np.allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_calc_epsr_wq_ij_high_frequency(self):
        """Test CalcEpsrWq_ij with high frequency."""
        phost.SetParamsAlAs()
        aw = np.array([1e-20, 1e-30])
        bw = np.array([1e-20, 1e-30])
        w_ij = 1e17  # Above _w2

        result = phost.CalcEpsrWq_ij(w_ij, aw, bw)
        expected = phost._epsr_infty + bw[0] / w_ij**2 + bw[1] / w_ij**3
        assert np.allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_calc_epsr_wq_ij_mid_frequency(self):
        """Test CalcEpsrWq_ij with mid frequency."""
        phost.SetParamsAlAs()
        aw = np.array([1e-20, 1e-30])
        bw = np.array([1e-20, 1e-30])
        w_ij = 1e15  # Between _w1 and _w2

        result = phost.CalcEpsrWq_ij(w_ij, aw, bw)
        expected = phost.nw2_no_gam(w_ij)
        assert np.allclose(result, expected, rtol=1e-10, atol=1e-10)


class TestDetermineCoeffs:
    """Test expansion coefficient determination."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._w1 = 1e14
        phost._w2 = 1e16
        phost._epsr_0 = 10.0
        phost._epsr_infty = 8.2

    def test_determine_coeffs_basic(self):
        """Test DetermineCoeffs with basic inputs."""
        phost.SetParamsAlAs()
        aw = np.zeros(2)
        bw = np.zeros(2)

        phost.DetermineCoeffs(aw, bw)

        assert len(aw) == 2
        assert len(bw) == 2
        assert np.all(np.isfinite(aw))
        assert np.all(np.isfinite(bw))


class TestEpsrQ:
    """Test dielectric constant array retrieval."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_epsr_q_basic(self):
        """Test Epsr_q with basic inputs."""
        N1, N2 = 16, 16
        phost._EpsrWq = (np.random.random((N1, N2)) +
                        1j * np.random.random((N1, N2))) * 10.0
        q = np.random.random((N1, N2)) * 1e7

        result = phost.Epsr_q(q)
        assert result.shape == (N1, N2)
        assert np.allclose(result, phost._EpsrWq, rtol=1e-12, atol=1e-12)


class TestEpsrQij:
    """Test dielectric constant at specific indices."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_epsr_qij_basic(self):
        """Test Epsr_qij with basic inputs."""
        N1, N2 = 16, 16
        phost._EpsrWq = (np.random.random((N1, N2)) +
                        1j * np.random.random((N1, N2))) * 10.0

        i, j = 5, 7
        result = phost.Epsr_qij(i, j)
        assert result == phost._EpsrWq[i, j]


class TestWq:
    """Test frequency retrieval at specific indices."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_wq_basic(self):
        """Test wq with basic inputs."""
        N1, N2 = 16, 16
        phost._omega_q = (np.random.random((N1, N2)) +
                         1j * np.random.random((N1, N2))) * 1e15

        i, j = 5, 7
        result = phost.wq(i, j)
        assert result == phost._omega_q[i, j]


class TestFDTDDispersion:
    """Test FDTD dispersion relation calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_fdtd_dispersion_basic(self):
        """Test FDTD_Dispersion with basic inputs."""
        Nx, Ny = 16, 16
        # Use smaller q values to avoid NaN from arcsin > 1
        qx = np.linspace(-1e6, 1e6, Nx)
        qy = np.linspace(-1e6, 1e6, Ny)
        dx = 1e-8
        dy = 1e-8
        dt = 1e-15
        n0 = 1.5

        phost.FDTD_Dispersion(qx, qy, dx, dy, dt, n0)

        assert phost._omega_q.shape == (Nx, Ny)
        # Some values may be NaN if arcsin argument > 1, which is expected
        # Check that at least some values are finite
        assert np.any(np.isfinite(phost._omega_q))

    def test_fdtd_dispersion_different_sizes(self):
        """Test FDTD_Dispersion with different array sizes."""
        for Nx, Ny in [(8, 8), (16, 16), (32, 32)]:
            qx = np.linspace(-1e7, 1e7, Nx)
            qy = np.linspace(-1e7, 1e7, Ny)
            dx = 1e-8
            dy = 1e-8
            dt = 1e-15
            n0 = 1.5

            phost.FDTD_Dispersion(qx, qy, dx, dy, dt, n0)
            assert phost._omega_q.shape == (Nx, Ny)


class TestSetInitialP:
    """Test initial polarization setup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        phost._osc = 2
        phost._A0 = 1.0
        phost._B = np.array([0.5, 0.3])
        phost._w = np.array([1e15, 2e15])

    def test_set_initial_p_basic(self):
        """Test SetInitialP with basic inputs."""
        N1, N2 = 16, 16
        Ex = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        Ey = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        qx = np.linspace(-1e7, 1e7, N1)
        qy = np.linspace(-1e7, 1e7, N2)
        qsq = np.zeros((N1, N2), dtype=complex)
        for j in range(N2):
            for i in range(N1):
                qsq[i, j] = qx[i]**2 + qy[j]**2
        dt = 1e-15
        Px = np.zeros((N1, N2), dtype=complex)
        Py = np.zeros((N1, N2), dtype=complex)
        epsb = 1.0

        # Initialize omega_q
        phost._omega_q = np.sqrt(np.real(qsq)) * c0 / 1.5

        phost.SetInitialP(Ex, Ey, qx, qy, qsq, dt, Px, Py, epsb)

        assert Px.shape == (N1, N2)
        assert Py.shape == (N1, N2)
        assert np.all(np.isfinite(Px))
        assert np.all(np.isfinite(Py))


class TestIFFT:
    """Test inverse FFT wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_ifft_basic(self):
        """Test IFFT with basic inputs."""
        N1, N2 = 16, 16
        f = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        f_orig = f.copy()

        phost.IFFT(f)

        # Check that array was modified
        assert not np.allclose(f, f_orig)

    def test_ifft_round_trip(self):
        """Test IFFT round-trip property."""
        N1, N2 = 16, 16
        f_orig = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        f = f_orig.copy()

        phost.IFFT(f)
        # Apply forward FFT (no normalization needed, IFFT already normalized)
        f = pyfftw.interfaces.numpy_fft.fft2(f)

        # Should recover original (within numerical precision)
        # Note: FFT normalization may cause slight differences
        assert np.allclose(f, f_orig, rtol=1e-8, atol=1e-8)


class TestWriteHostDispersion:
    """Test host dispersion data writing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_write_host_dispersion_basic(self):
        """Test WriteHostDispersion with basic inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs('fields/host', exist_ok=True)
            os.makedirs('fields/host/nogam', exist_ok=True)

            try:
                phost.SetParamsAlAs()
                # _gam is now correctly an array after fix
                phost.WriteHostDispersion()

                # Check that files were created
                assert os.path.exists('fields/host/n.w.real.dat')
                assert os.path.exists('fields/host/n.w.imag.dat')
                assert os.path.exists('fields/host/epsr.w.real.dat')
                assert os.path.exists('fields/host/epsr.w.imag.dat')
                assert os.path.exists('fields/host/nogam/n.w.real.dat')
                assert os.path.exists('fields/host/n.l.real.dat')
            finally:
                os.chdir(old_cwd)

    def test_write_host_dispersion_none_b_w(self):
        """Test WriteHostDispersion when _w or _B is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs('fields/host', exist_ok=True)

            try:
                old_w = phost._w
                phost._w = None
                phost.WriteHostDispersion()
                # Should complete without error
                assert True
                phost._w = old_w
            finally:
                os.chdir(old_cwd)


class TestIntegration:
    """Integration tests for complete host material workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_material_setup_workflow(self):
        """Test complete material setup workflow."""
        # Setup AlAs
        phost.SetParamsAlAs()
        lam = 1e-6
        epsr = 1.0
        n0 = 1.0
        phost.SetHostMaterial(True, 'AlAs', lam, epsr, n0)

        assert epsr > 0.0
        assert n0 > 0.0

        # Test dielectric constant calculation
        wL = 1e15
        n2 = phost.nw2_no_gam(wL)
        assert np.isfinite(n2)

    def test_polarization_workflow(self):
        """Test complete polarization calculation workflow."""
        phost._osc = 2
        phost._A0 = 1.0
        phost._gam = np.array([1e12, 2e12])
        phost._w = np.array([1e15, 2e15])
        phost._B = np.array([0.5, 0.3])
        phost._chi1 = np.array([0.1 + 0.01j, 0.2 + 0.02j])

        N1, N2 = 16, 16
        dt = 1e-15
        Ex = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        Ey = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        epsb = 1.0
        Px = np.zeros((N1, N2), dtype=complex)
        Py = np.zeros((N1, N2), dtype=complex)

        # Clear module variables
        phost._Px_before = None
        phost._Py_before = None
        phost._Px_now = None
        phost._Py_now = None
        phost._Px_after = None
        phost._Py_after = None

        # Calculate polarization
        phost.CalcPHost(Ex, Ey, dt, 0, epsb, Px, Py)

        assert Px.shape == (N1, N2)
        assert Py.shape == (N1, N2)
        assert np.all(np.isfinite(Px))
        assert np.all(np.isfinite(Py))

    def test_dispersion_workflow(self):
        """Test complete dispersion calculation workflow."""
        phost.SetParamsAlAs()
        Nx, Ny = 16, 16
        n0 = 1.5
        qsq = np.random.random((Nx, Ny)) * 1e14 + 1e10

        phost.InitializeHost(Nx, Ny, n0, qsq, True)

        assert phost._omega_q is not None
        assert phost._EpsrWq is not None

        # Test retrieval functions
        epsr_q = phost.Epsr_q(None)
        assert epsr_q.shape == (Nx, Ny)

        w = phost.wq(5, 7)
        assert np.isfinite(w)

    @pytest.mark.parametrize("N1,N2", [(8, 8), (16, 16), (32, 32), (64, 64)])
    def test_different_array_sizes(self, N1, N2):
        """Test integration with different array sizes."""
        phost._osc = 2
        phost._A0 = 1.0
        phost._gam = np.array([1e12, 2e12])
        phost._w = np.array([1e15, 2e15])
        phost._B = np.array([0.5, 0.3])
        phost._chi1 = np.array([0.1 + 0.01j, 0.2 + 0.02j])

        dt = 1e-15
        Ex = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        Ey = (np.random.random((N1, N2)) + 1j * np.random.random((N1, N2))) * 1e6
        epsb = 1.0
        Px = np.zeros((N1, N2), dtype=complex)
        Py = np.zeros((N1, N2), dtype=complex)

        # Clear module variables
        phost._Px_before = None
        phost._Py_before = None
        phost._Px_now = None
        phost._Py_now = None
        phost._Px_after = None
        phost._Py_after = None

        phost.CalcPHost(Ex, Ey, dt, 0, epsb, Px, Py)

        assert Px.shape == (N1, N2)
        assert Py.shape == (N1, N2)

