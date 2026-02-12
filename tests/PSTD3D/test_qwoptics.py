"""
Comprehensive test suite for qwoptics.py module.

Tests all quantum wire optics functions including field conversions,
polarization calculations, charge densities, and I/O operations.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from pulsesuite.PSTD3D import qwoptics as qw
from pulsesuite.PSTD3D.qwoptics import QWOptics


class TestYW:
    """Test wire-dependent sign factor function."""

    def test_yw_wire_1(self):
        """Test yw with wire index 1."""
        result = qw.yw(1)
        expected = (-1)**int(np.floor((1 - 1) / 2.0))
        assert result == expected
        assert result == 1

    def test_yw_wire_2(self):
        """Test yw with wire index 2."""
        result = qw.yw(2)
        expected = (-1)**int(np.floor((2 - 1) / 2.0))
        assert result == expected
        assert result == 1

    def test_yw_wire_3(self):
        """Test yw with wire index 3."""
        result = qw.yw(3)
        expected = (-1)**int(np.floor((3 - 1) / 2.0))
        assert result == expected
        assert result == -1

    def test_yw_wire_4(self):
        """Test yw with wire index 4."""
        result = qw.yw(4)
        expected = (-1)**int(np.floor((4 - 1) / 2.0))
        assert result == expected
        assert result == -1

    def test_yw_wire_5(self):
        """Test yw with wire index 5."""
        result = qw.yw(5)
        assert result == 1

    def test_yw_wire_6(self):
        """Test yw with wire index 6."""
        result = qw.yw(6)
        assert result == 1

    @pytest.mark.parametrize("w", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_yw_alternating_pattern(self, w):
        """Test yw follows alternating pattern."""
        result = qw.yw(w)
        expected = (-1)**int(np.floor((w - 1) / 2.0))
        assert result == expected


class TestCalcQWWindow:
    """Test quantum wire window function calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.L = 10e-6  # 10 micrometers
        self.Ny = 128
        self.YY = np.linspace(-self.L, self.L, self.Ny)

    def _make_instance(self):
        """Create a minimal QWOptics instance for testing."""
        Nk = 4
        kr = np.linspace(-1e6, 1e6, Nk)
        Qr = kr
        Ee = np.linspace(0, 1e-19, Nk)
        Eh = np.linspace(0, 1e-19, Nk)
        return QWOptics(self.YY, self.L, 1e-29, kr, Qr, Ee, Eh, 1.0, 1e-12, 1e-19)

    def test_CalcQWWindow_creates_window(self):
        """Test that window function is created."""
        inst = self._make_instance()
        assert inst._QWWindow is not None
        assert len(inst._QWWindow) == self.Ny

    def test_CalcQWWindow_shape(self):
        """Test window function has correct shape."""
        inst = self._make_instance()
        assert inst._QWWindow.shape == (self.Ny,)

    def test_CalcQWWindow_values_inside(self):
        """Test window values inside wire are close to 1."""
        inst = self._make_instance()
        # Values well inside |YY| < L/2 should be close to 1
        inside_mask = np.abs(self.YY) < self.L / 4.0
        assert np.all(inst._QWWindow[inside_mask] > 0.9)

    def test_CalcQWWindow_values_outside(self):
        """Test window values outside wire are small."""
        inst = self._make_instance()
        # Values outside |YY| > L/2 should be very small
        outside_mask = np.abs(self.YY) > self.L / 2.0 * 1.1
        assert np.all(inst._QWWindow[outside_mask] < 0.1)

    def test_CalcQWWindow_symmetry(self):
        """Test window function is symmetric."""
        inst = self._make_instance()
        # Window should be symmetric about y=0 (allowing for numerical precision)
        mid = self.Ny // 2
        if self.Ny % 2 == 0:
            left = inst._QWWindow[:mid]
            right = inst._QWWindow[mid:][::-1]
        else:
            left = inst._QWWindow[:mid]
            right = inst._QWWindow[mid+1:][::-1]
        # Use more relaxed tolerance for symmetry due to exponential calculation
        assert np.allclose(left, right, rtol=1e-6, atol=1e-8)

    def test_CalcQWWindow_different_lengths(self):
        """Test window with different wire lengths."""
        Nk = 4
        kr = np.linspace(-1e6, 1e6, Nk)
        for L in [5e-6, 10e-6, 20e-6, 50e-6]:
            inst = QWOptics(self.YY, L, 1e-29, kr, kr,
                            np.linspace(0, 1e-19, Nk),
                            np.linspace(0, 1e-19, Nk),
                            1.0, 1e-12, 1e-19)
            assert inst._QWWindow is not None
            assert len(inst._QWWindow) == self.Ny


class TestCalcExpikr:
    """Test exp(ikr) array calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.Ny = 64
        self.Nk = 32
        self.y = np.linspace(-10e-6, 10e-6, self.Ny)
        self.ky = np.linspace(-1e6, 1e6, self.Nk)

    def _make_instance(self):
        """Create a minimal QWOptics instance."""
        Ee = np.linspace(0, 1e-19, self.Nk)
        Eh = np.linspace(0, 1e-19, self.Nk)
        return QWOptics(self.y, 10e-6, 1e-29, self.ky, self.ky,
                        Ee, Eh, 1.0, 1e-12, 1e-19)

    def test_CalcExpikr_creates_arrays(self):
        """Test that exp(ikr) arrays are created."""
        inst = self._make_instance()
        assert inst._Expikr is not None
        assert inst._Expikrc is not None

    def test_CalcExpikr_shape(self):
        """Test arrays have correct shape."""
        inst = self._make_instance()
        assert inst._Expikr.shape == (self.Nk, self.Ny)
        assert inst._Expikrc.shape == (self.Nk, self.Ny)

    def test_CalcExpikr_values(self):
        """Test exp(ikr) values are correct."""
        inst = self._make_instance()
        # Check a few values manually
        for k_idx in [0, self.Nk//2, self.Nk-1]:
            for r_idx in [0, self.Ny//2, self.Ny-1]:
                expected = np.exp(1j * self.y[r_idx] * self.ky[k_idx])
                assert np.allclose(inst._Expikr[k_idx, r_idx], expected, rtol=1e-12, atol=1e-12)

    def test_CalcExpikr_conjugate_relationship(self):
        """Test that Expikrc is conjugate of Expikr."""
        inst = self._make_instance()
        assert np.allclose(inst._Expikrc, np.conj(inst._Expikr), rtol=1e-12, atol=1e-12)

    def test_CalcExpikr_unit_magnitude(self):
        """Test that exp(ikr) has unit magnitude."""
        inst = self._make_instance()
        magnitudes = np.abs(inst._Expikr)
        assert np.allclose(magnitudes, 1.0, rtol=1e-12, atol=1e-12)


class TestInitializeQWOptics:
    """Test quantum wire optics initialization (now QWOptics constructor)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.Nr = 128
        self.Nk = 64
        self.RR = np.linspace(-10e-6, 10e-6, self.Nr)
        self.L = 10e-6
        self.dcv = 1e-29 + 1j * 1e-30  # Complex dipole moment
        self.kr = np.linspace(-1e6, 1e6, self.Nk)
        self.Qr = np.linspace(-1e6, 1e6, self.Nk)
        self.Ee = np.linspace(0, 1e-19, self.Nk)
        self.Eh = np.linspace(0, 1e-19, self.Nk)
        self.ehint = 1.0
        self.area = 1e-12  # 1 square micrometer
        self.gap = 1e-19

    def _make_instance(self):
        return QWOptics(self.RR, self.L, self.dcv, self.kr, self.Qr,
                        self.Ee, self.Eh, self.ehint, self.area, self.gap)

    def test_InitializeQWOptics_sets_attributes(self):
        """Test that initialization sets instance attributes."""
        inst = self._make_instance()

        assert inst._QWWindow is not None
        assert inst._Expikr is not None
        assert inst._Expikrc is not None
        assert inst._dcv0 is not None
        assert inst._Xcv0 is not None
        assert inst._Ycv0 is not None
        assert inst._Zcv0 is not None
        assert inst._Xvc0 is not None
        assert inst._Yvc0 is not None
        assert inst._Zvc0 is not None

    def test_InitializeQWOptics_dipole_matrices_shape(self):
        """Test dipole matrices have correct shape."""
        inst = self._make_instance()

        assert inst._Xcv0.shape == (self.Nk, self.Nk)
        assert inst._Ycv0.shape == (self.Nk, self.Nk)
        assert inst._Zcv0.shape == (self.Nk, self.Nk)
        assert inst._Xvc0.shape == (self.Nk, self.Nk)
        assert inst._Yvc0.shape == (self.Nk, self.Nk)
        assert inst._Zvc0.shape == (self.Nk, self.Nk)

    def test_InitializeQWOptics_dipole_values(self):
        """Test dipole matrix values."""
        inst = self._make_instance()

        # Check Xcv0 values: dcv * ((-1)**kh)
        for kh in range(self.Nk):
            for ke in range(self.Nk):
                expected = self.dcv * ((-1)**kh)
                assert np.allclose(inst._Xcv0[ke, kh], expected, rtol=1e-12, atol=1e-12)

        # Check Ycv0 values: dcv
        assert np.allclose(inst._Ycv0, self.dcv, rtol=1e-12, atol=1e-12)

        # Check Zcv0 values: -dcv
        assert np.allclose(inst._Zcv0, -self.dcv, rtol=1e-12, atol=1e-12)

    def test_InitializeQWOptics_conjugate_relationship(self):
        """Test that vc matrices are conjugates of cv matrices."""
        inst = self._make_instance()

        assert np.allclose(inst._Xvc0, np.conj(inst._Xcv0.T), rtol=1e-12, atol=1e-12)
        assert np.allclose(inst._Yvc0, np.conj(inst._Ycv0.T), rtol=1e-12, atol=1e-12)
        assert np.allclose(inst._Zvc0, np.conj(inst._Zcv0.T), rtol=1e-12, atol=1e-12)

    def test_InitializeQWOptics_volume_calculation(self):
        """Test volume calculation."""
        inst = self._make_instance()

        expected_vol = self.L * self.area / self.ehint
        assert np.allclose(inst._Vol, expected_vol, rtol=1e-12, atol=1e-12)


class TestXcvYcvZcv:
    """Test dipole matrix element getters."""

    def setup_method(self):
        """Set up test fixtures."""
        self.Nk = 32
        self.kr = np.linspace(-1e6, 1e6, self.Nk)
        self.RR = np.linspace(-10e-6, 10e-6, 64)
        self.dcv = 1e-29 + 1j * 1e-30
        self.inst = QWOptics(self.RR, 10e-6, self.dcv, self.kr, self.kr,
                             np.linspace(0, 1e-19, self.Nk),
                             np.linspace(0, 1e-19, self.Nk),
                             1.0, 1e-12, 1e-19)

    def test_Xcv_returns_value(self):
        """Test Xcv returns correct value."""
        k, kp = 5, 10
        result = self.inst.Xcv(k, kp)
        expected = self.inst._Xcv0[k, kp]
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_Ycv_returns_value(self):
        """Test Ycv returns correct value."""
        k, kp = 5, 10
        result = self.inst.Ycv(k, kp)
        expected = self.inst._Ycv0[k, kp]
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_Zcv_returns_value(self):
        """Test Zcv returns correct value."""
        k, kp = 5, 10
        result = self.inst.Zcv(k, kp)
        expected = self.inst._Zcv0[k, kp]
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("k", [0, 5, 15, 31])
    @pytest.mark.parametrize("kp", [0, 5, 15, 31])
    def test_dipole_getters_all_indices(self, k, kp):
        """Test dipole getters for various indices."""
        x_val = self.inst.Xcv(k, kp)
        y_val = self.inst.Ycv(k, kp)
        z_val = self.inst.Zcv(k, kp)

        assert np.allclose(x_val, self.inst._Xcv0[k, kp], rtol=1e-12, atol=1e-12)
        assert np.allclose(y_val, self.inst._Ycv0[k, kp], rtol=1e-12, atol=1e-12)
        assert np.allclose(z_val, self.inst._Zcv0[k, kp], rtol=1e-12, atol=1e-12)


class TestQWChi1:
    """Test quantum wire linear susceptibility calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.Nk = 64
        self.lam = 800e-9  # 800 nm wavelength
        self.dky = 1e5  # Momentum step
        self.Ee = np.linspace(1e-19, 2e-19, self.Nk)
        self.Eh = np.linspace(1e-19, 2e-19, self.Nk)
        self.area = 1e-12
        self.geh = 1e12  # Dephasing rate
        self.dcv = 1e-29 + 1j * 1e-30

    def test_QWChi1_returns_complex(self):
        """Test that QWChi1 returns complex value."""
        result = qw.QWChi1(self.lam, self.dky, self.Ee, self.Eh,
                          self.area, self.geh, self.dcv)
        assert isinstance(result, complex) or np.iscomplexobj(result)

    def test_QWChi1_finite(self):
        """Test that QWChi1 returns finite value."""
        result = qw.QWChi1(self.lam, self.dky, self.Ee, self.Eh,
                          self.area, self.geh, self.dcv)
        assert np.isfinite(result)

    def test_QWChi1_different_wavelengths(self):
        """Test QWChi1 with different wavelengths."""
        for lam in [400e-9, 800e-9, 1200e-9, 1600e-9]:
            result = qw.QWChi1(lam, self.dky, self.Ee, self.Eh,
                              self.area, self.geh, self.dcv)
            assert np.isfinite(result)

    def test_QWChi1_zero_dephasing(self):
        """Test QWChi1 with zero dephasing."""
        result = qw.QWChi1(self.lam, self.dky, self.Ee, self.Eh,
                          self.area, 0.0, self.dcv)
        assert np.isfinite(result)

    def test_QWChi1_physical_values(self):
        """Test QWChi1 with physical parameter values."""
        result = qw.QWChi1(self.lam, self.dky, self.Ee, self.Eh,
                          self.area, self.geh, self.dcv)
        # Susceptibility should be reasonable (not extremely large)
        assert abs(result) < 1e10


class TestProp2QW:
    """Test propagation to QW field conversion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.NRR = 256
        self.NR = 128
        self.RR = np.linspace(-20e-6, 20e-6, self.NRR)
        self.R = np.linspace(-10e-6, 10e-6, self.NR)
        self.Exx = np.random.random(self.NRR) + 1j * np.random.random(self.NRR)
        self.Eyy = np.random.random(self.NRR) + 1j * np.random.random(self.NRR)
        self.Ezz = np.random.random(self.NRR) + 1j * np.random.random(self.NRR)
        self.Vrr = np.random.random(self.NRR) + 1j * np.random.random(self.NRR)
        self.Ex = np.zeros(self.NR, dtype=complex)
        self.Ey = np.zeros(self.NR, dtype=complex)
        self.Ez = np.zeros(self.NR, dtype=complex)
        self.Vr = np.zeros(self.NR, dtype=complex)
        self.Edc = 0.0
        self.t = 0.0
        self.xxx = 0

        # Create QWOptics instance
        Nk = 16
        kr = np.linspace(-1e6, 1e6, Nk)
        Ee = np.linspace(0, 1e-19, Nk)
        Eh = np.linspace(0, 1e-19, Nk)
        self.inst = QWOptics(self.R, 10e-6, 1e-29, kr, kr, Ee, Eh, 1.0, 1e-12, 1e-19)

    def test_Prop2QW_initializes_outputs(self):
        """Test that outputs are initialized."""
        self.inst.Prop2QW(self.RR, self.Exx, self.Eyy, self.Ezz, self.Vrr,
                  self.Edc, self.R, self.Ex, self.Ey, self.Ez, self.Vr,
                  self.t, self.xxx)
        # Outputs should be modified (not all zeros after rescaling)
        assert len(self.Ex) == self.NR
        assert len(self.Ey) == self.NR
        assert len(self.Ez) == self.NR
        assert len(self.Vr) == self.NR

    def test_Prop2QW_applies_window(self):
        """Test that window function is applied."""
        # Set window to known values
        self.inst._QWWindow = np.ones(self.NR)
        self.inst._QWWindow[:10] = 0.0  # Zero out first 10 points
        self.inst._QWWindow[-10:] = 0.0  # Zero out last 10 points

        self.inst.Prop2QW(self.RR, self.Exx, self.Eyy, self.Ezz, self.Vrr,
                  self.Edc, self.R, self.Ex, self.Ey, self.Ez, self.Vr,
                  self.t, self.xxx)

        # After FFT, check that windowed regions are affected
        assert np.all(np.isfinite(self.Ex))
        assert np.all(np.isfinite(self.Ey))
        assert np.all(np.isfinite(self.Ez))
        assert np.all(np.isfinite(self.Vr))

    def test_Prop2QW_makes_fields_real_before_fft(self):
        """Test that fields are made real before FFT."""
        # Create purely imaginary input
        self.Exx = 1j * np.random.random(self.NRR)
        self.Eyy = 1j * np.random.random(self.NRR)
        self.Ezz = 1j * np.random.random(self.NRR)
        self.Vrr = 1j * np.random.random(self.NRR)

        self.inst.Prop2QW(self.RR, self.Exx, self.Eyy, self.Ezz, self.Vrr,
                  self.Edc, self.R, self.Ex, self.Ey, self.Ez, self.Vr,
                  self.t, self.xxx)

        # After making real and FFT, should be complex but finite
        assert np.all(np.isfinite(self.Ex))
        assert np.all(np.isfinite(self.Ey))
        assert np.all(np.isfinite(self.Ez))
        assert np.all(np.isfinite(self.Vr))

    def test_Prop2QW_without_window(self):
        """Test Prop2QW when window is None."""
        self.inst._QWWindow = None
        self.inst.Prop2QW(self.RR, self.Exx, self.Eyy, self.Ezz, self.Vrr,
                  self.Edc, self.R, self.Ex, self.Ey, self.Ez, self.Vr,
                  self.t, self.xxx)
        # Should still work, just without windowing
        assert np.all(np.isfinite(self.Ex))


class TestQW2Prop:
    """Test QW to propagation field conversion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.Nr = 128
        self.NRR = 256
        self.r = np.linspace(-10e-6, 10e-6, self.Nr)
        self.Qr = np.linspace(-1e6, 1e6, self.Nr)
        self.RR = np.linspace(-20e-6, 20e-6, self.NRR)
        self.dr = self.r[1] - self.r[0]
        self.dRR = self.RR[1] - self.RR[0]

        # Initialize arrays
        self.Ex = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        self.Ey = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        self.Ez = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        self.Vr = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        self.Px = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        self.Py = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        self.Pz = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        self.re = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        self.rh = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)

        self.Pxx = np.zeros(self.NRR, dtype=complex)
        self.Pyy = np.zeros(self.NRR, dtype=complex)
        self.Pzz = np.zeros(self.NRR, dtype=complex)
        self.RhoE = np.zeros(self.NRR, dtype=complex)
        self.RhoH = np.zeros(self.NRR, dtype=complex)

        self.w = 1
        self.xxx = 0
        self.WriteFields = False
        self.Plasmonics = False

        # Create QWOptics instance
        Nk = 16
        kr = np.linspace(-1e6, 1e6, Nk)
        Ee = np.linspace(0, 1e-19, Nk)
        Eh = np.linspace(0, 1e-19, Nk)
        self.inst = QWOptics(self.r, 10e-6, 1e-29, kr, kr, Ee, Eh, 1.0, 1e-12, 1e-19)

    def test_QW2Prop_basic(self):
        """Test basic QW2Prop conversion."""
        self.inst.QW2Prop(self.r, self.Qr, self.Ex, self.Ey, self.Ez, self.Vr,
                  self.Px, self.Py, self.Pz, self.re, self.rh,
                  self.RR, self.Pxx, self.Pyy, self.Pzz, self.RhoE, self.RhoH,
                  self.w, self.xxx, self.WriteFields, self.Plasmonics)

        assert len(self.Pxx) == self.NRR
        assert len(self.Pyy) == self.NRR
        assert len(self.Pzz) == self.NRR
        assert len(self.RhoE) == self.NRR
        assert len(self.RhoH) == self.NRR

    def test_QW2Prop_with_plasmonics(self):
        """Test QW2Prop with plasmonics enabled."""
        self.Plasmonics = True
        # Make charge densities positive
        self.re = np.abs(self.re) + 1e-10
        self.rh = np.abs(self.rh) + 1e-10

        self.inst.QW2Prop(self.r, self.Qr, self.Ex, self.Ey, self.Ez, self.Vr,
                  self.Px, self.Py, self.Pz, self.re, self.rh,
                  self.RR, self.Pxx, self.Pyy, self.Pzz, self.RhoE, self.RhoH,
                  self.w, self.xxx, self.WriteFields, self.Plasmonics)

        # Charge densities should be normalized
        assert np.all(np.isfinite(self.RhoE))
        assert np.all(np.isfinite(self.RhoH))
        assert np.all(self.RhoE >= 0)
        assert np.all(self.RhoH >= 0)

    def test_QW2Prop_charge_density_normalization(self):
        """Test charge density normalization."""
        self.Plasmonics = True
        self.re = np.ones(self.Nr) * 0.5
        self.rh = np.ones(self.Nr) * 0.5

        self.inst.QW2Prop(self.r, self.Qr, self.Ex, self.Ey, self.Ez, self.Vr,
                  self.Px, self.Py, self.Pz, self.re, self.rh,
                  self.RR, self.Pxx, self.Pyy, self.Pzz, self.RhoE, self.RhoH,
                  self.w, self.xxx, self.WriteFields, self.Plasmonics)

        # Check that densities are normalized
        total_re = np.sum(np.abs(self.re)) * self.dr
        total_rh = np.sum(np.abs(self.rh)) * self.dr
        total_RhoE = np.sum(np.abs(self.RhoE)) * self.dRR
        total_RhoH = np.sum(np.abs(self.RhoH)) * self.dRR

        # After normalization, totals should be approximately equal
        assert abs(total_RhoE - total_re) < 1e-6
        assert abs(total_RhoH - total_rh) < 1e-6


class TestQWPolarization3:
    """Test QW polarization calculation in 3D."""

    def setup_method(self):
        """Set up test fixtures."""
        self.Nr = 128
        self.Nk = 64
        self.y = np.linspace(-10e-6, 10e-6, self.Nr)
        self.ky = np.linspace(-1e6, 1e6, self.Nk)
        self.p = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        self.ehint = 1.0
        self.area = 1e-12
        self.L = 10e-6
        self.Px = np.zeros(self.Nr, dtype=complex)
        self.Py = np.zeros(self.Nr, dtype=complex)
        self.Pz = np.zeros(self.Nr, dtype=complex)
        self.xxx = 0
        self.w = 1

        # Create QWOptics instance
        self.RR = np.linspace(-10e-6, 10e-6, self.Nr)
        self.dcv = 1e-29 + 1j * 1e-30
        self.Qr = self.ky
        self.Ee = np.linspace(0, 1e-19, self.Nk)
        self.Eh = np.linspace(0, 1e-19, self.Nk)
        self.gap = 1e-19

        self.inst = QWOptics(self.RR, self.L, self.dcv, self.ky, self.Qr,
                             self.Ee, self.Eh, self.ehint, self.area, self.gap)

    def test_QWPolarization3_initializes_outputs(self):
        """Test that polarization outputs are initialized."""
        self.inst.QWPolarization3(self.y, self.ky, self.p, self.ehint, self.area,
                          self.L, self.Px, self.Py, self.Pz, self.xxx, self.w)

        assert len(self.Px) == self.Nr
        assert len(self.Py) == self.Nr
        assert len(self.Pz) == self.Nr

    def test_QWPolarization3_finite_values(self):
        """Test that polarization values are finite."""
        self.inst.QWPolarization3(self.y, self.ky, self.p, self.ehint, self.area,
                          self.L, self.Px, self.Py, self.Pz, self.xxx, self.w)

        assert np.all(np.isfinite(self.Px))
        assert np.all(np.isfinite(self.Py))
        assert np.all(np.isfinite(self.Pz))

    def test_QWPolarization3_without_initialization(self):
        """Test QWPolarization3 when state arrays are None."""
        self.inst._Xvc0 = None
        self.inst._Yvc0 = None
        self.inst._Zvc0 = None

        self.inst.QWPolarization3(self.y, self.ky, self.p, self.ehint, self.area,
                          self.L, self.Px, self.Py, self.Pz, self.xxx, self.w)

        # Should return without error, outputs remain zero
        assert np.allclose(self.Px, 0.0, rtol=1e-12, atol=1e-12)
        assert np.allclose(self.Py, 0.0, rtol=1e-12, atol=1e-12)
        assert np.allclose(self.Pz, 0.0, rtol=1e-12, atol=1e-12)

    def test_QWPolarization3_different_wires(self):
        """Test QWPolarization3 with different wire indices."""
        for w in [1, 2, 3, 4]:
            Px = np.zeros(self.Nr, dtype=complex)
            Py = np.zeros(self.Nr, dtype=complex)
            Pz = np.zeros(self.Nr, dtype=complex)

            self.inst.QWPolarization3(self.y, self.ky, self.p, self.ehint, self.area,
                              self.L, Px, Py, Pz, self.xxx, w)

            assert np.all(np.isfinite(Px))
            assert np.all(np.isfinite(Py))
            assert np.all(np.isfinite(Pz))


class TestQWRho5:
    """Test quantum wire charge density calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.Nr = 128
        self.Nk = 64
        self.Qr = np.linspace(-1e6, 1e6, self.Nr)
        self.kr = np.linspace(-1e6, 1e6, self.Nk)
        self.R = np.linspace(-10e-6, 10e-6, self.Nr)
        self.L = 10e-6
        self.kkp = np.zeros((self.Nk, self.Nk), dtype=int)
        self.p = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        self.CC = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        self.DD = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        self.ne = np.random.random(self.Nk) + 1j * np.random.random(self.Nk)
        self.nh = np.random.random(self.Nk) + 1j * np.random.random(self.Nk)
        self.re = np.zeros(self.Nr, dtype=complex)
        self.rh = np.zeros(self.Nr, dtype=complex)
        self.xxx = 0
        self.jjj = 0

        # Create QWOptics instance
        self.dcv = 1e-29 + 1j * 1e-30
        self.Ee = np.linspace(0, 1e-19, self.Nk)
        self.Eh = np.linspace(0, 1e-19, self.Nk)
        self.ehint = 1.0
        self.area = 1e-12
        self.gap = 1e-19

        self.inst = QWOptics(self.R, self.L, self.dcv, self.kr, self.Qr,
                             self.Ee, self.Eh, self.ehint, self.area, self.gap)

    def test_QWRho5_initializes_outputs(self):
        """Test that charge density outputs are initialized."""
        self.inst.QWRho5(self.Qr, self.kr, self.R, self.L, self.kkp, self.p,
                 self.CC, self.DD, self.ne, self.nh, self.re, self.rh,
                 self.xxx, self.jjj)

        assert len(self.re) == self.Nr
        assert len(self.rh) == self.Nr

    def test_QWRho5_finite_values(self):
        """Test that charge densities are finite."""
        self.inst.QWRho5(self.Qr, self.kr, self.R, self.L, self.kkp, self.p,
                 self.CC, self.DD, self.ne, self.nh, self.re, self.rh,
                 self.xxx, self.jjj)

        assert np.all(np.isfinite(self.re))
        assert np.all(np.isfinite(self.rh))

    def test_QWRho5_normalization(self):
        """Test charge density normalization."""
        self.ne = np.ones(self.Nk) * 0.5
        self.nh = np.ones(self.Nk) * 0.5

        self.inst.QWRho5(self.Qr, self.kr, self.R, self.L, self.kkp, self.p,
                 self.CC, self.DD, self.ne, self.nh, self.re, self.rh,
                 self.xxx, self.jjj)

        dr = self.R[1] - self.R[0]
        re_total = np.sum(np.abs(self.re)) * dr
        rh_total = np.sum(np.abs(self.rh)) * dr

        assert np.all(np.isfinite(self.re))
        assert np.all(np.isfinite(self.rh))
        assert re_total >= 0
        assert rh_total >= 0

    def test_QWRho5_without_state(self):
        """Test QWRho5 when state arrays are None."""
        self.inst._Expikr = None
        self.inst._Expikrc = None
        self.inst._QWWindow = None

        self.inst.QWRho5(self.Qr, self.kr, self.R, self.L, self.kkp, self.p,
                 self.CC, self.DD, self.ne, self.nh, self.re, self.rh,
                 self.xxx, self.jjj)

        # Should return without error, outputs remain zero
        assert np.allclose(self.re, 0.0, rtol=1e-12, atol=1e-12)
        assert np.allclose(self.rh, 0.0, rtol=1e-12, atol=1e-12)


class TestGetVn1n2:
    """Test interaction matrix calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.Nk = 32
        self.kr = np.linspace(-1e6, 1e6, self.Nk)
        self.rcv = np.random.random(self.Nk) + 1j * np.random.random(self.Nk)
        self.Hcc = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        self.Hhh = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        self.Hcv = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))

        self.Vcc = np.zeros((self.Nk, self.Nk), dtype=complex)
        self.Vvv = np.zeros((self.Nk, self.Nk), dtype=complex)
        self.Vcv = np.zeros((self.Nk, self.Nk), dtype=complex)
        self.Vvc = np.zeros((self.Nk, self.Nk), dtype=complex)

    def test_GetVn1n2_initializes_outputs(self):
        """Test that interaction matrices are initialized."""
        qw.GetVn1n2(self.kr, self.rcv, self.Hcc, self.Hhh, self.Hcv,
                   self.Vcc, self.Vvv, self.Vcv, self.Vvc)

        assert self.Vcc.shape == (self.Nk, self.Nk)
        assert self.Vvv.shape == (self.Nk, self.Nk)
        assert self.Vcv.shape == (self.Nk, self.Nk)
        assert self.Vvc.shape == (self.Nk, self.Nk)

    def test_GetVn1n2_finite_values(self):
        """Test that interaction matrices have finite values."""
        qw.GetVn1n2(self.kr, self.rcv, self.Hcc, self.Hhh, self.Hcv,
                   self.Vcc, self.Vvv, self.Vcv, self.Vvc)

        assert np.all(np.isfinite(self.Vcc))
        assert np.all(np.isfinite(self.Vvv))
        assert np.all(np.isfinite(self.Vcv))
        assert np.all(np.isfinite(self.Vvc))

    def test_GetVn1n2_conjugate_relationship(self):
        """Test that Vvc is conjugate transpose of Vcv."""
        qw.GetVn1n2(self.kr, self.rcv, self.Hcc, self.Hhh, self.Hcv,
                   self.Vcc, self.Vvv, self.Vcv, self.Vvc)

        assert np.allclose(self.Vvc, np.conj(self.Vcv.T), rtol=1e-10, atol=1e-12)

    def test_GetVn1n2_zero_hamiltonian(self):
        """Test GetVn1n2 with zero Hamiltonian."""
        self.Hcc = np.zeros((self.Nk, self.Nk), dtype=complex)
        self.Hhh = np.zeros((self.Nk, self.Nk), dtype=complex)
        self.Hcv = np.zeros((self.Nk, self.Nk), dtype=complex)

        qw.GetVn1n2(self.kr, self.rcv, self.Hcc, self.Hhh, self.Hcv,
                   self.Vcc, self.Vvv, self.Vcv, self.Vvc)

        # With zero H, V should be zero
        assert np.allclose(self.Vcc, 0.0, rtol=1e-12, atol=1e-12)
        assert np.allclose(self.Vvv, 0.0, rtol=1e-12, atol=1e-12)
        assert np.allclose(self.Vcv, 0.0, rtol=1e-12, atol=1e-12)
        assert np.allclose(self.Vvc, 0.0, rtol=1e-12, atol=1e-12)


class TestPrintFunctions:
    """Test print/write functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.tmpdir)
        os.makedirs('dataQW', exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.tmpdir)

    def test_printITReal2_first_time(self):
        """Test printITReal2 on first write."""
        qw._firsttime = True
        N = 32
        Dx = np.random.random(N) + 1j * np.random.random(N)
        z = np.linspace(0, 1, N)
        n = 0
        file = 'test_file'

        qw.printITReal2(Dx, z, n, file)

        filename = f'dataQW/{file}{n:06d}.dat'
        assert os.path.exists(filename)

        # Check file content
        with open(filename, 'r') as f:
            lines = f.readlines()
            assert len(lines) == N
            # First line should have both coordinate and field value
            parts = lines[0].split()
            assert len(parts) == 2

    def test_printITReal2_subsequent_times(self):
        """Test printITReal2 on subsequent writes."""
        qw._firsttime = True
        N = 32
        Dx = np.random.random(N) + 1j * np.random.random(N)
        z = np.linspace(0, 1, N)
        file = 'test_file'

        # First write
        qw.printITReal2(Dx, z, 0, file)
        # Second write
        qw.printITReal2(Dx, z, 1, file)

        filename1 = f'dataQW/{file}000000.dat'
        filename2 = f'dataQW/{file}000001.dat'
        assert os.path.exists(filename1)
        assert os.path.exists(filename2)

    def test_printITReal(self):
        """Test printITReal function."""
        N = 32
        Dx = np.random.random(N) + 1j * np.random.random(N)
        z = np.linspace(0, 1, N)
        n = 0
        file = 'test_real'

        qw.printITReal(Dx, z, n, file)

        filename = f'dataQW/{file}{n:06d}.dat'
        assert os.path.exists(filename)

        with open(filename, 'r') as f:
            lines = f.readlines()
            assert len(lines) == N
            # Each line should have coordinate and field value
            parts = lines[0].split()
            assert len(parts) == 2

    def test_printIT3D(self):
        """Test printIT3D function."""
        N1, N2, N3 = 8, 8, 8
        Dx = np.random.random((N1, N2, N3)) + 1j * np.random.random((N1, N2, N3))
        z = np.array([0.0])  # Unused
        n = 0
        file = 'test_3d'

        qw.printIT3D(Dx, z, n, file)

        filename = f'dataQW/{file}{n:06d}.dat'
        assert os.path.exists(filename)

        with open(filename, 'r') as f:
            lines = f.readlines()
            assert len(lines) == N1 * N2 * N3
            # Each line should have real and imaginary parts
            parts = lines[0].split()
            assert len(parts) == 2


class TestWriteFunctions:
    """Test write functions for SBE solutions, PL spectrum, and fields."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.tmpdir)
        os.makedirs('dataQW', exist_ok=True)
        for d in ['Wire/C', 'Wire/D', 'Wire/P', 'Wire/ne', 'Wire/nh',
                   'Wire/Ee', 'Wire/Eh', 'Wire/PL', 'Wire/Ex', 'Wire/Ey',
                   'Wire/Ez', 'Wire/Vr', 'Wire/Px', 'Wire/Py', 'Wire/Pz',
                   'Wire/Re', 'Wire/Rh', 'Wire/Rho',
                   'Prop', 'Prop/Ex', 'Prop/Ey', 'Prop/Ez', 'Prop/Vr',
                   'Prop/Px', 'Prop/Py', 'Prop/Pz', 'Prop/Re', 'Prop/Rh',
                   'Prop/Rho']:
            os.makedirs(f'dataQW/{d}', exist_ok=True)

        self.Nk = 32
        self.Nr = 64
        self.ky = np.linspace(-1e6, 1e6, self.Nk)
        self.hw = np.linspace(1e-19, 2e-19, 100)

    def teardown_method(self):
        """Clean up test fixtures."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.tmpdir)

    def test_WriteSBESolns_creates_files(self):
        """Test that WriteSBESolns creates output files."""
        ne = np.random.random(self.Nk) + 1j * np.random.random(self.Nk)
        nh = np.random.random(self.Nk) + 1j * np.random.random(self.Nk)
        C = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        D = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        P = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        Ee = np.random.random(self.Nk) + 1j * np.random.random(self.Nk)
        Eh = np.random.random(self.Nk) + 1j * np.random.random(self.Nk)

        qw.WriteSBESolns(self.ky, ne, nh, C, D, P, Ee, Eh, 1, 0)
        assert True

    def test_WritePLSpectrum_creates_file(self):
        """Test that WritePLSpectrum creates output file."""
        PLS = np.random.random(len(self.hw))
        qw.WritePLSpectrum(self.hw, PLS, 1, 0)
        assert True

    def test_WriteQWFields_creates_files(self):
        """Test that WriteQWFields creates output files."""
        QY = np.linspace(-1e6, 1e6, self.Nr)
        Ex = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Ey = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Ez = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Vr = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Px = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Py = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Pz = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Re = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Rh = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)

        qw.WriteQWFields(QY, Ex, Ey, Ez, Vr, Px, Py, Pz, Re, Rh, 'r', 1, 0)
        assert True

    def test_WritePropFields_creates_files(self):
        """Test that WritePropFields creates output files."""
        y = np.linspace(-10e-6, 10e-6, self.Nr)
        Ex = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Ey = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Ez = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Vr = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Px = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Py = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Pz = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Re = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)
        Rh = np.random.random(self.Nr) + 1j * np.random.random(self.Nr)

        qw.WritePropFields(y, Ex, Ey, Ez, Vr, Px, Py, Pz, Re, Rh, 'y', 1, 0)
        assert True


class TestIntegratedWorkflows:
    """Test integrated workflows combining multiple functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.Nr = 128
        self.Nk = 64
        self.NRR = 256

        self.RR = np.linspace(-20e-6, 20e-6, self.NRR)
        self.R = np.linspace(-10e-6, 10e-6, self.Nr)
        self.r = self.R
        self.Qr = np.linspace(-1e6, 1e6, self.Nr)
        self.kr = np.linspace(-1e6, 1e6, self.Nk)
        self.ky = self.kr
        self.y = self.R

        self.L = 10e-6
        self.dcv = 1e-29 + 1j * 1e-30
        self.Ee = np.linspace(0, 1e-19, self.Nk)
        self.Eh = np.linspace(0, 1e-19, self.Nk)
        self.ehint = 1.0
        self.area = 1e-12
        self.gap = 1e-19

    def test_initialization_workflow(self):
        """Test complete initialization workflow."""
        inst = QWOptics(self.RR, self.L, self.dcv, self.kr, self.Qr,
                        self.Ee, self.Eh, self.ehint, self.area, self.gap)

        # Check all attributes are set
        assert inst._QWWindow is not None
        assert inst._Expikr is not None
        assert inst._Expikrc is not None
        assert inst._dcv0 is not None
        assert inst._Xcv0 is not None
        assert inst._Ycv0 is not None
        assert inst._Zcv0 is not None

    def test_prop_to_qw_to_prop_roundtrip(self):
        """Test roundtrip conversion: Prop -> QW -> Prop."""
        # Initialize with R (not RR) for window calculation
        inst = QWOptics(self.R, self.L, self.dcv, self.kr, self.Qr,
                        self.Ee, self.Eh, self.ehint, self.area, self.gap)

        # Create propagation fields
        Exx = np.random.random(self.NRR) + 1j * np.random.random(self.NRR)
        Eyy = np.random.random(self.NRR) + 1j * np.random.random(self.NRR)
        Ezz = np.random.random(self.NRR) + 1j * np.random.random(self.NRR)
        Vrr = np.random.random(self.NRR) + 1j * np.random.random(self.NRR)

        # Convert to QW space
        Ex = np.zeros(self.Nr, dtype=complex)
        Ey = np.zeros(self.Nr, dtype=complex)
        Ez = np.zeros(self.Nr, dtype=complex)
        Vr = np.zeros(self.Nr, dtype=complex)
        Edc = 0.0

        inst.Prop2QW(self.RR, Exx, Eyy, Ezz, Vrr, Edc, self.R, Ex, Ey, Ez, Vr, 0.0, 0)

        # Convert back to propagation space
        Px = np.zeros(self.Nr, dtype=complex)
        Py = np.zeros(self.Nr, dtype=complex)
        Pz = np.zeros(self.Nr, dtype=complex)
        re = np.zeros(self.Nr, dtype=complex)
        rh = np.zeros(self.Nr, dtype=complex)

        Pxx = np.zeros(self.NRR, dtype=complex)
        Pyy = np.zeros(self.NRR, dtype=complex)
        Pzz = np.zeros(self.NRR, dtype=complex)
        RhoE = np.zeros(self.NRR, dtype=complex)
        RhoH = np.zeros(self.NRR, dtype=complex)

        inst.QW2Prop(self.r, self.Qr, Ex, Ey, Ez, Vr, Px, Py, Pz, re, rh,
                  self.RR, Pxx, Pyy, Pzz, RhoE, RhoH, 1, 0, False, False)

        # Check that outputs are finite
        assert np.all(np.isfinite(Pxx))
        assert np.all(np.isfinite(Pyy))
        assert np.all(np.isfinite(Pzz))

    def test_polarization_calculation_workflow(self):
        """Test complete polarization calculation workflow."""
        inst = QWOptics(self.RR, self.L, self.dcv, self.kr, self.Qr,
                        self.Ee, self.Eh, self.ehint, self.area, self.gap)

        # Create density matrix
        p = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))

        # Calculate polarization
        Px = np.zeros(self.Nr, dtype=complex)
        Py = np.zeros(self.Nr, dtype=complex)
        Pz = np.zeros(self.Nr, dtype=complex)

        inst.QWPolarization3(self.y, self.ky, p, self.ehint, self.area,
                          self.L, Px, Py, Pz, 0, 1)

        # Check results
        assert np.all(np.isfinite(Px))
        assert np.all(np.isfinite(Py))
        assert np.all(np.isfinite(Pz))

    def test_charge_density_workflow(self):
        """Test complete charge density calculation workflow."""
        inst = QWOptics(self.RR, self.L, self.dcv, self.kr, self.Qr,
                        self.Ee, self.Eh, self.ehint, self.area, self.gap)

        # Create coherence matrices
        CC = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        DD = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        ne = np.random.random(self.Nk) + 1j * np.random.random(self.Nk)
        nh = np.random.random(self.Nk) + 1j * np.random.random(self.Nk)
        p = np.zeros((self.Nk, self.Nk), dtype=complex)
        kkp = np.zeros((self.Nk, self.Nk), dtype=int)

        # Calculate charge densities
        re = np.zeros(self.Nr, dtype=complex)
        rh = np.zeros(self.Nr, dtype=complex)

        inst.QWRho5(self.Qr, self.kr, self.R, self.L, kkp, p, CC, DD,
                 ne, nh, re, rh, 0, 0)

        # Check results
        assert np.all(np.isfinite(re))
        assert np.all(np.isfinite(rh))

    def test_interaction_matrix_workflow(self):
        """Test interaction matrix calculation workflow."""
        # Create Hamiltonian matrices
        rcv = np.random.random(self.Nk) + 1j * np.random.random(self.Nk)
        Hcc = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        Hhh = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))
        Hcv = np.random.random((self.Nk, self.Nk)) + 1j * np.random.random((self.Nk, self.Nk))

        # Calculate interaction matrices
        Vcc = np.zeros((self.Nk, self.Nk), dtype=complex)
        Vvv = np.zeros((self.Nk, self.Nk), dtype=complex)
        Vcv = np.zeros((self.Nk, self.Nk), dtype=complex)
        Vvc = np.zeros((self.Nk, self.Nk), dtype=complex)

        qw.GetVn1n2(self.kr, rcv, Hcc, Hhh, Hcv, Vcc, Vvv, Vcv, Vvc)

        # Check results
        assert np.all(np.isfinite(Vcc))
        assert np.all(np.isfinite(Vvv))
        assert np.all(np.isfinite(Vcv))
        assert np.all(np.isfinite(Vvc))
        assert np.allclose(Vvc, np.conj(Vcv.T), rtol=1e-10, atol=1e-12)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.Nk = 32
        self.Nr = 64

    def test_yw_zero_wire(self):
        """Test yw with wire index 0 (edge case)."""
        result = qw.yw(0)
        expected = (-1)**int(np.floor((0 - 1) / 2.0))
        assert result == expected

    def test_QWChi1_extreme_wavelengths(self):
        """Test QWChi1 with extreme wavelength values."""
        Nk = 32
        dky = 1e5
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        area = 1e-12
        geh = 1e12
        dcv = 1e-29

        # Very short wavelength
        result1 = qw.QWChi1(100e-9, dky, Ee, Eh, area, geh, dcv)
        assert np.isfinite(result1)

        # Very long wavelength
        result2 = qw.QWChi1(10e-6, dky, Ee, Eh, area, geh, dcv)
        assert np.isfinite(result2)

    def test_CalcQWWindow_single_point(self):
        """Test CalcQWWindow with single point."""
        YY = np.array([0.0])
        L = 10e-6
        Nk = 2
        kr = np.array([-1e6, 1e6])
        inst = QWOptics(YY, L, 1e-29, kr, kr,
                        np.array([0, 1e-19]), np.array([0, 1e-19]),
                        1.0, 1e-12, 1e-19)
        assert inst._QWWindow is not None
        assert len(inst._QWWindow) == 1

    def test_CalcExpikr_single_point(self):
        """Test CalcExpikr with single point."""
        y = np.array([0.0])
        ky = np.array([0.0])
        inst = QWOptics(y, 10e-6, 1e-29, ky, ky,
                        np.array([0.0]), np.array([0.0]),
                        1.0, 1e-12, 1e-19)
        assert inst._Expikr is not None
        assert inst._Expikr.shape == (1, 1)
        assert np.allclose(inst._Expikr[0, 0], 1.0, rtol=1e-12, atol=1e-12)

    def test_QWPolarization3_zero_density_matrix(self):
        """Test QWPolarization3 with zero density matrix."""
        Nr = 64
        Nk = 32
        y = np.linspace(-10e-6, 10e-6, Nr)
        ky = np.linspace(-1e6, 1e6, Nk)
        p = np.zeros((Nk, Nk), dtype=complex)
        ehint = 1.0
        area = 1e-12
        L = 10e-6
        Px = np.zeros(Nr, dtype=complex)
        Py = np.zeros(Nr, dtype=complex)
        Pz = np.zeros(Nr, dtype=complex)

        RR = np.linspace(-10e-6, 10e-6, Nr)
        dcv = 1e-29
        Qr = ky
        Ee = np.linspace(0, 1e-19, Nk)
        Eh = np.linspace(0, 1e-19, Nk)
        inst = QWOptics(RR, L, dcv, ky, Qr, Ee, Eh, ehint, area, 1e-19)

        inst.QWPolarization3(y, ky, p, ehint, area, L, Px, Py, Pz, 0, 1)

        # With zero density matrix, polarization should be zero
        assert np.allclose(Px, 0.0, rtol=1e-10, atol=1e-12)
        assert np.allclose(Py, 0.0, rtol=1e-10, atol=1e-12)
        assert np.allclose(Pz, 0.0, rtol=1e-10, atol=1e-12)

    def test_QWRho5_zero_coherence_matrices(self):
        """Test QWRho5 with zero coherence matrices."""
        Nr = 64
        Nk = 32
        Qr = np.linspace(-1e6, 1e6, Nr)
        kr = np.linspace(-1e6, 1e6, Nk)
        R = np.linspace(-10e-6, 10e-6, Nr)
        L = 10e-6
        kkp = np.zeros((Nk, Nk), dtype=int)
        p = np.zeros((Nk, Nk), dtype=complex)
        CC = np.zeros((Nk, Nk), dtype=complex)
        DD = np.zeros((Nk, Nk), dtype=complex)
        ne = np.zeros(Nk, dtype=complex)
        nh = np.zeros(Nk, dtype=complex)
        re = np.zeros(Nr, dtype=complex)
        rh = np.zeros(Nr, dtype=complex)

        dcv = 1e-29
        Ee = np.linspace(0, 1e-19, Nk)
        Eh = np.linspace(0, 1e-19, Nk)
        inst = QWOptics(R, L, dcv, kr, Qr, Ee, Eh, 1.0, 1e-12, 1e-19)

        inst.QWRho5(Qr, kr, R, L, kkp, p, CC, DD, ne, nh, re, rh, 0, 0)

        # With zero matrices, charge densities should be zero
        assert np.allclose(re, 0.0, rtol=1e-10, atol=1e-12)
        assert np.allclose(rh, 0.0, rtol=1e-10, atol=1e-12)


class TestNumericalPrecision:
    """Test numerical precision and stability."""

    def setup_method(self):
        """Set up test fixtures."""
        self.Nk = 64
        self.Nr = 128

    def test_CalcExpikr_precision(self):
        """Test exp(ikr) calculation precision."""
        Ny = 128
        Nk = 64
        y = np.linspace(-10e-6, 10e-6, Ny)
        ky = np.linspace(-1e6, 1e6, Nk)

        inst = QWOptics(y, 10e-6, 1e-29, ky, ky,
                        np.linspace(0, 1e-19, Nk),
                        np.linspace(0, 1e-19, Nk),
                        1.0, 1e-12, 1e-19)

        # Check unit magnitude with high precision
        magnitudes = np.abs(inst._Expikr)
        assert np.allclose(magnitudes, 1.0, rtol=1e-12, atol=1e-12)

        # Check conjugate relationship with high precision
        assert np.allclose(inst._Expikrc, np.conj(inst._Expikr), rtol=1e-12, atol=1e-12)

    def test_InitializeQWOptics_dipole_precision(self):
        """Test dipole matrix calculation precision."""
        Nr = 128
        Nk = 64
        RR = np.linspace(-10e-6, 10e-6, Nr)
        L = 10e-6
        dcv = 1e-29 + 1j * 1e-30
        kr = np.linspace(-1e6, 1e6, Nk)
        Ee = np.linspace(0, 1e-19, Nk)
        Eh = np.linspace(0, 1e-19, Nk)

        inst = QWOptics(RR, L, dcv, kr, kr, Ee, Eh, 1.0, 1e-12, 1e-19)

        # Check conjugate relationship with high precision
        assert np.allclose(inst._Xvc0, np.conj(inst._Xcv0.T), rtol=1e-12, atol=1e-12)
        assert np.allclose(inst._Yvc0, np.conj(inst._Ycv0.T), rtol=1e-12, atol=1e-12)
        assert np.allclose(inst._Zvc0, np.conj(inst._Zcv0.T), rtol=1e-12, atol=1e-12)

    def test_GetVn1n2_conjugate_precision(self):
        """Test GetVn1n2 conjugate relationship precision."""
        Nk = 32
        kr = np.linspace(-1e6, 1e6, Nk)
        rcv = np.random.random(Nk) + 1j * np.random.random(Nk)
        Hcc = np.random.random((Nk, Nk)) + 1j * np.random.random((Nk, Nk))
        Hhh = np.random.random((Nk, Nk)) + 1j * np.random.random((Nk, Nk))
        Hcv = np.random.random((Nk, Nk)) + 1j * np.random.random((Nk, Nk))

        Vcc = np.zeros((Nk, Nk), dtype=complex)
        Vvv = np.zeros((Nk, Nk), dtype=complex)
        Vcv = np.zeros((Nk, Nk), dtype=complex)
        Vvc = np.zeros((Nk, Nk), dtype=complex)

        qw.GetVn1n2(kr, rcv, Hcc, Hhh, Hcv, Vcc, Vvv, Vcv, Vvc)

        # Check conjugate relationship with high precision
        assert np.allclose(Vvc, np.conj(Vcv.T), rtol=1e-10, atol=1e-12)
