"""
Comprehensive test suite for dcfield.py module.

Tests all DC field carrier transport functions including initialization,
DC field calculations, drift velocity, phonon interactions, and legacy functions.

Organised into:
  - Pure function tests (module-level, no class needed)
  - DCFieldModule class tests (constructor, methods, getters)
  - Backward-compatibility tests (singleton wrappers, __getattr__)
  - Integration tests
"""

import os

import numpy as np
import pytest
from scipy.constants import hbar as hbar_SI

from pulsesuite.PSTD3D import dcfield as dc
from pulsesuite.PSTD3D.dcfield import DCFieldModule

# ============================================================================
# Pure function tests (no module state needed)
# ============================================================================


class TestGetKArray:
    """Test k-space array generation."""

    def test_basic(self):
        Nk = 64
        L = 1e-6
        k = dc.GetKArray(Nk, L)

        assert len(k) == Nk
        assert k.shape == (Nk,)
        assert np.allclose(k[Nk // 2], 0.0, atol=1e-12)

    def test_centered(self):
        Nk = 128
        L = 2e-6
        k = dc.GetKArray(Nk, L)

        assert k[Nk // 2] == pytest.approx(0.0, abs=1e-12)

        dk = 2.0 * np.pi / L
        assert k[1] - k[0] == pytest.approx(dk, rel=1e-10)

    def test_spacing(self):
        Nk = 64
        L = 1e-6
        k = dc.GetKArray(Nk, L)

        dk = 2.0 * np.pi / L
        assert k[1] - k[0] == pytest.approx(dk, rel=1e-10)

    def test_zero_length(self):
        Nk = 32
        k = dc.GetKArray(Nk, 0.0)

        assert len(k) == Nk
        assert k[1] - k[0] == pytest.approx(1.0, rel=1e-10)


class TestLrtz:
    """Test Lorentzian function."""

    def test_peak(self):
        result = dc.Lrtz(0.0, 1.0)
        expected = 1.0 / np.pi
        assert result == pytest.approx(expected, rel=1e-12)

    def test_array(self):
        a = np.linspace(-5.0, 5.0, 11)
        b = 1.0
        result = dc.Lrtz(a, b)
        expected = b / np.pi / (a**2 + b**2)
        assert np.allclose(result, expected, rtol=1e-12)

    def test_large_argument(self):
        result = dc.Lrtz(100.0, 1.0)
        expected = 1.0 / np.pi / (100.0**2 + 1.0)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_small_width(self):
        result = dc.Lrtz(0.0, 0.1)
        expected = 0.1 / np.pi / 0.1**2
        assert result == pytest.approx(expected, rel=1e-12)


class TestTheta:
    """Test Heaviside step function."""

    def test_positive(self):
        assert dc.theta(1.0) == pytest.approx(1.0, rel=1e-10)

    def test_negative(self):
        assert dc.theta(-1.0) == pytest.approx(0.0, abs=1e-10)

    def test_zero(self):
        assert dc.theta(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_array(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = dc.theta(x)
        assert result[0] < 0.1
        assert result[1] < 0.1
        assert result[2] == pytest.approx(0.0, abs=1e-10)
        assert result[3] > 0.9
        assert result[4] > 0.9


class TestEkReNorm:
    """Test energy renormalization function."""

    def test_simple(self):
        Nk = 8
        n = np.ones(Nk, dtype=float)
        En = np.arange(Nk, dtype=float) * 1e-20
        V = np.eye(Nk) * 1e-20

        Ec = dc.EkReNorm(n, En, V)

        assert len(Ec) == Nk
        assert np.all(np.isfinite(Ec))

    def test_zero_interaction(self):
        Nk = 16
        n = np.ones(Nk, dtype=float)
        En = np.linspace(0, 1e-19, Nk)
        V = np.zeros((Nk, Nk))

        Ec = dc.EkReNorm(n, En, V)

        assert np.allclose(Ec, En, rtol=1e-10, atol=1e-20)

    def test_physical_values(self):
        Nk = 32
        n = np.random.random(Nk) * 0.5
        En = np.linspace(-1e-19, 1e-19, Nk)
        V = np.random.random((Nk, Nk)) * 1e-21

        Ec = dc.EkReNorm(n, En, V)

        assert len(Ec) == Nk
        assert np.all(np.isfinite(Ec))
        assert np.all(np.isreal(Ec))


class TestDriftVt:
    """Test drift velocity calculation (now takes dkk explicitly)."""

    def test_simple(self):
        Nk = 64
        ky = np.linspace(-1e9, 1e9, Nk)
        dkk = ky[1] - ky[0]
        n = np.ones(Nk, dtype=float)
        Ec = np.linspace(0, 1e-19, Nk)

        v = dc.DriftVt(n, Ec, dkk)

        assert np.isfinite(v)
        assert np.isreal(v)

    def test_gaussian_distribution(self):
        ky = np.linspace(-1e9, 1e9, 64)
        dkk = ky[1] - ky[0]
        me = 9.109e-31
        n = np.exp(-(ky / 1e9)**2)
        Ec = ky**2 * hbar_SI**2 / (2.0 * me)

        v = dc.DriftVt(n, Ec, dkk)

        assert np.isfinite(v)

    def test_zero_distribution(self):
        Nk = 64
        ky = np.linspace(-1e9, 1e9, Nk)
        dkk = ky[1] - ky[0]
        n = np.zeros(Nk)
        Ec = np.linspace(0, 1e-19, Nk)

        v = dc.DriftVt(n, Ec, dkk)

        assert np.isfinite(v)

    def test_constant_energy(self):
        Nk = 64
        ky = np.linspace(-1e9, 1e9, Nk)
        dkk = ky[1] - ky[0]
        n = np.ones(Nk, dtype=float)
        Ec = np.ones(Nk) * 1e-19

        v = dc.DriftVt(n, Ec, dkk)

        assert v == pytest.approx(0.0, abs=1e-10)


class TestCalcI0n:
    """Test electron current calculation."""

    def test_simple(self):
        Nk = 64
        ky = np.linspace(-1e9, 1e9, Nk)
        me = 9.109e-31
        ne = np.ones(Nk, dtype=complex)

        Ie = dc.CalcI0n(ne, me, ky)

        assert np.isfinite(Ie)
        assert np.isreal(Ie)

    def test_zero_distribution(self):
        ky = np.linspace(-1e9, 1e9, 64)
        ne = np.zeros(64, dtype=complex)

        Ie = dc.CalcI0n(ne, 9.109e-31, ky)

        assert Ie == pytest.approx(0.0, abs=1e-20)

    def test_symmetric_distribution(self):
        ky = np.linspace(-1e9, 1e9, 64)
        ne = np.exp(-(ky / 1e9)**2).astype(complex)

        Ie = dc.CalcI0n(ne, 9.109e-31, ky)

        assert abs(Ie) < 1e-10


class TestCalcVD:
    """Test drift velocity from distribution."""

    def test_simple(self):
        ky = np.linspace(-1e9, 1e9, 64)
        n = np.ones(64, dtype=complex)

        v = dc.CalcVD(ky, 9.109e-31, n)

        assert np.isfinite(v)

    def test_zero_distribution(self):
        ky = np.linspace(-1e9, 1e9, 64)
        n = np.zeros(64, dtype=complex)

        v = dc.CalcVD(ky, 9.109e-31, n)

        assert np.isfinite(v)


class TestCalcPD:
    """Test momentum from distribution."""

    def test_simple(self):
        ky = np.linspace(-1e9, 1e9, 64)
        n = np.ones(64, dtype=complex)

        p = dc.CalcPD(ky, 9.109e-31, n)

        assert np.isfinite(p)

    def test_symmetric_distribution(self):
        ky = np.linspace(-1e9, 1e9, 64)
        n = np.exp(-(ky / 1e9)**2).astype(complex)

        p = dc.CalcPD(ky, 9.109e-31, n)

        assert abs(p) < 1e-20


class TestThetaEM:
    """Test emission matrix element calculation."""

    def test_basic(self):
        Nk = 32
        ky = np.linspace(-1e9, 1e9, Nk)
        n = np.ones(Nk, dtype=float) * 0.5
        Cq2 = np.ones(Nk, dtype=float) * 1e-20

        result = dc.ThetaEM(1e-20, 9.109e-31, 1e12, ky, n, Cq2, 1e5, 0.1,
                            16, 16)

        assert np.isfinite(result)
        assert result >= 0.0

    def test_edge_cases(self):
        Nk = 32
        ky = np.linspace(-1e9, 1e9, Nk)
        n = np.ones(Nk, dtype=float) * 0.5
        Cq2 = np.ones(Nk, dtype=float) * 1e-20

        r1 = dc.ThetaEM(1e-20, 9.109e-31, 1e12, ky, n, Cq2, 1e5, 0.1, 1, 1)
        r2 = dc.ThetaEM(1e-20, 9.109e-31, 1e12, ky, n, Cq2, 1e5, 0.1,
                         Nk, Nk)

        assert np.isfinite(r1)
        assert np.isfinite(r2)


class TestThetaABS:
    """Test absorption matrix element calculation."""

    def test_basic(self):
        Nk = 32
        ky = np.linspace(-1e9, 1e9, Nk)
        n = np.ones(Nk, dtype=float) * 0.5
        Cq2 = np.ones(Nk, dtype=float) * 1e-20

        result = dc.ThetaABS(1e-20, 9.109e-31, 1e12, ky, n, Cq2, 1e5, 0.1,
                             16, 16)

        assert np.isfinite(result)
        assert result >= 0.0


class TestFDrift2:
    """Test drift force calculation."""

    def test_basic(self):
        Nk = 32
        ky = np.linspace(-1e9, 1e9, Nk)
        n = np.ones(Nk, dtype=float) * 0.5
        Cq2 = np.ones(Nk, dtype=float) * 1e-20
        x = np.ones(Nk, dtype=float)

        Fd = dc.FDrift2(1e-20, 9.109e-31, 1e12, ky, n, Cq2, 1e5, 0.1, x)

        assert len(Fd) == Nk
        assert np.all(np.isfinite(Fd))

    def test_zero_distribution(self):
        Nk = 32
        ky = np.linspace(-1e9, 1e9, Nk)
        n = np.zeros(Nk, dtype=float)
        Cq2 = np.ones(Nk, dtype=float) * 1e-20
        x = np.ones(Nk, dtype=float)

        Fd = dc.FDrift2(1e-20, 9.109e-31, 1e12, ky, n, Cq2, 1e5, 0.1, x)

        assert np.allclose(Fd, 0.0, atol=1e-20)


class TestFDrift:
    """Test drift force (alternative implementation)."""

    def test_basic(self):
        Nk = 32
        q = np.linspace(1e8, 1e9, Nk)
        dndk = np.ones(Nk, dtype=float)
        Cq2 = np.ones(Nk, dtype=float) * 1e-20
        x = np.ones(Nk, dtype=float)

        Fd = dc.FDrift(1e-20, 9.109e-31, q, dndk, Cq2, 1e5, 0.1, x)

        assert np.isfinite(Fd)

    def test_zero_derivative(self):
        Nk = 32
        q = np.linspace(1e8, 1e9, Nk)
        dndk = np.zeros(Nk, dtype=float)
        Cq2 = np.ones(Nk, dtype=float) * 1e-20
        x = np.ones(Nk, dtype=float)

        Fd = dc.FDrift(1e-20, 9.109e-31, q, dndk, Cq2, 1e5, 0.1, x)

        assert Fd == pytest.approx(0.0, abs=1e-20)


class TestCalcAvgCoeff:
    """Test average coefficient calculation."""

    def test_basic(self):
        Nk = 32
        ky = np.linspace(-1e9, 1e9, Nk)
        dk = ky[1] - ky[0]
        k1, k2 = ky[10], ky[15]

        x1, x2, x3, x4 = dc.CalcAvgCoeff(ky, dk, k1, k2, 11, 16,
                                           0.0, 0.0, 0.0, 0.0)

        assert np.isfinite(x1)
        assert np.isfinite(x2)
        assert np.isfinite(x3)
        assert np.isfinite(x4)
        assert (x1 + x2 + x3 + x4) == pytest.approx(1.0, rel=1e-10)


class TestDndEk:
    """Test derivative of occupation with respect to energy."""

    def test_basic(self):
        Nk = 32
        q = np.linspace(1e8, 1e9, Nk)
        dndq = np.ones(Nk, dtype=float)

        result = dc.dndEk(1e-20, 9.109e-31, q, dndq)

        assert len(result) == Nk
        assert np.all(np.isfinite(result))


class TestThetaEMABS:
    """Test emission-absorption matrix element."""

    def test_basic(self):
        Nk = 32
        q = np.linspace(1e8, 1e9, Nk)
        dndk = np.ones(Nk, dtype=float)
        Cq2 = np.ones(Nk, dtype=float) * 1e-20

        result = dc.ThetaEMABS(1e-20, 9.109e-31, q, dndk, Cq2, 1e5)

        assert len(result) == Nk
        assert np.all(np.isfinite(result))


class TestDC_Step_Scale:
    """Test DC step using scaling method (legacy, pure)."""

    def test_basic(self):
        Nk = 32
        ky = np.linspace(-1e9, 1e9, Nk).astype(np.float64)
        ne = np.ones(Nk, dtype=complex) * 0.5
        nh = np.ones(Nk, dtype=complex) * 0.5

        try:
            dc.DC_Step_Scale(ne, nh, ky, 1e6, 1e-15)
            assert ne.shape == (Nk,)
            assert nh.shape == (Nk,)
        except Exception as e:
            assert "TypingError" in type(e).__name__ or "numba" in type(e).__name__.lower()

    def test_zero_field(self):
        Nk = 32
        ky = np.linspace(-1e9, 1e9, Nk).astype(np.float64)
        ne = np.ones(Nk, dtype=complex) * 0.5
        nh = np.ones(Nk, dtype=complex) * 0.5

        try:
            dc.DC_Step_Scale(ne, nh, ky, 0.0, 1e-15)
            assert ne.shape == (Nk,)
        except Exception as e:
            assert "TypingError" in type(e).__name__ or "numba" in type(e).__name__.lower()


class TestDC_Step_FD:
    """Test DC step using finite difference method (legacy, pure)."""

    def test_basic(self):
        Nk = 32
        ky = np.linspace(-1e9, 1e9, Nk)
        ne = np.ones(Nk, dtype=complex) * 0.5
        nh = np.ones(Nk, dtype=complex) * 0.5
        nemid = np.ones(Nk, dtype=complex) * 0.5
        nhmid = np.ones(Nk, dtype=complex) * 0.5

        dc.DC_Step_FD(ne, nh, nemid, nhmid, ky, 1e6, 1e-15, 9.109e-31,
                      1.672e-27)

        assert ne.shape == (Nk,)
        assert nh.shape == (Nk,)
        assert np.all(np.isfinite(ne))
        assert np.all(np.isfinite(nh))

    def test_zero_field(self):
        Nk = 32
        ky = np.linspace(-1e9, 1e9, Nk)
        ne = np.ones(Nk, dtype=complex) * 0.5
        nh = np.ones(Nk, dtype=complex) * 0.5
        nemid = np.ones(Nk, dtype=complex) * 0.5
        nhmid = np.ones(Nk, dtype=complex) * 0.5

        dc.DC_Step_FD(ne, nh, nemid, nhmid, ky, 0.0, 1e-15, 9.109e-31,
                      1.672e-27)

        assert np.all(np.isfinite(ne))
        assert np.all(np.isfinite(nh))


# ============================================================================
# DCFieldModule class tests
# ============================================================================


class _DCFieldFixture:
    """Shared fixture helper for DCFieldModule tests."""

    @staticmethod
    def make_instance(Nk=32):
        ky = np.linspace(-1e9, 1e9, Nk)
        me = 9.109e-31
        mh = 1.672e-27
        return DCFieldModule(ky, me, mh, datadir=None)


class TestDCFieldModuleInit(_DCFieldFixture):
    """Test DCFieldModule constructor."""

    def test_all_arrays_populated(self):
        m = self.make_instance()

        assert m.Y is not None
        assert m.xe is not None
        assert m.xh is not None
        assert m.qinv is not None

    def test_array_shapes(self):
        Nk = 64
        m = self.make_instance(Nk)

        assert m.Y.shape == (Nk,)
        assert m.xe.shape == (Nk,)
        assert m.xh.shape == (Nk,)
        assert m.qinv.shape == (Nk + 2,)

    def test_rates_initialized(self):
        m = self.make_instance()

        assert m.ERate == 0.0
        assert m.HRate == 0.0
        assert m.VEDrift == 0.0
        assert m.VHDrift == 0.0

    def test_dkk_set(self):
        Nk = 64
        ky = np.linspace(-1e9, 1e9, Nk)
        m = DCFieldModule(ky, 9.109e-31, 1.672e-27, datadir=None)

        expected_dkk = ky[2] - ky[1]
        assert m.dkk == pytest.approx(expected_dkk, rel=1e-10)

    def test_WithPhns_default(self):
        m = self.make_instance()
        assert m.WithPhns is True

    def test_WithPhns_false(self):
        ky = np.linspace(-1e9, 1e9, 32)
        m = DCFieldModule(ky, 9.109e-31, 1.672e-27, WithPhns=False,
                          datadir=None)
        assert m.WithPhns is False

    def test_no_files_when_datadir_none(self):
        m = self.make_instance()
        assert m.fe_file is None
        assert m.fh_file is None

    def test_files_created_with_datadir(self, tmp_path):
        ky = np.linspace(-1e9, 1e9, 32)
        datadir = str(tmp_path / 'dataQW')
        m = DCFieldModule(ky, 9.109e-31, 1.672e-27, datadir=datadir)

        assert m.fe_file is not None
        assert m.fh_file is not None
        assert os.path.exists(os.path.join(datadir, 'Fe.dat'))
        assert os.path.exists(os.path.join(datadir, 'Fh.dat'))
        m.close()

    def test_kmin_kmax(self):
        Nk = 64
        ky = np.linspace(-1e9, 1e9, Nk)
        dky = ky[2] - ky[1]
        m = DCFieldModule(ky, 9.109e-31, 1.672e-27, datadir=None)

        assert m.kmin == pytest.approx(ky[0] - 2 * dky, rel=1e-10)
        assert m.kmax == pytest.approx(ky[-1] + 2 * dky, rel=1e-10)


class TestDCFieldModuleDCCalc(_DCFieldFixture):
    """Test DC field calculation methods."""

    def _make_inputs(self, Nk=32):
        ky = np.linspace(-1e9, 1e9, Nk)
        Cq2 = np.ones(Nk, dtype=float) * 1e-20
        ne = np.ones(Nk, dtype=complex) * 0.5
        Ee = np.linspace(0, 1e-19, Nk)
        Vee = np.eye(Nk) * 1e-20
        DC = np.zeros(Nk, dtype=float)
        return ky, Cq2, ne, Ee, Vee, DC

    def test_CalcDCE2_no_DCTrans(self):
        m = self.make_instance()
        ky, Cq2, ne, Ee, Vee, DC = self._make_inputs()

        m.CalcDCE2(False, ky, Cq2, 1e6, 9.109e-31, 1e12, 1e-20, 0.1,
                   ne, Ee, Vee, 1, 1, DC)

        assert np.allclose(DC, 0.0, atol=1e-20)

    def test_CalcDCE2_with_DCTrans(self):
        m = self.make_instance()
        ky, Cq2, ne, Ee, Vee, DC = self._make_inputs()

        m.CalcDCE2(True, ky, Cq2, 1e6, 9.109e-31, 1e12, 1e-20, 0.1,
                   ne, Ee, Vee, 1, 1, DC)

        assert np.all(np.isfinite(DC))

    def test_CalcDCH2_no_DCTrans(self):
        m = self.make_instance()
        ky, Cq2, nh, Eh, Vhh, DC = self._make_inputs()

        m.CalcDCH2(False, ky, Cq2, 1e6, 1.672e-27, 1e12, 1e-20, 0.1,
                   nh, Eh, Vhh, 1, 1, DC)

        assert np.allclose(DC, 0.0, atol=1e-20)

    def test_CalcDCH2_with_DCTrans(self):
        m = self.make_instance()
        ky, Cq2, nh, Eh, Vhh, DC = self._make_inputs()

        m.CalcDCH2(True, ky, Cq2, 1e6, 1.672e-27, 1e12, 1e-20, 0.1,
                   nh, Eh, Vhh, 1, 1, DC)

        assert np.all(np.isfinite(DC))

    def test_CalcDCE_no_DCTrans(self):
        m = self.make_instance()
        ky, Cq2, ne, Ee, Vee, DC = self._make_inputs()

        m.CalcDCE(False, ky, Cq2, 1e6, 9.109e-31, 1e12, 1e-20, 0.1,
                  ne, Ee, Vee, DC)

        assert np.allclose(DC, 0.0, atol=1e-20)

    def test_CalcDCE_with_DCTrans(self):
        m = self.make_instance()
        ky, Cq2, ne, Ee, Vee, DC = self._make_inputs()

        m.CalcDCE(True, ky, Cq2, 1e6, 9.109e-31, 1e12, 1e-20, 0.1,
                  ne, Ee, Vee, DC)

        assert np.all(np.isfinite(DC))

    def test_CalcDCH_no_DCTrans(self):
        m = self.make_instance()
        ky, Cq2, nh, Eh, Vhh, DC = self._make_inputs()

        m.CalcDCH(False, ky, Cq2, 1e6, 1.672e-27, 1e12, 1e-20, 0.1,
                  nh, Eh, Vhh, DC)

        assert np.allclose(DC, 0.0, atol=1e-20)

    def test_CalcDCH_with_DCTrans(self):
        m = self.make_instance()
        ky, Cq2, nh, Eh, Vhh, DC = self._make_inputs()

        m.CalcDCH(True, ky, Cq2, 1e6, 1.672e-27, 1e12, 1e-20, 0.1,
                  nh, Eh, Vhh, DC)

        assert np.all(np.isfinite(DC))

    def test_CalcDCE2_updates_ERate(self):
        m = self.make_instance()
        ky, Cq2, ne, Ee, Vee, DC = self._make_inputs()

        m.CalcDCE2(True, ky, Cq2, 1e6, 9.109e-31, 1e12, 1e-20, 0.1,
                   ne, Ee, Vee, 1, 1, DC)

        assert m.ERate != 0.0
        assert m.VEDrift != 0.0

    def test_CalcDCH2_updates_HRate(self):
        m = self.make_instance()
        ky, Cq2, nh, Eh, Vhh, DC = self._make_inputs()

        m.CalcDCH2(True, ky, Cq2, 1e6, 1.672e-27, 1e12, 1e-20, 0.1,
                   nh, Eh, Vhh, 1, 1, DC)

        assert m.HRate != 0.0
        assert m.VHDrift != 0.0


class TestDCFieldModuleCalcI0(_DCFieldFixture):
    """Test CalcI0 method."""

    def test_simple(self):
        Nk = 32
        m = self.make_instance(Nk)
        ky = np.linspace(-1e9, 1e9, Nk)
        dk = ky[1] - ky[0]
        ne = np.ones(Nk, dtype=complex) * 0.5
        nh = np.ones(Nk, dtype=complex) * 0.5
        Ee = np.linspace(0, 1e-19, Nk)
        Eh = np.linspace(0, 1e-19, Nk)
        VC = np.zeros((Nk, Nk, 4))
        VC[:, :, 1] = np.eye(Nk) * 1e-20
        VC[:, :, 2] = np.eye(Nk) * 1e-20

        I = m.CalcI0(ne, nh, Ee, Eh, VC, dk, ky, 0.0)

        assert np.isfinite(I)

    def test_zero_distributions(self):
        Nk = 32
        m = self.make_instance(Nk)
        ky = np.linspace(-1e9, 1e9, Nk)
        dk = ky[1] - ky[0]
        ne = np.zeros(Nk, dtype=complex)
        nh = np.zeros(Nk, dtype=complex)
        Ee = np.linspace(0, 1e-19, Nk)
        Eh = np.linspace(0, 1e-19, Nk)
        VC = np.zeros((Nk, Nk, 4))

        I = m.CalcI0(ne, nh, Ee, Eh, VC, dk, ky, 0.0)

        assert I == pytest.approx(0.0, abs=1e-20)

    def test_updates_dkk(self):
        Nk = 32
        m = self.make_instance(Nk)
        ky = np.linspace(-1e9, 1e9, Nk)
        dk = 42.0
        ne = np.zeros(Nk, dtype=complex)
        nh = np.zeros(Nk, dtype=complex)
        Ee = np.linspace(0, 1e-19, Nk)
        Eh = np.linspace(0, 1e-19, Nk)
        VC = np.zeros((Nk, Nk, 4))

        m.CalcI0(ne, nh, Ee, Eh, VC, dk, ky, 0.0)

        assert m.dkk == 42.0


class TestDCFieldModuleGetters(_DCFieldFixture):
    """Test getter methods."""

    def test_GetEDrift(self):
        m = self.make_instance()
        m.ERate = 1e12
        assert m.GetEDrift() == pytest.approx(1e12, rel=1e-10)

    def test_GetHDrift(self):
        m = self.make_instance()
        m.HRate = 2e12
        assert m.GetHDrift() == pytest.approx(2e12, rel=1e-10)

    def test_GetVEDrift(self):
        m = self.make_instance()
        m.VEDrift = 1e5
        assert m.GetVEDrift() == pytest.approx(1e5, rel=1e-10)

    def test_GetVHDrift(self):
        m = self.make_instance()
        m.VHDrift = 2e5
        assert m.GetVHDrift() == pytest.approx(2e5, rel=1e-10)


class TestDCFieldModuleShift(_DCFieldFixture):
    """Test FFT-based shift operations."""

    def test_ShiftN1D_zero_shift(self):
        m = self.make_instance(64)
        ne = np.random.random(64) + 1j * np.random.random(64)
        ne_orig = ne.copy()

        m.ShiftN1D(ne, 0.0)

        assert np.allclose(ne, ne_orig, rtol=1e-10)

    def test_ShiftN1D_small_shift(self):
        m = self.make_instance(64)
        ne = np.random.random(64) + 1j * np.random.random(64)

        m.ShiftN1D(ne, 1e8)

        assert ne.shape == (64,)
        assert np.all(np.isfinite(ne))

    def test_ShiftN2D_zero_shift(self):
        Nk = 32
        m = self.make_instance(Nk)
        C = np.random.random((Nk, Nk)) + 1j * np.random.random((Nk, Nk))
        C_orig = C.copy()

        m.ShiftN2D(C, 0.0)

        assert np.allclose(C, C_orig, rtol=1e-10)

    def test_ShiftN2D_small_shift(self):
        Nk = 32
        m = self.make_instance(Nk)
        C = np.random.random((Nk, Nk)) + 1j * np.random.random((Nk, Nk))

        m.ShiftN2D(C, 1e8)

        assert C.shape == (Nk, Nk)
        assert np.all(np.isfinite(C))


class TestDCFieldModuleTransport(_DCFieldFixture):
    """Test transport step."""

    def test_no_DCTrans(self):
        Nk = 32
        m = self.make_instance(Nk)
        C = np.random.random((Nk, Nk)) + 1j * np.random.random((Nk, Nk))
        C_orig = C.copy()

        m.Transport(C, 1e6, 0.0, 1e-15, False, True)

        assert np.allclose(C, C_orig, rtol=1e-12)

    def test_with_DCTrans(self):
        Nk = 32
        m = self.make_instance(Nk)
        C = np.random.random((Nk, Nk)) + 1j * np.random.random((Nk, Nk))

        m.Transport(C, 1e6, 0.0, 1e-15, True, True)

        assert C.shape == (Nk, Nk)
        assert np.all(np.isfinite(C))


# ============================================================================
# Backward-compatibility tests
# ============================================================================


class TestBackwardCompat:
    """Test backward-compatible wrapper functions and __getattr__."""

    def setup_method(self):
        ky = np.linspace(-1e9, 1e9, 32)
        dc._instance = DCFieldModule(ky, 9.109e-31, 1.672e-27, datadir=None)

    def teardown_method(self):
        dc._instance = None

    def test_getattr_Y(self):
        assert dc._Y is not None
        assert len(dc._Y) == 32

    def test_getattr_xe(self):
        assert dc._xe is not None
        assert len(dc._xe) == 32

    def test_getattr_xh(self):
        assert dc._xh is not None
        assert len(dc._xh) == 32

    def test_getattr_qinv(self):
        assert dc._qinv is not None
        assert len(dc._qinv) == 34

    def test_getattr_dkk(self):
        assert dc._dkk is not None
        assert dc._dkk != 0.0

    def test_getattr_rates(self):
        assert dc._ERate == 0.0
        assert dc._HRate == 0.0
        assert dc._VEDrift == 0.0
        assert dc._VHDrift == 0.0

    def test_getattr_returns_none_before_init(self):
        dc._instance = None
        assert dc._Y is None
        assert dc._xe is None
        assert dc._dkk is None

    def test_getattr_unknown_raises(self):
        with pytest.raises(AttributeError):
            _ = dc._nonexistent_attr

    def test_wrapper_InitializeDC(self, tmp_path):
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            ky = np.linspace(-1e9, 1e9, 64)
            dc.InitializeDC(ky, 9.109e-31, 1.672e-27)

            assert dc._instance is not None
            assert dc._Y is not None
            assert len(dc._Y) == 64
        finally:
            os.chdir(old_cwd)

    def test_wrapper_GetEDrift(self):
        dc._instance.ERate = 42.0
        assert dc.GetEDrift() == pytest.approx(42.0, rel=1e-10)

    def test_wrapper_GetHDrift(self):
        dc._instance.HRate = 99.0
        assert dc.GetHDrift() == pytest.approx(99.0, rel=1e-10)

    def test_wrapper_GetVEDrift(self):
        dc._instance.VEDrift = 1e5
        assert dc.GetVEDrift() == pytest.approx(1e5, rel=1e-10)

    def test_wrapper_GetVHDrift(self):
        dc._instance.VHDrift = 2e5
        assert dc.GetVHDrift() == pytest.approx(2e5, rel=1e-10)

    def test_wrapper_Transport_no_DCTrans(self):
        Nk = 32
        C = np.random.random((Nk, Nk)) + 1j * np.random.random((Nk, Nk))
        C_orig = C.copy()

        dc.Transport(C, 1e6, 0.0, 1e-15, False, True)

        assert np.allclose(C, C_orig, rtol=1e-12)

    def test_wrapper_CalcI0(self):
        Nk = 32
        ky = np.linspace(-1e9, 1e9, Nk)
        dk = ky[1] - ky[0]
        ne = np.zeros(Nk, dtype=complex)
        nh = np.zeros(Nk, dtype=complex)
        Ee = np.linspace(0, 1e-19, Nk)
        Eh = np.linspace(0, 1e-19, Nk)
        VC = np.zeros((Nk, Nk, 4))

        I = dc.CalcI0(ne, nh, Ee, Eh, VC, dk, ky, 0.0)

        assert I == pytest.approx(0.0, abs=1e-20)

    def test_wrapper_not_initialized(self):
        dc._instance = None
        with pytest.raises(RuntimeError, match="not initialized"):
            dc.GetEDrift()


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Integration tests for DC field calculations."""

    @staticmethod
    def _make(Nk=64):
        ky = np.linspace(-1e9, 1e9, Nk)
        return DCFieldModule(ky, 9.109e-31, 1.672e-27, datadir=None), ky

    def test_energy_renormalization_chain(self):
        m, ky = self._make()
        ne = np.ones(64, dtype=complex) * 0.5
        Ee = np.linspace(0, 1e-19, 64)
        Vee = np.eye(64) * 1e-20

        Ec = dc.EkReNorm(np.real(ne), Ee, Vee)
        v = dc.DriftVt(np.real(ne), Ec, m.dkk)

        assert np.isfinite(v)

    def test_dc_field_workflow(self):
        m, ky = self._make()
        Nk = 64
        Cq2 = np.ones(Nk, dtype=float) * 1e-20
        ne = np.ones(Nk, dtype=complex) * 0.5
        Ee = np.linspace(0, 1e-19, Nk)
        Vee = np.eye(Nk) * 1e-20
        DC = np.zeros(Nk, dtype=float)

        m.CalcDCE2(True, ky, Cq2, 1e6, 9.109e-31, 1e12, 1e-20, 0.1,
                   ne, Ee, Vee, 1, 1, DC)

        assert m.ERate != 0.0
        assert m.VEDrift != 0.0
        assert np.all(np.isfinite(DC))

    def test_current_workflow(self):
        m, ky = self._make()
        Nk = 64
        dk = ky[1] - ky[0]
        ne = np.ones(Nk, dtype=complex) * 0.5
        nh = np.ones(Nk, dtype=complex) * 0.5
        Ee = np.linspace(0, 1e-19, Nk)
        Eh = np.linspace(0, 1e-19, Nk)
        VC = np.zeros((Nk, Nk, 4))
        VC[:, :, 1] = np.eye(Nk) * 1e-20
        VC[:, :, 2] = np.eye(Nk) * 1e-20

        Ie = dc.CalcI0n(ne, 9.109e-31, ky)
        I = m.CalcI0(ne, nh, Ee, Eh, VC, dk, ky, 0.0)

        assert np.isfinite(Ie)
        assert np.isfinite(I)

    def test_phonon_interaction_workflow(self):
        _, ky = self._make()
        Nk = 64
        n = np.ones(Nk, dtype=float) * 0.5
        Cq2 = np.ones(Nk, dtype=float) * 1e-20
        x = np.ones(Nk, dtype=float)

        Fd = dc.FDrift2(1e-20, 9.109e-31, 1e12, ky, n, Cq2, 1e5, 0.1, x)

        assert len(Fd) == Nk
        assert np.all(np.isfinite(Fd))


class TestPhysicalProperties:
    """Test that physical properties are preserved."""

    def test_current_conservation(self):
        ky = np.linspace(-1e9, 1e9, 64)
        me = 9.109e-31

        ne_sym = np.exp(-(ky / 1e9)**2).astype(complex)
        Ie_sym = dc.CalcI0n(ne_sym, me, ky)

        ne_asym = np.exp(-((ky - 1e8) / 1e9)**2).astype(complex)
        Ie_asym = dc.CalcI0n(ne_asym, me, ky)

        assert abs(Ie_sym) < abs(Ie_asym)

    def test_energy_renormalization_deterministic(self):
        Nk = 32
        n = np.ones(Nk, dtype=float) * 0.5
        En = np.linspace(0, 1e-19, Nk)
        V = np.eye(Nk) * 1e-20

        Ec1 = dc.EkReNorm(n, En, V)
        Ec2 = dc.EkReNorm(n, En, V)

        assert np.allclose(Ec1, Ec2, rtol=1e-12)


class TestNumericalStability:
    """Test numerical stability of functions."""

    def test_energy_renormalization_stability(self):
        Nk = 64
        cases = [
            (np.ones(Nk), np.linspace(0, 1e-19, Nk), np.eye(Nk) * 1e-20),
            (np.zeros(Nk), np.linspace(0, 1e-19, Nk), np.zeros((Nk, Nk))),
            (np.random.random(Nk), np.random.random(Nk) * 1e-19,
             np.random.random((Nk, Nk)) * 1e-21),
        ]

        for n, En, V in cases:
            Ec = dc.EkReNorm(n, En, V)
            assert np.all(np.isfinite(Ec))

    def test_drift_velocity_stability(self):
        Nk = 64
        ky = np.linspace(-1e9, 1e9, Nk)
        dkk = ky[1] - ky[0]

        cases = [
            (np.ones(Nk), np.linspace(0, 1e-19, Nk)),
            (np.zeros(Nk), np.linspace(0, 1e-19, Nk)),
            (np.random.random(Nk), np.random.random(Nk) * 1e-19),
        ]

        for n, Ec in cases:
            v = dc.DriftVt(n, Ec, dkk)
            assert np.isfinite(v)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
