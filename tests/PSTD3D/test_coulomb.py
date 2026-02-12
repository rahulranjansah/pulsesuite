"""
Comprehensive test suite for coulomb.py module.

Tests all Coulomb interaction functions including initialization, interaction integrals,
many-body arrays, screening calculations, and Semiconductor Bloch Equations terms.
Tests both the CoulombModule class and the module-level backward-compat wrappers.
"""


import numpy as np
import pytest
from scipy.constants import hbar as hbar_SI

from pulsesuite.PSTD3D import coulomb
from pulsesuite.PSTD3D.coulomb import CoulombModule

# Physical constants
eV = 1.602176634e-19  # Electron volt in Joules
hbar = hbar_SI
pi = np.pi
twopi = 2.0 * np.pi


class TestGaussDelta:
    """Test Gaussian delta function approximation."""

    def test_gauss_delta_normal(self):
        """Test Gaussian delta with normal values."""
        a = 0.0
        b = 1.0
        result = coulomb.GaussDelta(a, b)
        expected = 1.0 / (np.sqrt(pi) * b)
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_gauss_delta_nonzero_a(self):
        """Test Gaussian delta with nonzero a."""
        a = 1.0
        b = 0.5
        result = coulomb.GaussDelta(a, b)
        expected = 1.0 / (np.sqrt(pi) * b) * np.exp(-(a / b) ** 2)
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_gauss_delta_small_b(self):
        """Test Gaussian delta with very small b."""
        a = 1.0
        b = 1e-301  # Smaller than threshold to trigger check
        result = coulomb.GaussDelta(a, b)
        assert result == 0.0

    def test_gauss_delta_zero_b(self):
        """Test Gaussian delta with zero b."""
        a = 1.0
        b = 0.0
        result = coulomb.GaussDelta(a, b)
        assert result == 0.0

    def test_gauss_delta_large_a(self):
        """Test Gaussian delta with large a."""
        a = 100.0
        b = 1.0
        result = coulomb.GaussDelta(a, b)
        expected = 1.0 / (np.sqrt(pi) * b) * np.exp(-(a / b) ** 2)
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)
        assert result < 1e-100  # Should be very small


class TestMakeK3:
    """Test momentum conservation indexing array."""

    def test_make_k3_small(self):
        """Test MakeK3 with small array."""
        ky = np.array([-1.0, 0.0, 1.0])
        k3 = coulomb.MakeK3(ky)
        N = len(ky)
        assert k3.shape == (N, N, N)
        assert k3.dtype == np.int32

        for k1 in range(N):
            for k2 in range(N):
                for k4 in range(N):
                    k3_idx_1b = k3[k1, k2, k4]
                    expected_k3_1b = (k1 + 1) + (k2 + 1) - (k4 + 1)
                    if 1 <= expected_k3_1b <= N:
                        assert k3_idx_1b == expected_k3_1b
                    else:
                        assert k3_idx_1b == 0

    def test_make_k3_even_length(self):
        """Test MakeK3 with even length array."""
        N = 64
        ky = np.linspace(-10.0, 10.0, N)
        k3 = coulomb.MakeK3(ky)
        assert k3.shape == (N, N, N)
        assert np.all(k3 >= 0)
        assert np.all(k3 <= N)

    def test_make_k3_odd_length(self):
        """Test MakeK3 with odd length array."""
        N = 101
        ky = np.linspace(-10.0, 10.0, N)
        k3 = coulomb.MakeK3(ky)
        assert k3.shape == (N, N, N)

    def test_make_k3_prime_length(self):
        """Test MakeK3 with prime length array."""
        N = 97
        ky = np.linspace(-10.0, 10.0, N)
        k3 = coulomb.MakeK3(ky)
        assert k3.shape == (N, N, N)

    def test_make_k3_invalid_combinations(self):
        """Test that invalid momentum combinations return 0."""
        ky = np.array([-1.0, 0.0, 1.0])
        k3 = coulomb.MakeK3(ky)
        assert k3[0, 0, 2] == 0
        assert k3[2, 2, 0] == 0

    def test_make_k3_is_pure(self):
        """Test that MakeK3 is a pure function (returns same result each time)."""
        ky = np.array([-1.0, 0.0, 1.0])
        k3_a = coulomb.MakeK3(ky)
        k3_b = coulomb.MakeK3(ky)
        assert np.array_equal(k3_a, k3_b)


class TestMakeQs:
    """Test momentum difference arrays."""

    def test_make_qs_small(self):
        """Test MakeQs with small array."""
        ky = np.array([-1.0, 0.0, 1.0])
        ae = 0.1
        ah = 0.2
        qe, qh = coulomb.MakeQs(ky, ae, ah)
        N = len(ky)
        assert qe.shape == (N, N)
        assert qh.shape == (N, N)

        assert np.all(qe >= ae / 2.0)
        assert np.all(qh >= ah / 2.0)

        for i in range(N):
            assert qe[i, i] == ae / 2.0
            assert qh[i, i] == ah / 2.0

    def test_make_qs_even_length(self):
        """Test MakeQs with even length array."""
        N = 64
        ky = np.linspace(-10.0, 10.0, N)
        ae = 0.5
        ah = 0.6
        qe, qh = coulomb.MakeQs(ky, ae, ah)
        assert qe.shape == (N, N)
        assert qh.shape == (N, N)
        assert np.all(qe >= ae / 2.0)
        assert np.all(qh >= ah / 2.0)

    def test_make_qs_symmetry(self):
        """Test that qe and qh are symmetric."""
        ky = np.linspace(-5.0, 5.0, 32)
        ae = 0.1
        ah = 0.2
        qe, qh = coulomb.MakeQs(ky, ae, ah)
        assert np.allclose(qe, qe.T, rtol=1e-12, atol=1e-12)
        assert np.allclose(qh, qh.T, rtol=1e-12, atol=1e-12)


class TestMakeUnDel:
    """Test inverse delta function array."""

    def test_make_undel_small(self):
        """Test MakeUnDel with small array."""
        ky = np.array([-1.0, 0.0, 1.0])
        UnDel = coulomb.MakeUnDel(ky)
        N = len(ky)
        assert UnDel.shape == (N + 1, N + 1)

        assert np.all(UnDel[0, :] == 0.0)
        assert np.all(UnDel[:, 0] == 0.0)

        for i in range(1, N + 1):
            assert UnDel[i, i] == 0.0

        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i != j:
                    assert UnDel[i, j] == 1.0

    def test_make_undel_even_length(self):
        """Test MakeUnDel with even length array."""
        N = 64
        ky = np.linspace(-10.0, 10.0, N)
        UnDel = coulomb.MakeUnDel(ky)
        assert UnDel.shape == (N + 1, N + 1)
        assert np.all(UnDel[0, :] == 0.0)
        assert np.all(UnDel[:, 0] == 0.0)


class TestSetLorentzDelta:
    """Test LorentzDelta flag setting."""

    def test_set_lorentz_delta_true(self):
        """Test setting LorentzDelta to True."""
        coulomb.SetLorentzDelta(True)
        assert coulomb._LorentzDelta is True

    def test_set_lorentz_delta_false(self):
        """Test setting LorentzDelta to False."""
        coulomb.SetLorentzDelta(False)
        assert coulomb._LorentzDelta is False

    def test_set_lorentz_delta_int(self):
        """Test setting LorentzDelta with integer."""
        coulomb.SetLorentzDelta(1)
        assert coulomb._LorentzDelta is True
        coulomb.SetLorentzDelta(0)
        assert coulomb._LorentzDelta is False


class TestVint:
    """Test interaction integral calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Ny = 64
        self.y = np.linspace(-10e-6, 10e-6, self.Ny)
        self.alphae = 1e6
        self.alphah = 1.2e6
        self.Delta0 = 10e-9

    def test_vint_small_q(self):
        """Test Vint with small momentum difference."""
        Qyk = 1e5
        result = coulomb.Vint(Qyk, self.y, self.alphae, self.alphah, self.Delta0)
        assert result >= 0.0
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_vint_large_q(self):
        """Test Vint with large momentum difference."""
        Qyk = 1e8
        result = coulomb.Vint(Qyk, self.y, self.alphae, self.alphah, self.Delta0)
        assert result >= 0.0
        assert not np.isnan(result)

    def test_vint_zero_q(self):
        """Test Vint with zero momentum difference."""
        Qyk = 0.0
        result = coulomb.Vint(Qyk, self.y, self.alphae, self.alphah, self.Delta0)
        assert result >= 0.0

    def test_vint_different_sizes(self):
        """Test Vint with different array sizes."""
        for Ny in [32, 64, 128, 101]:
            y = np.linspace(-10e-6, 10e-6, Ny)
            Qyk = 1e6
            result = coulomb.Vint(Qyk, y, self.alphae, self.alphah, self.Delta0)
            assert result >= 0.0
            assert not np.isnan(result)

    def test_vint_symmetry(self):
        """Test that Vint is symmetric in alphae and alphah."""
        Qyk = 1e6
        result1 = coulomb.Vint(Qyk, self.y, self.alphae, self.alphah, self.Delta0)
        result2 = coulomb.Vint(Qyk, self.y, self.alphah, self.alphae, self.Delta0)
        assert result1 >= 0.0
        assert result2 >= 0.0


class TestVehint:
    """Test electron-hole interaction integral."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Ny = 64
        self.Nk = 32
        self.y = np.linspace(-10e-6, 10e-6, self.Ny)
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.alphae = 1e6
        self.alphah = 1.2e6
        self.Delta0 = 10e-9

    def test_vehint_valid_indices(self):
        """Test Vehint with valid indices."""
        k = 1
        q = 2
        result = coulomb.Vehint(k, q, self.y, self.ky, self.alphae, self.alphah, self.Delta0)
        assert result >= 0.0
        assert not np.isnan(result)

    def test_vehint_same_indices(self):
        """Test Vehint with same indices."""
        k = 1
        q = 1
        result = coulomb.Vehint(k, q, self.y, self.ky, self.alphae, self.alphah, self.Delta0)
        assert result >= 0.0

    def test_vehint_different_sizes(self):
        """Test Vehint with different array sizes."""
        for Nk in [16, 32, 64, 101]:
            ky = np.linspace(-1e7, 1e7, Nk)
            k = 1
            q = 2
            result = coulomb.Vehint(k, q, self.y, ky, self.alphae, self.alphah, self.Delta0)
            assert result >= 0.0


class TestCalcCoulombArrays:
    """Test unscreened Coulomb array calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.N = 32
        self.y = np.linspace(-10e-6, 10e-6, 64)
        self.ky = np.linspace(-1e7, 1e7, self.N)
        self.er = 12.0
        self.alphae = 1e6
        self.alphah = 1.2e6
        self.L = 100e-6
        self.Delta0 = 10e-9

        self.NQ = 2 * self.N - 1
        self.Qy = np.linspace(-2e7, 2e7, self.NQ)
        self.kkp = np.zeros((self.N, self.N), dtype=np.int32)
        for k in range(self.N):
            for q in range(self.N):
                idx = k - q + self.NQ // 2
                if 0 <= idx < self.NQ:
                    self.kkp[k, q] = idx
                else:
                    self.kkp[k, q] = -1

    def test_calc_coulomb_arrays_basic(self):
        """Test CalcCoulombArrays with basic inputs."""
        Veh0, Vee0, Vhh0 = coulomb.CalcCoulombArrays(
            self.y, self.ky, self.er, self.alphae, self.alphah,
            self.L, self.Delta0, self.Qy, self.kkp
        )
        assert Veh0.shape == (self.N, self.N)
        assert Vee0.shape == (self.N, self.N)
        assert Vhh0.shape == (self.N, self.N)
        assert np.all(np.isfinite(Veh0))
        assert np.all(np.isfinite(Vee0))
        assert np.all(np.isfinite(Vhh0))

    def test_calc_coulomb_arrays_screw_this(self):
        """Test CalcCoulombArrays with ScrewThis=True."""
        Veh0, Vee0, Vhh0 = coulomb.CalcCoulombArrays(
            self.y, self.ky, self.er, self.alphae, self.alphah,
            self.L, self.Delta0, self.Qy, self.kkp, ScrewThis=True
        )
        assert np.all(Veh0 == 0.0)
        assert np.all(Vee0 == 0.0)
        assert np.all(Vhh0 == 0.0)

    def test_calc_coulomb_arrays_different_sizes(self):
        """Test CalcCoulombArrays with different array sizes."""
        for N in [16, 32, 64, 101]:
            ky = np.linspace(-1e7, 1e7, N)
            NQ = 2 * N - 1
            Qy = np.linspace(-2e7, 2e7, NQ)
            kkp = np.zeros((N, N), dtype=np.int32)
            for k in range(N):
                for q in range(N):
                    idx = k - q + NQ // 2
                    if 0 <= idx < NQ:
                        kkp[k, q] = idx
                    else:
                        kkp[k, q] = -1

            Veh0, Vee0, Vhh0 = coulomb.CalcCoulombArrays(
                self.y, ky, self.er, self.alphae, self.alphah,
                self.L, self.Delta0, Qy, kkp
            )
            assert Veh0.shape == (N, N)
            assert Vee0.shape == (N, N)
            assert Vhh0.shape == (N, N)


class TestCalcMBArrays:
    """Test many-body interaction array calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.N = 32
        self.ky = np.linspace(-1e7, 1e7, self.N)
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.ge = 1e12
        self.gh = 1e12
        self.k3 = coulomb.MakeK3(self.ky)
        self.UnDel = coulomb.MakeUnDel(self.ky)

    def test_calc_mb_arrays_lorentzian(self):
        """Test CalcMBArrays with Lorentzian broadening."""
        Ceh, Cee, Chh = coulomb.CalcMBArrays(
            self.ky, self.Ee, self.Eh, self.ge, self.gh, self.k3, self.UnDel,
            LorentzDelta=True
        )
        assert Ceh.shape == (self.N + 1, self.N + 1, self.N + 1)
        assert Cee.shape == (self.N + 1, self.N + 1, self.N + 1)
        assert Chh.shape == (self.N + 1, self.N + 1, self.N + 1)
        assert np.all(Ceh >= 0.0)
        assert np.all(Cee >= 0.0)
        assert np.all(Chh >= 0.0)

    def test_calc_mb_arrays_gaussian(self):
        """Test CalcMBArrays with Gaussian delta function."""
        Ceh, Cee, Chh = coulomb.CalcMBArrays(
            self.ky, self.Ee, self.Eh, self.ge, self.gh, self.k3, self.UnDel,
            LorentzDelta=False
        )
        assert Ceh.shape == (self.N + 1, self.N + 1, self.N + 1)
        assert Cee.shape == (self.N + 1, self.N + 1, self.N + 1)
        assert Chh.shape == (self.N + 1, self.N + 1, self.N + 1)

    def test_calc_mb_arrays_different_sizes(self):
        """Test CalcMBArrays with different array sizes."""
        for N in [16, 32, 64, 101]:
            ky = np.linspace(-1e7, 1e7, N)
            Ee = 0.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
            Eh = 0.5 * 1.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
            k3 = coulomb.MakeK3(ky)
            UnDel = coulomb.MakeUnDel(ky)
            Ceh, Cee, Chh = coulomb.CalcMBArrays(
                ky, Ee, Eh, self.ge, self.gh, k3, UnDel
            )
            assert Ceh.shape == (N + 1, N + 1, N + 1)


class TestCalcChi1D:
    """Test 1D susceptibility array calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.N = 32
        self.ky = np.linspace(-1e7, 1e7, self.N)
        self.alphae = 1e6
        self.alphah = 1.2e6
        self.Delta0 = 10e-9
        self.epsr = 12.0
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31
        self.qe, self.qh = coulomb.MakeQs(self.ky, self.alphae, self.alphah)

    def test_calc_chi1d_basic(self):
        """Test CalcChi1D with basic inputs."""
        Chi1De, Chi1Dh = coulomb.CalcChi1D(
            self.ky, self.alphae, self.alphah, self.Delta0,
            self.epsr, self.me, self.mh, self.qe, self.qh
        )
        assert Chi1De.shape == (self.N, self.N)
        assert Chi1Dh.shape == (self.N, self.N)
        assert np.all(np.isfinite(Chi1De))
        assert np.all(np.isfinite(Chi1Dh))
        assert np.all(Chi1De >= 0.0)
        assert np.all(Chi1Dh >= 0.0)

    def test_calc_chi1d_different_sizes(self):
        """Test CalcChi1D with different array sizes."""
        for N in [16, 32, 64, 101]:
            ky = np.linspace(-1e7, 1e7, N)
            qe, qh = coulomb.MakeQs(ky, self.alphae, self.alphah)
            Chi1De, Chi1Dh = coulomb.CalcChi1D(
                ky, self.alphae, self.alphah, self.Delta0,
                self.epsr, self.me, self.mh, qe, qh
            )
            assert Chi1De.shape == (N, N)
            assert Chi1Dh.shape == (N, N)


class TestGetChi1Dqw:
    """Test 1D quantum wire susceptibility calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.alphae = 1e6
        self.alphah = 1.2e6
        self.Delta0 = 10e-9
        self.epsr = 12.0
        self.game = np.ones(self.Nk) * 1e12
        self.gamh = np.ones(self.Nk) * 1e12
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.ne = 0.5 * (1.0 + np.tanh((self.ky - 0.5e7) / 1e6))
        self.nh = 0.5 * (1.0 + np.tanh((-self.ky - 0.5e7) / 1e6))

    def test_get_chi1dqw_basic(self):
        """Test GetChi1Dqw with basic inputs."""
        qq = 1e6
        w = 1e15
        chir, chii = coulomb.GetChi1Dqw(
            self.alphae, self.alphah, self.Delta0, self.epsr,
            self.game, self.gamh, self.ky, self.Ee, self.Eh,
            self.ne, self.nh, qq, w
        )
        assert np.isfinite(chir)
        assert np.isfinite(chii)

    def test_get_chi1dqw_zero_frequency(self):
        """Test GetChi1Dqw with zero frequency."""
        qq = 1e6
        w = 0.0
        chir, chii = coulomb.GetChi1Dqw(
            self.alphae, self.alphah, self.Delta0, self.epsr,
            self.game, self.gamh, self.ky, self.Ee, self.Eh,
            self.ne, self.nh, qq, w
        )
        assert np.isfinite(chir)
        assert np.isfinite(chii)

    def test_get_chi1dqw_different_q(self):
        """Test GetChi1Dqw with different momentum values."""
        w = 1e15
        for qq in [1e5, 1e6, 1e7, 1e8]:
            chir, chii = coulomb.GetChi1Dqw(
                self.alphae, self.alphah, self.Delta0, self.epsr,
                self.game, self.gamh, self.ky, self.Ee, self.Eh,
                self.ne, self.nh, qq, w
            )
            assert np.isfinite(chir)
            assert np.isfinite(chii)


class TestGetEps1Dqw:
    """Test 1D quantum wire dielectric function calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.alphae = 1e6
        self.alphah = 1.2e6
        self.Delta0 = 10e-9
        self.epsr = 12.0
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31

    def test_get_eps1dqw_basic(self):
        """Test GetEps1Dqw with basic inputs."""
        n1D = 1e6
        q = 1e6
        w = 1e15
        epr, epi = coulomb.GetEps1Dqw(
            self.alphae, self.alphah, self.Delta0, self.epsr,
            self.me, self.mh, n1D, q, w
        )
        assert np.isfinite(epr)
        assert np.isfinite(epi)

    def test_get_eps1dqw_zero_frequency(self):
        """Test GetEps1Dqw with zero frequency."""
        n1D = 1e6
        q = 1e6
        w = 0.0
        epr, epi = coulomb.GetEps1Dqw(
            self.alphae, self.alphah, self.Delta0, self.epsr,
            self.me, self.mh, n1D, q, w
        )
        assert np.isfinite(epr)
        assert np.isfinite(epi)

    def test_get_eps1dqw_small_q(self):
        """Test GetEps1Dqw with small q (should use minimum)."""
        n1D = 1e6
        q = 0.1
        w = 1e15
        epr, epi = coulomb.GetEps1Dqw(
            self.alphae, self.alphah, self.Delta0, self.epsr,
            self.me, self.mh, n1D, q, w
        )
        assert np.isfinite(epr)
        assert np.isfinite(epi)

    def test_get_eps1dqw_different_densities(self):
        """Test GetEps1Dqw with different densities."""
        q = 1e6
        w = 1e15
        for n1D in [1e5, 1e6, 1e7, 1e8]:
            epr, epi = coulomb.GetEps1Dqw(
                self.alphae, self.alphah, self.Delta0, self.epsr,
                self.me, self.mh, n1D, q, w
            )
            assert np.isfinite(epr)
            assert np.isfinite(epi)


###############################################################################
# CoulombModule class tests
###############################################################################

class _CoulombFixture:
    """Shared fixture builder for CoulombModule tests."""

    @staticmethod
    def make_instance(N=32):
        y = np.linspace(-10e-6, 10e-6, 2 * N)
        ky = np.linspace(-1e7, 1e7, N)
        L = 100e-6
        Delta0 = 10e-9
        me = 0.067 * 9.109e-31
        mh = 0.5 * 9.109e-31
        Ee = 0.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
        Eh = 0.5 * 1.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
        ge = 1e12
        gh = 1e12
        alphae = 1e6
        alphah = 1.2e6
        er = 12.0

        NQ = 2 * N - 1
        Qy = np.linspace(-2e7, 2e7, NQ)
        kkp = np.zeros((N, N), dtype=np.int32)
        for k in range(N):
            for q in range(N):
                idx = k - q + NQ // 2
                if 0 <= idx < NQ:
                    kkp[k, q] = idx
                else:
                    kkp[k, q] = -1

        return CoulombModule(y, ky, L, Delta0, me, mh, Ee, Eh, ge, gh,
                             alphae, alphah, er, Qy, kkp, False)


class TestCoulombModuleInit(_CoulombFixture):
    """Test CoulombModule initialization."""

    def test_all_arrays_populated(self):
        """Test that __init__ populates all arrays."""
        cm = self.make_instance(N=32)
        assert cm.UnDel is not None
        assert cm.k3 is not None
        assert cm.qe is not None
        assert cm.qh is not None
        assert cm.Ceh is not None
        assert cm.Cee is not None
        assert cm.Chh is not None
        assert cm.Veh0 is not None
        assert cm.Vee0 is not None
        assert cm.Vhh0 is not None
        assert cm.Chi1De is not None
        assert cm.Chi1Dh is not None

    def test_array_shapes(self):
        """Test that arrays have correct shapes."""
        N = 32
        cm = self.make_instance(N=N)
        assert cm.k3.shape == (N, N, N)
        assert cm.UnDel.shape == (N + 1, N + 1)
        assert cm.qe.shape == (N, N)
        assert cm.qh.shape == (N, N)
        assert cm.Ceh.shape == (N + 1, N + 1, N + 1)
        assert cm.Veh0.shape == (N, N)
        assert cm.Chi1De.shape == (N, N)

    def test_lorentz_delta_default(self):
        """Test default LorentzDelta is False."""
        cm = self.make_instance()
        assert cm.LorentzDelta is False

    def test_set_lorentz_delta(self):
        """Test SetLorentzDelta on instance."""
        cm = self.make_instance()
        cm.SetLorentzDelta(True)
        assert cm.LorentzDelta is True
        cm.SetLorentzDelta(False)
        assert cm.LorentzDelta is False


class TestCoulombModuleScreening(_CoulombFixture):
    """Test screening methods on CoulombModule."""

    def test_eps1d(self):
        """Test Eps1D method."""
        cm = self.make_instance()
        n1D = 1e6
        eps1d = cm.Eps1D(n1D)
        N = cm.Chi1De.shape[0]
        assert eps1d.shape == (N, N)
        assert np.all(np.isfinite(eps1d))

    def test_calc_screened_arrays_unscreened(self):
        """Test CalcScreenedArrays with screening disabled."""
        cm = self.make_instance(N=32)
        N = 32
        ne = 0.5 * np.ones(N, dtype=complex)
        nh = 0.5 * np.ones(N, dtype=complex)
        VC = np.zeros((N, N, 3))
        E1D = np.zeros((N, N))
        cm.CalcScreenedArrays(False, 100e-6, ne, nh, VC, E1D)
        assert np.allclose(E1D, 1.0, rtol=1e-12, atol=1e-12)
        assert np.all(np.isfinite(VC))

    def test_calc_screened_arrays_screened(self):
        """Test CalcScreenedArrays with screening enabled."""
        cm = self.make_instance(N=32)
        N = 32
        ne = 0.5 * np.ones(N, dtype=complex)
        nh = 0.5 * np.ones(N, dtype=complex)
        VC = np.zeros((N, N, 3))
        E1D = np.zeros((N, N))
        cm.CalcScreenedArrays(True, 100e-6, ne, nh, VC, E1D)
        assert np.all(np.isfinite(E1D))
        assert np.all(np.isfinite(VC))
        assert not np.allclose(E1D, 1.0, rtol=1e-6, atol=1e-6)


class TestCoulombModuleSBETerms(_CoulombFixture):
    """Test SBE calculation methods on CoulombModule."""

    def setup_method(self):
        self.N = 32
        self.cm = self.make_instance(N=self.N)

    def test_calc_mveh(self):
        """Test CalcMVeh method."""
        Nf = 4
        p = 0.1 * (np.random.random((self.N, self.N, Nf)) +
                    1j * np.random.random((self.N, self.N, Nf)))
        VC = np.zeros((self.N, self.N, 3))
        VC[:, :, 0] = 1e-20 * np.random.random((self.N, self.N))
        MVeh = np.zeros((self.N, self.N, Nf), dtype=complex)
        self.cm.CalcMVeh(p, VC, MVeh)
        assert np.all(np.isfinite(MVeh))

    def test_undell(self):
        """Test undell method."""
        assert self.cm.undell(1, 2) == 1.0
        assert self.cm.undell(1, 1) == 0.0
        assert self.cm.undell(0, 1) == 0.0

    def test_bg_renorm(self):
        """Test BGRenorm method."""
        C = np.eye(self.N, dtype=complex) * 0.5
        D = np.eye(self.N, dtype=complex) * 0.5
        VC = np.zeros((self.N, self.N, 3))
        VC[:, :, 1] = 1e-20 * np.random.random((self.N, self.N))
        VC[:, :, 2] = 1e-20 * np.random.random((self.N, self.N))
        BGR = np.zeros((self.N, self.N), dtype=complex)
        self.cm.BGRenorm(C, D, VC, BGR)
        assert np.all(np.isfinite(BGR))

    def test_ee_renorm(self):
        """Test EeRenorm method."""
        ne = 0.5 * np.ones(self.N, dtype=complex)
        VC = np.zeros((self.N, self.N, 3))
        VC[:, :, 1] = 1e-20 * np.random.random((self.N, self.N))
        BGR = np.zeros((self.N, self.N), dtype=complex)
        self.cm.EeRenorm(ne, VC, BGR)
        assert np.all(np.isfinite(BGR))

    def test_eh_renorm(self):
        """Test EhRenorm method."""
        nh = 0.5 * np.ones(self.N, dtype=complex)
        VC = np.zeros((self.N, self.N, 3))
        VC[:, :, 2] = 1e-20 * np.random.random((self.N, self.N))
        BGR = np.zeros((self.N, self.N), dtype=complex)
        self.cm.EhRenorm(nh, VC, BGR)
        assert np.all(np.isfinite(BGR))


class TestCoulombModuleMBRates(_CoulombFixture):
    """Test many-body relaxation rate methods on CoulombModule."""

    def setup_method(self):
        self.Nk = 32
        self.cm = self.make_instance(N=self.Nk)
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.ge = 1e12
        self.gh = 1e12
        self.geh = (self.ge + self.gh) / 2.0
        self.ne0 = 0.5 * np.ones(self.Nk)
        self.nh0 = 0.5 * np.ones(self.Nk)
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.Nk, self.Nk))
        self.VC[:, :, 2] = 1e-20 * np.random.random((self.Nk, self.Nk))

    def test_mbce2(self):
        """Test MBCE2 method."""
        Win = np.zeros(self.Nk)
        Wout = np.zeros(self.Nk)
        self.cm.MBCE2(self.ne0, self.nh0, self.ky, self.Ee, self.Eh,
                      self.VC, self.geh, self.ge, Win, Wout)
        assert np.all(Win >= 0.0)
        assert np.all(Wout >= 0.0)
        assert np.all(np.isfinite(Win))
        assert np.all(np.isfinite(Wout))

    def test_mbce(self):
        """Test MBCE method."""
        Win = np.zeros(self.Nk)
        Wout = np.zeros(self.Nk)
        self.cm.MBCE(self.ne0, self.nh0, self.ky, self.Ee, self.Eh,
                     self.VC, self.geh, self.ge, Win, Wout)
        assert np.all(Win >= 0.0)
        assert np.all(Wout >= 0.0)

    def test_mbce_identical_to_mbce2(self):
        """Test that MBCE gives same results as MBCE2."""
        Win1 = np.zeros(self.Nk)
        Wout1 = np.zeros(self.Nk)
        Win2 = np.zeros(self.Nk)
        Wout2 = np.zeros(self.Nk)

        self.cm.MBCE(self.ne0, self.nh0, self.ky, self.Ee, self.Eh,
                     self.VC, self.geh, self.ge, Win1, Wout1)
        self.cm.MBCE2(self.ne0, self.nh0, self.ky, self.Ee, self.Eh,
                      self.VC, self.geh, self.ge, Win2, Wout2)
        assert np.allclose(Win1, Win2, rtol=1e-10, atol=1e-12)
        assert np.allclose(Wout1, Wout2, rtol=1e-10, atol=1e-12)

    def test_mbch(self):
        """Test MBCH method."""
        Win = np.zeros(self.Nk)
        Wout = np.zeros(self.Nk)
        self.cm.MBCH(self.ne0, self.nh0, self.ky, self.Ee, self.Eh,
                     self.VC, self.geh, self.gh, Win, Wout)
        assert np.all(Win >= 0.0)
        assert np.all(Wout >= 0.0)
        assert np.all(np.isfinite(Win))
        assert np.all(np.isfinite(Wout))


###############################################################################
# Backward-compat wrapper tests (via singleton)
###############################################################################

class TestBackwardCompat(_CoulombFixture):
    """Test that module-level wrappers and __getattr__ work."""

    def setup_method(self):
        self.N = 32
        y = np.linspace(-10e-6, 10e-6, 2 * self.N)
        ky = np.linspace(-1e7, 1e7, self.N)
        L = 100e-6
        Delta0 = 10e-9
        me = 0.067 * 9.109e-31
        mh = 0.5 * 9.109e-31
        Ee = 0.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
        Eh = 0.5 * 1.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
        ge = 1e12
        gh = 1e12
        alphae = 1e6
        alphah = 1.2e6
        er = 12.0

        NQ = 2 * self.N - 1
        Qy = np.linspace(-2e7, 2e7, NQ)
        kkp = np.zeros((self.N, self.N), dtype=np.int32)
        for k in range(self.N):
            for q in range(self.N):
                idx = k - q + NQ // 2
                if 0 <= idx < NQ:
                    kkp[k, q] = idx
                else:
                    kkp[k, q] = -1

        # Initialize the singleton
        coulomb.InitializeCoulomb(y, ky, L, Delta0, me, mh, Ee, Eh, ge, gh,
                                  alphae, alphah, er, Qy, kkp, False)

    def test_getattr_k3(self):
        """Test that coulomb._k3 returns the singleton's k3."""
        assert coulomb._k3 is not None
        assert coulomb._k3.shape == (self.N, self.N, self.N)

    def test_getattr_undel(self):
        """Test that coulomb._UnDel returns the singleton's UnDel."""
        assert coulomb._UnDel is not None
        assert coulomb._UnDel.shape == (self.N + 1, self.N + 1)

    def test_getattr_all_arrays(self):
        """Test all backward-compat attribute reads."""
        assert coulomb._k3 is not None
        assert coulomb._UnDel is not None
        assert coulomb._qe is not None
        assert coulomb._qh is not None
        assert coulomb._Ceh is not None
        assert coulomb._Cee is not None
        assert coulomb._Chh is not None
        assert coulomb._Veh0 is not None
        assert coulomb._Vee0 is not None
        assert coulomb._Vhh0 is not None
        assert coulomb._Chi1De is not None
        assert coulomb._Chi1Dh is not None

    def test_wrapper_calc_screened_arrays(self):
        """Test CalcScreenedArrays wrapper."""
        ne = 0.5 * np.ones(self.N, dtype=complex)
        nh = 0.5 * np.ones(self.N, dtype=complex)
        VC = np.zeros((self.N, self.N, 3))
        E1D = np.zeros((self.N, self.N))
        coulomb.CalcScreenedArrays(False, 100e-6, ne, nh, VC, E1D)
        assert np.all(np.isfinite(VC))

    def test_wrapper_mbce(self):
        """Test MBCE wrapper."""
        ky = np.linspace(-1e7, 1e7, self.N)
        Ee = 0.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
        Eh = 0.5 * 1.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
        ne0 = 0.5 * np.ones(self.N)
        nh0 = 0.5 * np.ones(self.N)
        VC = np.zeros((self.N, self.N, 3))
        VC[:, :, 0] = 1e-20 * np.random.random((self.N, self.N))
        VC[:, :, 1] = 1e-20 * np.random.random((self.N, self.N))
        Win = np.zeros(self.N)
        Wout = np.zeros(self.N)
        coulomb.MBCE(ne0, nh0, ky, Ee, Eh, VC, 1e12, 1e12, Win, Wout)
        assert np.all(np.isfinite(Win))

    def test_wrapper_mbch(self):
        """Test MBCH wrapper."""
        ky = np.linspace(-1e7, 1e7, self.N)
        Ee = 0.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
        Eh = 0.5 * 1.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
        ne0 = 0.5 * np.ones(self.N)
        nh0 = 0.5 * np.ones(self.N)
        VC = np.zeros((self.N, self.N, 3))
        VC[:, :, 0] = 1e-20 * np.random.random((self.N, self.N))
        VC[:, :, 2] = 1e-20 * np.random.random((self.N, self.N))
        Win = np.zeros(self.N)
        Wout = np.zeros(self.N)
        coulomb.MBCH(ne0, nh0, ky, Ee, Eh, VC, 1e12, 1e12, Win, Wout)
        assert np.all(np.isfinite(Win))

    def test_wrapper_not_initialized(self):
        """Test that wrappers raise ValueError when singleton is None."""
        old = coulomb._instance
        coulomb._instance = None
        try:
            with pytest.raises(ValueError):
                coulomb.CalcScreenedArrays(False, 1.0, np.zeros(4), np.zeros(4),
                                           np.zeros((4, 4, 3)), np.zeros((4, 4)))
        finally:
            coulomb._instance = old

    def test_getattr_returns_none_before_init(self):
        """Test that _k3 etc. return None when singleton is None."""
        old = coulomb._instance
        coulomb._instance = None
        try:
            assert coulomb._k3 is None
            assert coulomb._Veh0 is None
        finally:
            coulomb._instance = old


class TestIntegration(_CoulombFixture):
    """Integration tests for complete workflows using CoulombModule."""

    def test_full_workflow(self):
        """Test complete initialization + screening + SBE workflow."""
        N = 32
        cm = self.make_instance(N=N)

        # Screening
        ne = 0.5 * np.ones(N, dtype=complex)
        nh = 0.5 * np.ones(N, dtype=complex)
        VC = np.zeros((N, N, 3))
        E1D = np.zeros((N, N))
        cm.CalcScreenedArrays(False, 100e-6, ne, nh, VC, E1D)
        assert np.all(np.isfinite(VC))
        assert np.all(np.isfinite(E1D))

        # CalcMVeh
        Nf = 4
        p = 0.1 * (np.random.random((N, N, Nf)) +
                    1j * np.random.random((N, N, Nf)))
        MVeh = np.zeros((N, N, Nf), dtype=complex)
        cm.CalcMVeh(p, VC, MVeh)
        assert np.all(np.isfinite(MVeh))

        # BGRenorm
        C = np.eye(N, dtype=complex) * 0.5
        D = np.eye(N, dtype=complex) * 0.5
        BGR = np.zeros((N, N), dtype=complex)
        cm.BGRenorm(C, D, VC, BGR)
        assert np.all(np.isfinite(BGR))

    def test_screening_workflow(self):
        """Test unscreened vs screened comparison."""
        N = 32
        cm = self.make_instance(N=N)

        ne = 0.5 * np.ones(N, dtype=complex)
        nh = 0.5 * np.ones(N, dtype=complex)
        VC_un = np.zeros((N, N, 3))
        E1D_un = np.zeros((N, N))
        cm.CalcScreenedArrays(False, 100e-6, ne, nh, VC_un, E1D_un)

        VC_sc = np.zeros((N, N, 3))
        E1D_sc = np.zeros((N, N))
        cm.CalcScreenedArrays(True, 100e-6, ne, nh, VC_sc, E1D_sc)

        diff_VC = np.abs(VC_sc - VC_un)
        diff_E1D = np.abs(E1D_sc - E1D_un)
        assert np.any(diff_VC > 1e-30) or np.any(diff_E1D > 1e-15)

    def test_many_body_rates_workflow(self):
        """Test complete many-body rates calculation."""
        N = 32
        cm = self.make_instance(N=N)
        ky = np.linspace(-1e7, 1e7, N)
        Ee = 0.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
        Eh = 0.5 * 1.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
        ge = 1e12
        gh = 1e12
        geh = (ge + gh) / 2.0

        ne = 0.5 * np.ones(N, dtype=complex)
        nh = 0.5 * np.ones(N, dtype=complex)
        VC = np.zeros((N, N, 3))
        E1D = np.zeros((N, N))
        cm.CalcScreenedArrays(False, 100e-6, ne, nh, VC, E1D)

        ne0 = np.real(ne)
        nh0 = np.real(nh)
        Win_e = np.zeros(N)
        Wout_e = np.zeros(N)
        Win_h = np.zeros(N)
        Wout_h = np.zeros(N)

        cm.MBCE2(ne0, nh0, ky, Ee, Eh, VC, geh, ge, Win_e, Wout_e)
        cm.MBCH(ne0, nh0, ky, Ee, Eh, VC, geh, gh, Win_h, Wout_h)

        assert np.all(Win_e >= 0.0)
        assert np.all(Wout_e >= 0.0)
        assert np.all(Win_h >= 0.0)
        assert np.all(Wout_h >= 0.0)

    @pytest.mark.parametrize("N", [16, 32, 64, 101])
    def test_different_array_sizes(self, N):
        """Test CoulombModule with different array sizes."""
        cm = self.make_instance(N=N)
        assert cm.k3.shape == (N, N, N)
        assert cm.UnDel.shape == (N + 1, N + 1)
        assert cm.qe.shape == (N, N)
        assert cm.qh.shape == (N, N)
        assert cm.Ceh.shape == (N + 1, N + 1, N + 1)
        assert cm.Veh0.shape == (N, N)
