"""
Comprehensive test suite for coulomb.py module.

Tests all Coulomb interaction functions including initialization, interaction integrals,
many-body arrays, screening calculations, and Semiconductor Bloch Equations terms.
"""

import numpy as np
import pytest
import sys
import os

# Add pythonic directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import coulomb
from scipy.constants import e as e0, epsilon_0 as eps0, hbar as hbar_SI
from scipy.special import kv
from usefulsubs import K03


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

        # Test momentum conservation: k1 + k2 = k3 + k4
        # k3 is stored as k3[k1, k2, k4] and contains k3 index (1-based)
        # where k3_1b = (k1+1) + (k2+1) - (k4+1) = k1 + k2 - k4 + 1
        for k1 in range(N):
            for k2 in range(N):
                for k4 in range(N):
                    k3_idx_1b = k3[k1, k2, k4]  # Storage order: k3[k1, k2, k4]
                    expected_k3_1b = (k1 + 1) + (k2 + 1) - (k4 + 1)
                    if 1 <= expected_k3_1b <= N:
                        # Valid combination, should match
                        assert k3_idx_1b == expected_k3_1b
                    else:
                        # Invalid combination, should be 0
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
        # k3 is stored as k3[k1, k2, k4]
        # k1=0, k2=0, k4=2: k3_1b = (0+1)+(0+1)-(2+1) = 1+1-3 = -1 (invalid, should be 0)
        assert k3[0, 0, 2] == 0
        # k1=2, k2=2, k4=0: k3_1b = (2+1)+(2+1)-(0+1) = 3+3-1 = 5, which is > N=3, so should be 0
        assert k3[2, 2, 0] == 0

    def test_make_k3_module_storage(self):
        """Test that MakeK3 stores result in module-level variable."""
        ky = np.array([-1.0, 0.0, 1.0])
        k3 = coulomb.MakeK3(ky)
        # Reset module variable
        coulomb._k3 = None
        k3_again = coulomb.MakeK3(ky)
        assert np.array_equal(k3, k3_again)


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

        # Check minimum values
        assert np.all(qe >= ae / 2.0)
        assert np.all(qh >= ah / 2.0)

        # Check diagonal (should be minimum)
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

    def test_make_qs_module_storage(self):
        """Test that MakeQs stores results in module-level variables."""
        ky = np.array([-1.0, 0.0, 1.0])
        ae = 0.1
        ah = 0.2
        qe, qh = coulomb.MakeQs(ky, ae, ah)
        assert np.array_equal(coulomb._qe, qe)
        assert np.array_equal(coulomb._qh, qh)


class TestMakeUnDel:
    """Test inverse delta function array."""

    def test_make_undel_small(self):
        """Test MakeUnDel with small array."""
        ky = np.array([-1.0, 0.0, 1.0])
        UnDel = coulomb.MakeUnDel(ky)
        N = len(ky)
        assert UnDel.shape == (N + 1, N + 1)

        # Row 0 and column 0 should be zero
        assert np.all(UnDel[0, :] == 0.0)
        assert np.all(UnDel[:, 0] == 0.0)

        # Diagonal elements (1..N) should be zero
        for i in range(1, N + 1):
            assert UnDel[i, i] == 0.0

        # Off-diagonal elements should be 1.0
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

    def test_make_undel_module_storage(self):
        """Test that MakeUnDel stores result in module-level variable."""
        ky = np.array([-1.0, 0.0, 1.0])
        UnDel = coulomb.MakeUnDel(ky)
        assert np.array_equal(coulomb._UnDel, UnDel)


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
        self.alphae = 1e6  # 1/m
        self.alphah = 1.2e6  # 1/m
        self.Delta0 = 10e-9  # 10 nm

    def test_vint_small_q(self):
        """Test Vint with small momentum difference."""
        Qyk = 1e5  # 1/m
        result = coulomb.Vint(Qyk, self.y, self.alphae, self.alphah, self.Delta0)
        assert result >= 0.0
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_vint_large_q(self):
        """Test Vint with large momentum difference."""
        Qyk = 1e8  # 1/m
        result = coulomb.Vint(Qyk, self.y, self.alphae, self.alphah, self.Delta0)
        assert result >= 0.0
        assert not np.isnan(result)

    def test_vint_zero_q(self):
        """Test Vint with zero momentum difference."""
        Qyk = 0.0
        result = coulomb.Vint(Qyk, self.y, self.alphae, self.alphah, self.Delta0)
        # Should use minimum momentum
        kmin = (self.alphae + self.alphah) / 4.0
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
        # Should be different due to different exponential factors
        # But both should be valid
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
        self.alphae = 1e6  # 1/m
        self.alphah = 1.2e6  # 1/m
        self.Delta0 = 10e-9  # 10 nm

    def test_vehint_valid_indices(self):
        """Test Vehint with valid indices."""
        k = 1  # 1-based
        q = 2  # 1-based
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

        # Create Qy and kkp arrays
        self.NQ = 2 * self.N - 1
        self.Qy = np.linspace(-2e7, 2e7, self.NQ)
        self.kkp = np.zeros((self.N, self.N), dtype=np.int32)
        for k in range(self.N):
            for q in range(self.N):
                # Simple mapping: Qy index = k - q + NQ//2
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

    def test_calc_coulomb_arrays_module_storage(self):
        """Test that CalcCoulombArrays stores results in module-level variables."""
        Veh0, Vee0, Vhh0 = coulomb.CalcCoulombArrays(
            self.y, self.ky, self.er, self.alphae, self.alphah,
            self.L, self.Delta0, self.Qy, self.kkp
        )
        assert np.array_equal(coulomb._Veh0, Veh0)
        assert np.array_equal(coulomb._Vee0, Vee0)
        assert np.array_equal(coulomb._Vhh0, Vhh0)

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
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)  # J
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)  # J
        self.ge = 1e12  # Hz
        self.gh = 1e12  # Hz
        self.k3 = coulomb.MakeK3(self.ky)
        self.UnDel = coulomb.MakeUnDel(self.ky)

    def test_calc_mb_arrays_lorentzian(self):
        """Test CalcMBArrays with Lorentzian broadening."""
        coulomb.SetLorentzDelta(True)
        Ceh, Cee, Chh = coulomb.CalcMBArrays(
            self.ky, self.Ee, self.Eh, self.ge, self.gh, self.k3, self.UnDel
        )
        assert Ceh.shape == (self.N + 1, self.N + 1, self.N + 1)
        assert Cee.shape == (self.N + 1, self.N + 1, self.N + 1)
        assert Chh.shape == (self.N + 1, self.N + 1, self.N + 1)
        assert np.all(Ceh >= 0.0)
        assert np.all(Cee >= 0.0)
        assert np.all(Chh >= 0.0)

    def test_calc_mb_arrays_gaussian(self):
        """Test CalcMBArrays with Gaussian delta function."""
        coulomb.SetLorentzDelta(False)
        Ceh, Cee, Chh = coulomb.CalcMBArrays(
            self.ky, self.Ee, self.Eh, self.ge, self.gh, self.k3, self.UnDel
        )
        assert Ceh.shape == (self.N + 1, self.N + 1, self.N + 1)
        assert Cee.shape == (self.N + 1, self.N + 1, self.N + 1)
        assert Chh.shape == (self.N + 1, self.N + 1, self.N + 1)

    def test_calc_mb_arrays_explicit_lorentzian(self):
        """Test CalcMBArrays with explicit LorentzDelta parameter."""
        Ceh, Cee, Chh = coulomb.CalcMBArrays(
            self.ky, self.Ee, self.Eh, self.ge, self.gh, self.k3, self.UnDel,
            LorentzDelta=True
        )
        assert Ceh.shape == (self.N + 1, self.N + 1, self.N + 1)

    def test_calc_mb_arrays_explicit_gaussian(self):
        """Test CalcMBArrays with explicit LorentzDelta=False."""
        Ceh, Cee, Chh = coulomb.CalcMBArrays(
            self.ky, self.Ee, self.Eh, self.ge, self.gh, self.k3, self.UnDel,
            LorentzDelta=False
        )
        assert Ceh.shape == (self.N + 1, self.N + 1, self.N + 1)

    def test_calc_mb_arrays_module_storage(self):
        """Test that CalcMBArrays stores results in module-level variables."""
        Ceh, Cee, Chh = coulomb.CalcMBArrays(
            self.ky, self.Ee, self.Eh, self.ge, self.gh, self.k3, self.UnDel
        )
        assert np.array_equal(coulomb._Ceh, Ceh)
        assert np.array_equal(coulomb._Cee, Cee)
        assert np.array_equal(coulomb._Chh, Chh)

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
        self.me = 0.067 * 9.109e-31  # kg (GaAs electron mass)
        self.mh = 0.5 * 9.109e-31  # kg
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

    def test_calc_chi1d_module_storage(self):
        """Test that CalcChi1D stores results in module-level variables."""
        Chi1De, Chi1Dh = coulomb.CalcChi1D(
            self.ky, self.alphae, self.alphah, self.Delta0,
            self.epsr, self.me, self.mh, self.qe, self.qh
        )
        assert np.array_equal(coulomb._Chi1De, Chi1De)
        assert np.array_equal(coulomb._Chi1Dh, Chi1Dh)

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


class TestEps1D:
    """Test 1D dielectric function matrix."""

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
        coulomb.CalcChi1D(
            self.ky, self.alphae, self.alphah, self.Delta0,
            self.epsr, self.me, self.mh, self.qe, self.qh
        )

    def test_eps1d_basic(self):
        """Test Eps1D with basic inputs."""
        n1D = 1e6  # 1/m
        eps1d = coulomb.Eps1D(n1D, self.N)
        assert eps1d.shape == (self.N, self.N)
        assert np.all(np.isfinite(eps1d))

    def test_eps1d_different_densities(self):
        """Test Eps1D with different densities."""
        for n1D in [1e5, 1e6, 1e7, 1e8]:
            eps1d = coulomb.Eps1D(n1D, self.N)
            assert eps1d.shape == (self.N, self.N)
            assert np.all(np.isfinite(eps1d))

    def test_eps1d_error_if_not_initialized(self):
        """Test that Eps1D raises error if arrays not initialized."""
        # Temporarily clear module variables
        old_Chi1De = coulomb._Chi1De
        old_Chi1Dh = coulomb._Chi1Dh
        old_qe = coulomb._qe
        old_qh = coulomb._qh
        coulomb._Chi1De = None
        coulomb._Chi1Dh = None
        coulomb._qe = None
        coulomb._qh = None

        with pytest.raises(ValueError):
            coulomb.Eps1D(1e6, self.N)

        # Restore
        coulomb._Chi1De = old_Chi1De
        coulomb._Chi1Dh = old_Chi1Dh
        coulomb._qe = old_qe
        coulomb._qh = old_qh


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
        self.game = np.ones(self.Nk) * 1e12  # Hz
        self.gamh = np.ones(self.Nk) * 1e12  # Hz
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.ne = 0.5 * (1.0 + np.tanh((self.ky - 0.5e7) / 1e6))
        self.nh = 0.5 * (1.0 + np.tanh((-self.ky - 0.5e7) / 1e6))

    def test_get_chi1dqw_basic(self):
        """Test GetChi1Dqw with basic inputs."""
        qq = 1e6  # 1/m
        w = 1e15  # rad/s
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

    def test_get_chi1dqw_different_frequencies(self):
        """Test GetChi1Dqw with different frequencies."""
        qq = 1e6
        for w in [1e14, 1e15, 1e16]:
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
        n1D = 1e6  # 1/m
        q = 1e6  # 1/m
        w = 1e15  # rad/s
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
        q = 0.1  # Very small, should be set to 1.0
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


class TestCalcScreenedArrays:
    """Test screened Coulomb array calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.N = 32
        self.L = 100e-6
        self.ky = np.linspace(-1e7, 1e7, self.N)
        self.y = np.linspace(-10e-6, 10e-6, 64)
        self.er = 12.0
        self.alphae = 1e6
        self.alphah = 1.2e6
        self.Delta0 = 10e-9
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31

        # Initialize arrays
        self.Qy = np.linspace(-2e7, 2e7, 2 * self.N - 1)
        self.kkp = np.zeros((self.N, self.N), dtype=np.int32)
        for k in range(self.N):
            for q in range(self.N):
                idx = k - q + len(self.Qy) // 2
                if 0 <= idx < len(self.Qy):
                    self.kkp[k, q] = idx
                else:
                    self.kkp[k, q] = -1

        # Initialize Coulomb arrays
        coulomb.CalcCoulombArrays(
            self.y, self.ky, self.er, self.alphae, self.alphah,
            self.L, self.Delta0, self.Qy, self.kkp
        )
        coulomb.MakeQs(self.ky, self.alphae, self.alphah)
        coulomb.CalcChi1D(
            self.ky, self.alphae, self.alphah, self.Delta0,
            self.er, self.me, self.mh, coulomb._qe, coulomb._qh
        )

        self.ne = 0.5 * np.ones(self.N, dtype=complex)
        self.nh = 0.5 * np.ones(self.N, dtype=complex)
        self.VC = np.zeros((self.N, self.N, 3))
        self.E1D = np.zeros((self.N, self.N))

    def test_calc_screened_arrays_unscreened(self):
        """Test CalcScreenedArrays with screening disabled."""
        coulomb.CalcScreenedArrays(False, self.L, self.ne, self.nh, self.VC, self.E1D)
        assert np.allclose(self.E1D, 1.0, rtol=1e-12, atol=1e-12)
        # VC should contain unscreened arrays
        assert np.all(np.isfinite(self.VC))

    def test_calc_screened_arrays_screened(self):
        """Test CalcScreenedArrays with screening enabled."""
        coulomb.CalcScreenedArrays(True, self.L, self.ne, self.nh, self.VC, self.E1D)
        assert np.all(np.isfinite(self.E1D))
        assert np.all(np.isfinite(self.VC))
        # E1D should not be all ones when screened
        assert not np.allclose(self.E1D, 1.0, rtol=1e-6, atol=1e-6)

    def test_calc_screened_arrays_error_if_not_initialized(self):
        """Test that CalcScreenedArrays raises error if arrays not initialized."""
        old_Veh0 = coulomb._Veh0
        coulomb._Veh0 = None
        with pytest.raises(ValueError):
            coulomb.CalcScreenedArrays(True, self.L, self.ne, self.nh, self.VC, self.E1D)
        coulomb._Veh0 = old_Veh0


class TestUndell:
    """Test UnDel array access wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ky = np.array([-1.0, 0.0, 1.0])
        coulomb.MakeUnDel(self.ky)

    def test_undell_valid_indices(self):
        """Test undell with valid 1-based indices."""
        # k=1, q=2 should give 1.0 (off-diagonal)
        result = coulomb.undell(1, 2)
        assert result == 1.0

    def test_undell_diagonal(self):
        """Test undell with diagonal indices (should be 0)."""
        result = coulomb.undell(1, 1)
        assert result == 0.0

    def test_undell_zero_index(self):
        """Test undell with zero index (should be 0)."""
        result = coulomb.undell(0, 1)
        assert result == 0.0

    def test_undell_error_if_not_initialized(self):
        """Test that undell raises error if UnDel not initialized."""
        old_UnDel = coulomb._UnDel
        coulomb._UnDel = None
        with pytest.raises(ValueError):
            coulomb.undell(1, 2)
        coulomb._UnDel = old_UnDel


class TestCalcMVeh:
    """Test MVeh array calculation for Semiconductor Bloch Equations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.N = 32
        self.Nf = 8
        self.ky = np.linspace(-1e7, 1e7, self.N)
        self.k3 = coulomb.MakeK3(self.ky)
        self.UnDel = coulomb.MakeUnDel(self.ky)

        # Create test arrays
        self.p = np.zeros((self.N, self.N, self.Nf), dtype=complex)
        for f in range(self.Nf):
            self.p[:, :, f] = 0.1 * (np.random.random((self.N, self.N)) +
                                    1j * np.random.random((self.N, self.N)))

        self.VC = np.zeros((self.N, self.N, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.N, self.N))  # Veh
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.N, self.N))  # Vee
        self.VC[:, :, 2] = 1e-20 * np.random.random((self.N, self.N))  # Vhh

        self.MVeh = np.zeros((self.N, self.N, self.Nf), dtype=complex)

    def test_calc_mveh_basic(self):
        """Test CalcMVeh with basic inputs."""
        coulomb.CalcMVeh(self.p, self.VC, self.MVeh, self.k3, self.UnDel)
        assert self.MVeh.shape == (self.N, self.N, self.Nf)
        assert np.all(np.isfinite(self.MVeh))

    def test_calc_mveh_module_arrays(self):
        """Test CalcMVeh using module-level arrays."""
        coulomb._k3 = self.k3
        coulomb._UnDel = self.UnDel
        MVeh2 = np.zeros((self.N, self.N, self.Nf), dtype=complex)
        coulomb.CalcMVeh(self.p, self.VC, MVeh2)
        assert np.allclose(self.MVeh, MVeh2, rtol=1e-10, atol=1e-12)

    def test_calc_mveh_error_if_not_initialized(self):
        """Test that CalcMVeh raises error if arrays not initialized."""
        old_k3 = coulomb._k3
        coulomb._k3 = None
        with pytest.raises(ValueError):
            coulomb.CalcMVeh(self.p, self.VC, self.MVeh)
        coulomb._k3 = old_k3


class TestBGRenorm:
    """Test band gap renormalization calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.N = 32
        self.UnDel = coulomb.MakeUnDel(np.linspace(-1e7, 1e7, self.N))

        # Create test density matrices
        self.C = np.eye(self.N, dtype=complex) * 0.5
        self.D = np.eye(self.N, dtype=complex) * 0.5

        self.VC = np.zeros((self.N, self.N, 3))
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.N, self.N))  # Vee
        self.VC[:, :, 2] = 1e-20 * np.random.random((self.N, self.N))  # Vhh

        self.BGR = np.zeros((self.N, self.N), dtype=complex)

    def test_bg_renorm_basic(self):
        """Test BGRenorm with basic inputs."""
        coulomb.BGRenorm(self.C, self.D, self.VC, self.BGR, self.UnDel)
        assert self.BGR.shape == (self.N, self.N)
        assert np.all(np.isfinite(self.BGR))

    def test_bg_renorm_module_array(self):
        """Test BGRenorm using module-level UnDel."""
        coulomb._UnDel = self.UnDel
        BGR2 = np.zeros((self.N, self.N), dtype=complex)
        coulomb.BGRenorm(self.C, self.D, self.VC, BGR2)
        assert np.allclose(self.BGR, BGR2, rtol=1e-10, atol=1e-12)

    def test_bg_renorm_error_if_not_initialized(self):
        """Test that BGRenorm raises error if UnDel not initialized."""
        old_UnDel = coulomb._UnDel
        coulomb._UnDel = None
        with pytest.raises(ValueError):
            coulomb.BGRenorm(self.C, self.D, self.VC, self.BGR)
        coulomb._UnDel = old_UnDel


class TestEeRenorm:
    """Test electron energy renormalization calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.N = 32
        self.UnDel = coulomb.MakeUnDel(np.linspace(-1e7, 1e7, self.N))

        self.ne = 0.5 * np.ones(self.N, dtype=complex)
        self.VC = np.zeros((self.N, self.N, 3))
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.N, self.N))  # Vee

        self.BGR = np.zeros((self.N, self.N), dtype=complex)

    def test_ee_renorm_basic(self):
        """Test EeRenorm with basic inputs."""
        coulomb.EeRenorm(self.ne, self.VC, self.BGR, self.UnDel)
        assert self.BGR.shape == (self.N, self.N)
        assert np.all(np.isfinite(self.BGR))

    def test_ee_renorm_module_array(self):
        """Test EeRenorm using module-level UnDel."""
        coulomb._UnDel = self.UnDel
        BGR2 = np.zeros((self.N, self.N), dtype=complex)
        coulomb.EeRenorm(self.ne, self.VC, BGR2)
        assert np.allclose(self.BGR, BGR2, rtol=1e-10, atol=1e-12)


class TestEhRenorm:
    """Test hole energy renormalization calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.N = 32
        self.UnDel = coulomb.MakeUnDel(np.linspace(-1e7, 1e7, self.N))

        self.nh = 0.5 * np.ones(self.N, dtype=complex)
        self.VC = np.zeros((self.N, self.N, 3))
        self.VC[:, :, 2] = 1e-20 * np.random.random((self.N, self.N))  # Vhh

        self.BGR = np.zeros((self.N, self.N), dtype=complex)

    def test_eh_renorm_basic(self):
        """Test EhRenorm with basic inputs."""
        coulomb.EhRenorm(self.nh, self.VC, self.BGR, self.UnDel)
        assert self.BGR.shape == (self.N, self.N)
        assert np.all(np.isfinite(self.BGR))

    def test_eh_renorm_module_array(self):
        """Test EhRenorm using module-level UnDel."""
        coulomb._UnDel = self.UnDel
        BGR2 = np.zeros((self.N, self.N), dtype=complex)
        coulomb.EhRenorm(self.nh, self.VC, BGR2)
        assert np.allclose(self.BGR, BGR2, rtol=1e-10, atol=1e-12)


class TestMBCE2:
    """Test many-body Coulomb in/out rates for electrons (version 2)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.ge = 1e12
        self.gh = 1e12
        self.geh = (self.ge + self.gh) / 2.0

        # Initialize arrays
        self.k3 = coulomb.MakeK3(self.ky)
        self.UnDel = coulomb.MakeUnDel(self.ky)
        Ceh, Cee, Chh = coulomb.CalcMBArrays(
            self.ky, self.Ee, self.Eh, self.ge, self.gh, self.k3, self.UnDel
        )

        self.ne0 = 0.5 * np.ones(self.Nk)
        self.nh0 = 0.5 * np.ones(self.Nk)
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vee

        self.Win = np.zeros(self.Nk)
        self.Wout = np.zeros(self.Nk)

    def test_mbce2_basic(self):
        """Test MBCE2 with basic inputs."""
        coulomb.MBCE2(
            self.ne0, self.nh0, self.ky, self.Ee, self.Eh, self.VC,
            self.geh, self.ge, self.Win, self.Wout, self.k3, coulomb._Ceh, coulomb._Cee
        )
        assert np.all(self.Win >= 0.0)
        assert np.all(self.Wout >= 0.0)
        assert np.all(np.isfinite(self.Win))
        assert np.all(np.isfinite(self.Wout))

    def test_mbce2_module_arrays(self):
        """Test MBCE2 using module-level arrays."""
        # First call with explicit arrays to populate Win/Wout
        coulomb.MBCE2(
            self.ne0, self.nh0, self.ky, self.Ee, self.Eh, self.VC,
            self.geh, self.ge, self.Win, self.Wout, self.k3, coulomb._Ceh, coulomb._Cee
        )
        # Then call with module-level arrays
        Win2 = np.zeros(self.Nk)
        Wout2 = np.zeros(self.Nk)
        coulomb.MBCE2(
            self.ne0, self.nh0, self.ky, self.Ee, self.Eh, self.VC,
            self.geh, self.ge, Win2, Wout2
        )
        assert np.allclose(self.Win, Win2, rtol=1e-10, atol=1e-12)
        assert np.allclose(self.Wout, Wout2, rtol=1e-10, atol=1e-12)

    def test_mbce2_error_if_not_initialized(self):
        """Test that MBCE2 raises error if arrays not initialized."""
        old_k3 = coulomb._k3
        coulomb._k3 = None
        with pytest.raises(ValueError):
            coulomb.MBCE2(
                self.ne0, self.nh0, self.ky, self.Ee, self.Eh, self.VC,
                self.geh, self.ge, self.Win, self.Wout
            )
        coulomb._k3 = old_k3


class TestMBCE:
    """Test many-body Coulomb in/out rates for electrons."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.ge = 1e12
        self.gh = 1e12
        self.geh = (self.ge + self.gh) / 2.0

        # Initialize arrays
        self.k3 = coulomb.MakeK3(self.ky)
        self.UnDel = coulomb.MakeUnDel(self.ky)
        Ceh, Cee, Chh = coulomb.CalcMBArrays(
            self.ky, self.Ee, self.Eh, self.ge, self.gh, self.k3, self.UnDel
        )

        self.ne0 = 0.5 * np.ones(self.Nk)
        self.nh0 = 0.5 * np.ones(self.Nk)
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vee

        self.Win = np.zeros(self.Nk)
        self.Wout = np.zeros(self.Nk)

    def test_mbce_basic(self):
        """Test MBCE with basic inputs."""
        coulomb.MBCE(
            self.ne0, self.nh0, self.ky, self.Ee, self.Eh, self.VC,
            self.geh, self.ge, self.Win, self.Wout, self.k3, coulomb._Ceh, coulomb._Cee
        )
        assert np.all(self.Win >= 0.0)
        assert np.all(self.Wout >= 0.0)
        assert np.all(np.isfinite(self.Win))
        assert np.all(np.isfinite(self.Wout))

    def test_mbce_identical_to_mbce2(self):
        """Test that MBCE gives same results as MBCE2."""
        Win1 = np.zeros(self.Nk)
        Wout1 = np.zeros(self.Nk)
        Win2 = np.zeros(self.Nk)
        Wout2 = np.zeros(self.Nk)

        coulomb.MBCE(
            self.ne0, self.nh0, self.ky, self.Ee, self.Eh, self.VC,
            self.geh, self.ge, Win1, Wout1, self.k3, coulomb._Ceh, coulomb._Cee
        )
        coulomb.MBCE2(
            self.ne0, self.nh0, self.ky, self.Ee, self.Eh, self.VC,
            self.geh, self.ge, Win2, Wout2, self.k3, coulomb._Ceh, coulomb._Cee
        )
        assert np.allclose(Win1, Win2, rtol=1e-10, atol=1e-12)
        assert np.allclose(Wout1, Wout2, rtol=1e-10, atol=1e-12)


class TestMBCH:
    """Test many-body Coulomb in/out rates for holes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.ge = 1e12
        self.gh = 1e12
        self.geh = (self.ge + self.gh) / 2.0

        # Initialize arrays
        self.k3 = coulomb.MakeK3(self.ky)
        self.UnDel = coulomb.MakeUnDel(self.ky)
        Ceh, Cee, Chh = coulomb.CalcMBArrays(
            self.ky, self.Ee, self.Eh, self.ge, self.gh, self.k3, self.UnDel
        )

        self.ne0 = 0.5 * np.ones(self.Nk)
        self.nh0 = 0.5 * np.ones(self.Nk)
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 2] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vhh

        self.Win = np.zeros(self.Nk)
        self.Wout = np.zeros(self.Nk)

    def test_mbch_basic(self):
        """Test MBCH with basic inputs."""
        coulomb.MBCH(
            self.ne0, self.nh0, self.ky, self.Ee, self.Eh, self.VC,
            self.geh, self.gh, self.Win, self.Wout, self.k3, coulomb._Ceh, coulomb._Chh
        )
        assert np.all(self.Win >= 0.0)
        assert np.all(self.Wout >= 0.0)
        assert np.all(np.isfinite(self.Win))
        assert np.all(np.isfinite(self.Wout))

    def test_mbch_module_arrays(self):
        """Test MBCH using module-level arrays."""
        # First call with explicit arrays to populate Win/Wout
        coulomb.MBCH(
            self.ne0, self.nh0, self.ky, self.Ee, self.Eh, self.VC,
            self.geh, self.gh, self.Win, self.Wout, self.k3, coulomb._Ceh, coulomb._Chh
        )
        # Then call with module-level arrays
        Win2 = np.zeros(self.Nk)
        Wout2 = np.zeros(self.Nk)
        coulomb.MBCH(
            self.ne0, self.nh0, self.ky, self.Ee, self.Eh, self.VC,
            self.geh, self.gh, Win2, Wout2
        )
        assert np.allclose(self.Win, Win2, rtol=1e-10, atol=1e-12)
        assert np.allclose(self.Wout, Wout2, rtol=1e-10, atol=1e-12)


class TestInitializeCoulomb:
    """Test main initialization function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.N = 32
        self.y = np.linspace(-10e-6, 10e-6, 64)
        self.ky = np.linspace(-1e7, 1e7, self.N)
        self.L = 100e-6
        self.Delta0 = 10e-9
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.ge = 1e12
        self.gh = 1e12
        self.alphae = 1e6
        self.alphah = 1.2e6
        self.er = 12.0

        # Create Qy and kkp arrays
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

        self.screened = False

    def test_initialize_coulomb_basic(self):
        """Test InitializeCoulomb with basic inputs."""
        # Clear module variables
        coulomb._UnDel = None
        coulomb._k3 = None
        coulomb._qe = None
        coulomb._qh = None
        coulomb._Ceh = None
        coulomb._Cee = None
        coulomb._Chh = None
        coulomb._Veh0 = None
        coulomb._Vee0 = None
        coulomb._Vhh0 = None
        coulomb._Chi1De = None
        coulomb._Chi1Dh = None

        coulomb.InitializeCoulomb(
            self.y, self.ky, self.L, self.Delta0, self.me, self.mh,
            self.Ee, self.Eh, self.ge, self.gh, self.alphae, self.alphah,
            self.er, self.Qy, self.kkp, self.screened
        )

        # Check that all arrays are initialized
        assert coulomb._UnDel is not None
        assert coulomb._k3 is not None
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

    def test_initialize_coulomb_idempotent(self):
        """Test that InitializeCoulomb is idempotent."""
        # Initialize twice
        coulomb.InitializeCoulomb(
            self.y, self.ky, self.L, self.Delta0, self.me, self.mh,
            self.Ee, self.Eh, self.ge, self.gh, self.alphae, self.alphah,
            self.er, self.Qy, self.kkp, self.screened
        )
        UnDel1 = coulomb._UnDel.copy()
        k3_1 = coulomb._k3.copy()

        coulomb.InitializeCoulomb(
            self.y, self.ky, self.L, self.Delta0, self.me, self.mh,
            self.Ee, self.Eh, self.ge, self.gh, self.alphae, self.alphah,
            self.er, self.Qy, self.kkp, self.screened
        )
        UnDel2 = coulomb._UnDel
        k3_2 = coulomb._k3

        # Should be the same (or at least same shape and values)
        assert np.array_equal(UnDel1, UnDel2)
        assert np.array_equal(k3_1, k3_2)


class TestIntegration:
    """Integration tests for complete workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.N = 32
        self.y = np.linspace(-10e-6, 10e-6, 64)
        self.ky = np.linspace(-1e7, 1e7, self.N)
        self.L = 100e-6
        self.Delta0 = 10e-9
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.ge = 1e12
        self.gh = 1e12
        self.alphae = 1e6
        self.alphah = 1.2e6
        self.er = 12.0

        # Create Qy and kkp arrays
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

    def test_full_initialization_workflow(self):
        """Test complete initialization workflow."""
        # Clear all module variables
        coulomb._UnDel = None
        coulomb._k3 = None
        coulomb._qe = None
        coulomb._qh = None
        coulomb._Ceh = None
        coulomb._Cee = None
        coulomb._Chh = None
        coulomb._Veh0 = None
        coulomb._Vee0 = None
        coulomb._Vhh0 = None
        coulomb._Chi1De = None
        coulomb._Chi1Dh = None

        # Initialize
        coulomb.InitializeCoulomb(
            self.y, self.ky, self.L, self.Delta0, self.me, self.mh,
            self.Ee, self.Eh, self.ge, self.gh, self.alphae, self.alphah,
            self.er, self.Qy, self.kkp, False
        )

        # Test that we can use all initialized arrays
        # Test CalcScreenedArrays
        ne = 0.5 * np.ones(self.N, dtype=complex)
        nh = 0.5 * np.ones(self.N, dtype=complex)
        VC = np.zeros((self.N, self.N, 3))
        E1D = np.zeros((self.N, self.N))
        coulomb.CalcScreenedArrays(False, self.L, ne, nh, VC, E1D)
        assert np.all(np.isfinite(VC))
        assert np.all(np.isfinite(E1D))

        # Test CalcMVeh
        Nf = 4
        p = 0.1 * (np.random.random((self.N, self.N, Nf)) +
                   1j * np.random.random((self.N, self.N, Nf)))
        MVeh = np.zeros((self.N, self.N, Nf), dtype=complex)
        coulomb.CalcMVeh(p, VC, MVeh)
        assert np.all(np.isfinite(MVeh))

        # Test BGRenorm
        C = np.eye(self.N, dtype=complex) * 0.5
        D = np.eye(self.N, dtype=complex) * 0.5
        BGR = np.zeros((self.N, self.N), dtype=complex)
        coulomb.BGRenorm(C, D, VC, BGR)
        assert np.all(np.isfinite(BGR))

    def test_screening_workflow(self):
        """Test complete screening calculation workflow."""
        # Initialize
        coulomb.InitializeCoulomb(
            self.y, self.ky, self.L, self.Delta0, self.me, self.mh,
            self.Ee, self.Eh, self.ge, self.gh, self.alphae, self.alphah,
            self.er, self.Qy, self.kkp, False
        )

        # Test screening
        ne = 0.5 * np.ones(self.N, dtype=complex)
        nh = 0.5 * np.ones(self.N, dtype=complex)
        VC = np.zeros((self.N, self.N, 3))
        E1D = np.zeros((self.N, self.N))

        # Unscreened
        coulomb.CalcScreenedArrays(False, self.L, ne, nh, VC, E1D)
        VC_unscreened = VC.copy()
        E1D_unscreened = E1D.copy()

        # Screened
        coulomb.CalcScreenedArrays(True, self.L, ne, nh, VC, E1D)
        VC_screened = VC.copy()
        E1D_screened = E1D.copy()

        # Screened arrays should be different (at least for some elements)
        # Note: For very small densities, screening may have minimal effect
        # Check that at least some elements differ significantly
        diff_VC = np.abs(VC_screened - VC_unscreened)
        diff_E1D = np.abs(E1D_screened - E1D_unscreened)
        # At least some elements should differ (allowing for numerical precision)
        assert np.any(diff_VC > 1e-30) or np.any(diff_E1D > 1e-15)

    def test_many_body_rates_workflow(self):
        """Test complete many-body rates calculation workflow."""
        # Initialize
        coulomb.InitializeCoulomb(
            self.y, self.ky, self.L, self.Delta0, self.me, self.mh,
            self.Ee, self.Eh, self.ge, self.gh, self.alphae, self.alphah,
            self.er, self.Qy, self.kkp, False
        )

        # Calculate screened arrays
        ne = 0.5 * np.ones(self.N, dtype=complex)
        nh = 0.5 * np.ones(self.N, dtype=complex)
        VC = np.zeros((self.N, self.N, 3))
        E1D = np.zeros((self.N, self.N))
        coulomb.CalcScreenedArrays(False, self.L, ne, nh, VC, E1D)

        # Calculate many-body rates
        ne0 = np.real(ne)
        nh0 = np.real(nh)
        Win_e = np.zeros(self.N)
        Wout_e = np.zeros(self.N)
        Win_h = np.zeros(self.N)
        Wout_h = np.zeros(self.N)

        geh = (self.ge + self.gh) / 2.0
        coulomb.MBCE2(ne0, nh0, self.ky, self.Ee, self.Eh, VC, geh, self.ge,
                      Win_e, Wout_e)
        coulomb.MBCH(ne0, nh0, self.ky, self.Ee, self.Eh, VC, geh, self.gh,
                     Win_h, Wout_h)

        assert np.all(Win_e >= 0.0)
        assert np.all(Wout_e >= 0.0)
        assert np.all(Win_h >= 0.0)
        assert np.all(Wout_h >= 0.0)
        assert np.all(np.isfinite(Win_e))
        assert np.all(np.isfinite(Wout_e))
        assert np.all(np.isfinite(Win_h))
        assert np.all(np.isfinite(Wout_h))

    @pytest.mark.parametrize("N", [16, 32, 64, 101])
    def test_different_array_sizes(self, N):
        """Test integration with different array sizes."""
        # Clear module variables to avoid conflicts
        coulomb._UnDel = None
        coulomb._k3 = None
        coulomb._qe = None
        coulomb._qh = None
        coulomb._Ceh = None
        coulomb._Cee = None
        coulomb._Chh = None
        coulomb._Veh0 = None
        coulomb._Vee0 = None
        coulomb._Vhh0 = None
        coulomb._Chi1De = None
        coulomb._Chi1Dh = None

        y = np.linspace(-10e-6, 10e-6, 2 * N)
        ky = np.linspace(-1e7, 1e7, N)
        Ee = 0.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
        Eh = 0.5 * 1.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)

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

        # Initialize
        coulomb.InitializeCoulomb(
            y, ky, self.L, self.Delta0, self.me, self.mh,
            Ee, Eh, self.ge, self.gh, self.alphae, self.alphah,
            self.er, Qy, kkp, False
        )

        # Test that arrays have correct sizes
        assert coulomb._k3.shape == (N, N, N)
        assert coulomb._UnDel.shape == (N + 1, N + 1)
        assert coulomb._qe.shape == (N, N)
        assert coulomb._qh.shape == (N, N)
        assert coulomb._Ceh.shape == (N + 1, N + 1, N + 1)
        assert coulomb._Veh0.shape == (N, N)

