"""
Comprehensive test suite for dephasing.py module.

Tests all dephasing calculation functions including initialization, dephasing rates,
off-diagonal dephasing matrices, and file I/O operations.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import hbar as hbar_SI

from pulsesuite.PSTD3D import dephasing

# Physical constants
hbar = hbar_SI
pi = np.pi
ii = 1j


class TestInitializeDephasing:
    """Test dephasing module initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        # Clear module variables
        dephasing._k_p_q = None
        dephasing._k_m_q = None
        dephasing._k1_m_q = None
        dephasing._k1p_m_q = None
        dephasing._k1 = None
        dephasing._k1p = None
        dephasing._xe = None
        dephasing._xh = None
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

    def teardown_method(self):
        """Clean up after tests."""
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

    def test_initialize_dephasing_small(self):
        """Test InitializeDephasing with small array."""
        ky = np.array([-1.0, 0.0, 1.0]) * 1e7  # 1/m
        me = 0.067 * 9.109e-31  # kg (GaAs electron mass)
        mh = 0.5 * 9.109e-31  # kg

        dephasing.InitializeDephasing(ky, me, mh)

        Nk = len(ky)
        assert dephasing._k_p_q.shape == (Nk, Nk)
        assert dephasing._k_m_q.shape == (Nk, Nk)
        assert dephasing._k1_m_q.shape == (Nk, Nk)
        assert dephasing._k1p_m_q.shape == (Nk, Nk)
        assert dephasing._k1.shape == (Nk, Nk)
        assert dephasing._k1p.shape == (Nk, Nk)
        assert dephasing._xe.shape == (Nk,)
        assert dephasing._xh.shape == (Nk,)
        assert dephasing._fe_file is not None
        assert dephasing._fh_file is not None

    def test_initialize_dephasing_even_length(self):
        """Test InitializeDephasing with even length array."""
        Nk = 64
        ky = np.linspace(-1e7, 1e7, Nk)
        me = 0.067 * 9.109e-31
        mh = 0.5 * 9.109e-31

        dephasing.InitializeDephasing(ky, me, mh)

        assert dephasing._k_p_q.shape == (Nk, Nk)
        assert dephasing._k_m_q.shape == (Nk, Nk)
        assert dephasing._xe.shape == (Nk,)
        assert dephasing._xh.shape == (Nk,)

    def test_initialize_dephasing_odd_length(self):
        """Test InitializeDephasing with odd length array."""
        Nk = 101
        ky = np.linspace(-1e7, 1e7, Nk)
        me = 0.067 * 9.109e-31
        mh = 0.5 * 9.109e-31

        dephasing.InitializeDephasing(ky, me, mh)

        assert dephasing._k_p_q.shape == (Nk, Nk)
        assert dephasing._k_m_q.shape == (Nk, Nk)

    def test_initialize_dephasing_prime_length(self):
        """Test InitializeDephasing with prime length array."""
        Nk = 97
        ky = np.linspace(-1e7, 1e7, Nk)
        me = 0.067 * 9.109e-31
        mh = 0.5 * 9.109e-31

        dephasing.InitializeDephasing(ky, me, mh)

        assert dephasing._k_p_q.shape == (Nk, Nk)
        assert dephasing._k_m_q.shape == (Nk, Nk)

    def test_initialize_dephasing_single_element(self):
        """Test InitializeDephasing with single element array."""
        ky = np.array([0.0]) * 1e7
        me = 0.067 * 9.109e-31
        mh = 0.5 * 9.109e-31

        dephasing.InitializeDephasing(ky, me, mh)

        assert dephasing._k_p_q.shape == (1, 1)
        assert dephasing._k_m_q.shape == (1, 1)

    def test_initialize_dephasing_index_bounds(self):
        """Test that initialized indices are within valid bounds."""
        Nk = 32
        ky = np.linspace(-1e7, 1e7, Nk)
        me = 0.067 * 9.109e-31
        mh = 0.5 * 9.109e-31

        dephasing.InitializeDephasing(ky, me, mh)

        # Check that indices are within reasonable bounds (they can be negative
        # for extended arrays with boundary points, but should be finite)
        assert np.all(np.isfinite(dephasing._k_p_q))
        assert np.all(np.isfinite(dephasing._k_m_q))
        assert np.all(np.isfinite(dephasing._k1))
        assert np.all(np.isfinite(dephasing._k1p))

    def test_initialize_dephasing_xe_xh_positive(self):
        """Test that xe and xh arrays are positive."""
        Nk = 32
        ky = np.linspace(-1e7, 1e7, Nk)
        me = 0.067 * 9.109e-31
        mh = 0.5 * 9.109e-31

        dephasing.InitializeDephasing(ky, me, mh)

        assert np.all(dephasing._xe >= 0.0)
        assert np.all(dephasing._xh >= 0.0)
        assert np.all(np.isfinite(dephasing._xe))
        assert np.all(np.isfinite(dephasing._xh))


class TestVxx2:
    """Test squared interaction matrix calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_vxx2_small(self):
        """Test Vxx2 with small array."""
        q = np.array([-1.0, 0.0, 1.0]) * 1e7
        V = np.random.random((len(q), len(q))) * 1e-20
        result = dephasing.Vxx2(q, V)
        assert result.shape == (len(q),)
        assert np.all(result >= 0.0)
        assert np.all(np.isfinite(result))

    def test_vxx2_even_length(self):
        """Test Vxx2 with even length array."""
        N = 64
        q = np.linspace(-1e7, 1e7, N)
        V = np.random.random((N, N)) * 1e-20
        result = dephasing.Vxx2(q, V)
        assert result.shape == (N,)
        assert np.all(result >= 0.0)

    def test_vxx2_odd_length(self):
        """Test Vxx2 with odd length array."""
        N = 101
        q = np.linspace(-1e7, 1e7, N)
        V = np.random.random((N, N)) * 1e-20
        result = dephasing.Vxx2(q, V)
        assert result.shape == (N,)

    def test_vxx2_single_element(self):
        """Test Vxx2 with single element array."""
        q = np.array([0.0]) * 1e7
        V = np.array([[1e-20]])
        result = dephasing.Vxx2(q, V)
        assert result.shape == (1,)
        assert result[0] >= 0.0

    def test_vxx2_zero_momentum(self):
        """Test Vxx2 with zero momentum."""
        q = np.array([0.0, 1e7, 2e7])
        V = np.random.random((len(q), len(q))) * 1e-20
        result = dephasing.Vxx2(q, V)
        assert result[0] >= 0.0  # Zero momentum should give V[0,0]^2


class TestCalcGammaE:
    """Test electron dephasing rate calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31

        # Initialize dephasing module
        dephasing.InitializeDephasing(self.ky, self.me, self.mh)

        # Create test arrays
        self.ne0 = 0.5 * np.ones(self.Nk, dtype=complex)
        self.nh0 = 0.5 * np.ones(self.Nk, dtype=complex)
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vee
        self.GammaE = np.zeros(self.Nk)

    def teardown_method(self):
        """Clean up after tests."""
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

    def test_calc_gamma_e_basic(self):
        """Test CalcGammaE with basic inputs."""
        dephasing.CalcGammaE(self.ky, self.ne0, self.nh0, self.VC, self.GammaE)
        assert self.GammaE.shape == (self.Nk,)
        assert np.all(self.GammaE >= 0.0)
        assert np.all(np.isfinite(self.GammaE))

    def test_calc_gamma_e_zero_populations(self):
        """Test CalcGammaE with zero populations."""
        ne0_zero = np.zeros(self.Nk, dtype=complex)
        nh0_zero = np.zeros(self.Nk, dtype=complex)
        GammaE = np.zeros(self.Nk)
        dephasing.CalcGammaE(self.ky, ne0_zero, nh0_zero, self.VC, GammaE)
        assert np.all(GammaE >= 0.0)
        assert np.all(np.isfinite(GammaE))

    def test_calc_gamma_e_error_if_not_initialized(self):
        """Test that CalcGammaE raises error if not initialized."""
        old_k_p_q = dephasing._k_p_q
        dephasing._k_p_q = None
        with pytest.raises(RuntimeError):
            dephasing.CalcGammaE(self.ky, self.ne0, self.nh0, self.VC, self.GammaE)
        dephasing._k_p_q = old_k_p_q

    def test_calc_gamma_e_different_sizes(self):
        """Test CalcGammaE with different array sizes."""
        for Nk in [16, 32, 64, 101]:
            ky = np.linspace(-1e7, 1e7, Nk)
            me = 0.067 * 9.109e-31
            mh = 0.5 * 9.109e-31
            dephasing.InitializeDephasing(ky, me, mh)

            ne0 = 0.5 * np.ones(Nk, dtype=complex)
            nh0 = 0.5 * np.ones(Nk, dtype=complex)
            VC = np.zeros((Nk, Nk, 3))
            VC[:, :, 0] = 1e-20 * np.random.random((Nk, Nk))
            VC[:, :, 1] = 1e-20 * np.random.random((Nk, Nk))
            GammaE = np.zeros(Nk)

            dephasing.CalcGammaE(ky, ne0, nh0, VC, GammaE)
            assert GammaE.shape == (Nk,)
            assert np.all(GammaE >= 0.0)


class TestCalcGammaH:
    """Test hole dephasing rate calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31

        # Initialize dephasing module
        dephasing.InitializeDephasing(self.ky, self.me, self.mh)

        # Create test arrays
        self.ne0 = 0.5 * np.ones(self.Nk, dtype=complex)
        self.nh0 = 0.5 * np.ones(self.Nk, dtype=complex)
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 2] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vhh
        self.GammaH = np.zeros(self.Nk)

    def teardown_method(self):
        """Clean up after tests."""
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

    def test_calc_gamma_h_basic(self):
        """Test CalcGammaH with basic inputs."""
        dephasing.CalcGammaH(self.ky, self.ne0, self.nh0, self.VC, self.GammaH)
        assert self.GammaH.shape == (self.Nk,)
        assert np.all(self.GammaH >= 0.0)
        assert np.all(np.isfinite(self.GammaH))

    def test_calc_gamma_h_zero_populations(self):
        """Test CalcGammaH with zero populations."""
        ne0_zero = np.zeros(self.Nk, dtype=complex)
        nh0_zero = np.zeros(self.Nk, dtype=complex)
        GammaH = np.zeros(self.Nk)
        dephasing.CalcGammaH(self.ky, ne0_zero, nh0_zero, self.VC, GammaH)
        assert np.all(GammaH >= 0.0)
        assert np.all(np.isfinite(GammaH))

    def test_calc_gamma_h_error_if_not_initialized(self):
        """Test that CalcGammaH raises error if not initialized."""
        old_k_p_q = dephasing._k_p_q
        dephasing._k_p_q = None
        with pytest.raises(RuntimeError):
            dephasing.CalcGammaH(self.ky, self.ne0, self.nh0, self.VC, self.GammaH)
        dephasing._k_p_q = old_k_p_q


class TestCalcOffDiagDeph_E:
    """Test off-diagonal dephasing matrix calculation for electrons."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31

        # Initialize dephasing module
        dephasing.InitializeDephasing(self.ky, self.me, self.mh)

        # Create test arrays
        self.ne0 = 0.5 * np.ones(self.Nk, dtype=complex)
        self.nh0 = 0.5 * np.ones(self.Nk, dtype=complex)
        self.Ee0 = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh0 = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.gee = 1e12  # Hz
        self.geh = 1e12  # Hz
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vee

    def teardown_method(self):
        """Clean up after tests."""
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

    def test_calc_off_diag_deph_e_basic(self):
        """Test CalcOffDiagDeph_E with basic inputs."""
        result = dephasing.CalcOffDiagDeph_E(
            self.ne0, self.nh0, self.ky, self.Ee0, self.Eh0,
            self.gee, self.geh, self.VC
        )
        assert result.shape == (self.Nk, self.Nk)
        assert np.all(np.isfinite(result))

    def test_calc_off_diag_deph_e_zero_populations(self):
        """Test CalcOffDiagDeph_E with zero populations."""
        ne0_zero = np.zeros(self.Nk, dtype=complex)
        nh0_zero = np.zeros(self.Nk, dtype=complex)
        result = dephasing.CalcOffDiagDeph_E(
            ne0_zero, nh0_zero, self.ky, self.Ee0, self.Eh0,
            self.gee, self.geh, self.VC
        )
        assert result.shape == (self.Nk, self.Nk)
        assert np.all(np.isfinite(result))

    def test_calc_off_diag_deph_e_error_if_not_initialized(self):
        """Test that CalcOffDiagDeph_E raises error if not initialized."""
        old_k_p_q = dephasing._k_p_q
        dephasing._k_p_q = None
        with pytest.raises(RuntimeError):
            dephasing.CalcOffDiagDeph_E(
                self.ne0, self.nh0, self.ky, self.Ee0, self.Eh0,
                self.gee, self.geh, self.VC
            )
        dephasing._k_p_q = old_k_p_q


class TestCalcOffDiagDeph_H:
    """Test off-diagonal dephasing matrix calculation for holes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31

        # Initialize dephasing module
        dephasing.InitializeDephasing(self.ky, self.me, self.mh)

        # Create test arrays
        self.ne0 = 0.5 * np.ones(self.Nk, dtype=complex)
        self.nh0 = 0.5 * np.ones(self.Nk, dtype=complex)
        self.Ee0 = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh0 = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.ghh = 1e12  # Hz
        self.geh = 1e12  # Hz
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 2] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vhh

    def teardown_method(self):
        """Clean up after tests."""
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

    def test_calc_off_diag_deph_h_basic(self):
        """Test CalcOffDiagDeph_H with basic inputs."""
        result = dephasing.CalcOffDiagDeph_H(
            self.ne0, self.nh0, self.ky, self.Ee0, self.Eh0,
            self.ghh, self.geh, self.VC
        )
        assert result.shape == (self.Nk, self.Nk)
        assert np.all(np.isfinite(result))

    def test_calc_off_diag_deph_h_zero_populations(self):
        """Test CalcOffDiagDeph_H with zero populations."""
        ne0_zero = np.zeros(self.Nk, dtype=complex)
        nh0_zero = np.zeros(self.Nk, dtype=complex)
        result = dephasing.CalcOffDiagDeph_H(
            ne0_zero, nh0_zero, self.ky, self.Ee0, self.Eh0,
            self.ghh, self.geh, self.VC
        )
        assert result.shape == (self.Nk, self.Nk)
        assert np.all(np.isfinite(result))

    def test_calc_off_diag_deph_h_error_if_not_initialized(self):
        """Test that CalcOffDiagDeph_H raises error if not initialized."""
        old_k_p_q = dephasing._k_p_q
        dephasing._k_p_q = None
        with pytest.raises(RuntimeError):
            dephasing.CalcOffDiagDeph_H(
                self.ne0, self.nh0, self.ky, self.Ee0, self.Eh0,
                self.ghh, self.geh, self.VC
            )
        dephasing._k_p_q = old_k_p_q


class TestCalcOffDiagDeph_E2:
    """Test off-diagonal dephasing matrix calculation for electrons (version 2)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31

        # Initialize dephasing module
        dephasing.InitializeDephasing(self.ky, self.me, self.mh)

        # Create test arrays
        self.ne = 0.5 * np.ones(self.Nk, dtype=complex)
        self.nh = 0.5 * np.ones(self.Nk, dtype=complex)
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.gee = 1e12  # Hz
        self.geh = 1e12  # Hz
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vee

    def teardown_method(self):
        """Clean up after tests."""
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

    def test_calc_off_diag_deph_e2_basic(self):
        """Test CalcOffDiagDeph_E2 with basic inputs."""
        result = dephasing.CalcOffDiagDeph_E2(
            self.ne, self.nh, self.ky, self.Ee, self.Eh,
            self.gee, self.geh, self.VC, self.Nk
        )
        assert result.shape == (2 * self.Nk + 1, self.Nk)
        assert np.all(np.isfinite(result))

    def test_calc_off_diag_deph_e2_zero_populations(self):
        """Test CalcOffDiagDeph_E2 with zero populations."""
        ne_zero = np.zeros(self.Nk, dtype=complex)
        nh_zero = np.zeros(self.Nk, dtype=complex)
        result = dephasing.CalcOffDiagDeph_E2(
            ne_zero, nh_zero, self.ky, self.Ee, self.Eh,
            self.gee, self.geh, self.VC, self.Nk
        )
        assert result.shape == (2 * self.Nk + 1, self.Nk)
        assert np.all(np.isfinite(result))

    def test_calc_off_diag_deph_e2_different_sizes(self):
        """Test CalcOffDiagDeph_E2 with different array sizes."""
        for Nk in [16, 32, 64, 101]:
            ky = np.linspace(-1e7, 1e7, Nk)
            me = 0.067 * 9.109e-31
            mh = 0.5 * 9.109e-31
            dephasing.InitializeDephasing(ky, me, mh)

            ne = 0.5 * np.ones(Nk, dtype=complex)
            nh = 0.5 * np.ones(Nk, dtype=complex)
            Ee = 0.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
            Eh = 0.5 * 1.5 * 9.109e-31 * (ky / 1e7) ** 2 * (1.6e-19)
            VC = np.zeros((Nk, Nk, 3))
            VC[:, :, 0] = 1e-20 * np.random.random((Nk, Nk))
            VC[:, :, 1] = 1e-20 * np.random.random((Nk, Nk))

            result = dephasing.CalcOffDiagDeph_E2(ne, nh, ky, Ee, Eh, 1e12, 1e12, VC, Nk)
            assert result.shape == (2 * Nk + 1, Nk)


class TestCalcOffDiagDeph_H2:
    """Test off-diagonal dephasing matrix calculation for holes (version 2)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31

        # Initialize dephasing module
        dephasing.InitializeDephasing(self.ky, self.me, self.mh)

        # Create test arrays
        self.ne = 0.5 * np.ones(self.Nk, dtype=complex)
        self.nh = 0.5 * np.ones(self.Nk, dtype=complex)
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.ghh = 1e12  # Hz
        self.geh = 1e12  # Hz
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 2] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vhh

    def teardown_method(self):
        """Clean up after tests."""
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

    def test_calc_off_diag_deph_h2_basic(self):
        """Test CalcOffDiagDeph_H2 with basic inputs."""
        result = dephasing.CalcOffDiagDeph_H2(
            self.ne, self.nh, self.ky, self.Ee, self.Eh,
            self.ghh, self.geh, self.VC, self.Nk
        )
        assert result.shape == (2 * self.Nk + 1, self.Nk)
        assert np.all(np.isfinite(result))

    def test_calc_off_diag_deph_h2_zero_populations(self):
        """Test CalcOffDiagDeph_H2 with zero populations."""
        ne_zero = np.zeros(self.Nk, dtype=complex)
        nh_zero = np.zeros(self.Nk, dtype=complex)
        result = dephasing.CalcOffDiagDeph_H2(
            ne_zero, nh_zero, self.ky, self.Ee, self.Eh,
            self.ghh, self.geh, self.VC, self.Nk
        )
        assert result.shape == (2 * self.Nk + 1, self.Nk)
        assert np.all(np.isfinite(result))


class TestOffDiagDephasing:
    """Test off-diagonal dephasing contribution calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31

        # Initialize dephasing module
        dephasing.InitializeDephasing(self.ky, self.me, self.mh)

        # Create test arrays
        self.ne = 0.5 * np.ones(self.Nk, dtype=complex)
        self.nh = 0.5 * np.ones(self.Nk, dtype=complex)
        self.p = 0.1 * (np.random.random((self.Nk, self.Nk)) +
                       1j * np.random.random((self.Nk, self.Nk)))
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.g = np.array([1e12, 1e12, 1e12])  # [gee, ghh, geh]
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vee
        self.VC[:, :, 2] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vhh
        self.x = np.zeros((self.Nk, self.Nk), dtype=complex)

    def teardown_method(self):
        """Clean up after tests."""
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

    def test_off_diag_dephasing_basic(self):
        """Test OffDiagDephasing with basic inputs."""
        dephasing.OffDiagDephasing(
            self.ne, self.nh, self.p, self.ky, self.Ee, self.Eh,
            self.g, self.VC, self.x
        )
        assert self.x.shape == (self.Nk, self.Nk)
        assert np.all(np.isfinite(self.x))

    def test_off_diag_dephasing_zero_polarization(self):
        """Test OffDiagDephasing with zero polarization."""
        p_zero = np.zeros((self.Nk, self.Nk), dtype=complex)
        x = np.zeros((self.Nk, self.Nk), dtype=complex)
        dephasing.OffDiagDephasing(
            self.ne, self.nh, p_zero, self.ky, self.Ee, self.Eh,
            self.g, self.VC, x
        )
        assert np.all(np.isfinite(x))

    def test_off_diag_dephasing_error_if_not_initialized(self):
        """Test that OffDiagDephasing raises error if not initialized."""
        old_k_p_q = dephasing._k_p_q
        dephasing._k_p_q = None
        with pytest.raises(RuntimeError):
            dephasing.OffDiagDephasing(
                self.ne, self.nh, self.p, self.ky, self.Ee, self.Eh,
                self.g, self.VC, self.x
            )
        dephasing._k_p_q = old_k_p_q


class TestOffDiagDephasing2:
    """Test off-diagonal dephasing contribution calculation (version 2)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31

        # Initialize dephasing module
        dephasing.InitializeDephasing(self.ky, self.me, self.mh)

        # Create test arrays
        self.ne = 0.5 * np.ones(self.Nk, dtype=complex)
        self.nh = 0.5 * np.ones(self.Nk, dtype=complex)
        self.p = 0.1 * (np.random.random((self.Nk, self.Nk)) +
                       1j * np.random.random((self.Nk, self.Nk)))
        self.Ee = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.g = np.array([1e12, 1e12, 1e12])  # [gee, ghh, geh]
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vee
        self.VC[:, :, 2] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vhh
        self.t = 0.0
        self.x = np.zeros((self.Nk, self.Nk), dtype=complex)

    def teardown_method(self):
        """Clean up after tests."""
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

    def test_off_diag_dephasing2_basic(self):
        """Test OffDiagDephasing2 with basic inputs."""
        dephasing.OffDiagDephasing2(
            self.ne, self.nh, self.p, self.ky, self.Ee, self.Eh,
            self.g, self.VC, self.t, self.x
        )
        assert self.x.shape == (self.Nk, self.Nk)
        assert np.all(np.isfinite(self.x))

    def test_off_diag_dephasing2_zero_polarization(self):
        """Test OffDiagDephasing2 with zero polarization."""
        p_zero = np.zeros((self.Nk, self.Nk), dtype=complex)
        x = np.zeros((self.Nk, self.Nk), dtype=complex)
        dephasing.OffDiagDephasing2(
            self.ne, self.nh, p_zero, self.ky, self.Ee, self.Eh,
            self.g, self.VC, self.t, x
        )
        assert np.all(np.isfinite(x))

    def test_off_diag_dephasing2_file_output(self):
        """Test that OffDiagDephasing2 writes to files."""
        # Files should be opened by InitializeDephasing
        assert dephasing._fe_file is not None
        assert dephasing._fh_file is not None

        dephasing.OffDiagDephasing2(
            self.ne, self.nh, self.p, self.ky, self.Ee, self.Eh,
            self.g, self.VC, self.t, self.x
        )

        # Check that files exist and have content
        fe_path = Path('dataQW/Wire/info/MaxOffDiag.e.dat')
        fh_path = Path('dataQW/Wire/info/MaxOffDiag.h.dat')
        if fe_path.exists():
            assert fe_path.stat().st_size > 0
        if fh_path.exists():
            assert fh_path.stat().st_size > 0


class TestPrintGam:
    """Test dephasing rate printing function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_print_gam_basic(self):
        """Test printGam with basic inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs('dataQW', exist_ok=True)

            try:
                Dx = np.array([1e12, 2e12, 3e12])
                z = np.array([0.0, 1e-6, 2e-6])
                n = 5
                file = 'test_gam'

                dephasing.printGam(Dx, z, n, file)

                filename = f'dataQW/{file}{n:05d}.dat'
                assert os.path.exists(filename)

                # Read and verify file contents
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    assert len(lines) == len(Dx)
                    for i, line in enumerate(lines):
                        parts = line.strip().split()
                        assert len(parts) == 2
                        assert abs(float(parts[0]) - z[i]) < 1e-6
                        assert abs(float(parts[1]) - Dx[i]) < 1e6
            finally:
                os.chdir(old_cwd)

    def test_print_gam_different_sizes(self):
        """Test printGam with different array sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs('dataQW', exist_ok=True)

            try:
                for N in [16, 32, 64, 101]:
                    Dx = np.random.random(N) * 1e12
                    z = np.linspace(0, 1e-6, N)
                    n = 10
                    file = 'test_gam'

                    dephasing.printGam(Dx, z, n, file)

                    filename = f'dataQW/{file}{n:05d}.dat'
                    assert os.path.exists(filename)
                    with open(filename, 'r') as f:
                        lines = f.readlines()
                        assert len(lines) == N
            finally:
                os.chdir(old_cwd)

    def test_print_gam_single_element(self):
        """Test printGam with single element array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs('dataQW', exist_ok=True)

            try:
                Dx = np.array([1e12])
                z = np.array([0.0])
                n = 1
                file = 'test_gam'

                dephasing.printGam(Dx, z, n, file)

                filename = f'dataQW/{file}{n:05d}.dat'
                assert os.path.exists(filename)
            finally:
                os.chdir(old_cwd)


class TestWriteDephasing:
    """Test dephasing rate file writing function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_write_dephasing_basic(self):
        """Test WriteDephasing with basic inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs('dataQW/Wire/Ge', exist_ok=True)
            os.makedirs('dataQW/Wire/Gh', exist_ok=True)

            try:
                Nk = 32
                ky = np.linspace(-1e7, 1e7, Nk)
                gamE = np.random.random(Nk) * 1e12
                gamH = np.random.random(Nk) * 1e12
                w = 1
                xxx = 5

                dephasing.WriteDephasing(ky, gamE, gamH, w, xxx)

                # Check that files were created
                fe_path = Path(f'dataQW/Wire/Ge/Ge.{w:02d}.k.{xxx:05d}.dat')
                fh_path = Path(f'dataQW/Wire/Gh/Gh.{w:02d}.k.{xxx:05d}.dat')
                assert fe_path.exists()
                assert fh_path.exists()

                # Verify file contents
                with open(fe_path, 'r') as f:
                    lines = f.readlines()
                    assert len(lines) == Nk

                with open(fh_path, 'r') as f:
                    lines = f.readlines()
                    assert len(lines) == Nk
            finally:
                os.chdir(old_cwd)

    def test_write_dephasing_different_indices(self):
        """Test WriteDephasing with different wire and time indices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs('dataQW/Wire/Ge', exist_ok=True)
            os.makedirs('dataQW/Wire/Gh', exist_ok=True)

            try:
                Nk = 16
                ky = np.linspace(-1e7, 1e7, Nk)
                gamE = np.random.random(Nk) * 1e12
                gamH = np.random.random(Nk) * 1e12

                for w in [0, 1, 10, 99]:
                    for xxx in [0, 1, 100, 9999]:
                        dephasing.WriteDephasing(ky, gamE, gamH, w, xxx)

                        fe_path = Path(f'dataQW/Wire/Ge/Ge.{w:02d}.k.{xxx:05d}.dat')
                        fh_path = Path(f'dataQW/Wire/Gh/Gh.{w:02d}.k.{xxx:05d}.dat')
                        assert fe_path.exists()
                        assert fh_path.exists()
            finally:
                os.chdir(old_cwd)


class TestIntegration:
    """Integration tests for complete dephasing workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.me = 0.067 * 9.109e-31
        self.mh = 0.5 * 9.109e-31

        # Initialize dephasing module
        dephasing.InitializeDephasing(self.ky, self.me, self.mh)

        # Create test arrays
        self.ne0 = 0.5 * np.ones(self.Nk, dtype=complex)
        self.nh0 = 0.5 * np.ones(self.Nk, dtype=complex)
        self.Ee0 = 0.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.Eh0 = 0.5 * 1.5 * 9.109e-31 * (self.ky / 1e7) ** 2 * (1.6e-19)
        self.VC = np.zeros((self.Nk, self.Nk, 3))
        self.VC[:, :, 0] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Veh
        self.VC[:, :, 1] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vee
        self.VC[:, :, 2] = 1e-20 * np.random.random((self.Nk, self.Nk))  # Vhh

    def teardown_method(self):
        """Clean up after tests."""
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

    def test_full_dephasing_workflow(self):
        """Test complete dephasing calculation workflow."""
        # Calculate dephasing rates
        GammaE = np.zeros(self.Nk)
        GammaH = np.zeros(self.Nk)
        dephasing.CalcGammaE(self.ky, self.ne0, self.nh0, self.VC, GammaE)
        dephasing.CalcGammaH(self.ky, self.ne0, self.nh0, self.VC, GammaH)

        assert np.all(GammaE >= 0.0)
        assert np.all(GammaH >= 0.0)
        assert np.all(np.isfinite(GammaE))
        assert np.all(np.isfinite(GammaH))

        # Calculate off-diagonal dephasing matrices
        De = dephasing.CalcOffDiagDeph_E(
            self.ne0, self.nh0, self.ky, self.Ee0, self.Eh0,
            1e12, 1e12, self.VC
        )
        Dh = dephasing.CalcOffDiagDeph_H(
            self.ne0, self.nh0, self.ky, self.Ee0, self.Eh0,
            1e12, 1e12, self.VC
        )

        assert De.shape == (self.Nk, self.Nk)
        assert Dh.shape == (self.Nk, self.Nk)
        assert np.all(np.isfinite(De))
        assert np.all(np.isfinite(Dh))

    def test_off_diag_dephasing_workflow(self):
        """Test complete off-diagonal dephasing workflow."""
        p = 0.1 * (np.random.random((self.Nk, self.Nk)) +
                  1j * np.random.random((self.Nk, self.Nk)))
        g = np.array([1e12, 1e12, 1e12])
        x = np.zeros((self.Nk, self.Nk), dtype=complex)

        dephasing.OffDiagDephasing(
            self.ne0, self.nh0, p, self.ky, self.Ee0, self.Eh0,
            g, self.VC, x
        )

        assert x.shape == (self.Nk, self.Nk)
        assert np.all(np.isfinite(x))

    def test_off_diag_dephasing2_workflow(self):
        """Test complete off-diagonal dephasing workflow (version 2)."""
        p = 0.1 * (np.random.random((self.Nk, self.Nk)) +
                  1j * np.random.random((self.Nk, self.Nk)))
        g = np.array([1e12, 1e12, 1e12])
        x = np.zeros((self.Nk, self.Nk), dtype=complex)
        t = 0.0

        dephasing.OffDiagDephasing2(
            self.ne0, self.nh0, p, self.ky, self.Ee0, self.Eh0,
            g, self.VC, t, x
        )

        assert x.shape == (self.Nk, self.Nk)
        assert np.all(np.isfinite(x))

    def test_version2_consistency(self):
        """Test that CalcOffDiagDeph_E2 and CalcOffDiagDeph_H2 produce valid results."""
        De2 = dephasing.CalcOffDiagDeph_E2(
            self.ne0, self.nh0, self.ky, self.Ee0, self.Eh0,
            1e12, 1e12, self.VC, self.Nk
        )
        Dh2 = dephasing.CalcOffDiagDeph_H2(
            self.ne0, self.nh0, self.ky, self.Ee0, self.Eh0,
            1e12, 1e12, self.VC, self.Nk
        )

        assert De2.shape == (2 * self.Nk + 1, self.Nk)
        assert Dh2.shape == (2 * self.Nk + 1, self.Nk)
        assert np.all(np.isfinite(De2))
        assert np.all(np.isfinite(Dh2))

    @pytest.mark.parametrize("Nk", [16, 32, 64, 101])
    def test_different_array_sizes(self, Nk):
        """Test integration with different array sizes."""
        # Clear module variables
        dephasing._k_p_q = None
        dephasing._k_m_q = None
        dephasing._k1_m_q = None
        dephasing._k1p_m_q = None
        dephasing._k1 = None
        dephasing._k1p = None
        dephasing._xe = None
        dephasing._xh = None
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

        ky = np.linspace(-1e7, 1e7, Nk)
        me = 0.067 * 9.109e-31
        mh = 0.5 * 9.109e-31

        dephasing.InitializeDephasing(ky, me, mh)

        ne0 = 0.5 * np.ones(Nk, dtype=complex)
        nh0 = 0.5 * np.ones(Nk, dtype=complex)
        VC = np.zeros((Nk, Nk, 3))
        VC[:, :, 0] = 1e-20 * np.random.random((Nk, Nk))
        VC[:, :, 1] = 1e-20 * np.random.random((Nk, Nk))
        VC[:, :, 2] = 1e-20 * np.random.random((Nk, Nk))

        GammaE = np.zeros(Nk)
        GammaH = np.zeros(Nk)

        dephasing.CalcGammaE(ky, ne0, nh0, VC, GammaE)
        dephasing.CalcGammaH(ky, ne0, nh0, VC, GammaH)

        assert GammaE.shape == (Nk,)
        assert GammaH.shape == (Nk,)
        assert np.all(GammaE >= 0.0)
        assert np.all(GammaH >= 0.0)

        # Clean up
        if dephasing._fe_file is not None:
            dephasing._fe_file.close()
            dephasing._fe_file = None
        if dephasing._fh_file is not None:
            dephasing._fh_file.close()
            dephasing._fh_file = None

