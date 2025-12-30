"""
Comprehensive test suite for phonons.py module.

Tests all phonon interaction functions including initialization, many-body
phonon-electron/hole interactions, distribution functions, and utility functions.
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

import phonons
from scipy.constants import hbar as hbar_SI, k as kB_SI

# Physical constants
hbar = hbar_SI
kB = kB_SI
pi = np.pi
ii = 1j


class TestInitializePhonons:
    """Test phonon module initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        # Clear module variables
        phonons._EP = None
        phonons._EPT = None
        phonons._HP = None
        phonons._HPT = None
        phonons._idel = None
        phonons._NO = 0.0
        phonons._Vscale = 0.0

    def test_InitializePhonons_small_array(self):
        """Test InitializePhonons with small array."""
        Nk = 8
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12  # Phonon damping rate (Hz)
        Oph = 1e13  # Phonon frequency (Hz)

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        assert phonons._EP is not None
        assert phonons._EPT is not None
        assert phonons._HP is not None
        assert phonons._HPT is not None
        assert phonons._idel is not None
        assert phonons._NO > 0
        assert phonons._Vscale != 0.0

    def test_InitializePhonons_array_shapes(self):
        """Test that initialized arrays have correct shapes."""
        Nk = 32
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        assert phonons._EP.shape == (Nk, Nk)
        assert phonons._EPT.shape == (Nk, Nk)
        assert phonons._HP.shape == (Nk, Nk)
        assert phonons._HPT.shape == (Nk, Nk)
        assert phonons._idel.shape == (Nk, Nk)

    def test_InitializePhonons_transpose_relationship(self):
        """Test that EPT and HPT are transposes of EP and HP."""
        Nk = 16
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        assert np.allclose(phonons._EPT, phonons._EP.T, rtol=1e-12, atol=1e-12)
        assert np.allclose(phonons._HPT, phonons._HP.T, rtol=1e-12, atol=1e-12)

    def test_InitializePhonons_idel_matrix(self):
        """Test that idel matrix has zeros on diagonal."""
        Nk = 16
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        # Diagonal should be zero
        assert np.allclose(np.diag(phonons._idel), 0.0, rtol=1e-12, atol=1e-12)
        # Off-diagonal should be 1
        for i in range(Nk):
            for j in range(Nk):
                if i != j:
                    assert np.allclose(phonons._idel[i, j], 1.0, rtol=1e-12, atol=1e-12)

    def test_InitializePhonons_EP_values(self):
        """Test that EP matrix has correct structure."""
        Nk = 8
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        # EP should be finite and non-negative
        assert np.all(np.isfinite(phonons._EP))
        assert np.all(phonons._EP >= 0)

    def test_InitializePhonons_HP_values(self):
        """Test that HP matrix has correct structure."""
        Nk = 8
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        # HP should be finite and non-negative
        assert np.all(np.isfinite(phonons._HP))
        assert np.all(phonons._HP >= 0)

    def test_InitializePhonons_NO_calculation(self):
        """Test that NO (Bose distribution) is calculated correctly."""
        Nk = 8
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        # NO should be positive
        assert phonons._NO > 0
        # Check formula: NO = 1 / (exp(hbar*Oph / kB / Temp) - 1)
        expected_NO = 1.0 / (np.exp(hbar * Oph / kB / phonons._Temp) - 1.0)
        assert np.allclose(phonons._NO, expected_NO, rtol=1e-10, atol=1e-12)

    def test_InitializePhonons_Vscale_calculation(self):
        """Test that Vscale is calculated correctly."""
        Nk = 8
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        # Check formula: Vscale = hbar * Oph * epsr * (1/epsrINF - 1/epsr0)
        expected_Vscale = hbar * Oph * epsr * (1.0 / phonons._epsrINF - 1.0 / phonons._epsr0)
        assert np.allclose(phonons._Vscale, expected_Vscale, rtol=1e-10, atol=1e-12)

    def test_InitializePhonons_different_temperatures(self):
        """Test InitializePhonons with different temperatures."""
        Nk = 16
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        for temp in [77.0, 100.0, 300.0]:
            phonons._Temp = temp
            phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)
            assert phonons._NO > 0
            assert np.all(np.isfinite(phonons._EP))
            assert np.all(np.isfinite(phonons._HP))

    def test_InitializePhonons_different_frequencies(self):
        """Test InitializePhonons with different phonon frequencies."""
        Nk = 16
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12

        for Oph in [1e12, 1e13, 1e14]:
            phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)
            assert phonons._NO > 0
            assert np.all(np.isfinite(phonons._EP))
            assert np.all(np.isfinite(phonons._HP))


class TestMBPE:
    """Test many-body phonon-electron interaction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        Nk = 32
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        self.Nk = Nk
        self.ne = np.random.random(Nk) * 0.5
        self.VC = np.random.random((Nk, Nk, 3)) * 1e-20
        self.E1D = np.random.random((Nk, Nk)) * 1e-19 + 1e-20
        self.Win = np.zeros(Nk)
        self.Wout = np.zeros(Nk)

    def test_MBPE_initializes_outputs(self):
        """Test that MBPE initializes outputs correctly."""
        phonons.MBPE(self.ne, self.VC, self.E1D, self.Win, self.Wout)

        assert len(self.Win) == self.Nk
        assert len(self.Wout) == self.Nk

    def test_MBPE_finite_values(self):
        """Test that MBPE produces finite values."""
        phonons.MBPE(self.ne, self.VC, self.E1D, self.Win, self.Wout)

        assert np.all(np.isfinite(self.Win))
        assert np.all(np.isfinite(self.Wout))

    def test_MBPE_non_negative(self):
        """Test that MBPE produces non-negative rates."""
        phonons.MBPE(self.ne, self.VC, self.E1D, self.Win, self.Wout)

        # Rates should be non-negative
        assert np.all(self.Win >= 0)
        assert np.all(self.Wout >= 0)

    def test_MBPE_adds_to_existing_values(self):
        """Test that MBPE adds to existing Win/Wout values."""
        Win_initial = np.ones(self.Nk) * 0.5
        Wout_initial = np.ones(self.Nk) * 0.3

        phonons.MBPE(self.ne, self.VC, self.E1D, Win_initial, Wout_initial)

        # Values should be increased
        assert np.all(Win_initial >= 0.5)
        assert np.all(Wout_initial >= 0.3)

    def test_MBPE_zero_population(self):
        """Test MBPE with zero electron population."""
        ne_zero = np.zeros(self.Nk)
        Win = np.zeros(self.Nk)
        Wout = np.zeros(self.Nk)

        phonons.MBPE(ne_zero, self.VC, self.E1D, Win, Wout)

        # With zero population, Win should be zero
        assert np.allclose(Win, 0.0, rtol=1e-10, atol=1e-12)
        # Wout may be non-zero (out-scattering can still occur)

    def test_MBPE_full_population(self):
        """Test MBPE with full electron population."""
        ne_full = np.ones(self.Nk)
        Win = np.zeros(self.Nk)
        Wout = np.zeros(self.Nk)

        phonons.MBPE(ne_full, self.VC, self.E1D, Win, Wout)

        # With full population, Wout should be zero
        assert np.allclose(Wout, 0.0, rtol=1e-10, atol=1e-12)
        # Win may be non-zero (in-scattering can still occur)

    def test_MBPE_different_array_sizes(self):
        """Test MBPE with different array sizes."""
        for Nk in [8, 16, 32, 64]:
            ky = np.linspace(-1e7, 1e7, Nk)
            Ee = np.linspace(1e-19, 2e-19, Nk)
            Eh = np.linspace(1e-19, 2e-19, Nk)
            phonons.InitializePhonons(ky, Ee, Eh, 10e-6, 10.0, 1e12, 1e13)

            ne = np.random.random(Nk) * 0.5
            VC = np.random.random((Nk, Nk, 3)) * 1e-20
            E1D = np.random.random((Nk, Nk)) * 1e-19 + 1e-20
            Win = np.zeros(Nk)
            Wout = np.zeros(Nk)

            phonons.MBPE(ne, VC, E1D, Win, Wout)

            assert np.all(np.isfinite(Win))
            assert np.all(np.isfinite(Wout))


class TestMBPH:
    """Test many-body phonon-hole interaction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        Nk = 32
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        self.Nk = Nk
        self.nh = np.random.random(Nk) * 0.5
        self.VC = np.random.random((Nk, Nk, 3)) * 1e-20
        self.E1D = np.random.random((Nk, Nk)) * 1e-19 + 1e-20
        self.Win = np.zeros(Nk)
        self.Wout = np.zeros(Nk)

    def test_MBPH_initializes_outputs(self):
        """Test that MBPH initializes outputs correctly."""
        phonons.MBPH(self.nh, self.VC, self.E1D, self.Win, self.Wout)

        assert len(self.Win) == self.Nk
        assert len(self.Wout) == self.Nk

    def test_MBPH_finite_values(self):
        """Test that MBPH produces finite values."""
        phonons.MBPH(self.nh, self.VC, self.E1D, self.Win, self.Wout)

        assert np.all(np.isfinite(self.Win))
        assert np.all(np.isfinite(self.Wout))

    def test_MBPH_non_negative(self):
        """Test that MBPH produces non-negative rates."""
        phonons.MBPH(self.nh, self.VC, self.E1D, self.Win, self.Wout)

        # Rates should be non-negative
        assert np.all(self.Win >= 0)
        assert np.all(self.Wout >= 0)

    def test_MBPH_adds_to_existing_values(self):
        """Test that MBPH adds to existing Win/Wout values."""
        Win_initial = np.ones(self.Nk) * 0.5
        Wout_initial = np.ones(self.Nk) * 0.3

        phonons.MBPH(self.nh, self.VC, self.E1D, Win_initial, Wout_initial)

        # Values should be increased
        assert np.all(Win_initial >= 0.5)
        assert np.all(Wout_initial >= 0.3)

    def test_MBPH_zero_population(self):
        """Test MBPH with zero hole population."""
        nh_zero = np.zeros(self.Nk)
        Win = np.zeros(self.Nk)
        Wout = np.zeros(self.Nk)

        phonons.MBPH(nh_zero, self.VC, self.E1D, Win, Wout)

        # With zero population, Win should be zero
        assert np.allclose(Win, 0.0, rtol=1e-10, atol=1e-12)
        # Wout may be non-zero (out-scattering can still occur)

    def test_MBPH_full_population(self):
        """Test MBPH with full hole population."""
        nh_full = np.ones(self.Nk)
        Win = np.zeros(self.Nk)
        Wout = np.zeros(self.Nk)

        phonons.MBPH(nh_full, self.VC, self.E1D, Win, Wout)

        # With full population, Wout should be zero
        assert np.allclose(Wout, 0.0, rtol=1e-10, atol=1e-12)
        # Win may be non-zero (in-scattering can still occur)

    def test_MBPH_different_array_sizes(self):
        """Test MBPH with different array sizes."""
        for Nk in [8, 16, 32, 64]:
            ky = np.linspace(-1e7, 1e7, Nk)
            Ee = np.linspace(1e-19, 2e-19, Nk)
            Eh = np.linspace(1e-19, 2e-19, Nk)
            phonons.InitializePhonons(ky, Ee, Eh, 10e-6, 10.0, 1e12, 1e13)

            nh = np.random.random(Nk) * 0.5
            VC = np.random.random((Nk, Nk, 3)) * 1e-20
            E1D = np.random.random((Nk, Nk)) * 1e-19 + 1e-20
            Win = np.zeros(Nk)
            Wout = np.zeros(Nk)

            phonons.MBPH(nh, VC, E1D, Win, Wout)

            assert np.all(np.isfinite(Win))
            assert np.all(np.isfinite(Wout))


class TestCq2:
    """Test Cq2 function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        Nk = 32
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        self.Nk = Nk
        self.q = np.linspace(-1e7, 1e7, Nk)
        self.V = np.random.random((Nk, Nk)) * 1e-20
        self.E1D = np.random.random((Nk, Nk)) * 1e-19 + 1e-20

    def test_Cq2_returns_correct_shape(self):
        """Test that Cq2 returns correct shape."""
        result = phonons.Cq2(self.q, self.V, self.E1D)

        assert len(result) == len(self.q)
        assert result.shape == (len(self.q),)

    def test_Cq2_finite_values(self):
        """Test that Cq2 returns finite values."""
        result = phonons.Cq2(self.q, self.V, self.E1D)

        assert np.all(np.isfinite(result))

    def test_Cq2_zero_momentum(self):
        """Test Cq2 with zero momentum."""
        q_zero = np.array([0.0])
        V = np.random.random((self.Nk, self.Nk)) * 1e-20
        E1D = np.random.random((self.Nk, self.Nk)) * 1e-19 + 1e-20

        result = phonons.Cq2(q_zero, V, E1D)

        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_Cq2_single_element(self):
        """Test Cq2 with single element array."""
        q_single = np.array([1e7])
        V = np.random.random((self.Nk, self.Nk)) * 1e-20
        E1D = np.random.random((self.Nk, self.Nk)) * 1e-19 + 1e-20

        result = phonons.Cq2(q_single, V, E1D)

        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_Cq2_different_sizes(self):
        """Test Cq2 with different array sizes."""
        for Nq in [8, 16, 32, 64]:
            q = np.linspace(-1e7, 1e7, Nq)
            V = np.random.random((Nq, Nq)) * 1e-20
            E1D = np.random.random((Nq, Nq)) * 1e-19 + 1e-20

            result = phonons.Cq2(q, V, E1D)

            assert len(result) == Nq
            assert np.all(np.isfinite(result))


class TestFermiDistr:
    """Test Fermi-Dirac distribution function."""

    def setup_method(self):
        """Set up test fixtures."""
        phonons._Temp = 77.0

    def test_FermiDistr_zero_energy(self):
        """Test FermiDistr at zero energy."""
        result = phonons.FermiDistr(0.0)

        # At E=0, f = 1/(exp(0) + 1) = 1/2
        assert np.allclose(result, 0.5, rtol=1e-10, atol=1e-12)

    def test_FermiDistr_negative_energy(self):
        """Test FermiDistr with negative energy."""
        En = -1e-19
        result = phonons.FermiDistr(En)

        # For negative energy, f should be > 0.5
        assert result > 0.5
        assert result <= 1.0

    def test_FermiDistr_positive_energy(self):
        """Test FermiDistr with positive energy."""
        En = 1e-19
        result = phonons.FermiDistr(En)

        # For positive energy, f should be < 0.5
        assert result < 0.5
        assert result >= 0.0

    def test_FermiDistr_large_positive_energy(self):
        """Test FermiDistr with large positive energy."""
        En = 1e-18
        result = phonons.FermiDistr(En)

        # For large positive energy, f should be close to 0
        assert result < 0.1
        assert result >= 0.0

    def test_FermiDistr_large_negative_energy(self):
        """Test FermiDistr with large negative energy."""
        En = -1e-18
        result = phonons.FermiDistr(En)

        # For large negative energy, f should be close to 1
        assert result > 0.9
        assert result <= 1.0

    def test_FermiDistr_array_input(self):
        """Test FermiDistr with array input."""
        En = np.linspace(-1e-19, 1e-19, 10)
        result = phonons.FermiDistr(En)

        assert len(result) == len(En)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert np.all(np.isfinite(result))

    def test_FermiDistr_different_temperatures(self):
        """Test FermiDistr with different temperatures."""
        En = 1e-19

        for temp in [77.0, 100.0, 300.0]:
            phonons._Temp = temp
            result = phonons.FermiDistr(En)
            assert 0.0 <= result <= 1.0
            assert np.isfinite(result)


class TestBoseDistr:
    """Test Bose-Einstein distribution function."""

    def setup_method(self):
        """Set up test fixtures."""
        phonons._Temp = 77.0

    def test_BoseDistr_zero_energy(self):
        """Test BoseDistr at zero energy."""
        # At E=0, Bose distribution diverges, but function should handle it
        En = 0.0
        result = phonons.BoseDistr(En)

        # Should be finite (or very large)
        assert np.isfinite(result) or np.isinf(result)

    def test_BoseDistr_positive_energy(self):
        """Test BoseDistr with positive energy."""
        En = 1e-19
        result = phonons.BoseDistr(En)

        # For positive energy, result should be positive
        assert result > 0.0
        assert np.isfinite(result)

    def test_BoseDistr_large_positive_energy(self):
        """Test BoseDistr with large positive energy."""
        En = 1e-18
        result = phonons.BoseDistr(En)

        # For large positive energy, result should be small
        assert result >= 0.0
        assert np.isfinite(result)

    def test_BoseDistr_array_input(self):
        """Test BoseDistr with array input."""
        En = np.linspace(1e-20, 1e-19, 10)
        result = phonons.BoseDistr(En)

        assert len(result) == len(En)
        assert np.all(result >= 0.0)
        assert np.all(np.isfinite(result))

    def test_BoseDistr_different_temperatures(self):
        """Test BoseDistr with different temperatures."""
        En = 1e-19

        for temp in [77.0, 100.0, 300.0]:
            phonons._Temp = temp
            result = phonons.BoseDistr(En)
            assert result >= 0.0
            assert np.isfinite(result)


class TestN00:
    """Test N00 function."""

    def setup_method(self):
        """Set up test fixtures."""
        Nk = 16
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

    def test_N00_returns_NO(self):
        """Test that N00 returns the NO value."""
        result = phonons.N00()

        assert result == phonons._NO
        assert result > 0

    def test_N00_after_initialization(self):
        """Test N00 after initialization."""
        result = phonons.N00()

        # NO should be positive after initialization
        assert result > 0
        assert np.isfinite(result)


class TestIntegratedWorkflows:
    """Test integrated workflows combining multiple functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.Nk = 32
        self.ky = np.linspace(-1e7, 1e7, self.Nk)
        self.Ee = np.linspace(1e-19, 2e-19, self.Nk)
        self.Eh = np.linspace(1e-19, 2e-19, self.Nk)
        self.L = 10e-6
        self.epsr = 10.0
        self.Gph = 1e12
        self.Oph = 1e13

    def test_initialization_workflow(self):
        """Test complete initialization workflow."""
        phonons.InitializePhonons(self.ky, self.Ee, self.Eh, self.L, self.epsr,
                                  self.Gph, self.Oph)

        # Check all globals are set
        assert phonons._EP is not None
        assert phonons._EPT is not None
        assert phonons._HP is not None
        assert phonons._HPT is not None
        assert phonons._idel is not None
        assert phonons._NO > 0
        assert phonons._Vscale != 0.0

    def test_MBPE_workflow(self):
        """Test complete MBPE workflow."""
        phonons.InitializePhonons(self.ky, self.Ee, self.Eh, self.L, self.epsr,
                                  self.Gph, self.Oph)

        ne = np.random.random(self.Nk) * 0.5
        VC = np.random.random((self.Nk, self.Nk, 3)) * 1e-20
        E1D = np.random.random((self.Nk, self.Nk)) * 1e-19 + 1e-20
        Win = np.zeros(self.Nk)
        Wout = np.zeros(self.Nk)

        phonons.MBPE(ne, VC, E1D, Win, Wout)

        assert np.all(np.isfinite(Win))
        assert np.all(np.isfinite(Wout))
        assert np.all(Win >= 0)
        assert np.all(Wout >= 0)

    def test_MBPH_workflow(self):
        """Test complete MBPH workflow."""
        phonons.InitializePhonons(self.ky, self.Ee, self.Eh, self.L, self.epsr,
                                  self.Gph, self.Oph)

        nh = np.random.random(self.Nk) * 0.5
        VC = np.random.random((self.Nk, self.Nk, 3)) * 1e-20
        E1D = np.random.random((self.Nk, self.Nk)) * 1e-19 + 1e-20
        Win = np.zeros(self.Nk)
        Wout = np.zeros(self.Nk)

        phonons.MBPH(nh, VC, E1D, Win, Wout)

        assert np.all(np.isfinite(Win))
        assert np.all(np.isfinite(Wout))
        assert np.all(Win >= 0)
        assert np.all(Wout >= 0)

    def test_combined_electron_hole_workflow(self):
        """Test combined electron and hole phonon interactions."""
        phonons.InitializePhonons(self.ky, self.Ee, self.Eh, self.L, self.epsr,
                                  self.Gph, self.Oph)

        ne = np.random.random(self.Nk) * 0.5
        nh = np.random.random(self.Nk) * 0.5
        VC = np.random.random((self.Nk, self.Nk, 3)) * 1e-20
        E1D = np.random.random((self.Nk, self.Nk)) * 1e-19 + 1e-20

        Win_e = np.zeros(self.Nk)
        Wout_e = np.zeros(self.Nk)
        Win_h = np.zeros(self.Nk)
        Wout_h = np.zeros(self.Nk)

        phonons.MBPE(ne, VC, E1D, Win_e, Wout_e)
        phonons.MBPH(nh, VC, E1D, Win_h, Wout_h)

        assert np.all(np.isfinite(Win_e))
        assert np.all(np.isfinite(Wout_e))
        assert np.all(np.isfinite(Win_h))
        assert np.all(np.isfinite(Wout_h))

    def test_Cq2_workflow(self):
        """Test Cq2 workflow."""
        phonons.InitializePhonons(self.ky, self.Ee, self.Eh, self.L, self.epsr,
                                  self.Gph, self.Oph)

        q = np.linspace(-1e7, 1e7, self.Nk)
        V = np.random.random((self.Nk, self.Nk)) * 1e-20
        E1D = np.random.random((self.Nk, self.Nk)) * 1e-19 + 1e-20

        result = phonons.Cq2(q, V, E1D)

        assert len(result) == len(q)
        assert np.all(np.isfinite(result))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_InitializePhonons_single_element(self):
        """Test InitializePhonons with single element array."""
        ky = np.array([0.0])
        Ee = np.array([1e-19])
        Eh = np.array([1e-19])
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        assert phonons._EP.shape == (1, 1)
        assert phonons._HP.shape == (1, 1)
        assert phonons._idel[0, 0] == 0.0

    def test_InitializePhonons_identical_energies(self):
        """Test InitializePhonons with identical electron/hole energies."""
        Nk = 8
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.ones(Nk) * 1e-19
        Eh = np.ones(Nk) * 1e-19
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        assert np.all(np.isfinite(phonons._EP))
        assert np.all(np.isfinite(phonons._HP))

    def test_MBPE_extreme_populations(self):
        """Test MBPE with extreme population values."""
        Nk = 16
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        phonons.InitializePhonons(ky, Ee, Eh, 10e-6, 10.0, 1e12, 1e13)

        VC = np.random.random((Nk, Nk, 3)) * 1e-20
        E1D = np.random.random((Nk, Nk)) * 1e-19 + 1e-20

        # Test with very small population
        ne_small = np.ones(Nk) * 1e-10
        Win = np.zeros(Nk)
        Wout = np.zeros(Nk)
        phonons.MBPE(ne_small, VC, E1D, Win, Wout)
        assert np.all(np.isfinite(Win))
        assert np.all(np.isfinite(Wout))

        # Test with very large population (close to 1)
        ne_large = np.ones(Nk) * (1.0 - 1e-10)
        Win = np.zeros(Nk)
        Wout = np.zeros(Nk)
        phonons.MBPE(ne_large, VC, E1D, Win, Wout)
        assert np.all(np.isfinite(Win))
        assert np.all(np.isfinite(Wout))

    def test_FermiDistr_extreme_energies(self):
        """Test FermiDistr with extreme energy values."""
        # Very large positive energy
        result1 = phonons.FermiDistr(1e-15)
        assert 0.0 <= result1 <= 1.0

        # Very large negative energy
        result2 = phonons.FermiDistr(-1e-15)
        assert 0.0 <= result2 <= 1.0

    def test_BoseDistr_extreme_energies(self):
        """Test BoseDistr with extreme energy values."""
        # Very large positive energy
        result1 = phonons.BoseDistr(1e-15)
        assert result1 >= 0.0
        assert np.isfinite(result1)

        # Very small positive energy
        result2 = phonons.BoseDistr(1e-25)
        assert result2 >= 0.0


class TestNumericalPrecision:
    """Test numerical precision and stability."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def test_InitializePhonons_precision(self):
        """Test InitializePhonons numerical precision."""
        Nk = 32
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        L = 10e-6
        epsr = 10.0
        Gph = 1e12
        Oph = 1e13

        phonons.InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph)

        # Check transpose relationship with high precision
        assert np.allclose(phonons._EPT, phonons._EP.T, rtol=1e-12, atol=1e-12)
        assert np.allclose(phonons._HPT, phonons._HP.T, rtol=1e-12, atol=1e-12)

    def test_MBPE_precision(self):
        """Test MBPE numerical precision."""
        Nk = 16
        ky = np.linspace(-1e7, 1e7, Nk)
        Ee = np.linspace(1e-19, 2e-19, Nk)
        Eh = np.linspace(1e-19, 2e-19, Nk)
        phonons.InitializePhonons(ky, Ee, Eh, 10e-6, 10.0, 1e12, 1e13)

        ne = np.random.random(Nk) * 0.5
        VC = np.random.random((Nk, Nk, 3)) * 1e-20
        E1D = np.random.random((Nk, Nk)) * 1e-19 + 1e-20
        Win1 = np.zeros(Nk)
        Wout1 = np.zeros(Nk)
        Win2 = np.zeros(Nk)
        Wout2 = np.zeros(Nk)

        # Run twice with same inputs
        phonons.MBPE(ne, VC, E1D, Win1, Wout1)
        phonons.MBPE(ne, VC, E1D, Win2, Wout2)

        # Results should be identical
        assert np.allclose(Win1, Win2, rtol=1e-12, atol=1e-12)
        assert np.allclose(Wout1, Wout2, rtol=1e-12, atol=1e-12)

    def test_FermiDistr_precision(self):
        """Test FermiDistr numerical precision."""
        En = 1e-19
        result1 = phonons.FermiDistr(En)
        result2 = phonons.FermiDistr(En)

        # Results should be identical
        assert np.allclose(result1, result2, rtol=1e-12, atol=1e-12)

    def test_BoseDistr_precision(self):
        """Test BoseDistr numerical precision."""
        En = 1e-19
        result1 = phonons.BoseDistr(En)
        result2 = phonons.BoseDistr(En)

        # Results should be identical
        assert np.allclose(result1, result2, rtol=1e-12, atol=1e-12)

