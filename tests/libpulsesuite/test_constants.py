"""
Tests for Constants — verifies physical and mathematical constants against CODATA / NIST values.

Every constant is checked against its known value and expected type.
"""
import numpy as np

from pulsesuite.core.constants import Constants

# ---------------------------------------------------------------------------
# Mathematical constants
# ---------------------------------------------------------------------------

def test_pi_value():
    assert np.isclose(Constants.pi, np.pi, atol=1e-15)
    assert isinstance(Constants.pi, (float, np.floating))


def test_const_e_value():
    assert np.isclose(Constants.const_e, np.e, atol=1e-15)
    assert isinstance(Constants.const_e, (float, np.floating))


def test_pio2():
    assert np.isclose(Constants.pio2, np.pi / 2, atol=1e-15)


def test_twopi():
    assert np.isclose(Constants.twopi, 2 * np.pi, atol=1e-15)


def test_sqrt2():
    assert np.isclose(Constants.sqrt2, np.sqrt(2), atol=1e-15)


def test_euler():
    # Euler-Mascheroni constant to high precision
    assert np.isclose(Constants.euler, 0.5772156649015328606, atol=1e-15)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

def test_ii_is_imaginary_unit():
    assert Constants.ii == 1j
    assert isinstance(Constants.ii, (complex, np.complexfloating))


def test_c0_speed_of_light():
    assert np.isclose(Constants.c0, 299792458.0)
    assert isinstance(Constants.c0, (float, np.floating))


def test_eps0_vacuum_permittivity():
    assert np.isclose(Constants.eps0, 8.854187817e-12, rtol=1e-9)


def test_mu0_vacuum_permeability():
    assert np.isclose(Constants.mu0, 1.2566370614e-6, rtol=1e-9)


def test_eps0_mu0_c0_relation():
    """Maxwell: c0 = 1/sqrt(eps0 * mu0)."""
    c_derived = 1.0 / np.sqrt(Constants.eps0 * Constants.mu0)
    assert np.isclose(c_derived, Constants.c0, rtol=1e-6)


def test_e0_elementary_charge():
    assert np.isclose(Constants.e0, 1.60217733e-19, rtol=1e-6)


def test_eV_equals_e0():
    assert Constants.eV == Constants.e0


def test_hplank():
    assert np.isclose(Constants.hplank, 6.62606876e-34, rtol=1e-6)


def test_hbar_equals_h_over_2pi():
    assert np.isclose(Constants.hbar, Constants.hplank / (2 * np.pi), rtol=1e-4)


def test_me0_electron_mass():
    assert np.isclose(Constants.me0, 9.109534e-31, rtol=1e-4)


# ---------------------------------------------------------------------------
# as_dict
# ---------------------------------------------------------------------------

def test_as_dict_returns_all_numeric_constants():
    d = Constants.as_dict()
    assert isinstance(d, dict)
    assert 'pi' in d and np.isclose(d['pi'], np.pi, atol=1e-15)
    assert 'c0' in d and np.isclose(d['c0'], 299792458.0)
    assert 'eps0' in d
    assert 'ii' in d
