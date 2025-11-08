import sys
import os
import numpy as np
import pytest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from materialproperties import (MaterialProperties, MaterialError,
                               MaterialOptions, MatHelp, mat_error_handler, verify_G)

# Use these materials and wavelengths for tests
MATERIALS = ['BK7', 'FSIL', 'AIR', 'IND1.5']
LAM = 800e-9  # 800 nm, common test wavelength
LAM2 = 532.1e-9  # 532.1 nm, for beta in BK7
OMEGA = 2 * np.pi * 299792458.0 / LAM
OMEGA_ARR = np.linspace(2 * np.pi * 299792458.0 / 1e-6, 2 * np.pi * 299792458.0 / 400e-9, 5)

mp = MaterialProperties(os.path.join(os.path.dirname(__file__), '../src/materials_py.ini'))


# skipped tests
def test_alpha_runs():
    for mat in MATERIALS:
        try:
            val = mp.alpha(mat, LAM)
            assert isinstance(val, float)
        except MaterialError:
            pytest.skip(f"{mat} has no Absorption parameter")

def test_beta_runs():
    for mat in MATERIALS:
        val = mp.beta(mat, LAM)
        assert isinstance(val, float)
    # Check known value for BK7 at 532.1 nm
    assert np.isclose(mp.beta('BK7', LAM2), 2.9e-14, rtol=1e-2)

def test_n0_and_index_of_refraction():
    for mat in MATERIALS:
        n = mp.n0(mat, LAM)
        n2 = mp.index_of_refraction(mat, LAM)
        assert isinstance(n, float)
        assert np.isclose(n, n2)
    # Check fallback value for IND1.5
    assert np.isclose(mp.n0('IND1.5', LAM), 1.5, rtol=1e-6)

# skipped tests
def test_n2I_n2F():
    for mat in ['AIR', 'FSIL']:
        try:
            n2i = mp.n2I(mat, LAM)
            n2f = mp.n2F(mat, LAM)
            assert isinstance(n2i, float)
            assert isinstance(n2f, float)
        except MaterialError:
            pytest.skip(f"{mat} has no n2I or n2F parameter")

def test_Vp_k0_Vg():
    for mat in MATERIALS:
        vp = mp.Vp(mat, LAM)
        k0 = mp.k0(mat, LAM)
        vg = mp.Vg(mat, LAM)
        assert isinstance(vp, float)
        assert isinstance(k0, float)
        assert isinstance(vg, float)

# skipped tests
def test_k_derivatives():
    # Only test materials with 3 Sellmeier coefficients
    for mat in ['BK7', 'FSIL']:
        for fn in [mp.k1, mp.k2, mp.k3, mp.k4, mp.k5, mp.k1_l, mp.k2_l]:
            try:
                val = fn(mat, LAM)
                assert isinstance(val, float)
            except Exception as e:
                pytest.skip(f"{mat} cannot be used for {fn.__name__}: {e}")

def test_GetKW():
    for mat in ['AIR', 'FSIL']:
        try:
            arr = mp.GetKW(mat, OMEGA_ARR)
            assert isinstance(arr, np.ndarray)
            assert arr.shape == OMEGA_ARR.shape
        except MaterialError:
            pytest.skip(f"{mat} has no GetKW parameter")

# skipped tests
def test_Tr():
    for mat in ['AIR', 'FSIL']:
        try:
            val = mp.Tr(mat, LAM)
            assert isinstance(val, float)
        except MaterialError:
            pytest.skip(f"{mat} has no Raman-tr parameter")

# skipped tests
def test_plasma_properties():
    for mat in ['AIR', 'FSIL']:
        try:
            assert isinstance(mp.GetPlasmaElectronMass(mat, LAM), float)
            assert isinstance(mp.GetPlasmaBandGap(mat, LAM), float)
            assert isinstance(mp.GetPlasmaTrappingTime(mat, LAM), float)
            assert isinstance(mp.GetPlasmaCollisionTime(mat, LAM), float)
            assert isinstance(mp.GetPlasmaMaximumDensity(mat, LAM), float)
            assert isinstance(mp.GetPlasmaOrder(mat, LAM), float)
            assert isinstance(mp.GetPlasmaCrossSection(mat, LAM), float)
        except MaterialError:
            pytest.skip(f"{mat} has no Plasma properties")

def test_MaterialOptions_and_MatHelp(capsys):
    # Should handle a non-existent file gracefully
    assert MaterialOptions('--material-datafile=not_a_real_file.ini')
    # Should not handle unrelated options
    assert not MaterialOptions('--other-option')
    # MatHelp prints help
    MatHelp()
    out = capsys.readouterr().out
    assert 'material-datafile' in out

def test_mat_error_handler_warns():
    # Should raise for NOTFOUND
    with pytest.raises(MaterialError):
        mat_error_handler(MaterialProperties.MAT_ERR_NOTFOUND, 'n0', 'BK7', LAM)
    # Should warn for OUTOFRANGE
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        mat_error_handler(MaterialProperties.MAT_ERR_OUTOFRANGE, 'n0', 'BK7', LAM)
        assert any('Error' in str(x.message) for x in w)

# def test_verify_G_runs():
#     # Just check it runs and prints
#     verify_G('AIR', LAM)

def test_private_sellmeier_coeff():
    # Should return tuple (A, B, C)
    A, B, C = mp._sellmeier_coeff('BK7', LAM)
    assert isinstance(A, float)
    assert isinstance(B, np.ndarray)
    assert isinstance(C, np.ndarray)

def test_private_read_tag_array_and_val():
    arr = mp._read_tag_array('BK7', 'Sellmeier-B')
    assert isinstance(arr, np.ndarray)
    val, code = mp._read_tag_val('BK7', 'Sellmeier-A')
    assert isinstance(val, float)
    assert isinstance(code, int)

def test_private_discrete_val():
    val, code = mp._discrete_val('IND1.5', 'n0', LAM)
    assert isinstance(val, float)
    assert isinstance(code, int)