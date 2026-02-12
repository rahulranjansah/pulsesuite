"""
Tests for helpers — grounded in mathematical identities and physics formulae.

Covers all public functions from helpers.py matching Fortran helpers.F90:
sech, arg, gauss, magsq, constrain, AmpToInten, FldToInten, IntenToAmp,
l2f, l2w, w2l, GetSpaceArray, GetKArray, LinearInterp, BilinearInterp,
TrilinearInterp, dfdt, LAX, noLAX, unwrap, factorial, isnan.

Tolerances:
    algebraic  :  rtol=1e-12, atol=1e-12  (float64)
    physics    :  rtol=1e-8               (constant combinations)
"""
import numpy as np
import pytest

from pulsesuite.libpulsesuite.helpers import (
    LAX,
    AmpToInten,
    BilinearInterp,
    FldToInten,
    GetKArray,
    GetSpaceArray,
    IntenToAmp,
    LinearInterp,
    TrilinearInterp,
    arg,
    c0,
    constrain,
    dfdt_1D_dp,
    dfdt_1D_dpc,
    dfdt_dp,
    ec2,
    factorial,
    gauss,
    isnan_dp,
    isnan_sp,
    l2f,
    l2w,
    magsq,
    noLAX,
    pi,
    sech,
    unwrap,
    w2l,
)

RTOL = 1e-12
ATOL = 1e-12


# ===================================================================
# sech
# ===================================================================

class TestSech:
    def test_sech_zero(self):
        assert np.isclose(sech(np.float64(0.0)), 1.0, atol=ATOL)

    def test_sech_identity(self):
        """sech(x) = 1/cosh(x)."""
        x = np.linspace(-5, 5, 50)
        np.testing.assert_allclose(sech(x), 1.0 / np.cosh(x), rtol=RTOL, atol=ATOL)

    def test_sech_even(self):
        """sech is an even function: sech(-x) = sech(x)."""
        x = np.array([0.5, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(sech(-x), sech(x), rtol=RTOL, atol=ATOL)


# ===================================================================
# arg
# ===================================================================

class TestArg:
    def test_arg_real_positive(self):
        assert np.isclose(arg(np.complex128(1.0 + 0j)), 0.0, atol=ATOL)

    def test_arg_imaginary(self):
        assert np.isclose(arg(np.complex128(0.0 + 1j)), np.pi / 2, atol=ATOL)

    def test_arg_negative_real(self):
        assert np.isclose(arg(np.complex128(-1.0 + 0j)), np.pi, atol=ATOL)

    def test_arg_matches_numpy_angle(self):
        z = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j, 1 + 1j])
        np.testing.assert_allclose(arg(z), np.angle(z), rtol=RTOL, atol=ATOL)


# ===================================================================
# gauss
# ===================================================================

class TestGauss:
    def test_gauss_zero(self):
        assert np.isclose(gauss(np.float64(0.0)), 1.0, atol=ATOL)

    def test_gauss_identity(self):
        """gauss(x) = exp(-x^2)."""
        x = np.linspace(-3, 3, 30)
        np.testing.assert_allclose(gauss(x), np.exp(-x**2), rtol=RTOL, atol=ATOL)

    def test_gauss_even(self):
        x = np.array([0.5, 1.0, 2.0])
        np.testing.assert_allclose(gauss(-x), gauss(x), rtol=RTOL, atol=ATOL)

    def test_gauss_large_array(self):
        big = np.random.rand(500_000)
        out = gauss(big)
        assert out.size == big.size


# ===================================================================
# magsq
# ===================================================================

class TestMagsq:
    def test_magsq_real(self):
        """magsq(x+0j) = x^2."""
        z = np.array([1 + 0j, 2 + 0j, -3 + 0j])
        np.testing.assert_allclose(magsq(z), np.array([1, 4, 9], dtype=float), rtol=RTOL, atol=ATOL)

    def test_magsq_identity(self):
        """|z|^2 = Re(z)^2 + Im(z)^2."""
        z = np.array([1 + 1j, 3 - 4j, 0 + 2j])
        expected = np.real(z)**2 + np.imag(z)**2
        np.testing.assert_allclose(magsq(z), expected, rtol=RTOL, atol=ATOL)


# ===================================================================
# constrain
# ===================================================================

class TestConstrain:
    def test_constrain_clips(self):
        x = np.array([-5.0, 0.0, 5.0, 10.0])
        result = constrain(x, 8.0, -2.0)
        assert np.all(result >= -2.0) and np.all(result <= 8.0)

    def test_constrain_passthrough(self):
        """Values within bounds should be unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        result = constrain(x, 5.0, 0.0)
        np.testing.assert_allclose(result, x, rtol=RTOL, atol=ATOL)

    def test_constrain_integer(self):
        x = np.array([-10, 0, 10], dtype=np.int64)
        result = constrain(x, 5, -5)
        assert np.all(result >= -5) and np.all(result <= 5)


# ===================================================================
# Field conversion: AmpToInten, FldToInten, IntenToAmp
# ===================================================================

class TestFieldConversions:
    def test_amp_to_inten_roundtrip(self):
        """IntenToAmp(AmpToInten(E)) = E for positive E."""
        e = 2.5
        n0 = 1.33
        I = AmpToInten(e, n0)
        e_back = IntenToAmp(I, n0)
        assert np.isclose(e, e_back, rtol=1e-10)

    def test_amp_to_inten_formula(self):
        """I = n0 * 2 * eps0 * c0 * E^2."""
        e = 3.0
        n0 = 1.5
        expected = n0 * ec2 * e**2
        assert np.isclose(AmpToInten(e, n0), expected, rtol=RTOL)

    def test_fld_to_inten_complex(self):
        """FldToInten uses |E|^2."""
        e = 1.0 + 2.0j
        n0 = 1.0
        expected = n0 * ec2 * (1.0**2 + 2.0**2)
        assert np.isclose(FldToInten(e, n0), expected, rtol=1e-10)

    def test_no_n0_defaults(self):
        """When n0 is not passed, formula uses n0=1 effectively."""
        e = 2.0
        I_default = AmpToInten(e)
        I_n1 = AmpToInten(e, 1.0)
        assert np.isclose(I_default, I_n1, rtol=RTOL)


# ===================================================================
# Wavelength / frequency conversions
# ===================================================================

class TestWavelengthFrequency:
    def test_l2f_identity(self):
        """f = c0 / lam."""
        lam = 800e-9  # 800 nm
        assert np.isclose(l2f(lam), c0 / lam, rtol=RTOL)

    def test_l2w_w2l_roundtrip(self):
        """w2l(l2w(lam)) = lam."""
        lam = 1.55e-6
        assert np.isclose(w2l(l2w(lam)), lam, rtol=RTOL)

    def test_l2w_formula(self):
        """omega = 2*pi*c0 / lam."""
        lam = 532e-9
        assert np.isclose(l2w(lam), 2 * pi * c0 / lam, rtol=RTOL)


# ===================================================================
# GetSpaceArray, GetKArray
# ===================================================================

class TestGridArrays:
    def test_space_array_centered(self):
        """N=3, length=10 -> [-5, 0, 5]."""
        arr = GetSpaceArray(3, 10.0)
        np.testing.assert_allclose(arr, [-5.0, 0.0, 5.0], rtol=RTOL, atol=ATOL)

    def test_space_array_symmetry(self):
        arr = GetSpaceArray(11, 20.0)
        np.testing.assert_allclose(arr, -arr[::-1], atol=ATOL)

    def test_space_array_n1(self):
        arr = GetSpaceArray(1, 10.0)
        assert arr.shape == (1,)
        assert arr[0] == 0.0

    def test_k_array_shape_and_dc(self):
        k = GetKArray(8, 2 * np.pi)
        assert k.shape == (8,)
        assert np.isclose(k[0], 0.0, atol=ATOL)

    def test_k_array_fft_convention(self):
        """k array: positive freqs up to N//2, then negative freqs."""
        N = 8
        length = 2 * np.pi
        k = GetKArray(N, length)
        dk = 2 * pi / length
        # Implementation uses i <= N//2 (inclusive), so Nyquist is positive
        expected = np.array([0, 1, 2, 3, 4, -3, -2, -1], dtype=float) * dk
        np.testing.assert_allclose(k, expected, rtol=RTOL, atol=ATOL)


# ===================================================================
# LinearInterp, BilinearInterp, TrilinearInterp
# ===================================================================

class TestLinearInterp:
    def test_linear_interp_exact(self):
        """Interpolating a linear function should be exact."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        f = 2.0 * x + 1.0  # f(x) = 2x + 1
        assert np.isclose(LinearInterp(f, x, 1.5), 4.0, rtol=1e-10)

    def test_linear_interp_at_node(self):
        x = np.array([0.0, 1.0, 2.0])
        f = np.array([10.0, 20.0, 30.0])
        assert np.isclose(LinearInterp(f, x, 1.0), 20.0, rtol=1e-10)

    def test_linear_interp_complex(self):
        x = np.array([0.0, 1.0, 2.0])
        f = np.array([0 + 0j, 1 + 1j, 2 + 2j])
        val = LinearInterp(f, x, 0.5)
        assert np.isclose(val, 0.5 + 0.5j, rtol=1e-10)


class TestBilinearInterp:
    def test_bilinear_plane(self):
        """Bilinear interp of f(x,y)=x+y should be exact."""
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = X + Y
        val = BilinearInterp(f, x, y, 0.5, 0.5)
        assert np.isclose(val, 1.0, rtol=1e-8)


class TestTrilinearInterp:
    def test_trilinear_plane(self):
        """Trilinear interp of f(x,y,z)=x+y+z should be exact."""
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 5)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        f = X + Y + Z
        val = TrilinearInterp(f, x, y, z, 0.5, 0.5, 0.5)
        assert np.isclose(val, 1.5, rtol=1e-8)


# ===================================================================
# dfdt (five-point stencil derivative)
# ===================================================================

class TestDfdt:
    def test_dfdt_index_linear(self):
        """Derivative of f(t) = t is 1 everywhere."""
        f = np.arange(20, dtype=float)
        dt = 1.0
        val = dfdt_dp(f, dt, 10)
        assert np.isclose(val, 1.0, rtol=1e-8)

    def test_dfdt_1d_quadratic(self):
        """Derivative of f(t) = t^2 is 2t. Interior points should be exact for 5pt stencil."""
        t = np.linspace(0, 5, 100)
        dt = t[1] - t[0]
        f = t**2
        deriv = dfdt_1D_dp(f, dt)
        # Check interior (away from boundaries)
        interior = slice(5, 95)
        np.testing.assert_allclose(deriv[interior], 2 * t[interior], rtol=1e-4)

    def test_dfdt_1d_complex(self):
        """Derivative of f(t) = t + i*t is (1 + i) everywhere."""
        t = np.linspace(0, 5, 100)
        dt = t[1] - t[0]
        f = t + 1j * t
        deriv = dfdt_1D_dpc(f, dt)
        interior = slice(5, 95)
        np.testing.assert_allclose(deriv[interior], np.full(90, 1.0 + 1.0j), rtol=1e-4)

    def test_dfdt_shape_preserved(self):
        f = np.arange(20, dtype=float)
        result = dfdt_1D_dp(f, 1.0)
        assert result.shape == f.shape


# ===================================================================
# LAX / noLAX
# ===================================================================

class TestLaxSmoothing:
    def test_lax_average(self):
        """LAX returns average of 6 nearest neighbors in 3D."""
        u = np.arange(27, dtype=complex).reshape((3, 3, 3))
        val = LAX(u, 1, 1, 1)
        neighbors = [u[0, 1, 1], u[2, 1, 1], u[1, 0, 1], u[1, 2, 1], u[1, 1, 0], u[1, 1, 2]]
        assert np.isclose(val, sum(neighbors) / 6.0)

    def test_noLAX_identity(self):
        """noLAX returns the center value unchanged."""
        u = np.arange(27, dtype=complex).reshape((3, 3, 3))
        assert noLAX(u, 1, 1, 1) == u[1, 1, 1]


# ===================================================================
# unwrap
# ===================================================================

class TestUnwrap:
    def test_unwrap_monotone(self):
        """Unwrapping a linearly increasing phase should be monotone."""
        phase = np.linspace(0, 6 * np.pi, 100) % (2 * np.pi)
        u = unwrap(phase)
        # Should be monotonically increasing (up to floating point)
        diffs = np.diff(u)
        assert np.all(diffs >= -1e-10)

    def test_unwrap_no_wraps(self):
        """If phase is already smooth, unwrap should not change it."""
        phase = np.linspace(0, np.pi, 50)
        u = unwrap(phase)
        np.testing.assert_allclose(u, phase, atol=1e-10)


# ===================================================================
# factorial
# ===================================================================

class TestFactorial:
    @pytest.mark.parametrize("n, expected", [
        (0, 1), (1, 1), (5, 120), (6, 720), (10, 3628800),
    ])
    def test_factorial_values(self, n, expected):
        assert factorial(n) == expected


# ===================================================================
# isnan
# ===================================================================

class TestIsnan:
    def test_isnan_dp(self):
        assert isnan_dp(np.nan) == True
        assert isnan_dp(0.0) == False

    def test_isnan_sp(self):
        assert isnan_sp(np.float32(np.nan)) == True
        assert isnan_sp(np.float32(1.0)) == False

    def test_isnan_array(self):
        x = np.array([0.0, np.nan, 1.0])
        result = isnan_dp(x)
        assert result[1] == True and result[0] == False and result[2] == False
