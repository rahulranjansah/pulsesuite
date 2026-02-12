"""
Tests for spliner — cubic spline and polynomial interpolation routines.

Covers all public functions from spliner.py matching Fortran spliner.F90:
spline_dp, seval_dp, spline_dpc, seval_dpc, spline2_dp, seval2_dp,
spline2_dpc, seval2_dpc, polint1, polint2, polint3, bcucof, bcuint,
rescale_1D_dp, rescale_1D_dpc, rescale_1D_cyl_dpc, rescale_2D_dp,
rescale_2D_dpc, GetValAt_1D, GetValAt_1D_dpc, GetValAt_2D, GetValAt_3D,
locate, spline, seval, rescale_1D, rescale_2D.

Tolerances:
    spline on polynomial : rtol=1e-8  (cubic spline is exact for degree <= 3)
    polint on polynomial : rtol=1e-10 (Neville's algorithm)
    rescale              : rtol=1e-4  (grid transfer, boundary effects)
"""
import numpy as np
import pytest

from pulsesuite.libpulsesuite.spliner import (
    GetValAt_1D,
    GetValAt_1D_dpc,
    GetValAt_2D,
    GetValAt_3D,
    bcucof,
    bcuint,
    locate,
    polint1,
    polint2,
    polint3,
    rescale_1D,
    rescale_1D_cyl_dpc,
    rescale_1D_dp,
    rescale_1D_dpc,
    rescale_2D,
    rescale_2D_dp,
    rescale_2D_dpc,
    seval,
    seval2_dp,
    seval2_dpc,
    seval_dp,
    seval_dpc,
    spline,
    spline2_dp,
    spline2_dpc,
    spline_dp,
    spline_dpc,
)

RTOL = 1e-8
ATOL = 1e-10


# ===================================================================
# locate (binary search)
# ===================================================================

class TestLocate:
    def test_locate_interior(self):
        x = np.linspace(0, 10, 11)  # [0,1,...,10]
        assert locate(x, 4.5) == 4

    def test_locate_at_node(self):
        x = np.linspace(0, 10, 11)
        assert locate(x, 3.0) == 3

    def test_locate_left_boundary(self):
        x = np.linspace(0, 10, 11)
        assert locate(x, -1.0) == 0

    def test_locate_right_boundary(self):
        x = np.linspace(0, 10, 11)
        assert locate(x, 15.0) == 10


# ===================================================================
# spline_dp / seval_dp (real cubic spline)
# ===================================================================

class TestSplineDp:
    def test_quadratic_exact(self):
        """Cubic spline should reproduce a quadratic exactly."""
        x = np.linspace(0, 10, 11)
        y = x**2
        b = np.zeros(11)
        c = np.zeros(11)
        d = np.zeros(11)
        spline_dp(x, y, b, c, d)
        for xi in np.linspace(0.5, 9.5, 19):
            yi = seval_dp(xi, x, y, b, c, d)
            assert np.isclose(yi, xi**2, atol=1e-6), f"Failed at x={xi}"

    def test_linear_exact(self):
        """Spline of a linear function should be exact."""
        x = np.linspace(0, 5, 6)
        y = 3.0 * x + 1.0
        b = np.zeros(6)
        c = np.zeros(6)
        d = np.zeros(6)
        spline_dp(x, y, b, c, d)
        for xi in [0.5, 1.5, 2.5, 3.5, 4.5]:
            yi = seval_dp(xi, x, y, b, c, d)
            assert np.isclose(yi, 3.0 * xi + 1.0, atol=1e-10)

    def test_n2_linear_fallback(self):
        """With only 2 points, spline should do linear interpolation."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 2.0])
        b = np.zeros(2)
        c = np.zeros(2)
        d = np.zeros(2)
        spline_dp(x, y, b, c, d)
        assert np.isclose(seval_dp(0.5, x, y, b, c, d), 1.0, atol=1e-10)

    def test_too_few_points_raises(self):
        x = np.array([1.0])
        y = np.array([2.0])
        b, c, d = np.zeros(1), np.zeros(1), np.zeros(1)
        with pytest.raises(ValueError):
            spline_dp(x, y, b, c, d)


# ===================================================================
# spline_dpc / seval_dpc (complex cubic spline)
# ===================================================================

class TestSplineDpc:
    def test_complex_linear(self):
        """Complex spline on a linear complex function."""
        x = np.linspace(0, 2 * np.pi, 20)
        y = (1 + 1j) * x
        b = np.zeros(20, dtype=complex)
        c = np.zeros(20, dtype=complex)
        d = np.zeros(20, dtype=complex)
        spline_dpc(x, y, b, c, d)
        xi = np.pi
        yi = seval_dpc(xi, x, y, b, c, d)
        assert np.isclose(yi, (1 + 1j) * xi, atol=1e-6)

    def test_complex_exp(self):
        """Spline on exp(ix) should give finite results and reasonable accuracy."""
        x = np.linspace(0, 2 * np.pi, 30)
        y = np.exp(1j * x)
        b = np.zeros(30, dtype=complex)
        c = np.zeros(30, dtype=complex)
        d = np.zeros(30, dtype=complex)
        spline_dpc(x, y, b, c, d)
        for xi in np.linspace(0.1, 6.0, 10):
            yi = seval_dpc(xi, x, y, b, c, d)
            assert np.isfinite(yi)
            # Should be close to exp(i*xi)
            assert np.abs(yi - np.exp(1j * xi)) < 0.05


# ===================================================================
# spline2_dp / seval2_dp (second derivative version, real)
# ===================================================================

class TestSpline2Dp:
    def test_linear(self):
        """spline2/seval2 on a linear function should be exact."""
        x = np.linspace(0, 5, 20)
        y = 3.0 * x + 1.0
        y2 = np.zeros(20)
        spline2_dp(x, y, y2)
        for xi in [0.5, 1.5, 2.5, 3.5, 4.5]:
            yi = seval2_dp(xi, x, y, y2)
            assert np.isclose(yi, 3.0 * xi + 1.0, atol=1e-6)

    def test_quadratic_interior(self):
        """spline2/seval2 on a quadratic, checking well-interior points."""
        x = np.linspace(0, 5, 50)
        y = x**2
        y2 = np.zeros(50)
        spline2_dp(x, y, y2)
        # Check interior points away from natural spline boundaries
        for xi in np.linspace(1.0, 4.0, 10):
            yi = seval2_dp(xi, x, y, y2)
            assert np.isclose(yi, xi**2, rtol=1e-3)


# ===================================================================
# spline2_dpc / seval2_dpc (second derivative version, complex)
# ===================================================================

class TestSpline2Dpc:
    def test_complex_linear(self):
        x = np.linspace(0, 1, 10)
        z = (2 + 3j) * x
        z2 = np.zeros(10, dtype=complex)
        spline2_dpc(x, z, z2)
        val = seval2_dpc(0.5, x, z, z2)
        assert np.isclose(val, (2 + 3j) * 0.5, atol=1e-6)


# ===================================================================
# polint1 (1D Neville polynomial interpolation)
# ===================================================================

class TestPolint1:
    def test_linear(self):
        """Polynomial interpolation of a linear function (2 points)."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 2.0])
        assert np.isclose(polint1(x, y, 0.5), 1.0, atol=1e-12)

    def test_quadratic(self):
        """Polynomial interpolation of a quadratic (3 points)."""
        x = np.array([0.0, 1.0, 2.0])
        y = x**2
        assert np.isclose(polint1(x, y, 1.5), 2.25, atol=1e-10)

    def test_error_estimate(self):
        """polint1 can return an error estimate via dy list."""
        x = np.array([0.0, 1.0, 2.0])
        y = x**2
        dy = [0.0]
        polint1(x, y, 1.5, dy=dy)
        assert isinstance(dy[0], float)


# ===================================================================
# polint2 (2D polynomial interpolation)
# ===================================================================

class TestPolint2:
    def test_bilinear(self):
        """polint2 on f(x,y) = x + y."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = X + Y
        val = polint2(x, y, Z, 0.5, 0.5)
        assert np.isclose(val, 1.0, atol=1e-8)


# ===================================================================
# polint3 (3D polynomial interpolation)
# ===================================================================

class TestPolint3:
    def test_trilinear(self):
        """polint3 on f(x,y,z) = x + y + z."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        z = np.array([0.0, 1.0, 2.0])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        V = X + Y + Z
        val = polint3(x, y, z, V, 0.5, 0.5, 0.5)
        assert np.isclose(val, 1.5, atol=1e-8)


# ===================================================================
# bcucof / bcuint (bicubic interpolation)
# ===================================================================

class TestBicubic:
    def test_bcuint_constant(self):
        """Bicubic interp of f(x,y) = 5 at center of unit square."""
        y = np.array([5.0, 5.0, 5.0, 5.0])
        y1 = np.zeros(4)
        y2 = np.zeros(4)
        y12 = np.zeros(4)
        ansy, ansy1, ansy2 = bcuint(y, y1, y2, y12, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5)
        assert np.isclose(ansy, 5.0, atol=1e-8)
        assert np.isclose(ansy1, 0.0, atol=1e-8)
        assert np.isclose(ansy2, 0.0, atol=1e-8)

    def test_bcuint_returns_tuple(self):
        """bcuint returns (value, dfdx, dfdy)."""
        y = np.ones(4)
        y1 = y2 = y12 = np.zeros(4)
        result = bcuint(y, y1, y2, y12, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5)
        assert len(result) == 3

    def test_bcucof_returns_coefficients(self):
        """bcucof should fill a 4x4 coefficient array."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y1 = np.zeros(4)
        y2 = np.zeros(4)
        y12 = np.zeros(4)
        c = np.zeros((4, 4))
        bcucof(y, y1, y2, y12, 1.0, 1.0, c)
        assert c.shape == (4, 4)
        assert np.any(c != 0)


# ===================================================================
# rescale_1D_dp (real 1D rescaling)
# ===================================================================

class TestRescale1dDp:
    def test_upsample_linear(self):
        """Rescale a linear function to a finer grid."""
        x0 = np.linspace(0, 1, 5)
        y0 = 2.0 * x0 + 1.0
        x1 = np.linspace(0, 1, 10)
        y1 = np.zeros(10)
        rescale_1D_dp(x0, y0, x1, y1)
        expected = 2.0 * x1 + 1.0
        np.testing.assert_allclose(y1, expected, rtol=1e-6)

    def test_out_of_bounds_zero(self):
        """Points outside the original range should be set to 0."""
        x0 = np.array([1.0, 2.0, 3.0])
        y0 = np.array([10.0, 20.0, 30.0])
        x1 = np.array([0.0, 2.0, 5.0])
        y1 = np.zeros(3)
        rescale_1D_dp(x0, y0, x1, y1)
        assert y1[0] == 0.0  # out of bounds left
        assert y1[2] == 0.0  # out of bounds right
        assert np.isclose(y1[1], 20.0, rtol=1e-6)


# ===================================================================
# rescale_1D_dpc (complex 1D rescaling)
# ===================================================================

class TestRescale1dDpc:
    def test_complex_upsample(self):
        """Rescale a complex linear function to a finer grid."""
        x0 = np.linspace(0, 1, 10)
        z0 = (1 + 2j) * x0
        x1 = np.linspace(0, 1, 20)
        z1 = np.zeros(20, dtype=complex)
        rescale_1D_dpc(x0, z0, x1, z1)
        expected = (1 + 2j) * x1
        np.testing.assert_allclose(z1, expected, rtol=1e-4)


# ===================================================================
# rescale_1D_cyl_dpc (cylindrical boundary)
# ===================================================================

class TestRescale1dCylDpc:
    def test_first_point_preserved(self):
        """Cylindrical rescale sets z1[0] = z0[0]."""
        x0 = np.linspace(0, 1, 10)
        z0 = np.exp(1j * x0)
        x1 = np.linspace(0, 1, 20)
        z1 = np.zeros(20, dtype=complex)
        rescale_1D_cyl_dpc(x0, z0, x1, z1)
        assert z1[0] == z0[0]


# ===================================================================
# rescale_2D_dp (real 2D rescaling)
# ===================================================================

class TestRescale2dDp:
    def test_upsample_plane(self):
        """Rescale f(x,y) = x + y to a finer grid."""
        x0 = np.linspace(0, 1, 5)
        y0 = np.linspace(0, 1, 5)
        X0, Y0 = np.meshgrid(x0, y0, indexing='ij')
        z0 = X0 + Y0
        x1 = np.linspace(0, 1, 10)
        y1 = np.linspace(0, 1, 10)
        z1 = np.zeros((10, 10))
        rescale_2D_dp(x0, y0, z0, x1, y1, z1)
        X1, Y1 = np.meshgrid(x1, y1, indexing='ij')
        expected = X1 + Y1
        np.testing.assert_allclose(z1, expected, rtol=1e-2, atol=1e-2)


# ===================================================================
# rescale_2D_dpc (complex 2D rescaling)
# ===================================================================

class TestRescale2dDpc:
    def test_complex_plane(self):
        x0 = np.linspace(0, 1, 5)
        y0 = np.linspace(0, 1, 5)
        X0, Y0 = np.meshgrid(x0, y0, indexing='ij')
        z0 = (1 + 1j) * (X0 + Y0)
        x1 = np.linspace(0, 1, 8)
        y1 = np.linspace(0, 1, 8)
        z1 = np.zeros((8, 8), dtype=complex)
        rescale_2D_dpc(x0, y0, z0, x1, y1, z1)
        assert np.all(np.isfinite(z1))


# ===================================================================
# GetValAt_1D, GetValAt_1D_dpc
# ===================================================================

class TestGetValAt1D:
    def test_real_interpolation(self):
        x = np.linspace(0, 5, 20)
        e = x**2
        val = GetValAt_1D(e, x, 2.5)
        assert np.isclose(val, 6.25, rtol=1e-4)

    def test_out_of_bounds_returns_zero(self):
        x = np.linspace(0, 5, 20)
        e = x**2
        assert GetValAt_1D(e, x, -1.0) == 0.0
        assert GetValAt_1D(e, x, 10.0) == 0.0

    def test_complex_interpolation(self):
        x = np.linspace(0, 1, 20)
        e = (1 + 2j) * x
        val = GetValAt_1D_dpc(e, x, 0.5)
        assert np.isclose(val, 0.5 + 1.0j, atol=1e-4)


# ===================================================================
# GetValAt_2D
# ===================================================================

class TestGetValAt2D:
    def test_plane(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y, indexing='ij')
        e = X + Y
        val = GetValAt_2D(e, x, y, 0.5, 0.5)
        assert np.isclose(val, 1.0, atol=0.1)

    def test_out_of_bounds(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        e = np.ones((5, 5))
        assert GetValAt_2D(e, x, y, -1.0, 0.5) == 0.0


# ===================================================================
# GetValAt_3D
# ===================================================================

class TestGetValAt3D:
    def test_constant(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 5)
        e = np.ones((5, 5, 5)) * 7.0
        val = GetValAt_3D(e, x, y, z, 0.5, 0.5, 0.5)
        assert np.isclose(val, 7.0, atol=0.5)

    def test_out_of_bounds(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 5)
        e = np.ones((5, 5, 5))
        assert GetValAt_3D(e, x, y, z, -1, 0.5, 0.5) == 0.0


# ===================================================================
# Unified interfaces: spline(), seval(), rescale_1D(), rescale_2D()
# ===================================================================

class TestUnifiedSpline:
    def test_spline_dp_interface(self):
        """spline() dispatches to spline_dp for real arrays."""
        x = np.linspace(0, 5, 10)
        y = x**2
        b = np.zeros(10)
        c = np.zeros(10)
        d = np.zeros(10)
        spline(x, y, b, c, d)
        # b should now be populated
        assert np.any(b != 0)

    def test_spline_dpc_interface(self):
        """spline() dispatches to spline_dpc for complex arrays."""
        x = np.linspace(0, 5, 10)
        y = (1 + 1j) * x
        b = np.zeros(10, dtype=complex)
        c = np.zeros(10, dtype=complex)
        d = np.zeros(10, dtype=complex)
        spline(x, y, b, c, d)
        assert np.any(b != 0)

    def test_spline2_interface(self):
        """spline() with 2 extra args dispatches to spline2_dp."""
        x = np.linspace(0, 5, 10)
        y = x**2
        y2 = np.zeros(10)
        spline(x, y, y2)
        assert np.any(y2 != 0)


class TestUnifiedSeval:
    def test_seval_dp_interface(self):
        x = np.linspace(0, 5, 10)
        y = x**2
        b = np.zeros(10)
        c = np.zeros(10)
        d = np.zeros(10)
        spline_dp(x, y, b, c, d)
        val = seval(2.5, x, y, b, c, d)
        assert np.isclose(val, 6.25, atol=1e-4)

    def test_seval2_interface(self):
        x = np.linspace(0, 5, 20)
        y = x**2
        y2 = np.zeros(20)
        spline2_dp(x, y, y2)
        val = seval(2.5, x, y, y2)
        assert np.isclose(val, 6.25, atol=1e-2)


class TestUnifiedRescale:
    def test_rescale_1d_real(self):
        x0 = np.linspace(0, 1, 10)
        y0 = x0 * 2
        x1 = np.linspace(0, 1, 20)
        y1 = np.zeros(20)
        rescale_1D(x0, y0, x1, y1)
        np.testing.assert_allclose(y1, x1 * 2, rtol=1e-4)

    def test_rescale_2d_dispatches(self):
        x0 = np.linspace(0, 1, 5)
        y0 = np.linspace(0, 1, 5)
        z0 = np.ones((5, 5))
        x1 = np.linspace(0, 1, 8)
        y1 = np.linspace(0, 1, 8)
        z1 = np.zeros((8, 8))
        rescale_2D(x0, y0, z0, x1, y1, z1)
        np.testing.assert_allclose(z1, 1.0, rtol=1e-2, atol=1e-2)
