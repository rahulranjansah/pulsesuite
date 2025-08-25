"""
spliner.py

High-performance spline and polynomial interpolation routines, ported from Fortran-90 spliner.F90.
Vectorized with NumPy, JIT-accelerated with Numba where appropriate.
"""
import numpy as np
from math import log10, sqrt
try:
    from numba import njit, prange; _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False
try:
    from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, BarycentricInterpolator
    _USE_SCIPY = True
except ImportError:
    _USE_SCIPY = False

from constants import *
from helpers import *

# JIT helper
def _jit(fn):
    return njit(parallel=True, cache=True)(fn) if _USE_NUMBA else fn

class Spliner:
    """
    Spline and polynomial interpolation utilities for scientific computing.
    All methods are static and use camelCase naming.
    """

    @staticmethod
    @_jit
    def locate(x, u):
        """
        Locate index i such that x[i] <= u < x[i+1].
        Parameters
        ----------
        x : ndarray
            Sorted input array.
        u : float
            Value to locate in x.
        Returns
        -------
        int
            Index i satisfying x[i] <= u < x[i+1].
        """
        lo, hi = 0, x.size - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if u < x[mid]:
                hi = mid
            else:
                lo = mid
        return lo

    @staticmethod
    def spline1D(x, y):
        """
        Compute cubic spline coefficients for 1D real data.
        Parameters
        ----------
        x : ndarray
            1D grid points.
        y : ndarray
            1D function values.
        Returns
        -------
        b, c, d : ndarrays
            Spline coefficients.
        """
        n = x.size
        b = np.zeros(n, dtype=y.dtype)
        c = np.zeros(n, dtype=y.dtype)
        d = np.zeros(n, dtype=y.dtype)
        if n < 3:
            b[:] = (y[1] - y[0]) / (x[1] - x[0])
            return b, c, d
        d[0] = x[1] - x[0]
        c[1] = (y[1] - y[0]) / d[0]
        for i in range(1, n-1):
            d[i] = x[i+1] - x[i]
            b[i] = 2.0 * (d[i-1] + d[i])
            c[i+1] = (y[i+1] - y[i]) / d[i]
            c[i] = c[i+1] - c[i]
        b[0] = -d[0]
        b[-1] = -d[-2]
        c[0] = 0.0
        c[-1] = 0.0
        for i in range(1, n):
            t = d[i-1] / b[i-1]
            b[i] = b[i] - t * d[i-1]
            c[i] = c[i] - t * c[i-1]
        c[-1] = c[-1] / b[-1]
        for i in range(n-2, -1, -1):
            c[i] = (c[i] - d[i] * c[i+1]) / b[i]
        b[-1] = (y[-1] - y[-2]) / d[-2] + d[-2] * (c[-2] + 2.0 * c[-1])
        for i in range(n-1):
            b[i] = (y[i+1] - y[i]) / d[i] - d[i] * (c[i+1] + 2.0 * c[i])
            d[i] = (c[i+1] - c[i]) / d[i]
            c[i] = 3.0 * c[i]
        c[-1] = 3.0 * c[-1]
        d[-1] = d[-2]
        return b, c, d

    @staticmethod
    def seval1D(u, x, y, b, c, d):
        """
        Evaluate cubic spline at u for 1D real data.
        Parameters
        ----------
        u : float
            Point to evaluate.
        x, y, b, c, d : ndarrays
            Spline grid, values, and coefficients.
        Returns
        -------
        float
            Interpolated value.
        """
        i = Spliner.locate(x, u)
        dx = u - x[i]
        return y[i] + dx * (b[i] + dx * (c[i] + dx * d[i]))

    # @staticmethod
    # @_jit
    # def spline1DComplex(x, y):
    #     n = x.size
    #     b = np.zeros(n, dtype=y.dtype)
    #     c = np.zeros(n, dtype=y.dtype)
    #     d = np.zeros(n, dtype=y.dtype)
    #     if n < 3:
    #         b[:] = (y[1] - y[0]) / (x[1] - x[0])
    #         return b, c, d
    #     d[0] = x[1] - x[0]
    #     c[1] = (y[1] - y[0]) / d[0]
    #     for i in range(1, n-1):
    #         d[i] = x[i+1] - x[i]
    #         b[i] = 2.0 * (d[i-1] + d[i])
    #         c[i+1] = (y[i+1] - y[i]) / d[i]
    #         c[i] = c[i+1] - c[i]
    #     b[0] = -d[0]
    #     b[-1] = -d[-2]
    #     c[0] = 0.0
    #     c[-1] = 0.0
    #     for i in range(1, n):
    #         t = d[i-1] / b[i-1]
    #         b[i] = b[i] - t * d[i-1]
    #         c[i] = c[i] - t * c[i-1]
    #     c[-1] = c[-1] / b[-1]
    #     for i in range(n-2, -1, -1):
    #         c[i] = (c[i] - d[i] * c[i+1]) / b[i]
    #     b[-1] = (y[-1] - y[-2]) / d[-2] + d[-2] * (c[-2] + 2.0 * c[-1])
    #     for i in range(n-1):
    #         b[i] = (y[i+1] - y[i]) / d[i] - d[i] * (c[i+1] + 2.0 * c[i])
    #         d[i] = (c[i+1] - c[i]) / d[i]
    #         c[i] = 3.0 * c[i]
    #     c[-1] = 3.0 * c[-1]
    #     d[-1] = d[-2]
    #     return b, c, d

    @staticmethod
    def spline1DComplex(x: np.ndarray, y: np.ndarray):
        br, cr, dr = Spliner.spline1D(x, y.real)
        bi, ci, di = Spliner.spline1D(x, y.imag)
        return br + 1j*bi, cr + 1j*ci, dr + 1j*di

    @staticmethod
    def seval1DComplex(u, x, y, b, c, d):
        i = Spliner.locate(x, u)
        dx = u - x[i]
        return y[i] + dx * (b[i] + dx * (c[i] + dx * d[i]))

    @staticmethod
    def polint1(xa: np.ndarray, ya: np.ndarray, x: float):
        """
        1D polynomial interpolation. Uses SciPy's BarycentricInterpolator if available,
        else falls back to Lagrange form.
        """
        if _USE_SCIPY:
            interp = BarycentricInterpolator(xa, ya)
            return interp(x)
        # fallback: Lagrange form
        n = xa.size
        if n == 2:
            return ya[0] + (ya[1] - ya[0]) * (x - xa[0]) / (xa[1] - xa[0])
        res = 0.0
        for j in range(n):
            term = ya[j]
            for m in range(n):
                if m != j:
                    term *= (x - xa[m]) / (xa[j] - xa[m])
            res += term
        return res

    @staticmethod
    def polint2(x1a: np.ndarray, x2a: np.ndarray,
                ya: np.ndarray, x1: float, x2: float):
        """
        2D polynomial interpolation via nested 1D calls (row-wise Barycentric).
        """
        # interpolate along second axis for each x1a point
        rows = ya.shape[0]
        if _USE_SCIPY:
            from scipy.interpolate import BarycentricInterpolator
            vals = np.empty(rows, dtype=ya.dtype)
            for j in range(rows):
                bary = BarycentricInterpolator(x2a, ya[j, :])
                vals[j] = bary(x2)
            bary1 = BarycentricInterpolator(x1a, vals)
            return bary1(x1)
        # fallback nested Lagrange
        temp = np.array([Spliner.polint1(x2a, ya[j, :], x2) for j in range(rows)])
        return Spliner.polint1(x1a, temp, x1)(x1a, temp, x1)

    @staticmethod
    def polint3(x1a: np.ndarray, x2a: np.ndarray, x3a: np.ndarray,
                ya: np.ndarray, x1: float, x2: float, x3: float):
        """
        3D polynomial interpolation via nested 1D calls (using Barycentric for speed).
        """
        dim1 = ya.shape[0]
        if _USE_SCIPY:
            from scipy.interpolate import BarycentricInterpolator
            vals2 = np.empty(dim1, dtype=ya.dtype)
            for i in range(dim1):
                # 2D on slice
                slice2d = ya[i]
                dim2 = slice2d.shape[0]
                vals1 = np.empty(dim2, dtype=ya.dtype)
                for j in range(dim2):
                    bary2 = BarycentricInterpolator(x3a, slice2d[j, :])
                    vals1[j] = bary2(x3)
                bary1 = BarycentricInterpolator(x2a, vals1)
                vals2[i] = bary1(x2)
            bary0 = BarycentricInterpolator(x1a, vals2)
            return bary0(x1)
        # fallback nested Lagrange
        temp2 = np.array([Spliner.polint2(x2a, x3a, ya[j, :, :], x2, x3)
                          for j in range(ya.shape[0])])
        return Spliner.polint1(x1a, temp2, x1)(x1a, temp2, x1)

    @staticmethod
    def spline2D(x, y, z, kind='cubic'):
        """
        Compute 2D spline interpolator for real data using scipy.
        Parameters
        ----------
        x, y : 1D arrays
            Grid points.
        z : 2D array
            Function values.
        kind : str
            Spline kind ('cubic', 'linear', etc.)
        Returns
        -------
        callable
            Interpolator function.
        """
        if _USE_SCIPY:
            return RectBivariateSpline(x, y, z, kx=3 if kind=='cubic' else 1, ky=3 if kind=='cubic' else 1)
        else:
            with open('../future/spliner.txt', 'a') as f:
                f.write('scipy.interpolate not available for spline2D\n')
            raise ImportError('scipy.interpolate required for spline2D')

    @staticmethod
    def seval2D(x0, y0, x, y, z, kind='cubic'):
        """
        Evaluate 2D spline at (x0, y0) using scipy.
        """
        interp = Spliner.spline2D(x, y, z, kind=kind)
        return float(interp(x0, y0))

    @staticmethod
    def spline3D(x, y, z, values, method='linear'):
        """
        Compute 3D spline interpolator for real data using scipy.
        Parameters
        ----------
        x, y, z : 1D arrays
            Grid points.
        values : 3D array
            Function values.
        method : str
            Interpolation method ('linear', 'nearest')
        Returns
        -------
        callable
            Interpolator function.
        """
        if _USE_SCIPY:
            return RegularGridInterpolator((x, y, z), values, method=method)
        else:
            with open('../future/spliner.txt', 'a') as f:
                f.write('scipy.interpolate not available for spline3D\n')
            raise ImportError('scipy.interpolate required for spline3D')

    @staticmethod
    def seval3D(x0, y0, z0, x, y, z, values, method='linear'):
        """
        Evaluate 3D spline at (x0, y0, z0) using scipy.
        """
        interp = Spliner.spline3D(x, y, z, values, method=method)
        return float(interp((x0, y0, z0)))

    @staticmethod
    def bcuint(x, y, z):
        """
        Bicubic interpolation using scipy's interp2d (for legacy API).
        """
        if _USE_SCIPY:
            # build spline on original grid
            spline = RectBivariateSpline(x, y, z, kx=3, ky=3)
            return lambda xi, yi: float(spline(xi, yi))
        else:
            with open('../future/spliner.txt', 'a') as f:
                f.write('scipy.interpolate not available for bcuint\n')
            raise ImportError('scipy.interpolate required for bcuint')

    @staticmethod
    def bcucof(x, y, z):
        """
        Bicubic coefficients are not directly exposed in scipy; use bcuint for interpolation.
        """
        # Not directly available; log and raise
        with open('../future/spliner.txt', 'a') as f:
            f.write('bcucof (bicubic coefficients) not directly available in scipy\n')
        raise NotImplementedError('bcucof (bicubic coefficients) not directly available in scipy')

# cache for precomputed splines
_cache = {}