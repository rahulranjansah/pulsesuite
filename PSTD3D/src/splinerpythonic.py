"""
Spliner Module - Pythonic Implementation

A comprehensive spline interpolation module providing 1D, 2D, and 3D interpolation
capabilities for both real and complex arrays. This module maintains 1:1 naming
parity with the original Fortran spliner module while providing Pythonic interfaces
and performance optimizations.

Key Features:
- Cubic spline interpolation for 1D and 2D arrays
- Polynomial interpolation for 1D, 2D, and 3D arrays
- Bicubic interpolation for 2D arrays
- Support for both real and complex data types
- Numba JIT compilation for performance-critical sections
- Vectorized operations where possible

Mathematical Background:
- Cubic splines: S(x) = a + b(x-xi) + c(x-xi)² + d(x-xi)³
- Polynomial interpolation using Neville's algorithm
- Bicubic interpolation with 16-point stencil

Author: Rahul R. Sah
"""

import numpy as np
from typing import Union, Optional, Tuple, Any
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback decorator if numba is not available
    def njit(cache=True, parallel=False):
        def decorator(func):
            return func
        return decorator
    def prange(*args, **kwargs):
        return range(*args, **kwargs)
    NUMBA_AVAILABLE = False

try:
    from nptyping import NDArray
except ImportError:
    # Fallback type hint if nptyping is not available
    NDArray = np.ndarray

# Type aliases for clarity
_dp = np.float64
_dc = np.complex128
_dp_array = NDArray[_dp]
_dc_array = NDArray[_dc]

# Global variables for optimization
_cached_splines = {}
_cached_locate = {}

@njit(cache=True)
def _iminloc(arr: _dp_array) -> int:
    """Find index of minimum element in array."""
    return int(np.argmin(arr))

@njit(cache=True)
def _locate(x: _dp_array, u: _dp) -> int:
    """
    Locate position in array bracketing a value.

    Returns index i where x[i] and x[i+1] bracket u.
    Uses binary search for efficiency.

    Parameters
    ----------
    x : (N,) ndarray
        Sorted array of x values
    u : float
        Value to locate

    Returns
    -------
    i : int
        Index where x[i] <= u < x[i+1]
    """
    n = len(x)

    if u < x[0] or u > x[-1]:
        return 0 if u < x[0] else n-1

    # Binary search
    i = 0
    j = n

    while j - i > 1:
        k = (i + j) // 2
        if u < x[k]:
            j = k
        else:
            i = k

    return i

@njit(cache=True)
def _spline_dp(x: _dp_array, y: _dp_array, b: _dp_array, c: _dp_array, d: _dp_array) -> None:
    """
    Compute cubic spline coefficients for real arrays.

    Uses natural boundary conditions (second derivative = 0 at endpoints).

    Parameters
    ----------
    x : (N,) ndarray
        Input x coordinates (must be strictly increasing)
    y : (N,) ndarray
        Input y values
    b, c, d : (N,) ndarray
        Output spline coefficients

    Notes
    -----
    The spline is defined as: S(x) = y[i] + b[i]*(x-x[i]) + c[i]*(x-x[i])² + d[i]*(x-x[i])³
    """
    n = len(x)

    if n < 2:
        raise ValueError("Not enough points to spline")

    if n < 3:
        # Linear interpolation for 2 points
        b[0] = (y[1] - y[0]) / (x[1] - x[0])
        c[0] = 0.0
        d[0] = 0.0
        b[1] = b[0]
        c[1] = 0.0
        d[1] = 0.0
        return

    # Initialize arrays
    d[0] = x[1] - x[0]
    c[1] = (y[1] - y[0]) / d[0]

    for i in range(1, n-1):
        d[i] = x[i+1] - x[i]
        b[i] = 2.0 * (d[i-1] + d[i])
        c[i+1] = (y[i+1] - y[i]) / d[i]
        c[i] = c[i+1] - c[i]

    # Boundary conditions
    b[0] = -d[0]
    b[n-1] = -d[n-2]
    c[0] = 0.0
    c[n-1] = 0.0

    if n != 3:
        c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
        c[n-1] = c[n-2] / (x[n-1] - x[n-3]) - c[n-3] / (x[n-2] - x[n-4])
        c[0] = c[0] * d[0]**2 / (x[3] - x[0])
        c[n-1] = -c[n-1] * d[n-2]**2 / (x[n-1] - x[n-4])

    # Solve tridiagonal system
    for i in range(1, n):
        t = d[i-1] / b[i-1]
        b[i] = b[i] - t * d[i-1]
        c[i] = c[i] - t * c[i-1]

    c[n-1] = c[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        c[i] = (c[i] - d[i] * c[i+1]) / b[i]

    # Compute b and d coefficients
    b[n-1] = (y[n-1] - y[n-2]) / d[n-2] + d[n-2] * (c[n-2] + 2.0 * c[n-1])
    for i in range(n-1):
        b[i] = (y[i+1] - y[i]) / d[i] - d[i] * (c[i+1] + 2.0 * c[i])
        d[i] = (c[i+1] - c[i]) / d[i]
        c[i] = 3.0 * c[i]

    c[n-1] = 3.0 * c[n-1]
    d[n-1] = d[n-2]

@njit(cache=True)
def _spline_dpc(x: _dp_array, y: _dc_array, b: _dc_array, c: _dc_array, d: _dc_array) -> None:
    """
    Compute cubic spline coefficients for complex arrays.

    Uses natural boundary conditions (second derivative = 0 at endpoints).

    Parameters
    ----------
    x : (N,) ndarray
        Input x coordinates (must be strictly increasing)
    y : (N,) ndarray
        Input y values (complex)
    b, c, d : (N,) ndarray
        Output spline coefficients (complex)
    """
    n = len(x)

    if n < 2:
        raise ValueError("Not enough points to spline")

    if n < 3:
        # Linear interpolation for 2 points
        b[0] = (y[1] - y[0]) / (x[1] - x[0])
        c[0] = 0.0 + 0.0j
        d[0] = 0.0 + 0.0j
        b[1] = b[0]
        c[1] = 0.0 + 0.0j
        d[1] = 0.0 + 0.0j
        return

    # Initialize arrays
    d[0] = x[1] - x[0]
    c[1] = (y[1] - y[0]) / d[0]

    for i in range(1, n-1):
        d[i] = x[i+1] - x[i]
        b[i] = 2.0 * (d[i-1] + d[i])
        c[i+1] = (y[i+1] - y[i]) / d[i]
        c[i] = c[i+1] - c[i]

    # Boundary conditions
    b[0] = -d[0]
    b[n-1] = -d[n-2]
    c[0] = 0.0 + 0.0j
    c[n-1] = 0.0 + 0.0j

    if n != 3:
        c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
        c[n-1] = c[n-2] / (x[n-1] - x[n-3]) - c[n-3] / (x[n-2] - x[n-4])
        c[0] = c[0] * d[0]**2 / (x[3] - x[0])
        c[n-1] = -c[n-1] * d[n-2]**2 / (x[n-1] - x[n-4])

    # Solve tridiagonal system
    for i in range(1, n):
        t = d[i-1] / b[i-1]
        b[i] = b[i] - t * d[i-1]
        c[i] = c[i] - t * c[i-1]

    c[n-1] = c[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        c[i] = (c[i] - d[i] * c[i+1]) / b[i]

    # Compute b and d coefficients
    b[n-1] = (y[n-1] - y[n-2]) / d[n-2] + d[n-2] * (c[n-2] + 2.0 * c[n-1])
    for i in range(n-1):
        b[i] = (y[i+1] - y[i]) / d[i] - d[i] * (c[i+1] + 2.0 * c[i])
        d[i] = (c[i+1] - c[i]) / d[i]
        c[i] = 3.0 * c[i]

    c[n-1] = 3.0 * c[n-1]
    d[n-1] = d[n-2]

@njit(cache=True)
def _seval_dp(u: _dp, x: _dp_array, y: _dp_array, b: _dp_array, c: _dp_array, d: _dp_array) -> _dp:
    """
    Evaluate cubic spline at point u.

    Parameters
    ----------
    u : float
        Point to evaluate at
    x : (N,) ndarray
        Input x coordinates
    y : (N,) ndarray
        Input y values
    b, c, d : (N,) ndarray
        Spline coefficients

    Returns
    -------
    y_val : float
        Interpolated value at u
    """
    i = _locate(x, u)
    dx = u - x[i]
    return y[i] + dx * (b[i] + dx * (c[i] + dx * d[i]))

@njit(cache=True)
def _seval_dpc(u: _dp, x: _dp_array, y: _dc_array, b: _dc_array, c: _dc_array, d: _dc_array) -> _dc:
    """
    Evaluate cubic spline at point u for complex arrays.

    Parameters
    ----------
    u : float
        Point to evaluate at
    x : (N,) ndarray
        Input x coordinates
    y : (N,) ndarray
        Input y values (complex)
    b, c, d : (N,) ndarray
        Spline coefficients (complex)

    Returns
    -------
    y_val : complex
        Interpolated value at u
    """
    i = _locate(x, u)
    dx = u - x[i]
    return y[i] + dx * (b[i] + dx * (c[i] + dx * d[i]))

# Public interface functions
def spline_dp(x: _dp_array, y: _dp_array, b: _dp_array, c: _dp_array, d: _dp_array) -> None:
    """Compute cubic spline coefficients for real arrays."""
    _spline_dp(x, y, b, c, d)

def spline_dpc(x: _dp_array, y: _dc_array, b: _dc_array, c: _dc_array, d: _dc_array) -> None:
    """Compute cubic spline coefficients for complex arrays."""
    _spline_dpc(x, y, b, c, d)

def seval_dp(u: _dp, x: _dp_array, y: _dp_array, b: _dp_array, c: _dp_array, d: _dp_array) -> _dp:
    """Evaluate cubic spline for real arrays."""
    return _seval_dp(u, x, y, b, c, d)

def seval_dpc(u: _dp, x: _dp_array, y: _dc_array, b: _dc_array, c: _dc_array, d: _dc_array) -> _dc:
    """Evaluate cubic spline for complex arrays."""
    return _seval_dpc(u, x, y, b, c, d)

def locate(x: _dp_array, u: _dp) -> int:
    """Locate position in array bracketing a value."""
    return _locate(x, u)

def iminloc(arr: _dp_array) -> int:
    """Find index of minimum element in array."""
    return _iminloc(arr)

@njit(cache=True)
def _spline2_dp(x: _dp_array, y: _dp_array, y2: _dp_array) -> None:
    """
    Compute cubic spline second derivatives for real arrays.

    Parameters
    ----------
    x : (N,) ndarray
        Input x coordinates (must be strictly increasing)
    y : (N,) ndarray
        Input y values
    y2 : (N,) ndarray
        Output second derivatives
    """
    n = len(x)
    u = np.zeros(n, dtype=_dp)

    y2[0] = 0.0
    u[0] = 0.0

    for i in range(1, n-1):
        sig = (x[i] - x[i-1]) / (x[i+1] - x[i-1])
        p = sig * y2[i-1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (y[i+1] - y[i]) / (x[i+1] - x[i]) - (y[i] - y[i-1]) / (x[i] - x[i-1])
        u[i] = (6.0 * u[i] / (x[i+1] - x[i-1]) - sig * u[i-1]) / p

    y2[n-1] = 0.0

    for i in range(n-2, -1, -1):
        y2[i] = y2[i] * y2[i+1] + u[i]

@njit(cache=True)
def _spline2_dpc(x: _dp_array, y: _dc_array, y2: _dc_array) -> None:
    """
    Compute cubic spline second derivatives for complex arrays.

    Parameters
    ----------
    x : (N,) ndarray
        Input x coordinates (must be strictly increasing)
    y : (N,) ndarray
        Input y values (complex)
    y2 : (N,) ndarray
        Output second derivatives (complex)
    """
    n = len(x)
    u = np.zeros(n, dtype=_dc)

    y2[0] = 0.0 + 0.0j
    u[0] = 0.0 + 0.0j

    for i in range(1, n-1):
        sig = (x[i] - x[i-1]) / (x[i+1] - x[i-1])
        p = sig * y2[i-1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (y[i+1] - y[i]) / (x[i+1] - x[i]) - (y[i] - y[i-1]) / (x[i] - x[i-1])
        u[i] = (6.0 * u[i] / (x[i+1] - x[i-1]) - sig * u[i-1]) / p

    y2[n-1] = 0.0 + 0.0j

    for i in range(n-2, -1, -1):
        y2[i] = y2[i] * y2[i+1] + u[i]

@njit(cache=True)
def _seval2_dp(x0: _dp, x: _dp_array, y: _dp_array, y2: _dp_array) -> _dp:
    """
    Evaluate cubic spline using second derivatives for real arrays.

    Parameters
    ----------
    x0 : float
        Point to evaluate at
    x : (N,) ndarray
        Input x coordinates
    y : (N,) ndarray
        Input y values
    y2 : (N,) ndarray
        Second derivatives

    Returns
    -------
    y_val : float
        Interpolated value at x0
    """
    i = _locate(x, x0)
    h = x[i+1] - x[i]
    a = (x[i+1] - x0) / h
    b = (x0 - x[i]) / h

    return a * y[i] + b * y[i+1] + ((a**3 - a) * y2[i] + (b**3 - b) * y2[i+1]) * h**2 / 6.0

@njit(cache=True)
def _seval2_dpc(x0: _dp, x: _dp_array, y: _dc_array, y2: _dc_array) -> _dc:
    """
    Evaluate cubic spline using second derivatives for complex arrays.

    Parameters
    ----------
    x0 : float
        Point to evaluate at
    x : (N,) ndarray
        Input x coordinates
    y : (N,) ndarray
        Input y values (complex)
    y2 : (N,) ndarray
        Second derivatives (complex)

    Returns
    -------
    y_val : complex
        Interpolated value at x0
    """
    i = _locate(x, x0)
    h = x[i+1] - x[i]
    a = (x[i+1] - x0) / h
    b = (x0 - x[i]) / h

    return a * y[i] + b * y[i+1] + ((a**3 - a) * y2[i] + (b**3 - b) * y2[i+1]) * h**2 / 6.0

def spline2_dp(x: _dp_array, y: _dp_array, y2: _dp_array) -> None:
    """Compute cubic spline second derivatives for real arrays."""
    _spline2_dp(x, y, y2)

def spline2_dpc(x: _dp_array, y: _dc_array, y2: _dc_array) -> None:
    """Compute cubic spline second derivatives for complex arrays."""
    _spline2_dpc(x, y, y2)

def seval2_dp(x0: _dp, x: _dp_array, y: _dp_array, y2: _dp_array) -> _dp:
    """Evaluate cubic spline using second derivatives for real arrays."""
    return _seval2_dp(x0, x, y, y2)

def seval2_dpc(x0: _dp, x: _dp_array, y: _dc_array, y2: _dc_array) -> _dc:
    """Evaluate cubic spline using second derivatives for complex arrays."""
    return _seval2_dpc(x0, x, y, y2)

@njit(cache=True)
def _polint1(xa: _dp_array, ya: _dp_array, x: _dp) -> Tuple[_dp, _dp]:
    """
    Polynomial interpolation for 1D arrays using Neville's algorithm.

    Parameters
    ----------
    xa : (N,) ndarray
        Input x coordinates
    ya : (N,) ndarray
        Input y values
    x : float
        Point to interpolate at

    Returns
    -------
    y : float
        Interpolated value
    dy : float
        Error estimate
    """
    n = len(xa)
    c = ya.copy()
    d = ya.copy()
    ho = xa - x

    ns = _iminloc(np.abs(x - xa))
    y = ya[ns]
    ns = ns - 1

    for m in range(1, n):
        den = ho[:n-m] - ho[m:n]
        if np.any(den == 0.0):
            raise ValueError("polint: calculation failure")
        den = (c[1:n-m+1] - d[:n-m]) / den
        d[:n-m] = ho[m:n] * den
        c[:n-m] = ho[:n-m] * den
        if 2*ns < n-m:
            dyt = c[ns + 1]
        else:
            dyt = d[ns]
            ns = ns - 1
        y = y + dyt

    return y, dyt

@njit(cache=True)
def _polint2(x1a: _dp_array, x2a: _dp_array, ya: _dp_array, x1: _dp, x2: _dp) -> Tuple[_dp, _dp]:
    """
    Polynomial interpolation for 2D arrays.

    Parameters
    ----------
    x1a : (M,) ndarray
        Input x1 coordinates
    x2a : (N,) ndarray
        Input x2 coordinates
    ya : (M, N) ndarray
        Input y values
    x1, x2 : float
        Points to interpolate at

    Returns
    -------
    y : float
        Interpolated value
    dy : float
        Error estimate
    """
    m = len(x1a)
    ymtmp = np.zeros(m, dtype=_dp)

    for j in range(m):
        ymtmp[j], _ = _polint1(x2a, ya[j, :], x2)

    return _polint1(x1a, ymtmp, x1)

@njit(cache=True)
def _polint3(x1a: _dp_array, x2a: _dp_array, x3a: _dp_array, ya: _dp_array,
             x1: _dp, x2: _dp, x3: _dp) -> Tuple[_dp, _dp]:
    """
    Polynomial interpolation for 3D arrays.

    Parameters
    ----------
    x1a : (M,) ndarray
        Input x1 coordinates
    x2a : (N,) ndarray
        Input x2 coordinates
    x3a : (P,) ndarray
        Input x3 coordinates
    ya : (M, N, P) ndarray
        Input y values
    x1, x2, x3 : float
        Points to interpolate at

    Returns
    -------
    y : float
        Interpolated value
    dy : float
        Error estimate
    """
    m = len(x1a)
    ymtmp = np.zeros(m, dtype=_dp)

    for j in range(m):
        ymtmp[j], _ = _polint2(x2a, x3a, ya[j, :, :], x2, x3)

    return _polint1(x1a, ymtmp, x1)

def polint1(xa: _dp_array, ya: _dp_array, x: _dp, dy: Optional[_dp_array] = None) -> Union[_dp, Tuple[_dp, _dp]]:
    """
    Polynomial interpolation for 1D arrays.

    Parameters
    ----------
    xa : (N,) ndarray
        Input x coordinates
    ya : (N,) ndarray
        Input y values
    x : float
        Point to interpolate at
    dy : ndarray, optional
        Output array for error estimate

    Returns
    -------
    y : float or tuple
        Interpolated value, or (value, error) if dy is None
    """
    y, dyt = _polint1(xa, ya, x)
    if dy is not None:
        dy[0] = dyt
        return y
    else:
        return y, dyt

def polint2(x1a: _dp_array, x2a: _dp_array, ya: _dp_array, x1: _dp, x2: _dp,
            dy: Optional[_dp_array] = None) -> Union[_dp, Tuple[_dp, _dp]]:
    """
    Polynomial interpolation for 2D arrays.

    Parameters
    ----------
    x1a : (M,) ndarray
        Input x1 coordinates
    x2a : (N,) ndarray
        Input x2 coordinates
    ya : (M, N) ndarray
        Input y values
    x1, x2 : float
        Points to interpolate at
    dy : ndarray, optional
        Output array for error estimate

    Returns
    -------
    y : float or tuple
        Interpolated value, or (value, error) if dy is None
    """
    y, dyt = _polint2(x1a, x2a, ya, x1, x2)
    if dy is not None:
        dy[0] = dyt
        return y
    else:
        return y, dyt

def polint3(x1a: _dp_array, x2a: _dp_array, x3a: _dp_array, ya: _dp_array,
            x1: _dp, x2: _dp, x3: _dp, dy: Optional[_dp_array] = None) -> Union[_dp, Tuple[_dp, _dp]]:
    """
    Polynomial interpolation for 3D arrays.

    Parameters
    ----------
    x1a : (M,) ndarray
        Input x1 coordinates
    x2a : (N,) ndarray
        Input x2 coordinates
    x3a : (P,) ndarray
        Input x3 coordinates
    ya : (M, N, P) ndarray
        Input y values
    x1, x2, x3 : float
        Points to interpolate at
    dy : ndarray, optional
        Output array for error estimate

    Returns
    -------
    y : float or tuple
        Interpolated value, or (value, error) if dy is None
    """
    y, dyt = _polint3(x1a, x2a, x3a, ya, x1, x2, x3)
    if dy is not None:
        dy[0] = dyt
        return y
    else:
        return y, dyt

@njit(cache=True)
def _bcucof(y: _dp_array, y1: _dp_array, y2: _dp_array, y12: _dp_array,
            d1: _dp, d2: _dp) -> _dp_array:
    """
    Compute bicubic interpolation coefficients.

    Parameters
    ----------
    y : (4,) ndarray
        Function values at corners
    y1 : (4,) ndarray
        First derivatives w.r.t. x1
    y2 : (4,) ndarray
        First derivatives w.r.t. x2
    y12 : (4,) ndarray
        Mixed second derivatives
    d1, d2 : float
        Grid spacing in x1 and x2 directions

    Returns
    -------
    c : (4, 4) ndarray
        Bicubic coefficients
    """
    # Weight matrix for bicubic interpolation
    wt = np.array([
        [1, 0, -3, 2, 0, 0, 0, 0, -3, 0, 9, -6, 2, 0, -6, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -9, 6, -2, 0, 6, -4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -6, 0, 0, -6, 4],
        [0, 0, 3, -2, 0, 0, 0, 0, 0, 0, -9, 6, 0, 0, 6, -4],
        [0, 0, 0, 0, 1, 0, -3, 2, -2, 0, 6, -4, 1, 0, -3, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 3, -2, 1, 0, -3, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 2, 0, 0, 3, -2],
        [0, 0, 0, 0, 0, 0, 3, -2, 0, 0, -6, 4, 0, 0, 3, -2],
        [0, 1, -2, 1, 0, 0, 0, 0, 0, -3, 6, -3, 0, 2, -4, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -6, 3, 0, -2, 4, -2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -3, 0, 0, -2, 2],
        [0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 1, -2, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1]
    ], dtype=_dp)

    x = np.zeros(16, dtype=_dp)
    x[:4] = y
    x[4:8] = y1 * d1
    x[8:12] = y2 * d2
    x[12:16] = y12 * d1 * d2

    x = np.dot(wt, x)
    c = x.reshape(4, 4)

    return c

@njit(cache=True)
def _bcuint(y: _dp_array, y1: _dp_array, y2: _dp_array, y12: _dp_array,
            x1l: _dp, x1u: _dp, x2l: _dp, x2u: _dp, x1: _dp, x2: _dp) -> Tuple[_dp, _dp, _dp]:
    """
    Bicubic interpolation.

    Parameters
    ----------
    y : (4,) ndarray
        Function values at corners
    y1 : (4,) ndarray
        First derivatives w.r.t. x1
    y2 : (4,) ndarray
        First derivatives w.r.t. x2
    y12 : (4,) ndarray
        Mixed second derivatives
    x1l, x1u : float
        Lower and upper bounds in x1
    x2l, x2u : float
        Lower and upper bounds in x2
    x1, x2 : float
        Interpolation points

    Returns
    -------
    ansy : float
        Interpolated value
    ansy1 : float
        Interpolated first derivative w.r.t. x1
    ansy2 : float
        Interpolated first derivative w.r.t. x2
    """
    c = _bcucof(y, y1, y2, y12, x1u - x1l, x2u - x2l)

    t = (x1 - x1l) / (x1u - x1l)
    u = (x2 - x2l) / (x2u - x2l)

    ansy = 0.0
    ansy2 = 0.0
    ansy1 = 0.0

    for i in range(3, -1, -1):
        ansy = t * ansy + ((c[i, 3] * u + c[i, 2]) * u + c[i, 1]) * u + c[i, 0]
        ansy2 = t * ansy2 + (3.0 * c[i, 3] * u + 2.0 * c[i, 2]) * u + c[i, 1]
        ansy1 = u * ansy1 + (3.0 * c[3, i] * t + 2.0 * c[2, i]) * t + c[1, i]

    ansy1 = ansy1 / (x1u - x1l)
    ansy2 = ansy2 / (x2u - x2l)

    return ansy, ansy1, ansy2

def bcucof(y: _dp_array, y1: _dp_array, y2: _dp_array, y12: _dp_array,
           d1: _dp, d2: _dp) -> _dp_array:
    """Compute bicubic interpolation coefficients."""
    return _bcucof(y, y1, y2, y12, d1, d2)

def bcuint(y: _dp_array, y1: _dp_array, y2: _dp_array, y12: _dp_array,
           x1l: _dp, x1u: _dp, x2l: _dp, x2u: _dp, x1: _dp, x2: _dp) -> Tuple[_dp, _dp, _dp]:
    """Bicubic interpolation."""
    return _bcuint(y, y1, y2, y12, x1l, x1u, x2l, x2u, x1, x2)

def rescale_1D_dp(x0: _dp_array, y0: _dp_array, x1: _dp_array, y1: _dp_array) -> None:
    """
    Rescale 1D real array from grid x0 to grid x1.

    Parameters
    ----------
    x0 : (N,) ndarray
        Original x coordinates
    y0 : (N,) ndarray
        Original y values
    x1 : (M,) ndarray
        New x coordinates
    y1 : (M,) ndarray
        Output y values
    """
    if len(x0) != len(y0):
        raise ValueError("Bad sizes in rescale_1D_dp")

    n = len(x0)
    b = np.zeros(n, dtype=_dp)
    c = np.zeros(n, dtype=_dp)
    d = np.zeros(n, dtype=_dp)

    _spline_dp(x0, y0, b, c, d)

    for i in range(len(x1)):
        if x1[i] >= np.min(x0) and x1[i] <= np.max(x0):
            y1[i] = _seval_dp(x1[i], x0, y0, b, c, d)
        else:
            y1[i] = 0.0

def rescale_1D_dpc(x0: _dp_array, z0: _dc_array, x1: _dp_array, z1: _dc_array) -> None:
    """
    Rescale 1D complex array from grid x0 to grid x1.

    Parameters
    ----------
    x0 : (N,) ndarray
        Original x coordinates
    z0 : (N,) ndarray
        Original z values (complex)
    x1 : (M,) ndarray
        New x coordinates
    z1 : (M,) ndarray
        Output z values (complex)
    """
    if len(x0) != len(z0):
        raise ValueError("Bad sizes in rescale_1D_dpc")

    n = len(x0)
    b = np.zeros(n, dtype=_dc)
    c = np.zeros(n, dtype=_dc)
    d = np.zeros(n, dtype=_dc)

    _spline_dpc(x0, z0, b, c, d)

    for i in range(len(x1)):
        if x1[i] >= np.min(x0) and x1[i] <= np.max(x0):
            z1[i] = _seval_dpc(x1[i], x0, z0, b, c, d)
        else:
            z1[i] = 0.0 + 0.0j

def rescale_1D_cyl_dpc(x0: _dp_array, z0: _dc_array, x1: _dp_array, z1: _dc_array) -> None:
    """
    Rescale 1D complex array from grid x0 to grid x1 (cylindrical version).

    Special handling for cylindrical coordinates where z1[0] = z0[0].

    Parameters
    ----------
    x0 : (N,) ndarray
        Original x coordinates
    z0 : (N,) ndarray
        Original z values (complex)
    x1 : (M,) ndarray
        New x coordinates
    z1 : (M,) ndarray
        Output z values (complex)
    """
    if len(x0) != len(z0):
        raise ValueError("Bad sizes in rescale_1D_cyl_dpc")

    n = len(x0)
    b = np.zeros(n, dtype=_dc)
    c = np.zeros(n, dtype=_dc)
    d = np.zeros(n, dtype=_dc)

    _spline_dpc(x0, z0, b, c, d)

    z1[0] = z0[0]

    for i in range(1, len(x1)):
        if x1[i] >= np.min(x0) and x1[i] <= np.max(x0):
            z1[i] = _seval_dpc(x1[i], x0, z0, b, c, d)
        else:
            z1[i] = 0.0 + 0.0j

def rescale_2D_dp(x0: _dp_array, y0: _dp_array, z0: _dp_array,
                  x1: _dp_array, y1: _dp_array, z1: _dp_array) -> None:
    """
    Rescale 2D real array from grid (x0, y0) to grid (x1, y1).

    Parameters
    ----------
    x0, y0 : (M,), (N,) ndarray
        Original grid coordinates
    z0 : (M, N) ndarray
        Original function values
    x1, y1 : (P,), (Q,) ndarray
        New grid coordinates
    z1 : (P, Q) ndarray
        Output function values
    """
    zt = np.zeros((len(x1), len(y0)), dtype=_dp)

    # Interpolate along first dimension
    for i in range(len(y0)):
        rescale_1D_dp(x0, z0[:, i], x1, zt[:, i])

    # Interpolate along second dimension
    for i in range(len(x1)):
        rescale_1D_dp(y0, zt[i, :], y1, z1[i, :])

def rescale_2D_dpc(x0: _dp_array, y0: _dp_array, z0: _dc_array,
                   x1: _dp_array, y1: _dp_array, z1: _dc_array) -> None:
    """
    Rescale 2D complex array from grid (x0, y0) to grid (x1, y1).

    Parameters
    ----------
    x0, y0 : (M,), (N,) ndarray
        Original grid coordinates
    z0 : (M, N) ndarray
        Original function values (complex)
    x1, y1 : (P,), (Q,) ndarray
        New grid coordinates
    z1 : (P, Q) ndarray
        Output function values (complex)
    """
    zt = np.zeros((len(x1), len(y0)), dtype=_dc)

    # Interpolate along first dimension
    for i in range(len(y0)):
        rescale_1D_dpc(x0, z0[:, i], x1, zt[:, i])

    # Interpolate along second dimension
    for i in range(len(x1)):
        rescale_1D_dpc(y0, zt[i, :], y1, z1[i, :])

def GetValAt_1D(e: _dp_array, x0: _dp_array, x1: _dp) -> _dp:
    """
    Interpolate 1D array at arbitrary point using cubic spline.

    Parameters
    ----------
    e : (N,) ndarray
        Function values
    x0 : (N,) ndarray
        Grid coordinates
    x1 : float
        Interpolation point

    Returns
    -------
    y : float
        Interpolated value
    """
    if x1 < np.min(x0) or x1 > np.max(x0):
        return 0.0

    n = len(e)
    b = np.zeros(n, dtype=_dp)
    c = np.zeros(n, dtype=_dp)
    d = np.zeros(n, dtype=_dp)

    _spline_dp(x0, e, b, c, d)
    return _seval_dp(x1, x0, e, b, c, d)

def GetValAt_1D_dpc(e: _dc_array, x0: _dp_array, x1: _dp) -> _dc:
    """
    Interpolate 1D complex array at arbitrary point using cubic spline.

    Parameters
    ----------
    e : (N,) ndarray
        Function values (complex)
    x0 : (N,) ndarray
        Grid coordinates
    x1 : float
        Interpolation point

    Returns
    -------
    y : complex
        Interpolated value
    """
    if x1 < np.min(x0) or x1 > np.max(x0):
        return 0.0 + 0.0j

    n = len(e)
    b = np.zeros(n, dtype=_dc)
    c = np.zeros(n, dtype=_dc)
    d = np.zeros(n, dtype=_dc)

    _spline_dpc(x0, e, b, c, d)
    return _seval_dpc(x1, x0, e, b, c, d)

def GetValAt_2D(e: _dp_array, x0a: _dp_array, x1a: _dp_array, x0: _dp, x1: _dp, N: int = 2) -> _dp:
    """
    Interpolate 2D array at arbitrary point using polynomial interpolation.

    Parameters
    ----------
    e : (M, N) ndarray
        Function values
    x0a, x1a : (M,), (N,) ndarray
        Grid coordinates
    x0, x1 : float
        Interpolation points
    N : int, optional
        Interpolation order (default: 2)

    Returns
    -------
    y : float
        Interpolated value
    """
    if x0 < np.min(x0a) or x0 > np.max(x0a):
        return 0.0
    if x1 < np.min(x1a) or x1 > np.max(x1a):
        return 0.0

    i = _iminloc(np.abs(x0a - x0))
    j = _iminloc(np.abs(x1a - x1))

    i0 = max(0, i - N)
    j0 = max(0, j - N)
    i1 = min(len(x0a), i + N + 1)
    j1 = min(len(x1a), j + N + 1)

    return _polint2(x0a[i0:i1], x1a[j0:j1], e[i0:i1, j0:j1], x0, x1)[0]

def GetValAt_3D(e: _dp_array, x0a: _dp_array, x1a: _dp_array, x2a: _dp_array,
                x0: _dp, x1: _dp, x2: _dp, N: int = 2) -> _dp:
    """
    Interpolate 3D array at arbitrary point using polynomial interpolation.

    Parameters
    ----------
    e : (M, N, P) ndarray
        Function values
    x0a, x1a, x2a : (M,), (N,), (P,) ndarray
        Grid coordinates
    x0, x1, x2 : float
        Interpolation points
    N : int, optional
        Interpolation order (default: 2)

    Returns
    -------
    y : float
        Interpolated value
    """
    if x0 < np.min(x0a) or x0 > np.max(x0a):
        return 0.0
    if x1 < np.min(x1a) or x1 > np.max(x1a):
        return 0.0
    if x2 < np.min(x2a) or x2 > np.max(x2a):
        return 0.0

    i = _iminloc(np.abs(x0a - x0))
    j = _iminloc(np.abs(x1a - x1))
    k = _iminloc(np.abs(x2a - x2))

    i0 = max(0, i - N)
    j0 = max(0, j - N)
    k0 = max(0, k - N)
    i1 = min(len(x0a), i + N + 1)
    j1 = min(len(x1a), j + N + 1)
    k1 = min(len(x2a), k + N + 1)

    return _polint3(x0a[i0:i1], x1a[j0:j1], x2a[k0:k1], e[i0:i1, j0:j1, k0:k1], x0, x1, x2)[0]

# Interface functions for backward compatibility
def spline(x: _dp_array, y: _dp_array, b: _dp_array, c: _dp_array, d: _dp_array) -> None:
    """Interface function for spline (calls spline_dp)."""
    spline_dp(x, y, b, c, d)

def seval(u: _dp, x: _dp_array, y: _dp_array, b: _dp_array, c: _dp_array, d: _dp_array) -> _dp:
    """Interface function for seval (calls seval_dp)."""
    return seval_dp(u, x, y, b, c, d)

def rescale_1D(x0: _dp_array, y0: _dp_array, x1: _dp_array, y1: _dp_array) -> None:
    """Interface function for rescale_1D (calls rescale_1D_dp)."""
    rescale_1D_dp(x0, y0, x1, y1)

def rescale_2D(x0: _dp_array, y0: _dp_array, z0: _dp_array,
               x1: _dp_array, y1: _dp_array, z1: _dp_array) -> None:
    """Interface function for rescale_2D (calls rescale_2D_dp)."""
    rescale_2D_dp(x0, y0, z0, x1, y1, z1)
