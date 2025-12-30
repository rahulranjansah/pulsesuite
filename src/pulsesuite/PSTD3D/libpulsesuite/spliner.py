"""
Spline interpolation routines for arrays.

This module provides routines for interpolation of arrays using cubic splines
and polynomial interpolation methods. Supports both real and complex arrays,
1D and 2D rescaling, and multi-dimensional interpolation.

Author: Rahul R. Sah
"""

import numpy as np

try:
    from numba import jit
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False
    # Fallback: create a no-op decorator
    def jit(*args, **kwargs):  # noqa: ARG001, ARG002
        def decorator(func):
            return func
        if args and callable(args[0]):
            # Called as @jit without parentheses
            return args[0]
        return decorator


# Helper function: iminloc - find index of minimum value
# Note: Cannot use nopython=True with np.argmin, so using fallback
@jit
def _iminloc_core(arr):
    """
    JIT-compiled core for finding minimum index. (doesnt belong here)
    (Port it in nrutils.py)
    """
    n = len(arr)
    min_val = arr[0]
    min_idx = 0
    for i in range(1, n):
        if arr[i] < min_val:
            min_val = arr[i]
            min_idx = i
    return min_idx


def iminloc(arr):
    """
    Find the index of the minimum value in an array.

    Parameters
    ----------
    arr : ndarray
        1D array of real values

    Returns
    -------
    int
        Index (1-based in Fortran, 0-based in Python) of minimum value.
        Note: Returns 0-based index to match Python convention.
    """
    if JIT_AVAILABLE:
        try:
            return _iminloc_core(arr)
        except (TypeError, ValueError, RuntimeError):
            # Fallback to numpy if JIT fails (e.g., unsupported types)
            return np.argmin(arr)
    else:
        return np.argmin(arr)


# Main functions from lines 34-258

def rescale_2D_dp(x0, y0, z0, x1, y1, z1):
    """
    Rescale a 2D real array from one grid to another.

    Rescales the 2D array z0 defined on grid (x0, y0) to a new grid (x1, y1),
    storing the result in z1. Uses cubic spline interpolation.

    Parameters
    ----------
    x0 : ndarray
        Original X-coordinates, 1D array
    y0 : ndarray
        Original Y-coordinates, 1D array
    z0 : ndarray
        Original 2D array, shape (len(x0), len(y0))
    x1 : ndarray
        New X-coordinates, 1D array
    y1 : ndarray
        New Y-coordinates, 1D array
    z1 : ndarray
        Output 2D array, shape (len(x1), len(y1)), will be modified in-place

    Notes
    -----
    The function first interpolates along the x-direction for each y0 value,
    then interpolates along the y-direction for each x1 value.
    """
    zt = np.zeros((len(x1), len(y0)))

    # Interpolate along x-direction for each y0
    for i in range(len(y0)):
        rescale_1D_dp(x0, z0[:, i], x1, zt[:, i])

    # Interpolate along y-direction for each x1
    for i in range(len(x1)):
        rescale_1D_dp(y0, zt[i, :], y1, z1[i, :])


def rescale_2D_dpc(x0, y0, z0, x1, y1, z1):
    """
    Rescale a 2D complex array from one grid to another.

    Rescales the 2D complex array z0 defined on grid (x0, y0) to a new grid (x1, y1),
    storing the result in z1. Uses cubic spline interpolation.

    Parameters
    ----------
    x0 : ndarray
        Original X-coordinates, 1D array
    y0 : ndarray
        Original Y-coordinates, 1D array
    z0 : ndarray
        Original 2D complex array, shape (len(x0), len(y0))
    x1 : ndarray
        New X-coordinates, 1D array
    y1 : ndarray
        New Y-coordinates, 1D array
    z1 : ndarray
        Output 2D complex array, shape (len(x1), len(y1)), will be modified in-place

    Notes
    -----
    The function first interpolates along the x-direction for each y0 value,
    then interpolates along the y-direction for each x1 value.
    """
    zt = np.zeros((len(x1), len(y0)), dtype=complex)

    # Interpolate along x-direction for each y0
    for i in range(len(y0)):
        rescale_1D_dpc(x0, z0[:, i], x1, zt[:, i])

    # Interpolate along y-direction for each x1
    for i in range(len(x1)):
        rescale_1D_dpc(y0, zt[i, :], y1, z1[i, :])


def rescale_1D_dpc(x0, z0, x1, z1):
    """
    Rescale a 1D complex array from one grid to another.

    Rescales the 1D complex array z0 defined on grid x0 to a new grid x1,
    storing the result in z1. Uses cubic spline interpolation.

    Parameters
    ----------
    x0 : ndarray
        Original X-coordinates, 1D array
    z0 : ndarray
        Original 1D complex array
    x1 : ndarray
        New X-coordinates, 1D array
    z1 : ndarray
        Output 1D complex array, will be modified in-place

    Notes
    -----
    Values outside the range [min(x0), max(x0)] are set to 0.
    """
    if len(x0) != len(z0):
        raise ValueError("Bad sizes in rescale_1D_dpc")

    b = np.zeros(len(x0), dtype=complex)
    c = np.zeros(len(x0), dtype=complex)
    d = np.zeros(len(x0), dtype=complex)

    spline_dpc(x0, z0, b, c, d)

    for i in range(len(x1)):
        if x1[i] >= np.min(x0) and x1[i] <= np.max(x0):
            z1[i] = seval_dpc(x1[i], x0, z0, b, c, d)
        else:
            z1[i] = 0.0 + 0.0j


def rescale_1D_cyl_dpc(x0, z0, x1, z1):
    """
    Rescale a 1D complex array from one grid to another with cylindrical boundary condition.

    Rescales the 1D complex array z0 defined on grid x0 to a new grid x1,
    storing the result in z1. Uses cubic spline interpolation.
    The first point z1[0] is set to z0[0] (cylindrical boundary condition).

    Parameters
    ----------
    x0 : ndarray
        Original X-coordinates, 1D array
    z0 : ndarray
        Original 1D complex array
    x1 : ndarray
        New X-coordinates, 1D array
    z1 : ndarray
        Output 1D complex array, will be modified in-place

    Notes
    -----
    Values outside the range [min(x0), max(x0)] are set to 0.
    The first point z1[0] is always set to z0[0].
    """
    if len(x0) != len(z0):
        raise ValueError("Bad sizes in rescale_1D_cyl_dpc")

    b = np.zeros(len(x0), dtype=complex)
    c = np.zeros(len(x0), dtype=complex)
    d = np.zeros(len(x0), dtype=complex)

    spline_dpc(x0, z0, b, c, d)

    z1[0] = z0[0]

    for i in range(1, len(x1)):
        if x1[i] >= np.min(x0) and x1[i] <= np.max(x0):
            z1[i] = seval_dpc(x1[i], x0, z0, b, c, d)
        else:
            z1[i] = 0.0 + 0.0j


def rescale_1D_dp(x0, y0, x1, y1):
    """
    Rescale a 1D real array from one grid to another.

    Rescales the 1D real array y0 defined on grid x0 to a new grid x1,
    storing the result in y1. Uses cubic spline interpolation.

    Parameters
    ----------
    x0 : ndarray
        Original X-coordinates, 1D array
    y0 : ndarray
        Original 1D real array
    x1 : ndarray
        New X-coordinates, 1D array
    y1 : ndarray
        Output 1D real array, will be modified in-place

    Notes
    -----
    Values outside the range [min(x0), max(x0)] are set to 0.
    """
    if len(x0) != len(y0):
        raise ValueError("Bad sizes in rescale_1D_dp")

    b = np.zeros(len(x0))
    c = np.zeros(len(x0))
    d = np.zeros(len(x0))

    spline_dp(x0, y0, b, c, d)

    for i in range(len(x1)):
        if x1[i] >= np.min(x0) and x1[i] <= np.max(x0):
            y1[i] = seval_dp(x1[i], x0, y0, b, c, d)
        else:
            y1[i] = 0.0


def GetValAt_3D(e, x0a, x1a, x2a, x0, x1, x2, N=None):
    """
    Interpolate a 3D array at an arbitrary point.

    Uses polynomial interpolation to interpolate the 3D array e at the
    point (x0, x1, x2). Returns 0 if the point is outside the array bounds.

    Parameters
    ----------
    e : ndarray
        3D array to interpolate
    x0a : ndarray
        X-coordinates for first dimension, 1D array
    x1a : ndarray
        X-coordinates for second dimension, 1D array
    x2a : ndarray
        X-coordinates for third dimension, 1D array
    x0 : float
        Position in first dimension
    x1 : float
        Position in second dimension
    x2 : float
        Position in third dimension
    N : int, optional
        Order of interpolation (default: 2)

    Returns
    -------
    float
        Interpolated value at (x0, x1, x2), or 0 if outside bounds
    """
    N0 = 2
    Nt = N if N is not None else N0

    Z = 0.0

    if x0 < np.min(x0a) or x0 > np.max(x0a):
        return Z
    if x1 < np.min(x1a) or x1 > np.max(x1a):
        return Z
    if x2 < np.min(x2a) or x2 > np.max(x2a):
        return Z

    i = iminloc(np.abs(x0a - x0))
    j = iminloc(np.abs(x1a - x1))
    k = iminloc(np.abs(x2a - x2))

    i0 = i - Nt
    j0 = j - Nt
    k0 = k - Nt

    i1 = i + Nt
    j1 = j + Nt
    k1 = k + Nt

    if i0 < 0:
        i0 = 0
    if j0 < 0:
        j0 = 0
    if k0 < 0:
        k0 = 0

    if i1 >= len(x0a):
        i1 = len(x0a) - 1
    if j1 >= len(x1a):
        j1 = len(x1a) - 1
    if k1 >= len(x2a):
        k1 = len(x2a) - 1

    # Note: Fortran uses 1-based indexing, so we need to adjust
    # i0:i1+1 in Python corresponds to i0+1:i1+1 in Fortran (1-based)
    Z = polint3(x0a[i0:i1 + 1], x1a[j0:j1 + 1], x2a[k0:k1 + 1],
                e[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1], x0, x1, x2)

    return Z


def GetValAt_2D(e, x0a, x1a, x0, x1, N=None):
    """
    Interpolate a 2D array at an arbitrary point.

    Uses polynomial interpolation to interpolate the 2D array e at the
    point (x0, x1). Returns 0 if the point is outside the array bounds.

    Parameters
    ----------
    e : ndarray
        2D array to interpolate
    x0a : ndarray
        X-coordinates for first dimension, 1D array
    x1a : ndarray
        X-coordinates for second dimension, 1D array
    x0 : float
        Position in first dimension
    x1 : float
        Position in second dimension
    N : int, optional
        Order of interpolation (default: 2)

    Returns
    -------
    float
        Interpolated value at (x0, x1), or 0 if outside bounds
    """
    N0 = 2
    Nt = N if N is not None else N0

    Z = 0.0

    if x0 < np.min(x0a) or x0 > np.max(x0a):
        return Z
    if x1 < np.min(x1a) or x1 > np.max(x1a):
        return Z

    i = iminloc(np.abs(x0a - x0))
    j = iminloc(np.abs(x1a - x1))

    i0 = i - Nt
    j0 = j - Nt

    i1 = i + Nt
    j1 = j + Nt

    if i0 < 0:
        i0 = 0
    if j0 < 0:
        j0 = 0

    if i1 >= len(x0a):
        i1 = len(x0a) - 1
    if j1 >= len(x1a):
        j1 = len(x1a) - 1

    Z = polint2(x0a[i0:i1 + 1], x1a[j0:j1 + 1],
                e[i0:i1 + 1, j0:j1 + 1], x0, x1)

    return Z


def GetValAt_1D(e, x0, x1):
    """
    Interpolate a 1D real array at an arbitrary point.

    Uses cubic spline interpolation to interpolate the 1D array e at the
    point x1. Returns 0 if the point is outside the array bounds.

    Parameters
    ----------
    e : ndarray
        1D array to interpolate
    x0 : ndarray
        X-coordinates for the array, 1D array
    x1 : float
        Position to interpolate

    Returns
    -------
    float
        Interpolated value at x1, or 0 if outside bounds
    """
    Z = 0.0

    if x1 < np.min(x0) or x1 > np.max(x0):
        return Z

    b = np.zeros(len(e))
    c = np.zeros(len(e))
    d = np.zeros(len(e))

    spline_dp(x0, e, b, c, d)
    Z = seval_dp(x1, x0, e, b, c, d)

    return Z


def GetValAt_1D_dpc(e, x0, x1):
    """
    Interpolate a 1D complex array at an arbitrary point.

    Uses cubic spline interpolation to interpolate the 1D complex array e
    at the point x1. Returns 0 if the point is outside the array bounds.

    Parameters
    ----------
    e : ndarray
        1D complex array to interpolate
    x0 : ndarray
        X-coordinates for the array, 1D array
    x1 : float
        Position to interpolate

    Returns
    -------
    complex
        Interpolated complex value at x1, or 0 if outside bounds
    """
    Z = 0.0 + 0.0j

    if x1 < np.min(x0) or x1 > np.max(x0):
        return Z

    b = np.zeros(len(e), dtype=complex)
    c = np.zeros(len(e), dtype=complex)
    d = np.zeros(len(e), dtype=complex)

    spline_dpc(x0, e, b, c, d)
    Z = seval_dpc(x1, x0, e, b, c, d)

    return Z

#######################################################################
# Polint functions
#######################################################################

# Helper function: polint1 - polynomial interpolation for 1D arrays
def polint1(xa, ya, x, dy=None):
    """
    Interpolate a 1D array at an arbitrary point using polynomial interpolation.

    Uses Neville's algorithm for polynomial interpolation. The polynomial
    order is determined by the array size (N-1).

    Parameters
    ----------
    xa : ndarray
        X-coordinates, 1D array
    ya : ndarray
        Y-values at x coordinates, 1D array
    x : float
        Position to interpolate
    dy : list or None, optional
        If provided as a list, will store error estimate in dy[0]

    Returns
    -------
    float
        Interpolated value at x
    """
    n = len(xa)
    c = ya.copy()
    d = ya.copy()
    ho = xa - x
    ns = iminloc(np.abs(x - xa))
    y = ya[ns]
    ns = ns - 1

    dyt = 0.0
    for m in range(n - 1):
        den = ho[:n - m] - ho[1 + m:n]
        if np.any(den == 0.0):
            raise ValueError('polint: calculation failure')
        den = (c[1:n - m + 1] - d[:n - m]) / den
        d[:n - m] = ho[1 + m:n] * den
        c[:n - m] = ho[:n - m] * den
        if 2 * ns < n - m:
            dyt = c[ns + 1]
        else:
            dyt = d[ns]
            ns = ns - 1
        y = y + dyt

    if dy is not None:
        if isinstance(dy, list):
            dy[0] = dyt

    return y


# Helper function: polint2 - polynomial interpolation for 2D arrays
def polint2(x1a, x2a, ya, x1, x2, dy=None):
    """
    Interpolate a 2D array at an arbitrary point using polynomial interpolation.

    Uses polynomial interpolation by first interpolating along the second
    dimension, then along the first dimension.

    Parameters
    ----------
    x1a : ndarray
        X-coordinates for first dimension, 1D array
    x2a : ndarray
        X-coordinates for second dimension, 1D array
    ya : ndarray
        2D array of values
    x1 : float
        Position in first dimension
    x2 : float
        Position in second dimension
    dy : float, optional
        Output parameter for error estimate (if provided)

    Returns
    -------
    float
        Interpolated value at (x1, x2)
    """
    m = len(x1a)
    ymtmp = np.zeros(m)
    for j in range(m):
        ymtmp[j] = polint1(x2a, ya[j, :], x2, dy)
    return polint1(x1a, ymtmp, x1, dy)


# Helper function: polint3 - polynomial interpolation for 3D arrays
def polint3(x1a, x2a, x3a, ya, x1, x2, x3, dy=None):
    """
    Interpolate a 3D array at an arbitrary point using polynomial interpolation.

    Uses polynomial interpolation by first interpolating along the third
    and second dimensions, then along the first dimension.

    Parameters
    ----------
    x1a : ndarray
        X-coordinates for first dimension, 1D array
    x2a : ndarray
        X-coordinates for second dimension, 1D array
    x3a : ndarray
        X-coordinates for third dimension, 1D array
    ya : ndarray
        3D array of values
    x1 : float
        Position in first dimension
    x2 : float
        Position in second dimension
    x3 : float
        Position in third dimension
    dy : float, optional
        Output parameter for error estimate (if provided)

    Returns
    -------
    float
        Interpolated value at (x1, x2, x3)
    """
    m = len(x1a)
    ymtmp = np.zeros(m)
    for j in range(m):
        ymtmp[j] = polint2(x2a, x3a, ya[j, :, :], x2, x3, dy)

    if dy is not None:
        return polint1(x1a, ymtmp, x1, dy)
    else:
        return polint1(x1a, ymtmp, x1)


#######################################################################
# Bicubic Interpolation ###############################################
#######################################################################
def bcucof(y, y1, y2, y12, d1, d2, c):
    """
    Compute bicubic interpolation coefficients.

    Computes the coefficients for bicubic interpolation given function values,
    first derivatives, and cross derivatives at the four corners of a rectangle.

    Parameters
    ----------
    y : ndarray
        Function values at four corners, shape (4,)
    y1 : ndarray
        First derivatives in x-direction at four corners, shape (4,)
    y2 : ndarray
        First derivatives in y-direction at four corners, shape (4,)
    y12 : ndarray
        Cross derivatives (d^2/dxdy) at four corners, shape (4,)
    d1 : float
        Width of rectangle in x-direction
    d2 : float
        Width of rectangle in y-direction
    c : ndarray
        Output array for bicubic coefficients, shape (4, 4), modified in-place

    Notes
    -----
    The coefficients are stored in c such that the interpolated value is:
    sum over i,j of c[i,j] * t^i * u^j
    where t and u are normalized coordinates in [0,1].
    """
    # Weight matrix for bicubic interpolation (16x16)
    # Parsed from Fortran data statement with multipliers expanded
    # Format: 1,0,-3,2,4*0 means [1,0,-3,2,0,0,0,0]
    # Full 256-element array (16x16) from Fortran data statement
    wt_data = [
        1, 0, -3, 2, 0, 0, 0, 0, -3, 0, 9, -6, 2, 0, -6, 4,
        0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -9, 6, -2, 0, 6, -4,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -6, 0, 0, -6, 4,
        0, 0, 3, -2, 0, 0, 0, 0, 0, 0, -9, 6, 0, 0, 6, -4,
        0, 0, 0, 0, 1, 0, -3, 2, -2, 0, 6, -4, 1, 0, -3, 2,
        0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 3, -2, 1, 0, -3, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 2, 0, 0, 3, -2,
        0, 0, 0, 0, 0, 0, 3, -2, 0, 0, -6, 4, 0, 0, 3, -2,
        0, 1, -2, 1, 0, 0, 0, 0, 0, -3, 6, -3, 0, 2, -4, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -6, 3, 0, -2, 4, -2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 2, -2,
        0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 3, -3, 0, 0, -2, 2,
        0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 1, -2, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1,
        0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 2, -2, 0, 0, -1, 1
    ]

    wt = np.array(wt_data, dtype=np.float64).reshape((16, 16))

    x = np.zeros(16)
    x[0:4] = y
    x[4:8] = y1 * d1
    x[8:12] = y2 * d2
    x[12:16] = y12 * d1 * d2

    x = np.dot(wt, x)
    # Fortran reshape with order=(/2,1/) means transpose after reshape
    c[:, :] = x.reshape((4, 4), order='F').T


def bcuint(y, y1, y2, y12, x1l, x1u, x2l, x2u, x1, x2):
    """
    Bicubic interpolation.

    Performs bicubic interpolation at the point (x1, x2) within a rectangle
    defined by [x1l, x1u] x [x2l, x2u]. Returns the interpolated value.

    Parameters
    ----------
    y : ndarray
        Function values at four corners, shape (4,)
    y1 : ndarray
        First derivatives in x-direction at four corners, shape (4,)
    y2 : ndarray
        First derivatives in y-direction at four corners, shape (4,)
    y12 : ndarray
        Cross derivatives (d^2/dxdy) at four corners, shape (4,)
    x1l : float
        Lower bound of x1 coordinate
    x1u : float
        Upper bound of x1 coordinate
    x2l : float
        Lower bound of x2 coordinate
    x2u : float
        Upper bound of x2 coordinate
    x1 : float
        X1 coordinate to interpolate
    x2 : float
        X2 coordinate to interpolate

    Returns
    -------
    tuple
        (ansy, ansy1, ansy2) where:
        - ansy: Interpolated value at (x1, x2)
        - ansy1: Partial derivative d/dx1
        - ansy2: Partial derivative d/dx2
    """
    c = np.zeros((4, 4))
    bcucof(y, y1, y2, y12, x1u - x1l, x2u - x2l, c)

    t = (x1 - x1l) / (x1u - x1l)
    u = (x2 - x2l) / (x2u - x2l)

    ansy = 0.0
    ansy2 = 0.0
    ansy1 = 0.0

    # Fortran: do i=4,1,-1 means i = 4, 3, 2, 1 (1-based)
    # Python: range(3, -1, -1) means i = 3, 2, 1, 0 (0-based)
    for i in range(3, -1, -1):
        # Fortran indices: c(i,4), c(i,3), c(i,2), c(i,1)
        # Python indices: c[i, 3], c[i, 2], c[i, 1], c[i, 0]
        ansy = t * ansy + ((c[i, 3] * u + c[i, 2]) * u + c[i, 1]) * u + c[i, 0]
        ansy2 = t * ansy2 + (3.0 * c[i, 3] * u + 2.0 * c[i, 2]) * u + c[i, 1]
        # Fortran: c(4,i), c(3,i), c(2,i)
        # Python: c[3, i], c[2, i], c[1, i]
        ansy1 = u * ansy1 + (3.0 * c[3, i] * t + 2.0 * c[2, i]) * t + c[1, i]

    ansy1 = ansy1 / (x1u - x1l)
    ansy2 = ansy2 / (x2u - x2l)

    return ansy, ansy1, ansy2


def spline2_dpc(x, z, z2):
    """
    Compute the cubic spline interpolant for complex arrays (second derivative version).

    Computes the second derivatives z2 for cubic spline interpolation of the
    complex function z(x). This version uses natural boundary conditions
    (zero second derivatives at endpoints).

    Parameters
    ----------
    x : ndarray
        X-coordinates, 1D array, must be in ascending order
    z : ndarray
        Complex function values at x coordinates, 1D array
    z2 : ndarray
        Output array for second derivatives, shape (len(x)), complex,
        modified in-place

    Notes
    -----
    Uses natural spline boundary conditions: z2[0] = z2[N-1] = 0.
    The spline can be evaluated using seval2_dpc.
    """
    N = len(x)

    u = np.zeros(N, dtype=complex)

    z2[0] = 0.0 + 0.0j
    u[0] = 0.0 + 0.0j

    for i in range(1, N - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * z2[i - 1] + 2.0
        z2[i] = (sig - 1.0) / p
        u[i] = ((z[i + 1] - z[i]) / (x[i + 1] - x[i]) - (z[i] - z[i - 1]) / (x[i] - x[i - 1]))
        u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p

    z2[N - 1] = 0.0 + 0.0j

    for i in range(N - 2, -1, -1):
        z2[i] = z2[i] * z2[i + 1] + u[i]


def seval2_dpc(x0, x, z, z2):
    """
    Evaluate the cubic spline for complex arrays (second derivative version).

    Evaluates the cubic spline at point x0 using the pre-computed second
    derivatives z2 from spline2_dpc.

    Parameters
    ----------
    x0 : float
        Position to interpolate
    x : ndarray
        X-coordinates used to compute spline, 1D array
    z : ndarray
        Complex function values at x coordinates, 1D array
    z2 : ndarray
        Second derivatives from spline2_dpc, 1D complex array

    Returns
    -------
    complex
        Interpolated complex value at x0
    """
    i = locate(x, x0)

    h = x[i + 1] - x[i]
    a = (x[i + 1] - x0) / h
    b = (x0 - x[i]) / h

    z0 = (a * z[i] + b * z[i + 1] +
          ((a ** 3 - a) * z2[i] + (b ** 3 - b) * z2[i + 1]) * h ** 2 / 6.0)

    return z0


def spline2_dp(x, y, y2):
    """
    Compute the cubic spline interpolant for real arrays (second derivative version).

    Computes the second derivatives y2 for cubic spline interpolation of the
    function y(x). This version uses natural boundary conditions
    (zero second derivatives at endpoints).

    Parameters
    ----------
    x : ndarray
        X-coordinates, 1D array, must be in ascending order
    y : ndarray
        Function values at x coordinates, 1D array
    y2 : ndarray
        Output array for second derivatives, shape (len(x)),
        modified in-place

    Notes
    -----
    Uses natural spline boundary conditions: y2[0] = y2[N-1] = 0.
    The spline can be evaluated using seval2_dp.
    """
    N = len(x)

    u = np.zeros(N)

    y2[0] = 0.0
    u[0] = 0.0

    for i in range(1, N - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
        u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p

    y2[N - 1] = 0.0

    for i in range(N - 2, -1, -1):
        y2[i] = y2[i] * y2[i + 1] + u[i]

# sevals ###
#######################################################################


# Helper function: locate - find position in sorted array
@jit(nopython=True)
def locate(xx, x):
    """
    Find the index i such that xx[i] <= x < xx[i+1].

    Uses binary search for efficiency. Returns the index of the lower bound.

    Parameters
    ----------
    xx : ndarray
        Sorted 1D array (ascending order)
    x : float
        Value to locate

    Returns
    -------
    int
        Index i such that xx[i] <= x < xx[i+1], or len(xx)-1 if x >= xx[-1]
    """
    n = len(xx)
    if x < xx[0]:
        return 0
    if x >= xx[n-1]:
        return n - 1

    # Binary search
    jl = 0
    ju = n
    while ju - jl > 1:
        jm = (ju + jl) // 2
        if x >= xx[jm]:
            jl = jm
        else:
            ju = jm
    return jl


# Helper function: spline_dp - compute cubic spline coefficients for real arrays
def spline_dp(x, y, b, c, d):
    """
    Compute the cubic spline interpolant for real arrays.

    Computes the coefficients b, c, d for cubic spline interpolation
    of the function y(x). The spline can then be evaluated using seval_dp.

    Parameters
    ----------
    x : ndarray
        X-coordinates, 1D array, must be in ascending order
    y : ndarray
        Y-values at x coordinates, 1D array
    b : ndarray
        Output array for spline coefficients (linear term)
    c : ndarray
        Output array for spline coefficients (quadratic term)
    d : ndarray
        Output array for spline coefficients (cubic term)

    Notes
    -----
    The arrays b, c, d are modified in-place. The spline evaluation
    uses: y(x) = y[i] + dx*(b[i] + dx*(c[i] + dx*d[i]))
    where dx = x - x[i] and i is the index such that x[i] <= x < x[i+1].
    """
    n = len(x)

    if n < 2:
        raise ValueError("Not enough points to spline.")

    if n < 3:
        b[0] = (y[1] - y[0]) / (x[1] - x[0])
        c[0] = 0.0
        d[0] = 0.0
        b[1] = b[0]
        c[1] = 0.0
        d[1] = 0.0
        return

    d[0] = x[1] - x[0]
    c[1] = (y[1] - y[0]) / d[0]

    for i in range(1, n - 1):
        d[i] = x[i + 1] - x[i]
        b[i] = 2.0 * (d[i - 1] + d[i])
        c[i + 1] = (y[i + 1] - y[i]) / d[i]
        c[i] = c[i + 1] - c[i]

    b[0] = -d[0]
    b[n - 1] = -d[n - 2]
    c[0] = 0.0
    c[n - 1] = 0.0

    if n != 3:
        c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
        c[n - 1] = c[n - 2] / (x[n - 1] - x[n - 3]) - c[n - 3] / (x[n - 2] - x[n - 4])
        c[0] = c[0] * d[0] ** 2 / (x[3] - x[0])
        c[n - 1] = -c[n - 1] * d[n - 2] ** 2 / (x[n - 1] - x[n - 4])

    for i in range(1, n):
        t = d[i - 1] / b[i - 1]
        b[i] = b[i] - t * d[i - 1]
        c[i] = c[i] - t * c[i - 1]

    c[n - 1] = c[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i]

    b[n - 1] = (y[n - 1] - y[n - 2]) / d[n - 2] + d[n - 2] * (c[n - 2] + 2.0 * c[n - 1])
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
        d[i] = (c[i + 1] - c[i]) / d[i]
        c[i] = 3.0 * c[i]

    c[n - 1] = 3.0 * c[n - 1]
    d[n - 1] = d[n - 2]


# Helper function: seval_dp - evaluate cubic spline for real arrays
def seval_dp(u, x, y, b, c, d):
    """
    Evaluate the cubic spline for real arrays.

    Evaluates the cubic spline at point u using the pre-computed
    spline coefficients b, c, d from spline_dp.

    Parameters
    ----------
    u : float
        Position to interpolate
    x : ndarray
        X-coordinates used to compute spline, 1D array
    y : ndarray
        Y-values at x coordinates, 1D array
    b : ndarray
        Spline coefficients (linear term) from spline_dp
    c : ndarray
        Spline coefficients (quadratic term) from spline_dp
    d : ndarray
        Spline coefficients (cubic term) from spline_dp

    Returns
    -------
    float
        Interpolated value at u
    """
    i = locate(x, u)
    dx = u - x[i]
    return y[i] + dx * (b[i] + dx * (c[i] + dx * d[i]))


# Helper function: spline_dpc - compute cubic spline coefficients for complex arrays
def spline_dpc(x, y, b, c, d):
    """
    Compute the cubic spline interpolant for complex arrays.

    Computes the coefficients b, c, d for cubic spline interpolation
    of the complex function y(x). The spline can then be evaluated using seval_dpc.

    Parameters
    ----------
    x : ndarray
        X-coordinates, 1D array, must be in ascending order
    y : ndarray
        Complex Y-values at x coordinates, 1D array
    b : ndarray
        Output array for complex spline coefficients (linear term)
    c : ndarray
        Output array for complex spline coefficients (quadratic term)
    d : ndarray
        Output array for complex spline coefficients (cubic term)

    Notes
    -----
    The arrays b, c, d are modified in-place. The spline evaluation
    uses: y(x) = y[i] + dx*(b[i] + dx*(c[i] + dx*d[i]))
    where dx = x - x[i] and i is the index such that x[i] <= x < x[i+1].
    """
    n = len(x)

    if n < 2:
        raise ValueError("Not enough points to spline.")

    if n < 3:
        b[0] = (y[1] - y[0]) / (x[1] - x[0])
        c[0] = 0.0 + 0.0j
        d[0] = 0.0 + 0.0j
        b[1] = b[0]
        c[1] = 0.0 + 0.0j
        d[1] = 0.0 + 0.0j
        return

    d[0] = x[1] - x[0]
    c[1] = (y[1] - y[0]) / d[0]

    for i in range(1, n - 1):
        d[i] = x[i + 1] - x[i]
        b[i] = 2.0 * (d[i - 1] + d[i])
        c[i + 1] = (y[i + 1] - y[i]) / d[i]
        c[i] = c[i + 1] - c[i]

    b[0] = -d[0]
    b[n - 1] = -d[n - 2]
    c[0] = 0.0 + 0.0j
    c[n - 1] = 0.0 + 0.0j

    if n != 3:
        c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
        c[n - 1] = c[n - 2] / (x[n - 1] - x[n - 3]) - c[n - 3] / (x[n - 2] - x[n - 4])
        c[0] = c[0] * d[0] ** 2 / (x[3] - x[0])
        c[n - 1] = -c[n - 1] * d[n - 2] ** 2 / (x[n - 1] - x[n - 4])

    for i in range(1, n):
        t = d[i - 1] / b[i - 1]
        b[i] = b[i] - t * d[i - 1]
        c[i] = c[i] - t * c[i - 1]

    c[n - 1] = c[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i]

    b[n - 1] = (y[n - 1] - y[n - 2]) / d[n - 2] + d[n - 2] * (c[n - 2] + 2.0 * c[n - 1])
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
        d[i] = (c[i + 1] - c[i]) / d[i]
        c[i] = 3.0 * c[i]

    c[n - 1] = 3.0 * c[n - 1]
    d[n - 1] = d[n - 2]


# Helper function: seval_dpc - evaluate cubic spline for complex arrays
def seval_dpc(u, x, y, b, c, d):
    """
    Evaluate the cubic spline for complex arrays.

    Evaluates the cubic spline at point u using the pre-computed
    spline coefficients b, c, d from spline_dpc.

    Parameters
    ----------
    u : float
        Position to interpolate
    x : ndarray
        X-coordinates used to compute spline, 1D array
    y : ndarray
        Complex Y-values at x coordinates, 1D array
    b : ndarray
        Complex spline coefficients (linear term) from spline_dpc
    c : ndarray
        Complex spline coefficients (quadratic term) from spline_dpc
    d : ndarray
        Complex spline coefficients (cubic term) from spline_dpc

    Returns
    -------
    complex
        Interpolated complex value at u
    """
    i = locate(x, u)
    dx = u - x[i]
    return y[i] + dx * (b[i] + dx * (c[i] + dx * d[i]))


def seval2_dp(x0, x, y, y2):
    """
    Evaluate the cubic spline for real arrays (second derivative version).

    Evaluates the cubic spline at point x0 using the pre-computed second
    derivatives y2 from spline2_dp.

    Parameters
    ----------
    x0 : float
        Position to interpolate
    x : ndarray
        X-coordinates used to compute spline, 1D array
    y : ndarray
        Function values at x coordinates, 1D array
    y2 : ndarray
        Second derivatives from spline2_dp, 1D array

    Returns
    -------
    float
        Interpolated value at x0
    """
    i = locate(x, x0)

    h = x[i + 1] - x[i]
    a = (x[i + 1] - x0) / h
    b = (x0 - x[i]) / h

    y0 = (a * y[i] + b * y[i + 1] +
          ((a ** 3 - a) * y2[i] + (b ** 3 - b) * y2[i + 1]) * h ** 2 / 6.0)

    return y0


#######################################################################
# Interface Functions (similar to Fortran interfaces)
#######################################################################


def spline(x, *args):
    """
    Unified interface for spline functions.

    Automatically dispatches to the appropriate spline function based on
    argument count and types:
    - 3 args: (x, y, y2) or (x, z, z2) -> uses spline2_* (second derivative)
    - 5 args: (x, y, b, c, d) or (x, z, b, c, d) -> uses spline_* (coefficients)
    - Complex arrays -> uses *_dpc version
    - Real arrays -> uses *_dp version

    Parameters
    ----------
    x : ndarray
        X-coordinates, 1D array
    *args : variable
        Positional arguments:
        - For spline2: (y, y2) or (z, z2)
        - For spline: (y, b, c, d) or (z, b, c, d)
    **kwargs : dict
        Keyword arguments (alternative to positional args)

    Notes
    -----
    This interface matches the Fortran interface behavior, allowing
    the same function name to be used for different spline variants.

    Examples
    --------
    >>> # Using spline2 (second derivative version)
    >>> spline(x, y, y2)  # calls spline2_dp
    >>> spline(x, z, z2)  # calls spline2_dpc (if z is complex)

    >>> # Using regular spline (coefficient version)
    >>> spline(x, y, b, c, d)  # calls spline_dp
    >>> spline(x, z, b, c, d)  # calls spline_dpc (if z is complex)
    """
    n_args = len(args)

    # Handle spline2 version (3 arguments total: x, y/z, y2/z2)
    if n_args == 2:
        arr1, arr2 = args
        # Check if arr1 is complex
        if np.iscomplexobj(arr1):
            # spline2_dpc
            return spline2_dpc(x, arr1, arr2)
        else:
            # spline2_dp
            return spline2_dp(x, arr1, arr2)
    # Handle regular spline version (5 arguments total: x, y/z, b, c, d)
    elif n_args == 4:
        arr1, b, c, d = args
        # Check if arr1 is complex
        if np.iscomplexobj(arr1):
            # spline_dpc
            return spline_dpc(x, arr1, b, c, d)
        else:
            # spline_dp
            return spline_dp(x, arr1, b, c, d)
    else:
        raise ValueError(f"spline: Invalid number of arguments ({n_args + 1} total). "
                         "Expected either 3 args (x, y, y2) or 5 args (x, y, b, c, d).")


def seval(pos, x, *args):
    """
    Unified interface for spline evaluation functions.

    Automatically dispatches to the appropriate seval function based on
    argument count and types:
    - 4 args: (pos, x, y, y2) or (pos, x, z, z2) -> uses seval2_*
    - 6 args: (pos, x, y, b, c, d) or (pos, x, z, b, c, d) -> uses seval_*
    - Complex arrays -> uses *_dpc version
    - Real arrays -> uses *_dp version

    Parameters
    ----------
    pos : float
        Position to interpolate
    x : ndarray
        X-coordinates used to compute spline, 1D array
    *args : variable
        Positional arguments:
        - For seval2: (y, y2) or (z, z2)
        - For seval: (y, b, c, d) or (z, b, c, d)

    Returns
    -------
    float or complex
        Interpolated value

    Notes
    -----
    This interface matches the Fortran interface behavior, allowing
    the same function name to be used for different seval variants.

    Examples
    --------
    >>> # Using seval2 (second derivative version)
    >>> seval(x0, x, y, y2)  # calls seval2_dp
    >>> seval(x0, x, z, z2)  # calls seval2_dpc (if z is complex)

    >>> # Using regular seval (coefficient version)
    >>> seval(u, x, y, b, c, d)  # calls seval_dp
    >>> seval(u, x, z, b, c, d)  # calls seval_dpc (if z is complex)
    """
    n_args = len(args)

    # Handle seval2 version (4 arguments total: pos, x, y/z, y2/z2)
    if n_args == 2:
        arr1, arr2 = args
        # Check if arr1 is complex
        if np.iscomplexobj(arr1):
            # seval2_dpc
            return seval2_dpc(pos, x, arr1, arr2)
        else:
            # seval2_dp
            return seval2_dp(pos, x, arr1, arr2)
    # Handle regular seval version (6 arguments total: pos, x, y/z, b, c, d)
    elif n_args == 4:
        arr1, b, c, d = args
        # Check if arr1 is complex
        if np.iscomplexobj(arr1):
            # seval_dpc
            return seval_dpc(pos, x, arr1, b, c, d)
        else:
            # seval_dp
            return seval_dp(pos, x, arr1, b, c, d)
    else:
        raise ValueError(f"seval: Invalid number of arguments ({n_args + 2} total). "
                         "Expected either 4 args (pos, x, y, y2) or 6 args (pos, x, y, b, c, d).")

def rescale_1D(x0, y0=None, z0=None, x1=None, y1=None, z1=None):
    """
    Unified interface for 1D rescaling functions.

    Automatically dispatches to the appropriate rescale_1D function based on
    argument types:
    - If z0 is provided: uses rescale_1D_dpc (complex version)
    - Otherwise: uses rescale_1D_dp (real version)

    Parameters
    ----------
    x0 : ndarray
        Original X-coordinates, 1D array
    y0 : ndarray, optional
        Original 1D real array (for rescale_1D_dp)
    z0 : ndarray, optional
        Original 1D complex array (for rescale_1D_dpc)
    x1 : ndarray
        New X-coordinates, 1D array
    y1 : ndarray, optional
        Output 1D real array (for rescale_1D_dp), modified in-place
    z1 : ndarray, optional
        Output 1D complex array (for rescale_1D_dpc), modified in-place

    Notes
    -----
    This interface matches the Fortran interface behavior, allowing
    the same function name to be used for different rescale_1D variants.
    """
    # Detect 4-positional-argument call: rescale_1D(x0, y0, x1, y1)
    # Python binds this as: x0, y0, z0=R, x1=Ex, y1=None, z1=None
    # We detect this pattern: z0 is not None, x1 is not None, but y1 is None
    if z0 is not None and x1 is not None and y1 is None and z1 is None:
        # This is a 4-arg call: (x0, y0, x1, y1) where x1=z0 and y1=x1
        actual_x1 = z0  # The third positional arg is actually x1
        actual_y1 = x1  # The fourth positional arg is actually y1

        # Check if arrays are complex to determine which function to use
        if np.iscomplexobj(y0) or np.iscomplexobj(actual_y1):
            # Complex version: rescale_1D_dpc(x0, y0, actual_x1, actual_y1)
            return rescale_1D_dpc(x0, y0, actual_x1, actual_y1)
        else:
            # Real version: rescale_1D_dp(x0, y0, actual_x1, actual_y1)
            return rescale_1D_dp(x0, y0, actual_x1, actual_y1)

    # Handle explicit keyword arguments or 6-arg positional call
    if z0 is not None:
        # rescale_1D_dpc with explicit z0
        if z1 is None:
            raise ValueError("rescale_1D_dpc requires z1 to be provided when z0 is used")
        return rescale_1D_dpc(x0, z0, x1, z1)
    else:
        # rescale_1D_dp with y0
        if y1 is None:
            raise ValueError("rescale_1D_dp requires y1 to be provided when y0 is used")
        return rescale_1D_dp(x0, y0, x1, y1)

def rescale_2D(x0, y0, z0, x1, y1, z1):
    """
    Unified interface for 2D rescaling functions.

    Automatically dispatches to the appropriate rescale_2D function based on
    argument types:
    - If z0 is complex: uses rescale_2D_dpc (complex version)
    - Otherwise: uses rescale_2D_dp (real version)

    Parameters
    ----------
    x0 : ndarray
        Original X-coordinates, 1D array
    y0 : ndarray
        Original Y-coordinates, 1D array
    z0 : ndarray
        Original 2D array (real or complex)
    x1 : ndarray
        New X-coordinates, 1D array
    y1 : ndarray
        New Y-coordinates, 1D array
    z1 : ndarray
        Output 2D array (real or complex), modified in-place

    Notes
    -----
    This interface matches the Fortran interface behavior, allowing
    the same function name to be used for different rescale_2D variants.
    """
    if np.iscomplexobj(z0):
        # rescale_2D_dpc
        return rescale_2D_dpc(x0, y0, z0, x1, y1, z1)
    else:
        # rescale_2D_dp
        return rescale_2D_dp(x0, y0, z0, x1, y1, z1)

