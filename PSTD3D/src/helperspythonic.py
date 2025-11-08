"""
helperspythonic.py - Python port of helpers.F90 module

This module provides useful utility functions that didn't fit anywhere else.
Includes mathematical functions, interpolation routines, and utility functions.

Math cue: Provides sech, gauss, arg, magsq, constrain, interpolation, and derivative functions.

Author: Rahul R. Sah
"""

import numpy as np
from typing import Union, Optional, Tuple, Any
from numpy.typing import NDArray

# Import constants
try:
    from .constants import pi, twopi, c0, eps0, e0
except ImportError:
    try:
        from constants import pi, twopi, c0, eps0, e0
    except ImportError:
        # Fallback to direct values if constants module not available
        pi = np.float64(3.141592653589793238462643383279502884197)
        twopi = np.float64(6.283185307179586476925286766559005768394)
        c0 = np.float64(299792458.0)
        eps0 = np.float64(8.8541878176203898505365630317107e-12)
        e0 = np.float64(1.60217733e-19)

# Try to import Numba for JIT compilation
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    def njit(cache=True, parallel=False):
        def decorator(func):
            return func
        return decorator
    def prange(*args, **kwargs):
        return range(*args, **kwargs)
    NUMBA_AVAILABLE = False

# Type aliases
_dp = np.float64
_dc = np.complex128
_sp = np.float32
_sc = np.complex64

# Type arrays
_dp_array = NDArray[_dp]
_dc_array = NDArray[_dc]
_sp_array = NDArray[_sp]
_sc_array = NDArray[_sc]

# Constants
ec2 = 2.0 * eps0 * c0


# =============================================================================
# Mathematical Functions
# =============================================================================

@njit(cache=True)
def sech_dp(t: _dp) -> _dp:
    """Double precision sech function."""
    return 1.0 / np.cosh(t)

@njit(cache=True)
def sech_sp(t: _sp) -> _sp:
    """Single precision sech function."""
    return 1.0 / np.cosh(t)

def sech(t: Union[_dp, _sp]) -> Union[_dp, _sp]:
    """Sech function with automatic precision detection."""
    if isinstance(t, np.float32):
        return sech_sp(t)
    else:
        return sech_dp(t)

@njit(cache=True)
def arg_dp(Z: _dc) -> _dp:
    """Returns the angle of a complex number (double precision)."""
    return np.arctan2(Z.imag, Z.real)

@njit(cache=True)
def arg_sp(Z: _sc) -> _sp:
    """Returns the angle of a complex number (single precision)."""
    return np.arctan2(Z.imag, Z.real)

def arg(Z: Union[_dc, _sc]) -> Union[_dp, _sp]:
    """Returns the angle of a complex number with automatic precision detection."""
    if isinstance(Z, np.complex64):
        return arg_sp(Z)
    else:
        return arg_dp(Z)

@njit(cache=True)
def gauss_dp(x: _dp) -> _dp:
    """Gaussian function (double precision)."""
    return np.exp(-x**2)

@njit(cache=True)
def gauss_sp(x: _sp) -> _sp:
    """Gaussian function (single precision)."""
    return np.exp(-x**2)

def gauss(x: Union[_dp, _sp]) -> Union[_dp, _sp]:
    """Gaussian function with automatic precision detection."""
    if isinstance(x, np.float32):
        return gauss_sp(x)
    else:
        return gauss_dp(x)

@njit(cache=True)
def magsq_dp(Z: _dc) -> _dp:
    """Computes the magnitude squared of a complex number (double precision)."""
    return Z.real**2 + Z.imag**2

@njit(cache=True)
def magsq_sp(Z: _sc) -> _sp:
    """Computes the magnitude squared of a complex number (single precision)."""
    return Z.real**2 + Z.imag**2

def magsq(Z: Union[_dc, _sc]) -> Union[_dp, _sp]:
    """Computes the magnitude squared of a complex number with automatic precision detection."""
    if isinstance(Z, np.complex64):
        return magsq_sp(Z)
    else:
        return magsq_dp(Z)

@njit(cache=True)
def constrain_dp(x: _dp, H: _dp, L: _dp) -> _dp:
    """Constrains a number between high and low values (double precision)."""
    return max(min(L, H), min(max(L, H), x))

@njit(cache=True)
def constrain_int(x: int, H: int, L: int) -> int:
    """Constrains a number between high and low values (integer)."""
    return max(min(L, H), min(max(L, H), x))

def constrain(x: Union[_dp, int], H: Union[_dp, int], L: Union[_dp, int]) -> Union[_dp, int]:
    """Constrains a number between high and low values with automatic type detection."""
    if isinstance(x, int):
        return constrain_int(x, H, L)
    else:
        return constrain_dp(x, H, L)


# =============================================================================
# Intensity and Field Conversion Functions
# =============================================================================

@njit(cache=True)
def AmpToInten(e: _dp, n0: _dp = 1.0) -> _dp:
    """Converts a real amplitude into intensity in a medium with refractive index n0."""
    return n0 * ec2 * e**2

@njit(cache=True)
def FldToInten(e: _dc, n0: _dp = 1.0) -> _dp:
    """Converts a complex amplitude into intensity in a medium with refractive index n0."""
    return n0 * ec2 * magsq_dp(e)

@njit(cache=True)
def IntenToAmp(inten: _dp, n0: _dp = 1.0) -> _dp:
    """Converts medium intensity to a real amplitude."""
    return np.sqrt(inten / ec2 / n0)


# =============================================================================
# Lax Method Functions
# =============================================================================

@njit(cache=True)
def LAX(u: _dc_array, i: int, j: int, k: int) -> _dc:
    """Implements the Lax method in finite differencing for improved stability."""
    return (u[i-1, j, k] + u[i+1, j, k] +
            u[i, j-1, k] + u[i, j+1, k] +
            u[i, j, k-1] + u[i, j, k+1]) / 6.0

@njit(cache=True)
def noLAX(u: _dc_array, i: int, j: int, k: int) -> _dc:
    """Function with same signature as LAX for direct replacement (no Lax method)."""
    return u[i, j, k]


# =============================================================================
# Wavelength and Frequency Conversion
# =============================================================================

@njit(cache=True)
def l2f(lam: _dp) -> _dp:
    """Converts wavelength to frequency."""
    return c0 / lam

@njit(cache=True)
def l2w(lam: _dp) -> _dp:
    """Converts wavelength to angular frequency."""
    return twopi * c0 / lam

@njit(cache=True)
def w2l(w: _dp) -> _dp:
    """Converts angular frequency to wavelength."""
    return twopi * c0 / w


# =============================================================================
# Array Generation Functions
# =============================================================================

@njit(cache=True)
def GetSpaceArray(N: int, length: _dp) -> _dp_array:
    """Helper function to calculate the spatial array."""
    X = np.zeros(N, dtype=_dp)
    if N == 1:
        return X

    dl = length / (N - 1)
    for i in range(N):
        X[i] = float(i) * dl - length / 2.0
    return X

@njit(cache=True)
def GetKArray(N: int, length: _dp) -> _dp_array:
    """Calculates the wavevector array for the FFT."""
    k_array = np.zeros(N, dtype=_dp)
    dl = twopi / length

    for i in range(N):
        if i <= N // 2:
            k_array[i] = float(i) * dl
        else:
            k_array[i] = float(i - N) * dl
    return k_array


# =============================================================================
# Phase Unwrapping
# =============================================================================

@njit(cache=True)
def unwrap(phase: _dp_array) -> _dp_array:
    """Reconstructs a smooth phase from one that has been wrapped by modulo 2π."""
    N = len(phase)
    unwrapped = np.zeros(N, dtype=_dp)

    pm1 = phase[0]
    unwrapped[0] = pm1
    p0 = 0.0
    thr = pi - np.finfo(_dp).eps
    pi2 = twopi

    for i in range(1, N):
        cp = phase[i] + p0
        dpp = cp - pm1
        pm1 = cp

        if dpp > thr:
            while dpp > thr:
                p0 = p0 - pi2
                dpp = dpp - pi2
        if dpp < -thr:
            while dpp < -thr:
                p0 = p0 + pi2
                dpp = dpp + pi2

        cp = phase[i] + p0
        pm1 = cp
        unwrapped[i] = cp

    return unwrapped


# =============================================================================
# Factorial Function
# =============================================================================

@njit(cache=True)
def factorial(p: int) -> int:
    """Computes the factorial of an integer."""
    if p <= 1:
        return 1
    result = 1
    for i in range(2, p + 1):
        result *= i
    return result


# =============================================================================
# Locator Function
# =============================================================================

@njit(cache=True)
def locator(x: _dp_array, u: _dp) -> int:
    """Locates the position in array bracketing a position using binary search."""
    n = len(x)
    i = 1

    if i >= n:
        i = n - 1

    if u < x[i] or u > x[i + 1]:
        # Binary search
        i = 1
        j = n + 1
        while True:
            k = (i + j) // 2
            if u < x[k]:
                j = k
            if u >= x[k]:
                i = k
            if j - i == 1:
                break

    return i


# =============================================================================
# Linear Interpolation Functions
# =============================================================================

@njit(cache=True)
def LinearInterp_dp(f: _dp_array, x: _dp_array, x0: _dp) -> _dp:
    """Linear interpolation of real 1D array at specified position."""
    i = locator(x, x0)
    return (f[i] * (x[i+1] - x0) + f[i+1] * (x0 - x[i])) / (x[i+1] - x[i])

@njit(cache=True)
def LinearInterp_dpc(f: _dc_array, x: _dp_array, x0: _dp) -> _dc:
    """Linear interpolation of complex 1D array at specified position."""
    i = locator(x, x0)
    return (f[i] * (x[i+1] - x0) + f[i+1] * (x0 - x[i])) / (x[i+1] - x[i])

def LinearInterp(f: Union[_dp_array, _dc_array], x: _dp_array, x0: _dp) -> Union[_dp, _dc]:
    """Linear interpolation with automatic type detection."""
    if np.iscomplexobj(f):
        return LinearInterp_dpc(f, x, x0)
    else:
        return LinearInterp_dp(f, x, x0)

@njit(cache=True)
def BilinearInterp_dp(f: _dp_array, x: _dp_array, y: _dp_array, x0: _dp, y0: _dp) -> _dp:
    """Bilinear interpolation of real 2D array at specified position."""
    j = locator(y, y0)
    f1 = LinearInterp_dp(f[:, j], x, x0)
    f2 = LinearInterp_dp(f[:, j+1], x, x0)
    return (f1 * (y[j+1] - y0) + f2 * (y0 - y[j])) / (y[j+1] - y[j])

@njit(cache=True)
def BilinearInterp_dpc(f: _dc_array, x: _dp_array, y: _dp_array, x0: _dp, y0: _dp) -> _dc:
    """Bilinear interpolation of complex 2D array at specified position."""
    j = locator(y, y0)
    f1 = LinearInterp_dpc(f[:, j], x, x0)
    f2 = LinearInterp_dpc(f[:, j+1], x, x0)
    return (f1 * (y[j+1] - y0) + f2 * (y0 - y[j])) / (y[j+1] - y[j])

def BilinearInterp(f: Union[_dp_array, _dc_array], x: _dp_array, y: _dp_array,
                   x0: _dp, y0: _dp) -> Union[_dp, _dc]:
    """Bilinear interpolation with automatic type detection."""
    if np.iscomplexobj(f):
        return BilinearInterp_dpc(f, x, y, x0, y0)
    else:
        return BilinearInterp_dp(f, x, y, x0, y0)

@njit(cache=True)
def TrilinearInterp_dp(f: _dp_array, x: _dp_array, y: _dp_array, z: _dp_array,
                       x0: _dp, y0: _dp, z0: _dp) -> _dp:
    """Trilinear interpolation of real 3D array at specified position."""
    k = locator(z, z0)
    f1 = BilinearInterp_dp(f[:, :, k], x, y, x0, y0)
    f2 = BilinearInterp_dp(f[:, :, k+1], x, y, x0, y0)
    return (f1 * (z[k+1] - z0) + f2 * (z0 - z[k])) / (z[k+1] - z[k])

@njit(cache=True)
def TrilinearInterp_dpc(f: _dc_array, x: _dp_array, y: _dp_array, z: _dp_array,
                        x0: _dp, y0: _dp, z0: _dp) -> _dc:
    """Trilinear interpolation of complex 3D array at specified position."""
    k = locator(z, z0)
    f1 = BilinearInterp_dpc(f[:, :, k], x, y, x0, y0)
    f2 = BilinearInterp_dpc(f[:, :, k+1], x, y, x0, y0)
    return (f1 * (z[k+1] - z0) + f2 * (z0 - z[k])) / (z[k+1] - z[k])

def TrilinearInterp(f: Union[_dp_array, _dc_array], x: _dp_array, y: _dp_array, z: _dp_array,
                    x0: _dp, y0: _dp, z0: _dp) -> Union[_dp, _dc]:
    """Trilinear interpolation with automatic type detection."""
    if np.iscomplexobj(f):
        return TrilinearInterp_dpc(f, x, y, z, x0, y0, z0)
    else:
        return TrilinearInterp_dp(f, x, y, z, x0, y0, z0)


# =============================================================================
# Derivative Functions
# =============================================================================

@njit(cache=True)
def dfdt_dp(f: _dp_array, dt: _dp, k: int) -> _dc:
    """Five-point stencil derivative for real array at index k."""
    N = len(f)

    if k > 2 and k < N - 1:
        return (-f[k+2] + 8.0*f[k+1] - 8.0*f[k-1] + f[k-2]) / dt / 12.0
    elif k == 1:
        return (f[2] - 0.0) / 2.0 / dt
    elif k == 2:
        return (f[3] - f[1]) / 2.0 / dt
    elif k == N:
        return (0.0 - f[N-1]) / 2.0 / dt
    elif k == N - 1:
        return (f[N] - f[N-2]) / 2.0 / dt
    else:
        return 0.0

@njit(cache=True)
def dfdt_dpc(f: _dc_array, dt: _dp, k: int) -> _dc:
    """Five-point stencil derivative for complex array at index k."""
    N = len(f)

    if k > 2 and k < N - 1:
        return (-f[k+2] + 8.0*f[k+1] - 8.0*f[k-1] + f[k-2]) / dt / 12.0
    elif k == 1:
        return (f[2] - 0.0) / 2.0 / dt
    elif k == 2:
        return (f[3] - f[1]) / 2.0 / dt
    elif k == N:
        return (0.0 - f[N-1]) / 2.0 / dt
    elif k == N - 1:
        return (f[N] - f[N-2]) / 2.0 / dt
    else:
        return 0.0

@njit(cache=True)
def dfdt_1D_dp(f: _dp_array, dt: _dp) -> _dp_array:
    """Five-point stencil derivative for entire real array."""
    N = len(f)
    result = np.zeros(N, dtype=_dp)

    for k in range(N):
        if k > 2 and k < N - 2:
            result[k] = (-f[k+2] + 8.0*f[k+1] - 8.0*f[k-1] + f[k-2]) / dt / 12.0
        elif k == 1:
            result[k] = (f[2] - 0.0) / 2.0 / dt
        elif k == 2:
            result[k] = (f[3] - f[1]) / 2.0 / dt
        elif k == N - 1:
            result[k] = (0.0 - f[N-2]) / 2.0 / dt
        elif k == N - 2:
            result[k] = (f[N-1] - f[N-3]) / 2.0 / dt
        else:
            result[k] = 0.0

    return result

@njit(cache=True)
def dfdt_1D_dpc(f: _dc_array, dt: _dp) -> _dc_array:
    """Five-point stencil derivative for entire complex array."""
    N = len(f)
    result = np.zeros(N, dtype=_dc)

    for k in range(N):
        if k > 2 and k < N - 2:
            result[k] = (-f[k+2] + 8.0*f[k+1] - 8.0*f[k-1] + f[k-2]) / dt / 12.0
        elif k == 1:
            result[k] = (f[2] - 0.0) / 2.0 / dt
        elif k == 2:
            result[k] = (f[3] - f[1]) / 2.0 / dt
        elif k == N - 1:
            result[k] = (0.0 - f[N-2]) / 2.0 / dt
        elif k == N - 2:
            result[k] = (f[N-1] - f[N-3]) / 2.0 / dt
        else:
            result[k] = 0.0

    return result

def dfdt(f: Union[_dp_array, _dc_array], dt: _dp, k: Optional[int] = None) -> Union[_dc, _dp_array, _dc_array]:
    """Derivative function with automatic type detection."""
    if k is not None:
        if np.iscomplexobj(f):
            return dfdt_dpc(f, dt, k)
        else:
            return dfdt_dp(f, dt, k)
    else:
        if np.iscomplexobj(f):
            return dfdt_1D_dpc(f, dt)
        else:
            return dfdt_1D_dp(f, dt)


# =============================================================================
# NaN Detection Functions
# =============================================================================

@njit(cache=True)
def isnan_dp(X: _dp) -> bool:
    """Double precision NaN detection."""
    return np.isnan(X)

@njit(cache=True)
def isnan_sp(X: _sp) -> bool:
    """Single precision NaN detection."""
    return np.isnan(X)

def isnan(X: Union[_dp, _sp]) -> bool:
    """NaN detection with automatic precision detection."""
    if isinstance(X, np.float32):
        return isnan_sp(X)
    else:
        return isnan_dp(X)


# =============================================================================
# Interface Functions for Backward Compatibility
# =============================================================================

# These functions provide the same interface as the Fortran module
# but delegate to the appropriate Python implementations

def sech_interface(t: Union[_dp, _sp]) -> Union[_dp, _sp]:
    """Interface function for sech."""
    return sech(t)

def arg_interface(Z: Union[_dc, _sc]) -> Union[_dp, _sp]:
    """Interface function for arg."""
    return arg(Z)

def gauss_interface(x: Union[_dp, _sp]) -> Union[_dp, _sp]:
    """Interface function for gauss."""
    return gauss(x)

def magsq_interface(Z: Union[_dc, _sc]) -> Union[_dp, _sp]:
    """Interface function for magsq."""
    return magsq(Z)

def constrain_interface(x: Union[_dp, int], H: Union[_dp, int], L: Union[_dp, int]) -> Union[_dp, int]:
    """Interface function for constrain."""
    return constrain(x, H, L)

def LinearInterp_interface(f: Union[_dp_array, _dc_array], x: _dp_array, x0: _dp) -> Union[_dp, _dc]:
    """Interface function for LinearInterp."""
    return LinearInterp(f, x, x0)

def BilinearInterp_interface(f: Union[_dp_array, _dc_array], x: _dp_array, y: _dp_array,
                             x0: _dp, y0: _dp) -> Union[_dp, _dc]:
    """Interface function for BilinearInterp."""
    return BilinearInterp(f, x, y, x0, y0)

def TrilinearInterp_interface(f: Union[_dp_array, _dc_array], x: _dp_array, y: _dp_array, z: _dp_array,
                              x0: _dp, y0: _dp, z0: _dp) -> Union[_dp, _dc]:
    """Interface function for TrilinearInterp."""
    return TrilinearInterp(f, x, y, z, x0, y0, z0)

def dfdt_interface(f: Union[_dp_array, _dc_array], dt: _dp, k: Optional[int] = None) -> Union[_dc, _dp_array, _dc_array]:
    """Interface function for dfdt."""
    return dfdt(f, dt, k)

def isnan_interface(X: Union[_dp, _sp]) -> bool:
    """Interface function for isnan."""
    return isnan(X)


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Example usage of the helpers module."""
    import numpy as np

    # Test mathematical functions
    x = np.linspace(-2, 2, 100, dtype=np.float64)
    sech_vals = [sech_dp(xi) for xi in x]
    gauss_vals = [gauss_dp(xi) for xi in x]

    # Test complex functions
    z = 1.0 + 1.0j
    angle = arg_dp(z)
    mag_sq = magsq_dp(z)

    # Test interpolation
    x_grid = np.linspace(0, 10, 11, dtype=np.float64)
    f_grid = np.sin(x_grid)
    x_interp = 5.5
    f_interp = LinearInterp_dp(f_grid, x_grid, x_interp)

    # Test array generation
    space_array = GetSpaceArray(10, 1.0)
    k_array = GetKArray(10, 1.0)

    print("Helpers module example completed successfully!")
    print(f"sech(1.0) = {sech_dp(1.0)}")
    print(f"gauss(1.0) = {gauss_dp(1.0)}")
    print(f"arg(1+1j) = {angle}")
    print(f"magsq(1+1j) = {mag_sq}")
    print(f"Linear interpolation at {x_interp}: {f_interp}")


if __name__ == "__main__":
    example_usage()
