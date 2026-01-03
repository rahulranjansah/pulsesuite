"""
Helper functions module.

Mostly useful functions that did not fit anywhere else.

This module provides utility functions for mathematical operations,
interpolation, and field conversions.

Author: Rahul R. Sah
"""

import numpy as np
from scipy.constants import epsilon_0 as eps0_SI, c as c0_SI
from numba import jit
from typing import Union, Optional

from pulsesuite.PSTD3D.usefulsubs import locator

# Constants
pi = np.pi
eps0 = eps0_SI
c0 = c0_SI
ec2 = 2.0 * eps0 * c0  # Stores the result of 2.0 * eps0 * c0 for later use


# def locator(x: np.ndarray, x0: float) -> int:
#     """
#     Find index in sorted array where value should be inserted.

#     Finds the index i such that x[i] <= x0 < x[i+1] (0-based indexing).

#     Parameters
#     ----------
#     x : ndarray
#         Sorted array (real), 1D array
#     x0 : float
#         Value to locate

#     Returns
#     -------
#     int
#         Index i such that x[i] <= x0 < x[i+1] (0-based)

#     Notes
#     -----
#     Uses binary search to find the insertion point.
#     Returns 0 if x0 < x[0], and len(x)-2 if x0 >= x[-1].
#     """
#     # Use numpy's searchsorted which finds the right insertion point
#     i = np.searchsorted(x, x0, side='right') - 1
#     # Ensure i is within valid range for interpolation
#     i = max(0, min(i, len(x) - 2))
#     return i


#######################################################
################ INTERFACE DISPATCHERS #################
#######################################################

def sech(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Hyperbolic secant function.

    Single and double precision sech function.

    Parameters
    ----------
    t : float or ndarray
        Input value(s)

    Returns
    -------
    float or ndarray
        sech(t) = 1.0 / cosh(t)
    """
    t_arr = np.asarray(t)
    if t_arr.dtype == np.float32:
        return sech_sp(t_arr)
    else:
        return sech_dp(t_arr)


def arg(Z: Union[complex, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Returns the angle of a complex number.

    Returns the angle of a complex number with respect to the real axis.

    Parameters
    ----------
    Z : complex or ndarray
        Complex number(s)

    Returns
    -------
    float or ndarray
        Angle in radians: atan2(imag(Z), real(Z))
    """
    Z_arr = np.asarray(Z)
    if Z_arr.dtype == np.complex64:
        return arg_sp(Z_arr)
    else:
        return arg_dp(Z_arr)


def gauss(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Gaussian function.

    Returns exp(-x^2).

    Parameters
    ----------
    x : float or ndarray
        Input value(s)

    Returns
    -------
    float or ndarray
        exp(-x^2)
    """
    x_arr = np.asarray(x)
    if x_arr.dtype == np.float32:
        return gauss_sp(x_arr)
    else:
        return gauss_dp(x_arr)


def magsq(Z: Union[complex, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Magnitude squared of a complex number.

    Provides a faster abs(Z)**2 by computing real(Z)^2 + imag(Z)^2.

    Parameters
    ----------
    Z : complex or ndarray
        Complex number(s)

    Returns
    -------
    float or ndarray
        |Z|^2 = real(Z)^2 + imag(Z)^2
    """
    Z_arr = np.asarray(Z)
    if Z_arr.dtype == np.complex64:
        return magsq_sp(Z_arr)
    else:
        return magsq_dp(Z_arr)


def constrain(x: Union[float, int, np.ndarray], H: Union[float, int], L: Union[float, int]) -> Union[float, int, np.ndarray]:
    """
    Constrains values between limits.

    Constrains a number between a high and a low value.

    Parameters
    ----------
    x : float, int, or ndarray
        Value(s) to constrain
    H : float or int
        High limit
    L : float or int
        Low limit

    Returns
    -------
    float, int, or ndarray
        Constrained value(s): max(min(L,H), min(max(L,H), x))
    """
    x_arr = np.asarray(x)
    if np.issubdtype(x_arr.dtype, np.integer):
        return constrain_int(x_arr, H, L)
    else:
        return constrain_dp(x_arr, H, L)


def LinearInterp(f: np.ndarray, x: np.ndarray, x0: float) -> Union[float, complex]:
    """
    Linear interpolation.

    Returns the linearly interpolated value of the array f(:) at the position x0.

    Parameters
    ----------
    f : ndarray
        1D array to be interpolated
    x : ndarray
        1D position array corresponding to 'f'
    x0 : float
        Position at which 'f' is to be interpolated

    Returns
    -------
    float or complex
        Interpolated value at x0
    """
    if np.iscomplexobj(f):
        return LinearInterp_dpc(f, x, x0)
    else:
        return LinearInterp_dp(f, x, x0)


def BilinearInterp(f: np.ndarray, x: np.ndarray, y: np.ndarray, x0: float, y0: float) -> Union[float, complex]:
    """
    Bilinear interpolation.

    Returns the linearly interpolated value of the array f(:,:) at the position (x0,y0).

    Parameters
    ----------
    f : ndarray
        2D array to be interpolated
    x : ndarray
        1D X position array corresponding to 'f'
    y : ndarray
        1D Y position array corresponding to 'f'
    x0 : float
        X position at which 'f' is to be interpolated
    y0 : float
        Y position at which 'f' is to be interpolated

    Returns
    -------
    float or complex
        Interpolated value at (x0, y0)
    """
    if np.iscomplexobj(f):
        return BilinearInterp_dpc(f, x, y, x0, y0)
    else:
        return BilinearInterp_dp(f, x, y, x0, y0)


def TrilinearInterp(f: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, x0: float, y0: float, z0: float) -> Union[float, complex]:
    """
    Trilinear interpolation.

    Returns the linearly interpolated value of the array f(:,:,:) at the position (x0,y0,z0).

    Parameters
    ----------
    f : ndarray
        3D array to be interpolated
    x : ndarray
        1D X position array corresponding to 'f'
    y : ndarray
        1D Y position array corresponding to 'f'
    z : ndarray
        1D Z position array corresponding to 'f'
    x0 : float
        X position at which 'f' is to be interpolated
    y0 : float
        Y position at which 'f' is to be interpolated
    z0 : float
        Z position at which 'f' is to be interpolated

    Returns
    -------
    float or complex
        Interpolated value at (x0, y0, z0)
    """
    if np.iscomplexobj(f):
        return TrilinearInterp_dpc(f, x, y, z, x0, y0, z0)
    else:
        return TrilinearInterp_dp(f, x, y, z, x0, y0, z0)


def dfdt(f: np.ndarray, dt: float, k: Optional[int] = None) -> Union[float, complex, np.ndarray]:
    """
    First derivative with respect to t.

    Returns the first derivative value of the array f(:) with respect to t
    at the index k with a five-point stencil method.

    Parameters
    ----------
    f : ndarray
        Function of t, 1D array
    dt : float
        t differential
    k : int, optional
        t-index for dfdt. If None, returns derivative for all points.

    Returns
    -------
    float, complex, or ndarray
        Derivative value(s)
    """
    if k is None:
        if np.iscomplexobj(f):
            return dfdt_1D_dpc(f, dt)
        else:
            return dfdt_1D_dp(f, dt)
    else:
        if np.iscomplexobj(f):
            return dfdt_dpc(f, dt, k)
        else:
            return dfdt_dp(f, dt, k)


#######################################################
################ FIELD CONVERSION FUNCTIONS #################
######################################################

def AmpToInten(e: float, n0: Optional[float] = None) -> float:
    """
    Converts a real amplitude into the intensity.

    Converts a real amplitude into the intensity in a medium with refractive index n0.
    Default n0 = 1.0.

    Uses the formula: I = n0 * 2 * eps0 * c0 * E^2

    Parameters
    ----------
    e : float
        The amplitude
    n0 : float, optional
        The linear refractive index (default: 1.0)

    Returns
    -------
    float
        The corresponding intensity
    """
    if n0 is not None:
        return n0 * ec2 * e**2
    else:
        return ec2 * e**2


def FldToInten(e: complex, n0: Optional[float] = None) -> float:
    """
    Converts a complex amplitude into the intensity.

    Converts a complex amplitude into the intensity in a medium with refractive index n0.
    Default n0 = 1.0.

    Uses the formula: I = n0 * 2 * eps0 * c0 * |E|^2

    Parameters
    ----------
    e : complex
        The complex amplitude
    n0 : float, optional
        The linear refractive index (default: 1.0)

    Returns
    -------
    float
        The corresponding intensity
    """
    if n0 is not None:
        return n0 * ec2 * magsq_dp(e)
    else:
        return ec2 * magsq_dp(e)


def IntenToAmp(inten: float, n0: Optional[float] = None) -> float:
    """
    Converts the medium intensity to a real amplitude.

    Converts the medium intensity to a real amplitude.
    Default n0 = 1.0.

    Uses the formula: E = sqrt(I / (n0 * 2 * eps0 * c0))

    Parameters
    ----------
    inten : float
        The intensity
    n0 : float, optional
        The linear refractive index (default: 1.0)

    Returns
    -------
    float
        The corresponding amplitude
    """
    if n0 is not None:
        return np.sqrt(inten / ec2 / n0)
    else:
        return np.sqrt(inten / ec2)


#######################################################
################ ELEMENTAL FUNCTIONS - DOUBLE PRECISION #################
#######################################################

def arg_dp(Z: Union[complex, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Returns the angle of a complex number wrt the real axis.

    Parameters
    ----------
    Z : complex or ndarray
        Complex number(s)

    Returns
    -------
    float or ndarray
        Angle in radians: atan2(imag(Z), real(Z))
    """
    Z_arr = np.asarray(Z, dtype=np.complex128)
    return np.arctan2(np.imag(Z_arr), np.real(Z_arr))


def arg_sp(Z: Union[complex, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Returns the angle of a complex number wrt the real axis.

    Parameters
    ----------
    Z : complex or ndarray
        Complex number(s), single precision

    Returns
    -------
    float or ndarray
        Angle in radians: atan2(imag(Z), real(Z))
    """
    Z_arr = np.asarray(Z, dtype=np.complex64)
    return np.arctan2(np.imag(Z_arr), np.real(Z_arr))


@jit(nopython=True, cache=True)
def _sech_sp_jit(t: float) -> float:
    """JIT-compiled single precision sech for scalar."""
    return 1.0 / np.cosh(t)


@jit(nopython=True, cache=True)
def _sech_dp_jit(t: float) -> float:
    """JIT-compiled double precision sech for scalar."""
    return 1.0 / np.cosh(t)


def sech_sp(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Single precision sech.

    Parameters
    ----------
    t : float or ndarray
        Input value(s), single precision

    Returns
    -------
    float or ndarray
        sech(t) = 1.0 / cosh(t)
    """
    t_arr = np.asarray(t, dtype=np.float32)
    if t_arr.ndim == 0:
        try:
            return _sech_sp_jit(float(t_arr))
        except:
            return 1.0 / np.cosh(t_arr)
    else:
        return 1.0 / np.cosh(t_arr)


def sech_dp(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Double precision sech.

    Parameters
    ----------
    t : float or ndarray
        Input value(s), double precision

    Returns
    -------
    float or ndarray
        sech(t) = 1.0 / cosh(t)
    """
    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.ndim == 0:
        try:
            return _sech_dp_jit(float(t_arr))
        except:
            return 1.0 / np.cosh(t_arr)
    else:
        return 1.0 / np.cosh(t_arr)


@jit(nopython=True, cache=True)
def _gauss_dp_jit(x: float) -> float:
    """JIT-compiled double precision gauss for scalar."""
    return np.exp(-x * x)


@jit(nopython=True, cache=True)
def _gauss_sp_jit(x: float) -> float:
    """JIT-compiled single precision gauss for scalar."""
    return np.exp(-x * x)


def gauss_dp(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    The Gaussian function.

    Returns exp(-x^2).

    Parameters
    ----------
    x : float or ndarray
        Input value(s), double precision

    Returns
    -------
    float or ndarray
        exp(-x^2)
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 0:
        try:
            return _gauss_dp_jit(float(x_arr))
        except:
            return np.exp(-x_arr**2)
    else:
        return np.exp(-x_arr**2)


def gauss_sp(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    The Gaussian function.

    Returns exp(-x^2).

    Parameters
    ----------
    x : float or ndarray
        Input value(s), single precision

    Returns
    -------
    float or ndarray
        exp(-x^2)
    """
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 0:
        try:
            return _gauss_sp_jit(float(x_arr))
        except:
            return np.exp(-x_arr**2)
    else:
        return np.exp(-x_arr**2)


@jit(nopython=True, cache=True)
def _magsq_dp_jit(z_real: float, z_imag: float) -> float:
    """JIT-compiled double precision magsq for scalar."""
    return z_real * z_real + z_imag * z_imag


@jit(nopython=True, cache=True)
def _magsq_sp_jit(z_real: float, z_imag: float) -> float:
    """JIT-compiled single precision magsq for scalar."""
    return z_real * z_real + z_imag * z_imag


def magsq_dp(Z: Union[complex, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the magnitude squared of a complex number.

    Computes real(Z)^2 + imag(Z)^2.

    Parameters
    ----------
    Z : complex or ndarray
        Complex number(s), double precision

    Returns
    -------
    float or ndarray
        |Z|^2 = real(Z)^2 + imag(Z)^2
    """
    Z_arr = np.asarray(Z, dtype=np.complex128)
    if Z_arr.ndim == 0:
        try:
            return _magsq_dp_jit(float(np.real(Z_arr)), float(np.imag(Z_arr)))
        except:
            return np.real(Z_arr)**2 + np.imag(Z_arr)**2
    else:
        return np.real(Z_arr)**2 + np.imag(Z_arr)**2


def magsq_sp(Z: Union[complex, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the magnitude squared of a complex number.

    Computes real(Z)^2 + imag(Z)^2.

    Parameters
    ----------
    Z : complex or ndarray
        Complex number(s), single precision

    Returns
    -------
    float or ndarray
        |Z|^2 = real(Z)^2 + imag(Z)^2
    """
    Z_arr = np.asarray(Z, dtype=np.complex64)
    if Z_arr.ndim == 0:
        try:
            return _magsq_sp_jit(float(np.real(Z_arr)), float(np.imag(Z_arr)))
        except:
            return np.real(Z_arr)**2 + np.imag(Z_arr)**2
    else:
        return np.real(Z_arr)**2 + np.imag(Z_arr)**2


#######################################################
################ LAX FUNCTIONS #################
#######################################################
def LAX(u: np.ndarray, i: int, j: int, k: int) -> complex:
    """
    Implements the Lax method in finite differencing.

    Improves the stability of explicit methods.

    Parameters
    ----------
    u : ndarray
        3D complex array
    i : int
        Index in first dimension
    j : int
        Index in second dimension
    k : int
        Index in third dimension

    Returns
    -------
    complex
        Lax average: (u(i-1,j,k) + u(i+1,j,k) + u(i,j-1,k) + u(i,j+1,k) + u(i,j,k-1) + u(i,j,k+1)) / 6.0
    """
    return (u[i-1, j, k] + u[i+1, j, k] +
            u[i, j-1, k] + u[i, j+1, k] +
            u[i, j, k-1] + u[i, j, k+1]) / 6.0


def noLAX(u: np.ndarray, i: int, j: int, k: int) -> complex:
    """
    Function with the same signature as LAX for a direct replacement.

    Does not implement the Lax method.

    Parameters
    ----------
    u : ndarray
        3D complex array
    i : int
        Index in first dimension
    j : int
        Index in second dimension
    k : int
        Index in third dimension

    Returns
    -------
    complex
        Direct value: u(i,j,k)
    """
    return u[i, j, k]


# ============================================================================
# CONSTRAINT FUNCTIONS
# ============================================================================

def constrain_dp(x: Union[float, np.ndarray], H: float, L: float) -> Union[float, np.ndarray]:
    """
    Constrains a number between a high and a low value.

    Parameters
    ----------
    x : float or ndarray
        Value(s) to constrain, double precision
    H : float
        High limit
    L : float
        Low limit

    Returns
    -------
    float or ndarray
        Constrained value(s): max(min(L,H), min(max(L,H), x))
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return np.maximum(np.minimum(L, H), np.minimum(np.maximum(L, H), x_arr))


def constrain_int(x: Union[int, np.ndarray], H: int, L: int) -> Union[int, np.ndarray]:
    """
    Constrains a number between a high and a low value.

    Parameters
    ----------
    x : int or ndarray
        Value(s) to constrain, integer
    H : int
        High limit
    L : int
        Low limit

    Returns
    -------
    int or ndarray
        Constrained value(s): max(min(L,H), min(max(L,H), x))
    """
    x_arr = np.asarray(x, dtype=np.int64)
    return np.maximum(np.minimum(L, H), np.minimum(np.maximum(L, H), x_arr))



#######################################################
################ WAVELENGTH/FREQUENCY CONVERSION FUNCTIONS #################
#######################################################

def l2f(lam: float) -> float:
    """
    Converts a wavelength into its corresponding frequency.

    Uses the formula: f = c0 / λ

    Parameters
    ----------
    lam : float
        Wavelength

    Returns
    -------
    float
        Corresponding frequency: c0 / lam
    """
    return c0 / lam


def l2w(lam: float) -> float:
    """
    Converts a wavelength into its corresponding angular frequency.

    Uses the formula: ω = 2π * c0 / λ

    Parameters
    ----------
    lam : float
        Wavelength

    Returns
    -------
    float
        Corresponding angular frequency: 2π * c0 / lam
    """
    return 2.0 * pi * c0 / lam


def w2l(w: float) -> float:
    """
    Converts an angular frequency into its corresponding wavelength.

    Uses the formula: λ = 2π * c0 / ω

    Parameters
    ----------
    w : float
        Angular frequency

    Returns
    -------
    float
        Corresponding wavelength: 2π * c0 / w
    """
    return 2.0 * pi * c0 / w


#######################################################
################ SPACE ARRAY FUNCTIONS #################
#######################################################

def GetSpaceArray(N: int, length: float) -> np.ndarray:
    """
    Helper function to calculate the spatial array.

    Change this function to change the centering of the field.

    Parameters
    ----------
    N : int
        Number of points
    length : float
        Total length

    Returns
    -------
    ndarray
        Spatial array X of size N, centered around 0
    """
    if N == 1:
        return np.array([0.0])

    dl = length / (N - 1)  # 0 is between the middle two points
    i_arr = np.arange(N, dtype=np.float64)
    X = i_arr * dl - length / 2.0
    return X


def GetKArray(N: int, length: float) -> np.ndarray:
    """
    Calculates the wavevector array for the FFT.

    Parameters
    ----------
    N : int
        Number of points
    length : float
        Total length

    Returns
    -------
    ndarray
        Wavevector array of size N
    """
    dl = 2.0 * pi / length
    k_arr = np.zeros(N, dtype=np.float64)

    for i in range(N):
        if i <= N // 2:
            k_arr[i] = float(i) * dl
        else:
            k_arr[i] = float(i - N) * dl

    return k_arr


#######################################################
################ PHASE UNWRAPPING #################
#######################################################

def unwrap(phase: np.ndarray) -> np.ndarray:
    """
    Reconstructs a smooth phase from one that has been wrapped by modulo 2π.

    Parameters
    ----------
    phase : ndarray
        Wrapped phase array

    Returns
    -------
    ndarray
        Unwrapped phase array
    """
    phase_arr = np.asarray(phase, dtype=np.float64)
    N = len(phase_arr)
    unwrapped = np.zeros(N, dtype=np.float64)

    pm1 = phase_arr[0]
    unwrapped[0] = pm1
    p0 = 0.0
    thr = pi - np.finfo(np.float64).eps
    pi2 = 2.0 * pi

    for i in range(1, N):
        cp = phase_arr[i] + p0
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

        cp = phase_arr[i] + p0
        pm1 = cp
        unwrapped[i] = cp

    return unwrapped


#######################################################
################ FACTORIAL FUNCTIONS #################
#######################################################

def factorial(p: int) -> int:
    """
    Computes the factorial of an integer.

    Parameters
    ----------
    p : int
        Input integer

    Returns
    -------
    int
        Factorial of p (p!)
    """
    if p <= 1:
        return 1

    result = 1
    for i in range(2, p + 1):
        result = result * i

    return result


#######################################################
################ INTERPOLATION FUNCTIONS #############
#######################################################

def LinearInterp_dpc(f: np.ndarray, x: np.ndarray, x0: float) -> complex:
    """
    Computes a linear interpolation of a complex 1D array at a specified position.

    Parameters
    ----------
    f : ndarray
        1D complex array to be interpolated
    x : ndarray
        1D position array corresponding to 'f'
    x0 : float
        Position at which 'f' is to be interpolated

    Returns
    -------
    complex
        Interpolated value at x0
    """
    i = locator(x, x0)
    f0 = (f[i] * (x[i+1] - x0) + f[i+1] * (x0 - x[i])) / (x[i+1] - x[i])
    return f0


def LinearInterp_dp(f: np.ndarray, x: np.ndarray, x0: float) -> float:
    """
    Computes a linear interpolation of a real 1D array at a specified position.

    Parameters
    ----------
    f : ndarray
        1D real array to be interpolated
    x : ndarray
        1D position array corresponding to 'f'
    x0 : float
        Position at which 'f' is to be interpolated

    Returns
    -------
    float
        Interpolated value at x0
    """
    i = locator(x, x0)
    f0 = (f[i] * (x[i+1] - x0) + f[i+1] * (x0 - x[i])) / (x[i+1] - x[i])
    return f0


def BilinearInterp_dpc(f: np.ndarray, x: np.ndarray, y: np.ndarray, x0: float, y0: float) -> complex:
    """
    Computes a linear interpolation of a complex 2D array at a specified position.

    Parameters
    ----------
    f : ndarray
        2D complex array to be interpolated
    x : ndarray
        1D X position array corresponding to 'f'
    y : ndarray
        1D Y position array corresponding to 'f'
    x0 : float
        X position at which 'f' is to be interpolated
    y0 : float
        Y position at which 'f' is to be interpolated

    Returns
    -------
    complex
        Interpolated value at (x0, y0)
    """
    j = locator(y, y0)
    f1 = LinearInterp_dpc(f[:, j], x, x0)
    f2 = LinearInterp_dpc(f[:, j+1], x, x0)
    f0 = (f1 * (y[j+1] - y0) + f2 * (y0 - y[j])) / (y[j+1] - y[j])
    return f0


def BilinearInterp_dp(f: np.ndarray, x: np.ndarray, y: np.ndarray, x0: float, y0: float) -> float:
    """
    Computes a linear interpolation of a real 2D array at a specified position.

    Parameters
    ----------
    f : ndarray
        2D real array to be interpolated
    x : ndarray
        1D X position array corresponding to 'f'
    y : ndarray
        1D Y position array corresponding to 'f'
    x0 : float
        X position at which 'f' is to be interpolated
    y0 : float
        Y position at which 'f' is to be interpolated

    Returns
    -------
    float
        Interpolated value at (x0, y0)
    """
    j = locator(y, y0)
    f1 = LinearInterp_dp(f[:, j], x, x0)
    f2 = LinearInterp_dp(f[:, j+1], x, x0)
    f0 = (f1 * (y[j+1] - y0) + f2 * (y0 - y[j])) / (y[j+1] - y[j])
    return f0


def TrilinearInterp_dpc(f: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, x0: float, y0: float, z0: float) -> complex:
    """
    Computes a linear interpolation of a complex 3D array at a specified position.

    Parameters
    ----------
    f : ndarray
        3D complex array to be interpolated
    x : ndarray
        1D X position array corresponding to 'f'
    y : ndarray
        1D Y position array corresponding to 'f'
    z : ndarray
        1D Z position array corresponding to 'f'
    x0 : float
        X position at which 'f' is to be interpolated
    y0 : float
        Y position at which 'f' is to be interpolated
    z0 : float
        Z position at which 'f' is to be interpolated

    Returns
    -------
    complex
        Interpolated value at (x0, y0, z0)
    """
    k = locator(z, z0)
    f1 = BilinearInterp_dpc(f[:, :, k], x, y, x0, y0)
    f2 = BilinearInterp_dpc(f[:, :, k+1], x, y, x0, y0)
    f0 = (f1 * (z[k+1] - z0) + f2 * (z0 - z[k])) / (z[k+1] - z[k])
    return f0


def TrilinearInterp_dp(f: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, x0: float, y0: float, z0: float) -> float:
    """
    Computes a linear interpolation of a real 3D array at a specified position.

    Parameters
    ----------
    f : ndarray
        3D real array to be interpolated
    x : ndarray
        1D X position array corresponding to 'f'
    y : ndarray
        1D Y position array corresponding to 'f'
    z : ndarray
        1D Z position array corresponding to 'f'
    x0 : float
        X position at which 'f' is to be interpolated
    y0 : float
        Y position at which 'f' is to be interpolated
    z0 : float
        Z position at which 'f' is to be interpolated

    Returns
    -------
    float
        Interpolated value at (x0, y0, z0)
    """
    k = locator(z, z0)
    f1 = BilinearInterp_dp(f[:, :, k], x, y, x0, y0)
    f2 = BilinearInterp_dp(f[:, :, k+1], x, y, x0, y0)
    f0 = (f1 * (z[k+1] - z0) + f2 * (z0 - z[k])) / (z[k+1] - z[k])
    return f0


#######################################################
################ DERIVATIVE FUNCTIONS #################
#######################################################

def dfdt_dp(f: np.ndarray, dt: float, k: int) -> float:
    """
    Returns the first derivative value of the array f(:) with respect to t
    at the index k with a five-point stencil method.

    Parameters
    ----------
    f : ndarray
        Function of t, 1D real array
    dt : float
        t differential
    k : int
        t-index for dfdt (0-based, but FORTRAN logic uses 1-based)

    Returns
    -------
    float
        Derivative value at index k
    """
    N = len(f)
    # FORTRAN uses 1-based indexing: k>2 means k>=3 (1-based) = k>=2 (0-based)
    # k<N-1 means k<=N-2 (1-based) = k<=N-3 (0-based)

    if k >= 2 and k < N - 2:
        # Five-point stencil: (-f(k+2) + 8*f(k+1) - 8*f(k-1) + f(k-2)) / (12*dt)
        return (-f[k+2] + 8.0 * f[k+1] - 8.0 * f[k-1] + f[k-2]) / dt / 12.0
    elif k == 0:
        # k==1 in FORTRAN (1-based) = k==0 (0-based): Forward difference
        return (f[1] - 0.0) / 2.0 / dt
    elif k == 1:
        # k==2 in FORTRAN (1-based) = k==1 (0-based): Central difference
        return (f[2] - f[0]) / 2.0 / dt
    elif k == N - 1:
        # k==N in FORTRAN (1-based) = k==N-1 (0-based): Backward difference
        return (0.0 - f[N-2]) / 2.0 / dt
    elif k == N - 2:
        # k==N-1 in FORTRAN (1-based) = k==N-2 (0-based): Central difference
        return (f[N-1] - f[N-3]) / 2.0 / dt
    else:
        return 0.0


def dfdt_dpc(f: np.ndarray, dt: float, k: int) -> complex:
    """
    Returns the first derivative value of the array f(:) with respect to t
    at the index k with a five-point stencil method.

    Parameters
    ----------
    f : ndarray
        Function of t, 1D complex array
    dt : float
        t differential
    k : int
        t-index for dfdt (0-based, but FORTRAN logic uses 1-based)

    Returns
    -------
    complex
        Derivative value at index k
    """
    N = len(f)
    # FORTRAN uses 1-based indexing: k>2 means k>=3 (1-based) = k>=2 (0-based)
    # k<N-1 means k<=N-2 (1-based) = k<=N-3 (0-based)

    if k >= 2 and k < N - 2:
        # Five-point stencil: (-f(k+2) + 8*f(k+1) - 8*f(k-1) + f(k-2)) / (12*dt)
        return (-f[k+2] + 8.0 * f[k+1] - 8.0 * f[k-1] + f[k-2]) / dt / 12.0
    elif k == 0:
        # k==1 in FORTRAN (1-based) = k==0 (0-based): Forward difference
        return (f[1] - 0.0) / 2.0 / dt
    elif k == 1:
        # k==2 in FORTRAN (1-based) = k==1 (0-based): Central difference
        return (f[2] - f[0]) / 2.0 / dt
    elif k == N - 1:
        # k==N in FORTRAN (1-based) = k==N-1 (0-based): Backward difference
        return (0.0 - f[N-2]) / 2.0 / dt
    elif k == N - 2:
        # k==N-1 in FORTRAN (1-based) = k==N-2 (0-based): Central difference
        return (f[N-1] - f[N-3]) / 2.0 / dt
    else:
        return 0.0


@jit(nopython=True, cache=True)
def _dfdt_1D_dp_jit(f: np.ndarray, dt: float) -> np.ndarray:
    """JIT-compiled version of dfdt_1D_dp."""
    N = len(f)
    dfdt_arr = np.zeros(N, dtype=np.float64)

    for k in range(N):
        if k >= 2 and k < N - 2:
            dfdt_arr[k] = (-f[k+2] + 8.0 * f[k+1] - 8.0 * f[k-1] + f[k-2]) / dt / 12.0
        elif k == 0:
            dfdt_arr[k] = (f[1] - 0.0) / 2.0 / dt
        elif k == 1:
            dfdt_arr[k] = (f[2] - f[0]) / 2.0 / dt
        elif k == N - 1:
            dfdt_arr[k] = (0.0 - f[N-2]) / 2.0 / dt
        elif k == N - 2:
            dfdt_arr[k] = (f[N-1] - f[N-3]) / 2.0 / dt
        else:
            dfdt_arr[k] = 0.0

    return dfdt_arr


@jit(nopython=True, cache=True)
def _dfdt_1D_dpc_jit(f: np.ndarray, dt: float) -> np.ndarray:
    """JIT-compiled version of dfdt_1D_dpc."""
    N = len(f)
    dfdt_arr = np.zeros(N, dtype=np.complex128)

    for k in range(N):
        if k >= 2 and k < N - 2:
            dfdt_arr[k] = (-f[k+2] + 8.0 * f[k+1] - 8.0 * f[k-1] + f[k-2]) / dt / 12.0
        elif k == 0:
            dfdt_arr[k] = (f[1] - 0.0) / 2.0 / dt
        elif k == 1:
            dfdt_arr[k] = (f[2] - f[0]) / 2.0 / dt
        elif k == N - 1:
            dfdt_arr[k] = (0.0 - f[N-2]) / 2.0 / dt
        elif k == N - 2:
            dfdt_arr[k] = (f[N-1] - f[N-3]) / 2.0 / dt
        else:
            dfdt_arr[k] = 0.0

    return dfdt_arr


def dfdt_1D_dp(f: np.ndarray, dt: float) -> np.ndarray:
    """
    Returns the first derivative of the array f(:) with respect to t
    for all points using a five-point stencil method.

    Parameters
    ----------
    f : ndarray
        Function of t, 1D real array
    dt : float
        t differential

    Returns
    -------
    ndarray
        Derivative array
    """
    try:
        return _dfdt_1D_dp_jit(f, dt)
    except:
        N = len(f)
        dfdt_arr = np.zeros(N, dtype=np.float64)
        for k in range(N):
            dfdt_arr[k] = dfdt_dp(f, dt, k)
        return dfdt_arr


def dfdt_1D_dpc(f: np.ndarray, dt: float) -> np.ndarray:
    """
    Returns the first derivative of the array f(:) with respect to t
    for all points using a five-point stencil method.

    Parameters
    ----------
    f : ndarray
        Function of t, 1D complex array
    dt : float
        t differential

    Returns
    -------
    ndarray
        Derivative array
    """
    try:
        return _dfdt_1D_dpc_jit(f, dt)
    except:
        N = len(f)
        dfdt_arr = np.zeros(N, dtype=np.complex128)
        for k in range(N):
            dfdt_arr[k] = dfdt_dpc(f, dt, k)
        return dfdt_arr


####################################################
################### ISNAN FUNCTIONS ################
####################################################

def isnan_dp(X: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
    """
    A double precision isnan.

    Wraps numpy's isnan for double precision.

    Parameters
    ----------
    X : float or ndarray
        Value(s) to check, double precision

    Returns
    -------
    bool or ndarray
        True if X is NaN, False otherwise
    """
    return np.isnan(X)


def isnan_sp(X: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
    """
    A single precision isnan.

    Wraps numpy's isnan for single precision.

    Parameters
    ----------
    X : float or ndarray
        Value(s) to check, single precision

    Returns
    -------
    bool or ndarray
        True if X is NaN, False otherwise
    """
    return np.isnan(X)

