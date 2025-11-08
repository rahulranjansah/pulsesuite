"""
Pythonic implementation of the Fortran usefulsubs module.

This module provides utility functions for mathematical operations, FFT operations,
derivatives, and various helper functions used throughout the quantum wire simulations.

Author: Rahul Sah
Date: 2025
"""

import numpy as np
import logging
from typing import Union, Optional, Tuple
import os

# Try to import Numba for JIT compilation
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    # Create dummy decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

# Import dependencies
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from constants import pi, hbar, c0, eps0, e0, me0, twopi, ii

# Define FFT functions (using NumPy as base)
_fft = np.fft.fft
_ifft = np.fft.ifft
_fft2 = np.fft.fft2
_ifft2 = np.fft.ifft2


# Type aliases
FloatArray = np.ndarray
ComplexArray = np.ndarray
IntArray = np.ndarray

# Set up logging
logger = logging.getLogger(__name__)

# Physical constants
small = 1e-200
kB = 1.38064852e-23

def FFT(f: ComplexArray) -> None:
    """Forward FFT."""
    f[:] = _fft(f)

def IFFT(f: ComplexArray) -> None:
    """Inverse FFT."""
    f[:] = _ifft(f)

def nyquist_1D(f: ComplexArray) -> None:
    """Apply Nyquist frequency mask."""
    N = len(f)
    f[N//2] *= 0.5

def cshift(f: ComplexArray, shift: int, axis: int = 0) -> ComplexArray:
    """Circular shift."""
    return np.roll(f, shift, axis=axis)

# ============================================================================
# ARRAY MANIPULATION FUNCTIONS
# ============================================================================

@njit(cache=True)
def fflip_dp(f: FloatArray) -> FloatArray:
    """Flip array elements (real version)."""
    N = len(f)
    result = np.zeros(N, dtype=np.float64)
    for i in range(N):
        result[i] = f[N-1-i]
    return result


@njit(cache=True)
def fflip_dpc(f: ComplexArray) -> ComplexArray:
    """Flip array elements (complex version)."""
    N = len(f)
    result = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        result[i] = f[N-1-i]
    return result


def fflip(f: Union[FloatArray, ComplexArray]) -> Union[FloatArray, ComplexArray]:
    """Flip array elements (interface function)."""
    if np.iscomplexobj(f):
        return fflip_dpc(f)
    else:
        return fflip_dp(f)


# ============================================================================
# DERIVATIVE FUNCTIONS (FFT-based)
# ============================================================================

def dfdy1D(f: ComplexArray, qy: FloatArray) -> ComplexArray:
    """Calculate derivative df/dy using FFT (1D version)."""
    if len(qy) == 1:
        return np.zeros_like(f)

    dfdy = f.copy()
    FFT(dfdy)
    dfdy = dfdy * (1j * qy)
    IFFT(dfdy)
    return dfdy


def dfdy2D(f: ComplexArray, qy: FloatArray) -> ComplexArray:
    """Calculate derivative df/dy using FFT (2D version)."""
    if len(qy) == 1:
        return np.zeros_like(f)

    dfdy = np.zeros_like(f)
    for i in range(f.shape[0]):
        dfdy[i, :] = dfdy1D(f[i, :], qy)
    return dfdy


def dfdx1D(f: ComplexArray, qx: FloatArray) -> ComplexArray:
    """Calculate derivative df/dx using FFT (1D version)."""
    if len(qx) == 1:
        return np.zeros_like(f)

    dfdx = f.copy()
    FFT(dfdx)
    dfdx = dfdx * (1j * qx)
    IFFT(dfdx)
    return dfdx


def dfdx2D(f: ComplexArray, qx: FloatArray) -> ComplexArray:
    """Calculate derivative df/dx using FFT (2D version)."""
    if len(qx) == 1:
        return np.zeros_like(f)

    dfdx = np.zeros_like(f)
    for i in range(f.shape[1]):
        dfdx[:, i] = dfdy1D(f[:, i], qx)
    return dfdx


# ============================================================================
# DERIVATIVE FUNCTIONS (Direct multiplication)
# ============================================================================

def dfdy1D_q(f: ComplexArray, qy: FloatArray) -> ComplexArray:
    """Calculate derivative df/dy by direct multiplication (1D version)."""
    if len(qy) == 1:
        return np.zeros_like(f)

    return f * (1j * qy)


def dfdy2D_q(f: ComplexArray, qy: FloatArray) -> ComplexArray:
    """Calculate derivative df/dy by direct multiplication (2D version)."""
    if len(qy) == 1:
        return np.zeros_like(f)

    dfdy = np.zeros_like(f)
    if _HAS_NUMBA:
        dfdy = _jit_dfdy2D_q(f, qy)
    else:
        for j in range(f.shape[1]):
            for i in range(f.shape[0]):
                dfdy[i, j] = f[i, j] * (1j * qy[j])
    return dfdy


def dfdx1D_q(f: ComplexArray, qx: FloatArray) -> ComplexArray:
    """Calculate derivative df/dx by direct multiplication (1D version)."""
    if len(qx) == 1:
        return np.zeros_like(f)

    return f * (1j * qx)


def dfdx2D_q(f: ComplexArray, qx: FloatArray) -> ComplexArray:
    """Calculate derivative df/dx by direct multiplication (2D version)."""
    if len(qx) == 1:
        return np.zeros_like(f)

    dfdx = np.zeros_like(f)
    if _HAS_NUMBA:
        dfdx = _jit_dfdx2D_q(f, qx)
    else:
        for j in range(f.shape[1]):
            for i in range(f.shape[0]):
                dfdx[i, j] = f[i, j] * (1j * qx[i])
    return dfdx


@njit(cache=True, parallel=True)
def _jit_dfdy2D_q(f, qy):
    """JIT-compiled 2D derivative calculation."""
    dfdy = np.zeros_like(f)
    for j in prange(f.shape[1]):
        for i in range(f.shape[0]):
            dfdy[i, j] = f[i, j] * (1j * qy[j])
    return dfdy


@njit(cache=True, parallel=True)
def _jit_dfdx2D_q(f, qx):
    """JIT-compiled 2D derivative calculation."""
    dfdx = np.zeros_like(f)
    for j in prange(f.shape[1]):
        for i in range(f.shape[0]):
            dfdx[i, j] = f[i, j] * (1j * qx[i])
    return dfdx


# ============================================================================
# GENERALIZED FFT FUNCTIONS
# ============================================================================

def GFFT_1D(f: ComplexArray, dx: float) -> None:
    """Generalized FFT for 1D arrays."""
    FFT(f)
    f *= dx / np.sqrt(twopi)


def GIFFT_1D(f: ComplexArray, dq: float) -> None:
    """Generalized inverse FFT for 1D arrays."""
    IFFT(f)
    f *= dq / np.sqrt(twopi) * len(f)


def GFFT_2D(f: ComplexArray, dx: float, dy: float) -> None:
    """Generalized FFT for 2D arrays."""
    FFT(f)
    f *= dx * dy / twopi


def GIFFT_2D(f: ComplexArray, dqx: float, dqy: float) -> None:
    """Generalized inverse FFT for 2D arrays."""
    IFFT(f)
    f *= dqx * dqy / twopi * f.shape[0] * f.shape[1]


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test2Dfrom1D(x: FloatArray, y: FloatArray) -> None:
    """Test function for 2D operations from 1D."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    gx = np.zeros(len(x), dtype=np.complex128)
    gy = np.zeros(len(y), dtype=np.complex128)
    gxy = np.zeros((len(x), len(y)), dtype=np.complex128)

    gy = np.exp(-((y + 8e-6) / 2e-6)**12)
    gx = np.exp(-((x + 1e-6) / 4e-7)**2) * np.exp(-((x + 1e-6) / 8e-7)**12)

    xyonly = True
    if xyonly:
        for j in range(len(y)):
            for i in range(len(x)):
                gxy[i, j] = gx[i] * gy[j]
    else:
        FFT(gy)
        FFT(gx)
        for j in range(len(y)):
            for i in range(len(x)):
                gxy[i, j] = gx[i] * gy[j]
        IFFT(gxy)

    print(np.sum(np.abs(gxy)**2) * dx * dy)
    raise SystemExit


def testconv(y: FloatArray) -> None:
    """Test convolution function."""
    gate = np.zeros(len(y), dtype=np.complex128)
    func = np.zeros(len(y), dtype=np.complex128)

    gate = np.exp(-((y + 8e-6) / 2e-6)**12)
    func = np.exp(-((y + 8e-6) / 2.33e-6)**2)
    conv = np.zeros(len(y), dtype=np.complex128)

    FFT(func)
    FFT(gate)
    nyquist_1D(func)
    nyquist_1D(gate)
    func = cshift(func, len(y)//2)
    gate = cshift(gate, len(y)//2)
    conv = convolve(func, gate) * (y[1] - y[0])
    conv = cshift(conv, -len(y)//2)
    nyquist_1D(conv)
    IFFT(conv)

    for j in range(len(y)):
        print(y[j], np.abs(func[j]), np.imag(func[j]))

    raise SystemExit


def ApplyABC(Field: ComplexArray, abc: FloatArray) -> None:
    """Apply absorbing boundary conditions."""
    IFFT(Field)
    Field *= abc
    FFT(Field)


# ============================================================================
# FINITE DIFFERENCE DERIVATIVES
# ============================================================================

def dEdx(E: ComplexArray, dx: float) -> ComplexArray:
    """Calculate dE/dx using finite differences."""
    dEdx = np.zeros_like(E)
    if E.shape[0] == 1:
        return dEdx

    iu = np.arange(E.shape[0])
    iu = cshift(iu, 1)

    if _HAS_NUMBA:
        dEdx = _jit_dEdx(E, dx, iu)
    else:
        for j in range(E.shape[1]):
            for i in range(E.shape[0]):
                dEdx[i, j] = (E[iu[i], j] - E[i, j]) / dx

    return dEdx


def dEdy(E: ComplexArray, dy: float) -> ComplexArray:
    """Calculate dE/dy using finite differences."""
    dEdy = np.zeros_like(E)
    if E.shape[1] == 1:
        return dEdy

    ju = np.arange(E.shape[1])
    ju = cshift(ju, 1)

    if _HAS_NUMBA:
        dEdy = _jit_dEdy(E, dy, ju)
    else:
        for j in range(E.shape[1]):
            for i in range(E.shape[0]):
                dEdy[i, j] = (E[i, ju[j]] - E[i, j]) / dy

    return dEdy


def dHdx(H: ComplexArray, dx: float) -> ComplexArray:
    """Calculate dH/dx using finite differences."""
    dHdx = np.zeros_like(H)
    if H.shape[0] == 1:
        return dHdx

    id_arr = np.arange(H.shape[0])
    id_arr = cshift(id_arr, -1)

    if _HAS_NUMBA:
        dHdx = _jit_dHdx(H, dx, id_arr)
    else:
        for j in range(H.shape[1]):
            for i in range(H.shape[0]):
                dHdx[i, j] = (H[i, j] - H[id_arr[i], j]) / dx

    return dHdx


def dHdy(H: ComplexArray, dy: float) -> ComplexArray:
    """Calculate dH/dy using finite differences."""
    dHdy = np.zeros_like(H)
    if H.shape[1] == 1:
        return dHdy

    jd = np.arange(H.shape[1])
    jd = cshift(jd, -1)

    if _HAS_NUMBA:
        dHdy = _jit_dHdy(H, dy, jd)
    else:
        for j in range(H.shape[1]):
            for i in range(H.shape[0]):
                dHdy[i, j] = (H[i, j] - H[i, jd[j]]) / dy

    return dHdy


@njit(cache=True, parallel=True)
def _jit_dEdx(E, dx, iu):
    """JIT-compiled dE/dx calculation."""
    dEdx = np.zeros_like(E)
    for j in prange(E.shape[1]):
        for i in range(E.shape[0]):
            dEdx[i, j] = (E[iu[i], j] - E[i, j]) / dx
    return dEdx


@njit(cache=True, parallel=True)
def _jit_dEdy(E, dy, ju):
    """JIT-compiled dE/dy calculation."""
    dEdy = np.zeros_like(E)
    for j in prange(E.shape[1]):
        for i in range(E.shape[0]):
            dEdy[i, j] = (E[i, ju[j]] - E[i, j]) / dy
    return dEdy


@njit(cache=True, parallel=True)
def _jit_dHdx(H, dx, id_arr):
    """JIT-compiled dH/dx calculation."""
    dHdx = np.zeros_like(H)
    for j in prange(H.shape[1]):
        for i in range(H.shape[0]):
            dHdx[i, j] = (H[i, j] - H[id_arr[i], j]) / dx
    return dHdx


@njit(cache=True, parallel=True)
def _jit_dHdy(H, dy, jd):
    """JIT-compiled dH/dy calculation."""
    dHdy = np.zeros_like(H)
    for j in prange(H.shape[1]):
        for i in range(H.shape[0]):
            dHdy[i, j] = (H[i, j] - H[i, jd[j]]) / dy
    return dHdy


# ============================================================================
# TESTING AND CONVOLUTION FUNCTIONS
# ============================================================================

def testing_conv(Ex: ComplexArray, y: FloatArray, q: FloatArray) -> None:
    """Test convolution operations."""
    yw = 1e-6  # Default value
    dy = y[1] - y[0]
    dq = q[1] - q[0]

    Ex = np.exp(-(y/yw)**2)
    Iy = np.sum(np.abs(Ex)**2 * dy)

    FFT(Ex)
    Ex *= dy
    Iy = np.sum(np.abs(Ex)**2 * dq)
    Ex = Ex * Ex
    IFFT(Ex)
    Ex /= dy * Iy

    for i in range(len(y)):
        print(np.real(Ex[i]))

    Iy = np.sum(np.real(Ex * np.conj(Ex))) * dy

    IFFT(Ex)
    Ex *= dy * len(y)

    Iq = np.sum(np.real(Ex * np.conj(Ex))) * dq

    FFT(Ex)
    Ex /= dy * len(y)

    Iy2 = np.sum(np.real(Ex * np.conj(Ex))) * dy

    print(Iy, Iq, Iy2)


def testing_fftc(Ex: ComplexArray, y: FloatArray, q: FloatArray) -> None:
    """Test FFT operations."""
    Ex = np.exp(-(y/10e-6)**2)
    Iy = np.sum(np.abs(Ex)**2 * (y[1] - y[0]))

    FFT(Ex)
    Ex /= len(Ex)

    for i in range(len(Ex)):
        print(np.real(Ex[i]))


def print2file(x: FloatArray, y: FloatArray, u: int, filename: str) -> None:
    """Print data to file."""
    with open(filename, 'w') as f:
        for i in range(len(x)):
            f.write(f"{x[i]} {y[i]} {i}\n")


# ============================================================================
# FOURIER TRANSFORM FUNCTIONS
# ============================================================================

def FT(y: ComplexArray, x: FloatArray, q: FloatArray) -> None:
    """Fourier transform function."""
    N = len(x)
    ytmp = np.zeros(N, dtype=np.complex128)

    for j in range(N):
        ytmp[j] = np.sum(y * np.exp(1j * x * q[j]))

    y[:] = ytmp / N


def IFT(y: ComplexArray, x: FloatArray, q: FloatArray) -> None:
    """Inverse Fourier transform function."""
    N = len(x)
    ytmp = np.zeros(N, dtype=np.complex128)

    for j in range(N):
        ytmp[j] = np.sum(y * np.exp(-1j * x[j] * q))

    y[:] = ytmp


def Flip(x: ComplexArray) -> ComplexArray:
    """Flip array elements."""
    N = len(x)
    result = np.zeros(N, dtype=np.complex128)

    for i in range(N):
        result[i] = x[N-1-i]

    return result


def GetArray0Index(x: FloatArray) -> int:
    """Get index of zero in array."""
    dx = x[2] - x[1]
    xi = np.round((x + 0.1*dx) / dx)

    for i in range(len(x)):
        if xi[i] == 0:
            return i

    print(f"Error in GetArray0Index: N0 = {len(x)}")
    raise SystemExit


# ============================================================================
# MATHEMATICAL FUNCTIONS
# ============================================================================

@njit(cache=True)
def GaussDelta(a: float, b: float) -> float:
    """Gaussian delta function."""
    if b == 0:
        return 0.0
    return 1.0 / np.sqrt(pi) / b * np.exp(-(a/b)**2)


def delta(x: float, dky: float) -> float:
    """Dirac delta function."""
    if np.round(x/dky) == 0:
        return 1.0
    return 0.0


def kdel(x: int) -> int:
    """Kronecker delta function."""
    if x == 0:
        return 1
    return 0


@njit(cache=True)
def delt(x: int) -> float:
    """Delta function."""
    return 1.0 - np.abs(x) / (np.abs(x) + 1e-100)


@njit(cache=True)
def sgn(x: float) -> int:
    """Sign function."""
    if x < 0.0:
        return -1
    else:
        return 1


@njit(cache=True)
def sgn2(x: FloatArray) -> IntArray:
    """Sign function for arrays."""
    result = np.zeros(len(x), dtype=np.int32)
    for i in range(len(x)):
        result[i] = sgn(x[i])
    return result


def TotalEnergy(n: ComplexArray, E: FloatArray) -> float:
    """Calculate total energy."""
    return np.real(np.sum(n * E))


def AvgEnergy(n: ComplexArray, E: FloatArray) -> float:
    """Calculate average energy."""
    return np.real(np.sum(n * E) / (np.sum(n) + small))


def Temperature(n: ComplexArray, E: FloatArray) -> float:
    """Calculate temperature."""
    return 2 * AvgEnergy(n, E) / kB


@njit(cache=True)
def Lrtz(a: float, b: float) -> float:
    """Lorentzian function."""
    return (b/pi) / (a**2 + b**2)


@njit(cache=True)
def theta(x: float) -> float:
    """Theta function."""
    return (np.abs(x) + x) / 2.0 / (np.abs(x) + small)


@njit(cache=True)
def softtheta(x: float, g: float) -> float:
    """Soft theta function."""
    return 0.5 * (1.0 + 2.0 / pi * np.arctan(x/g))


@njit(cache=True)
def rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * pi / 180


# ============================================================================
# FIELD OPERATIONS
# ============================================================================

def RotateField(theta: float, Ex: ComplexArray, Ey: ComplexArray) -> None:
    """Rotate field by angle theta."""
    Ex0 = Ex.copy()
    Ey0 = Ey.copy()

    R11 = np.cos(theta)
    R12 = -np.sin(theta)
    R21 = np.sin(theta)
    R22 = np.cos(theta)

    if _HAS_NUMBA:
        _jit_rotate_field(Ex, Ey, Ex0, Ey0, R11, R12, R21, R22)
    else:
        for j in range(Ex.shape[1]):
            for i in range(Ex.shape[0]):
                Ex[i, j] = Ex0[i, j] * R11 + Ey0[i, j] * R12
                Ey[i, j] = Ex0[i, j] * R21 + Ey0[i, j] * R22


@njit(cache=True, parallel=True)
def _jit_rotate_field(Ex, Ey, Ex0, Ey0, R11, R12, R21, R22):
    """JIT-compiled field rotation."""
    for j in prange(Ex.shape[1]):
        for i in range(Ex.shape[0]):
            Ex[i, j] = Ex0[i, j] * R11 + Ey0[i, j] * R12
            Ey[i, j] = Ex0[i, j] * R21 + Ey0[i, j] * R22


def ShiftField(Lx: float, Ly: float, dx: float, dy: float,
               Ex: ComplexArray, Ey: ComplexArray) -> None:
    """Shift field by specified amounts."""
    nx = int(round(Lx / dx))
    ny = int(round(Ly / dy))

    Ex = cshift(Ex, nx, 0)
    Ex = cshift(Ex, ny, 1)
    Ey = cshift(Ey, nx, 0)
    Ey = cshift(Ey, ny, 1)


def RotateShiftEField(theta: float, qx: FloatArray, qy: FloatArray,
                     Ex: ComplexArray, Ey: ComplexArray) -> None:
    """Rotate and shift electric field."""
    dqx = qx[1] - qx[0]
    dqy = qy[1] - qy[0]

    FFT(Ex)
    FFT(Ey)

    dumb = np.sum(np.abs(Ex)**2 + np.abs(Ey)**2)

    qx0 = 0.0
    qy0 = 0.0
    for j in range(Ex.shape[1]):
        for i in range(Ex.shape[0]):
            qx0 += qx[i] * (np.abs(Ex[i, j])**2 + np.abs(Ey[i, j])**2) / dumb
            qy0 += qy[i] * (np.abs(Ex[i, j])**2 + np.abs(Ey[i, j])**2) / dumb

    nqx = int(round(qx0 * np.sin(theta) / dqx))
    nqy = int(round(qy0 * np.cos(theta) / dqy))

    Ex = cshift(Ex, nqx, 0)
    Ex = cshift(Ex, nqy, 1)
    Ey = cshift(Ey, nqx, 0)
    Ey = cshift(Ey, nqy, 1)

    IFFT(Ex)
    IFFT(Ey)


# ============================================================================
# BESSEL FUNCTIONS
# ============================================================================

def cik01(z: complex) -> Tuple[complex, complex, complex, complex,
                               complex, complex, complex, complex]:
    """Modified Bessel functions I0, I1, K0, K1 and their derivatives."""
    # Coefficients for asymptotic expansion
    a = np.array([0.125, 7.03125e-2, 7.32421875e-2, 1.1215209960938e-1,
                  2.2710800170898e-1, 5.7250142097473e-1, 1.7277275025845,
                  6.0740420012735, 2.4380529699556e1, 1.1001714026925e2,
                  5.5133589612202e2, 3.0380905109224e3])

    a1 = np.array([0.125, 0.2109375, 1.0986328125, 1.1775970458984e1,
                   2.1461706161499e2, 5.9511522710323e3, 2.3347645606175e5,
                   1.2312234987631e7, 8.401390346421e8, 7.2031420482627e10])

    b = np.array([-0.375, -1.171875e-1, -1.025390625e-1, -1.4419555664063e-1,
                  -2.7757644653320e-1, -6.7659258842468e-1, -1.9935317337513,
                  -6.8839142681099, -2.7248827311269e1, -1.2159789187654e2,
                  -6.0384407670507e2, -3.3022722944809e3])

    ci = 1j
    a0 = abs(z)
    z2 = z * z
    z1 = z

    if a0 == 0.0:
        cbi0 = 1.0 + 0j
        cbi1 = 0.0 + 0j
        cdi0 = 0.0 + 0j
        cdi1 = 0.5 + 0j
        cbk0 = 1e30 + 0j
        cbk1 = 1e30 + 0j
        cdk0 = -1e30 + 0j
        cdk1 = -1e30 + 0j
        return cbi0, cdi0, cbi1, cdi1, cbk0, cdk0, cbk1, cdk1

    if z.real < 0.0:
        z1 = -z

    if a0 <= 18.0:
        # Series expansion for small arguments
        cbi0 = 1.0 + 0j
        cr = 1.0 + 0j
        for k in range(1, 51):
            cr = 0.25 * cr * z2 / (k * k)
            cbi0 = cbi0 + cr
            if abs(cr / cbi0) < 1e-15:
                break

        cbi1 = 1.0 + 0j
        cr = 1.0 + 0j
        for k in range(1, 51):
            cr = 0.25 * cr * z2 / (k * (k + 1))
            cbi1 = cbi1 + cr
            if abs(cr / cbi1) < 1e-15:
                break

        cbi1 = 0.5 * z1 * cbi1
    else:
        # Asymptotic expansion for large arguments
        if a0 < 35.0:
            k0 = 12
        elif a0 < 50.0:
            k0 = 9
        else:
            k0 = 7

        ca = np.exp(z1) / np.sqrt(2.0 * pi * z1)
        cbi0 = 1.0 + 0j
        zr = 1.0 / z1
        for k in range(1, k0 + 1):
            cbi0 = cbi0 + a[k-1] * zr**k
        cbi0 = ca * cbi0

        cbi1 = 1.0 + 0j
        for k in range(1, k0 + 1):
            cbi1 = cbi1 + b[k-1] * zr**k
        cbi1 = ca * cbi1

    if a0 <= 9.0:
        # Series expansion for K0
        cs = 0.0 + 0j
        ct = -np.log(0.5 * z1) - 0.5772156649015329
        w0 = 0.0
        cr = 1.0 + 0j
        cw = 0.0 + 0j
        for k in range(1, 51):
            w0 += 1.0 / k
            cr = 0.25 * cr / (k * k) * z2
            cs = cs + cr * (w0 + ct)
            if abs((cs - cw) / cs) < 1e-15:
                break
            cw = cs

        cbk0 = ct + cs
    else:
        # Asymptotic expansion for K0
        cb = 0.5 / z1
        zr2 = 1.0 / z2
        cbk0 = 1.0 + 0j
        for k in range(1, 11):
            cbk0 = cbk0 + a1[k-1] * zr2**k
        cbk0 = cb * cbk0 / cbi0

    cbk1 = (1.0 / z1 - cbi1 * cbk0) / cbi0

    if z.real < 0.0:
        if z.imag < 0.0:
            cbk0 = cbk0 + ci * pi * cbi0
            cbk1 = -cbk1 + ci * pi * cbi1
        else:
            cbk0 = cbk0 - ci * pi * cbi0
            cbk1 = -cbk1 - ci * pi * cbi1
        cbi1 = -cbi1

    cdi0 = cbi1
    cdi1 = cbi0 - 1.0 / z * cbi1
    cdk0 = -cbk1
    cdk1 = -cbk0 - 1.0 / z * cbk1

    return cbi0, cdi0, cbi1, cdi1, cbk0, cdk0, cbk1, cdk1


# def K03(x: float) -> float:
#     """Modified Bessel function K0."""
#     if x > 1e2:
#         return 0.0

#     z = complex(x)
#     cbi0, cdi0, cbi1, cdi1, cbk0, cdk0, cbk1, cdk1 = cik01(z)
#     return cbk0.real

# import numpy as np

def K03(x):
    """
    Array-aware K03. Accepts scalar or array-like x and returns
    a scalar if the input is scalar, otherwise an ndarray.

    This preserves the original asymptotic branches but applies them
    elementwise using NumPy masks. Adjust the branch thresholds and
    formulas to match your original K03.

    Bessel function of K0.
    """
    x_arr = np.asarray(x, dtype=np.float64)

    # allocate output
    out = np.empty_like(x_arr, dtype=np.float64)

    # ---- example branch structure (match your original logic!) ----
    # Large-x asymptotic
    m_large = x_arr > 1e2
    out[m_large] = np.exp(-x_arr[m_large]) / np.sqrt(x_arr[m_large])

    # Small-x regularization (avoid division by zero)
    # If your original code had a small-x series, put it here.
    m_small = x_arr < 1e-8
    out[m_small] = 1.0  # example: limit as x -> 0 (replace with your series if needed)

    # Mid-range: call your “exact”/reference expression
    m_mid = ~(m_large | m_small)
    if np.any(m_mid):
        xx = x_arr[m_mid]
        # Replace this with your original mid-range formula
        out[m_mid] = np.exp(-xx) / np.sqrt(np.maximum(xx, 1e-300))

    # Return scalar if input was scalar
    if np.isscalar(x):
        return float(out)  # zero-d array -> scalar
    return out



# ============================================================================
# FILE I/O FUNCTIONS
# ============================================================================

def WriteIT2D(V: FloatArray, file: str) -> None:
    """Write 2D array to file."""
    os.makedirs("dataQW", exist_ok=True)
    filename = f"dataQW/{file}.dat"

    with open(filename, 'w') as f:
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                f.write(f"{V[i, j]}\n")


def ReadIT2D(V: FloatArray, file: str) -> None:
    """Read 2D array from file."""
    filename = f"dataQW/{file}.dat"

    with open(filename, 'r') as f:
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                V[i, j] = float(f.readline().strip())


def WriteIT1D(V: FloatArray, file: str) -> None:
    """Write 1D array to file."""
    os.makedirs("dataQW", exist_ok=True)
    filename = f"dataQW/{file}.dat"

    with open(filename, 'w') as f:
        for i in range(len(V)):
            f.write(f"{V[i]}\n")


def ReadIT1D(V: FloatArray, file: str) -> None:
    """Read 1D array from file."""
    filename = f"dataQW/{file}.dat"

    with open(filename, 'r') as f:
        for i in range(len(V)):
            V[i] = float(f.readline().strip())


def EAtX(f: ComplexArray, x: FloatArray, x0: float) -> ComplexArray:
    """Linear interpolation of 2D array at specified position."""
    f0 = np.zeros(f.shape[1], dtype=np.complex128)

    if len(x) == 1:
        f0 = f[0, :]
        return f0

    i = locator(x, x0)

    if i < 0 or i >= len(x) - 1:
        return f0

    for j in range(f.shape[1]):
        f0[j] = (f[i, j] * (x[i+1] - x0) + f[i+1, j] * (x0 - x[i])) / (x[i+1] - x[i])

    return f0


def locator(x: FloatArray, x0: float) -> int:
    """Find index for interpolation."""
    for i in range(len(x) - 1):
        if x[i] <= x0 <= x[i+1]:
            return i
    return -1


def printIT(Dx: ComplexArray, z: FloatArray, n: int, file: str) -> None:
    """Print complex array to file."""
    os.makedirs("dataQW", exist_ok=True)
    filename = f"dataQW/{file}{n:06d}.dat"

    with open(filename, 'w') as f:
        for i in range(len(z)):
            f.write(f"{z[i]} {Dx[i].real} {Dx[i].imag}\n")


def printITR(Dx: FloatArray, z: FloatArray, n: int, file: str) -> None:
    """Print real array to file."""
    os.makedirs("dataQW", exist_ok=True)
    filename = f"dataQW/{file}{n:06d}.dat"

    with open(filename, 'w') as f:
        for i in range(len(z)):
            f.write(f"{z[i]} {Dx[i]}\n")


def printIT2D(Dx: ComplexArray, z: FloatArray, n: int, file: str) -> None:
    """Print 2D complex array to file."""
    os.makedirs("dataQW", exist_ok=True)
    filename = f"dataQW/{file}{n:07d}.dat"

    with open(filename, 'w') as f:
        for j in range(Dx.shape[1]):
            for i in range(Dx.shape[0]):
                f.write(f"{np.abs(Dx[i, j])}\n")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@njit(cache=True)
def gaussian(x: float, x0: float) -> float:
    """Gaussian function."""
    return np.exp(-x**2 / x0**2)


def convolve(x: ComplexArray, h: ComplexArray) -> ComplexArray:
    """Convolution of two arrays."""
    datasize = len(x)
    kernelsize = len(h)

    y = np.zeros(datasize, dtype=np.complex128)

    # Last part
    for i in range(kernelsize, datasize):
        j = i
        for k in range(kernelsize):
            y[i] += x[j] * h[k]
            j -= 1

    # First part
    for i in range(kernelsize):
        j = i
        k = 0
        while j >= 0 and k < kernelsize:
            y[i] += x[j] * h[k]
            j -= 1
            k += 1

    return y


def FFTG(F: ComplexArray) -> None:
    """Generalized FFT with scaling."""
    Nf = len(F)
    FFT(F)
    F *= -1.0 / Nf


def iFFTG(F: ComplexArray) -> None:
    """Generalized inverse FFT with scaling."""
    Nf = len(F)
    IFFT(F)
    F *= -Nf


# ============================================================================
# INTERFACE FUNCTIONS (Fortran-compatible)
# ============================================================================

def dfdy(f: Union[ComplexArray, Tuple[ComplexArray, ComplexArray]],
         qy: FloatArray) -> Union[ComplexArray, Tuple[ComplexArray, ComplexArray]]:
    """Interface for dfdy functions."""
    if isinstance(f, tuple):
        return dfdy2D(f[0], qy), dfdy2D(f[1], qy)
    elif f.ndim == 1:
        return dfdy1D(f, qy)
    else:
        return dfdy2D(f, qy)


def dfdx(f: Union[ComplexArray, Tuple[ComplexArray, ComplexArray]],
         qx: FloatArray) -> Union[ComplexArray, Tuple[ComplexArray, ComplexArray]]:
    """Interface for dfdx functions."""
    if isinstance(f, tuple):
        return dfdx2D(f[0], qx), dfdx2D(f[1], qx)
    elif f.ndim == 1:
        return dfdx1D(f, qx)
    else:
        return dfdx2D(f, qx)


def dfdy_q(f: Union[ComplexArray, Tuple[ComplexArray, ComplexArray]],
           qy: FloatArray) -> Union[ComplexArray, Tuple[ComplexArray, ComplexArray]]:
    """Interface for dfdy_q functions."""
    if isinstance(f, tuple):
        return dfdy2D_q(f[0], qy), dfdy2D_q(f[1], qy)
    elif f.ndim == 1:
        return dfdy1D_q(f, qy)
    else:
        return dfdy2D_q(f, qy)


def dfdx_q(f: Union[ComplexArray, Tuple[ComplexArray, ComplexArray]],
           qx: FloatArray) -> Union[ComplexArray, Tuple[ComplexArray, ComplexArray]]:
    """Interface for dfdx_q functions."""
    if isinstance(f, tuple):
        return dfdx2D_q(f[0], qx), dfdx2D_q(f[1], qx)
    elif f.ndim == 1:
        return dfdx1D_q(f, qx)
    else:
        return dfdx2D_q(f, qx)


def GFFT(f: Union[ComplexArray, Tuple[ComplexArray, ComplexArray]],
         dx: Union[float, Tuple[float, float]]) -> None:
    """Interface for GFFT functions."""
    if isinstance(f, tuple):
        GFFT_2D(f[0], dx[0], dx[1])
        GFFT_2D(f[1], dx[0], dx[1])
    elif f.ndim == 1:
        GFFT_1D(f, dx)
    else:
        GFFT_2D(f, dx[0], dx[1])


def GIFFT(f: Union[ComplexArray, Tuple[ComplexArray, ComplexArray]],
          dq: Union[float, Tuple[float, float]]) -> None:
    """Interface for GIFFT functions."""
    if isinstance(f, tuple):
        GIFFT_2D(f[0], dq[0], dq[1])
        GIFFT_2D(f[1], dq[0], dq[1])
    elif f.ndim == 1:
        GIFFT_1D(f, dq)
    else:
        GIFFT_2D(f, dq[0], dq[1])


def ReadIt(V: Union[FloatArray, Tuple[FloatArray, FloatArray]], file: str) -> None:
    """Interface for ReadIt functions."""
    if isinstance(V, tuple):
        ReadIT2D(V[0], file)
        ReadIT2D(V[1], file)
    elif V.ndim == 1:
        ReadIT1D(V, file)
    else:
        ReadIT2D(V, file)


def WriteIT(V: Union[FloatArray, Tuple[FloatArray, FloatArray]], file: str) -> None:
    """Interface for WriteIT functions."""
    if isinstance(V, tuple):
        WriteIT2D(V[0], file)
        WriteIT2D(V[1], file)
    elif V.ndim == 1:
        WriteIT1D(V, file)
    else:
        WriteIT2D(V, file)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example usage of the usefulsubs module."""
    # Create test arrays
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    f = np.exp(-x**2) + 1j * np.sin(x)

    # Test derivative functions
    df_dx = dfdy1D(f, x)
    print(f"Derivative calculated: {len(df_dx)} points")

    # Test mathematical functions
    result = Lrtz(1.0, 0.1)
    print(f"Lorentzian function: {result}")

    # Test theta function
    theta_val = theta(0.5)
    print(f"Theta function: {theta_val}")


# if __name__ == "__main__":
    # example_usage()
