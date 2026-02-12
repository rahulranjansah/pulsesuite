"""
Useful subroutines for quantum wire simulations.

This module provides utility functions for array manipulation and
derivative calculations using FFT methods.

Author: Rahul R. Sah

Bug fixes:
    Fortran cshift(array, -1): shifts right (wraps left to right)
    NumPy np.roll(array, 1): shifts left (wraps right to left)

"""

import os

import numpy as np
import pyfftw
from numba import jit

pyfftw.interfaces.cache.enable()

# Imaginary unit
ii = 1j
pi = np.pi
twopi = 2.0 * np.pi

# Small constant for numerical stability
small = 1e-100

# Boltzmann constant (J/K)
kB = 1.380649e-23


@jit(nopython=True)
def _fflip_dp_jit(f):
    """JIT-compiled version of fflip_dp."""
    N = len(f)
    result = np.zeros_like(f)
    for i in range(N):
        result[i] = f[N - 1 - i]
    return result


@jit(nopython=True)
def _fflip_dpc_jit(f):
    """JIT-compiled version of fflip_dpc."""
    N = len(f)
    result = np.zeros_like(f, dtype=np.complex128)
    for i in range(N):
        result[i] = f[N - 1 - i]
    return result


def fflip_dp(f):
    """
    Flip a real array.

    Reverses the order of elements in a real array.

    Parameters
    ----------
    f : ndarray
        Input real array, 1D array

    Returns
    -------
    ndarray
        Flipped array, 1D array
    """
    return _fflip_dp_jit(f)


def fflip_dpc(f):
    """
    Flip a complex array.

    Reverses the order of elements in a complex array.

    Parameters
    ----------
    f : ndarray
        Input complex array, 1D array

    Returns
    -------
    ndarray
        Flipped array, 1D array
    """
    return _fflip_dpc_jit(f)


def dfdy1D(f, qy):
    """
    Calculate derivative in y direction for 1D array using FFT.

    Computes the derivative df/dy using FFT method:
    df/dy = IFFT(ii * qy * FFT(f))

    Parameters
    ----------
    f : ndarray
        Input function (complex), 1D array
    qy : ndarray
        Momentum coordinates in y direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dy (complex), 1D array

    Notes
    -----
    Uses pyfftw FFT functions for computation.
    Returns zero array if qy has only one element.
    """
    if len(qy) == 1:
        return np.zeros_like(f)

    result = f.copy()
    result = pyfftw.interfaces.numpy_fft.fft(result)
    result = result * (ii * qy)
    result = pyfftw.interfaces.numpy_fft.ifft(result)
    return result


def dfdy2D(f, qy):
    """
    Calculate derivative in y direction for 2D array using FFT.

    Computes the derivative df/dy for each row of a 2D array using
    the 1D derivative function.

    Parameters
    ----------
    f : ndarray
        Input function (complex), 2D array
    qy : ndarray
        Momentum coordinates in y direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dy (complex), 2D array with same shape as f

    Notes
    -----
    This function calls dfdy1D for each row of the 2D array.
    Returns zero array if qy has only one element.
    """
    if len(qy) == 1:
        return np.zeros_like(f)

    Nx = f.shape[0]
    result = np.zeros_like(f)

    for i in range(Nx):
        result[i, :] = dfdy1D(f[i, :], qy)

    return result


def dfdx1D(f, qx):
    """
    Calculate derivative in x direction for 1D array using FFT.

    Computes the derivative df/dx using FFT method:
    df/dx = IFFT(ii * qx * FFT(f))

    Parameters
    ----------
    f : ndarray
        Input function (complex), 1D array
    qx : ndarray
        Momentum coordinates in x direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dx (complex), 1D array

    Notes
    -----
    Uses pyfftw FFT functions for computation.
    Returns zero array if qx has only one element.
    """
    if len(qx) == 1:
        return np.zeros_like(f)

    result = f.copy()
    result = pyfftw.interfaces.numpy_fft.fft(result)
    result = result * (ii * qx)
    result = pyfftw.interfaces.numpy_fft.ifft(result)
    return result


def dfdx2D(f, qx):
    """
    Calculate derivative in x direction for 2D array using FFT.

    Computes the derivative df/dx for each column of a 2D array using
    the 1D derivative function.

    Parameters
    ----------
    f : ndarray
        Input function (complex), 2D array
    qx : ndarray
        Momentum coordinates in x direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dx (complex), 2D array with same shape as f

    Notes
    -----
    This function calls dfdy1D for each column of the 2D array.
    Returns zero array if qx has only one element.
    """
    if len(qx) == 1:
        return np.zeros_like(f)

    Ny = f.shape[1]
    result = np.zeros_like(f)

    for i in range(Ny):
        result[:, i] = dfdy1D(f[:, i], qx)

    return result


def dfdy1D_q(f, qy):
    """
    Calculate derivative in y direction for 1D array in q-space.

    Computes the derivative df/dy in q-space (Fourier space):
    df/dy = f * (ii * qy)

    Parameters
    ----------
    f : ndarray
        Input function in q-space (complex), 1D array
    qy : ndarray
        Momentum coordinates in y direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dy in q-space (complex), 1D array

    Notes
    -----
    This is the q-space version that operates directly on Fourier-transformed
    arrays without requiring FFT/IFFT calls.
    Returns zero array if qy has only one element.
    """
    if len(qy) == 1:
        return np.zeros_like(f)

    return f * (ii * qy)


@jit(nopython=True)
def _dfdy2D_q_jit(f, qy, ii_val):
    """JIT-compiled version of dfdy2D_q."""
    Nx = f.shape[0]
    Ny = f.shape[1]
    result = np.zeros_like(f, dtype=np.complex128)

    for j in range(Ny):
        for i in range(Nx):
            result[i, j] = f[i, j] * (ii_val * qy[j])

    return result


def dfdy2D_q(f, qy):
    """
    Calculate derivative in y direction for 2D array in q-space.

    Computes the derivative df/dy in q-space (Fourier space) for each element:
    df/dy(i,j) = f(i,j) * (ii * qy(j))

    Parameters
    ----------
    f : ndarray
        Input function in q-space (complex), 2D array
    qy : ndarray
        Momentum coordinates in y direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dy in q-space (complex), 2D array with same shape as f

    Notes
    -----
    This is the q-space version that operates directly on Fourier-transformed
    arrays without requiring FFT/IFFT calls.
    Returns zero array if qy has only one element.
    """
    if len(qy) == 1:
        return np.zeros_like(f)

    return _dfdy2D_q_jit(f, qy, ii)


def dfdx1D_q(f, qx):
    """
    Calculate derivative in x direction for 1D array in q-space.

    Computes the derivative df/dx in q-space (Fourier space):
    df/dx = f * (ii * qx)

    Parameters
    ----------
    f : ndarray
        Input function in q-space (complex), 1D array
    qx : ndarray
        Momentum coordinates in x direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dx in q-space (complex), 1D array

    Notes
    -----
    This is the q-space version that operates directly on Fourier-transformed
    arrays without requiring FFT/IFFT calls.
    Returns zero array if qx has only one element.
    """
    if len(qx) == 1:
        return np.zeros_like(f)

    return f * (ii * qx)


@jit(nopython=True)
def _dfdx2D_q_jit(f, qx, ii_val):
    """JIT-compiled version of dfdx2D_q."""
    Nx = f.shape[0]
    Ny = f.shape[1]
    result = np.zeros_like(f, dtype=np.complex128)

    for j in range(Ny):
        for i in range(Nx):
            result[i, j] = f[i, j] * (ii_val * qx[i])

    return result


def dfdx2D_q(f, qx):
    """
    Calculate derivative in x direction for 2D array in q-space.

    Computes the derivative df/dx in q-space (Fourier space) for each element:
    df/dx(i,j) = f(i,j) * (ii * qx(i))

    Parameters
    ----------
    f : ndarray
        Input function in q-space (complex), 2D array
    qx : ndarray
        Momentum coordinates in x direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dx in q-space (complex), 2D array with same shape as f

    Notes
    -----
    This is the q-space version that operates directly on Fourier-transformed
    arrays without requiring FFT/IFFT calls.
    Returns zero array if qx has only one element.
    """
    if len(qx) == 1:
        return np.zeros_like(f)

    return _dfdx2D_q_jit(f, qx, ii)


def GFFT_1D(f, dx):
    """
    Gaussian FFT for 1D array.

    Performs FFT and scales by dx/sqrt(2*pi) for Gaussian normalization.

    Parameters
    ----------
    f : ndarray
        Input function (complex), modified in-place, 1D array
    dx : float
        Spatial step size (m)

    Returns
    -------
    None
        f array is modified in-place.

    Notes
    -----
    Uses pyfftw FFT function for computation.
    The function performs: FFT(f) then f = f * dx / sqrt(2*pi)
    """
    f[:] = pyfftw.interfaces.numpy_fft.fft(f)
    f[:] = f[:] * dx / np.sqrt(twopi)


def GIFFT_1D(f, dq):
    """
    Gaussian IFFT for 1D array.

    Performs IFFT and scales by dq/sqrt(2*pi) * N for Gaussian normalization.

    Parameters
    ----------
    f : ndarray
        Input function (complex), modified in-place, 1D array
    dq : float
        Momentum step size (1/m)

    Returns
    -------
    None
        f array is modified in-place.

    Notes
    -----
    Uses pyfftw IFFT function for computation.
    The function performs: IFFT(f) then f = f * dq / sqrt(2*pi) * N
    """
    f[:] = pyfftw.interfaces.numpy_fft.ifft(f)
    N = len(f)
    f[:] = f[:] * dq / np.sqrt(twopi) * N


def GFFT_2D(f, dx, dy):
    """
    Gaussian FFT for 2D array.

    Performs FFT and scales by dx*dy/(2*pi) for Gaussian normalization.

    Parameters
    ----------
    f : ndarray
        Input function (complex), modified in-place, 2D array
    dx : float
        Spatial step size in x direction (m)
    dy : float
        Spatial step size in y direction (m)

    Returns
    -------
    None
        f array is modified in-place.

    Notes
    -----
    Uses pyfftw FFT function for computation.
    The function performs: FFT(f) then f = f * dx * dy / (2*pi)
    """
    f[:, :] = pyfftw.interfaces.numpy_fft.fft2(f)
    f[:, :] = f[:, :] * dx * dy / twopi


def GIFFT_2D(f, dqx, dqy):
    """
    Gaussian IFFT for 2D array.

    Performs IFFT and scales by dqx*dqy/(2*pi) * Nx * Ny for Gaussian normalization.

    Parameters
    ----------
    f : ndarray
        Input function (complex), modified in-place, 2D array
    dqx : float
        Momentum step size in x direction (1/m)
    dqy : float
        Momentum step size in y direction (1/m)

    Returns
    -------
    None
        f array is modified in-place.

    Notes
    -----
    Uses pyfftw IFFT function for computation.
    The function performs: IFFT(f) then f = f * dqx * dqy / (2*pi) * Nx * Ny
    """
    f[:, :] = pyfftw.interfaces.numpy_fft.ifft2(f)
    Nx = f.shape[0]
    Ny = f.shape[1]
    f[:, :] = f[:, :] * dqx * dqy / twopi * Nx * Ny


def test2Dfrom1D(x, y):
    """
    Test function for 2D array construction from 1D arrays.

    Creates 2D Gaussian functions from 1D arrays and tests FFT operations.

    Parameters
    ----------
    x : ndarray
        X coordinates (m), 1D array
    y : ndarray
        Y coordinates (m), 1D array

    Notes
    -----
    This is a test function. The function prints the sum of squared
    absolute values and stops execution.
    """
    xyonly = True
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    gx = np.zeros(len(x), dtype=np.complex128)
    gy = np.zeros(len(y), dtype=np.complex128)
    gxy = np.zeros((len(x), len(y)), dtype=np.complex128)

    gy[:] = np.exp(-((y + 8e-6) / 2e-6) ** 12)
    gx[:] = np.exp(-((x + 1e-6) / 4e-7) ** 2) * np.exp(-((x + 1e-6) / 8e-7) ** 12)

    if xyonly:
        for j in range(len(y)):
            for i in range(len(x)):
                gxy[i, j] = gx[i] * gy[j]
    else:
        gy[:] = pyfftw.interfaces.numpy_fft.fft(gy)
        gx[:] = pyfftw.interfaces.numpy_fft.fft(gx)
        for j in range(len(y)):
            for i in range(len(x)):
                gxy[i, j] = gx[i] * gy[j]
        gxy[:, :] = pyfftw.interfaces.numpy_fft.ifft2(gxy)

    print(np.sum(np.abs(gxy) ** 2) * dx * dy)
    # Note: stop statement removed in Python version


def testconv(y):
    """
    Test function for convolution operations.

    Tests convolution of two Gaussian functions using FFT methods.

    Parameters
    ----------
    y : ndarray
        Y coordinates (m), 1D array

    Notes
    -----
    This is a test function that uses nyquist_1D and convolve functions
    which need to be defined elsewhere. The function prints results and stops.
    """
    gate = np.zeros(len(y), dtype=np.complex128)
    func = np.zeros(len(y), dtype=np.complex128)
    conv = np.zeros(len(y), dtype=np.complex128)

    gate[:] = np.exp(-((y + 8e-6) / 2e-6) ** 12)
    func[:] = np.exp(-((y + 8e-6) / 2.33e-6) ** 2)

    func[:] = pyfftw.interfaces.numpy_fft.fft(func)
    gate[:] = pyfftw.interfaces.numpy_fft.fft(gate)
    # Note: nyquist_1D, convolve, Ny, dyy need to be defined
    # call nyquist_1D(func)
    # call nyquist_1D(gate)
    # func = cshift(func, Ny/2)
    # gate = cshift(gate, Ny/2)
    # conv = convolve(func, gate) * dyy
    # conv = cshift(conv, -Ny/2)
    # call nyquist_1D(conv)
    conv[:] = pyfftw.interfaces.numpy_fft.ifft(conv)

    for j in range(len(y)):
        print(y[j], np.abs(func[j]), np.imag(func[j]))
    # Note: stop statement removed in Python version


def ApplyABC(Field, abc):
    """
    Apply absorbing boundary conditions (ABC) to a field.

    Transforms field to real space, multiplies by ABC coefficients,
    and transforms back to Fourier space.

    Parameters
    ----------
    Field : ndarray
        Field array (complex), modified in-place, 2D array
    abc : ndarray
        Absorbing boundary condition coefficients (real), 2D array

    Returns
    -------
    None
        Field array is modified in-place.

    Notes
    -----
    The function performs: IFFT(Field), Field = Field * abc, FFT(Field)
    """
    Field[:, :] = pyfftw.interfaces.numpy_fft.ifft2(Field)
    Field[:, :] = Field[:, :] * abc
    Field[:, :] = pyfftw.interfaces.numpy_fft.fft2(Field)


@jit(nopython=True)
def _dEdx_jit(E, dx, iu):
    """JIT-compiled version of dEdx."""
    Nx = E.shape[0]
    Ny = E.shape[1]
    result = np.zeros_like(E, dtype=np.complex128)

    for j in range(Ny):
        for i in range(Nx):
            result[i, j] = (E[iu[i], j] - E[i, j]) / dx

    return result


def dEdx(E, dx):
    """
    Calculate derivative of E field in x direction.

    Computes forward difference derivative: dE/dx = (E(i+1,j) - E(i,j)) / dx

    Parameters
    ----------
    E : ndarray
        Electric field (complex), 2D array
    dx : float
        Spatial step size in x direction (m)

    Returns
    -------
    ndarray
        Derivative dE/dx (complex), 2D array with same shape as E

    Notes
    -----
    Returns zero array if E has only one row.
    Uses forward difference scheme with shifted indices.
    """
    if E.shape[0] == 1:
        return np.zeros_like(E)

    iu = np.arange(E.shape[0], dtype=np.int32)
    iu = np.roll(iu, -1)

    return _dEdx_jit(E, dx, iu)


@jit(nopython=True)
def _dEdy_jit(E, dy, ju):
    """JIT-compiled version of dEdy."""
    Nx = E.shape[0]
    Ny = E.shape[1]
    result = np.zeros_like(E, dtype=np.complex128)

    for j in range(Ny):
        for i in range(Nx):
            result[i, j] = (E[i, ju[j]] - E[i, j]) / dy

    return result


def dEdy(E, dy):
    """
    Calculate derivative of E field in y direction.

    Computes forward difference derivative: dE/dy = (E(i,j+1) - E(i,j)) / dy

    Parameters
    ----------
    E : ndarray
        Electric field (complex), 2D array
    dy : float
        Spatial step size in y direction (m)

    Returns
    -------
    ndarray
        Derivative dE/dy (complex), 2D array with same shape as E

    Notes
    -----
    Returns zero array if E has only one column.
    Uses forward difference scheme with shifted indices.
    """
    if E.shape[1] == 1:
        return np.zeros_like(E)

    ju = np.arange(E.shape[1], dtype=np.int32)
    ju = np.roll(ju, -1)

    return _dEdy_jit(E, dy, ju)


@jit(nopython=True)
def _dHdx_jit(H, dx, id_arr):
    """JIT-compiled version of dHdx."""
    Nx = H.shape[0]
    Ny = H.shape[1]
    result = np.zeros_like(H, dtype=np.complex128)

    for j in range(Ny):
        for i in range(Nx):
            result[i, j] = (H[i, j] - H[id_arr[i], j]) / dx

    return result


def dHdx(H, dx):
    """
    Calculate derivative of H field in x direction.

    Computes backward difference derivative: dH/dx = (H(i,j) - H(i-1,j)) / dx

    Parameters
    ----------
    H : ndarray
        Magnetic field (complex), 2D array
    dx : float
        Spatial step size in x direction (m)

    Returns
    -------
    ndarray
        Derivative dH/dx (complex), 2D array with same shape as H

    Notes
    -----
    Returns zero array if H has only one row.
    Uses backward difference scheme with shifted indices.
    """
    if H.shape[0] == 1:
        return np.zeros_like(H)

    id_arr = np.arange(H.shape[0], dtype=np.int32)
    id_arr = np.roll(id_arr, 1)

    return _dHdx_jit(H, dx, id_arr)


@jit(nopython=True)
def _dHdy_jit(H, dy, jd):
    """JIT-compiled version of dHdy."""
    Nx = H.shape[0]
    Ny = H.shape[1]
    result = np.zeros_like(H, dtype=np.complex128)

    for j in range(Ny):
        for i in range(Nx):
            result[i, j] = (H[i, j] - H[i, jd[j]]) / dy

    return result


def dHdy(H, dy):
    """
    Calculate derivative of H field in y direction.

    Computes backward difference derivative: dH/dy = (H(i,j) - H(i,j-1)) / dy

    Parameters
    ----------
    H : ndarray
        Magnetic field (complex), 2D array
    dy : float
        Spatial step size in y direction (m)

    Returns
    -------
    ndarray
        Derivative dH/dy (complex), 2D array with same shape as H

    Notes
    -----
    Returns zero array if H has only one column.
    Uses backward difference scheme with shifted indices.
    """
    if H.shape[1] == 1:
        return np.zeros_like(H)

    jd = np.arange(H.shape[1], dtype=np.int32)
    jd = np.roll(jd, 1)

    return _dHdy_jit(H, dy, jd)


def testing_conv(Ex, y, q):  # noqa: ARG001
    """
    Test function for convolution operations.

    Tests FFT/IFFT operations and normalization for convolution.

    Parameters
    ----------
    Ex : ndarray
        Field array (complex), modified in-place, 1D array
    y : ndarray
        Spatial coordinates (m), 1D array
    q : ndarray
        Momentum coordinates (1/m), 1D array

    Notes
    -----
    This is a test function that uses module variables yw, dy, dq, Ny
    and functions FFTc, IFFTc which need to be defined elsewhere.
    The function prints results for testing purposes.
    """
    # Note: yw, dy, dq, Ny, FFTc, IFFTc need to be defined
    # Ex = exp(-(y/yw)**2)
    # Iy = sum(abs(Ex)**2 * dy)
    # call FFTc(Ex)
    # Ex = Ex * dy
    # Iy = sum(abs(Ex)**2 * dq)
    # Ex = Ex * Ex
    # call IFFTc(Ex)
    # Ex = Ex / dy / Iy
    # for i in range(Ny):
    #     print(np.real(Ex[i]))
    # Iy = sum(real(Ex * conjg(Ex))) * dy
    # call IFFTc(Ex)
    # Ex = Ex * dy * Ny
    # Iq = sum(real(Ex * conjg(Ex))) * dq
    # call FFTc(Ex)
    # Ex = Ex / dy / Ny
    # Iy2 = sum(real(Ex * conjg(Ex))) * dy
    # print(Iy, Iq, Iy2)
    raise NotImplementedError("This is a test function requiring module variables")


def testing_fftc(Ex, y, q):  # noqa: ARG001
    """
    Test function for FFTc operations.

    Tests FFTc function with normalization.

    Parameters
    ----------
    Ex : ndarray
        Field array (complex), modified in-place, 1D array
    y : ndarray
        Spatial coordinates (m), 1D array
    q : ndarray
        Momentum coordinates (1/m), 1D array

    Notes
    -----
    This is a test function that uses module variables dy and function FFTc
    which need to be defined elsewhere. The function prints results.
    """
    # Note: dy, FFTc need to be defined
    # Ex = exp(-(y/10e-6)**2)
    # Iy = sum(abs(Ex)**2 * dy)
    # call FFTc(Ex)
    # Ex = Ex / len(Ex)
    # for i in range(len(Ex)):
    #     print(np.real(Ex[i]))
    raise NotImplementedError("This is a test function requiring module variables")


def print2file(x, y, filename):
    """
    Print arrays to file.

    Writes x, y arrays and index to a file.

    Parameters
    ----------
    x : ndarray
        First array (real), 1D array
    y : ndarray
        Second array (real), 1D array
    filename : str
        Output filename

    Returns
    -------
    None

    Notes
    -----
    The function writes: x(i), y(i), i for each element.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(len(x)):
            f.write(f"{x[i]} {y[i]} {i}\n")


@jit(nopython=True)
def _FT_jit(y, x, q):
    """JIT-compiled version of FT."""
    N = len(x)
    ytmp = np.zeros(N, dtype=np.complex128)
    ii_val = 1j

    for j in range(N):
        sum_val = 0.0 + 0.0j
        for k in range(len(y)):
            sum_val += y[k] * np.exp(ii_val * x[k] * q[j])
        ytmp[j] = sum_val / N

    return ytmp


def FT(y, x, q):
    """
    Fourier Transform using direct summation.

    Computes FT: Y(q) = (1/N) * sum(y(x) * exp(ii*x*q))

    Parameters
    ----------
    y : ndarray
        Input function (complex), modified in-place, 1D array
    x : ndarray
        Spatial coordinates (m), 1D array
    q : ndarray
        Momentum coordinates (1/m), 1D array

    Returns
    -------
    None
        y array is modified in-place with transformed values.

    Notes
    -----
    Uses direct summation method for Fourier transform.
    The result is normalized by N (length of arrays).
    """
    y[:] = _FT_jit(y, x, q)


@jit(nopython=True)
def _IFT_jit(y, x, q):
    """JIT-compiled version of IFT."""
    N = len(x)
    ytmp = np.zeros(N, dtype=np.complex128)
    ii_val = 1j

    for j in range(N):
        sum_val = 0.0 + 0.0j
        for k in range(len(q)):
            sum_val += y[k] * np.exp(-ii_val * x[j] * q[k])
        ytmp[j] = sum_val

    return ytmp


def IFT(y, x, q):
    """
    Inverse Fourier Transform using direct summation.

    Computes IFT: Y(x) = sum(y(q) * exp(-ii*x*q))

    Parameters
    ----------
    y : ndarray
        Input function in q-space (complex), modified in-place, 1D array
    x : ndarray
        Spatial coordinates (m), 1D array
    q : ndarray
        Momentum coordinates (1/m), 1D array

    Returns
    -------
    None
        y array is modified in-place with transformed values.

    Notes
    -----
    Uses direct summation method for inverse Fourier transform.
    No normalization factor is applied.
    """
    y[:] = _IFT_jit(y, x, q)


def Flip(x):
    """
    Flip a complex array.

    Reverses the order of elements in a complex array.

    Parameters
    ----------
    x : ndarray
        Input complex array, 1D array

    Returns
    -------
    ndarray
        Flipped array, 1D array

    Notes
    -----
    The function performs: Flip[i] = x[N-i] for i=0 to N-1
    """
    N = len(x)
    result = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        result[i] = x[N - 1 - i]
    return result


def GetArray0Index(x):
    """
    Get index of array element closest to zero.

    Finds the index where the array value is closest to zero
    by rounding and checking for zero.

    Parameters
    ----------
    x : ndarray
        Input array (real), 1D array

    Returns
    -------
    int
        Index (1-based in Fortran, 0-based in Python) where value is closest to zero

    Notes
    -----
    The function rounds array values and finds the first index
    where the rounded value is zero. Returns error message if not found.
    """
    dx = x[2] - x[1]
    xi = np.round((x + 0.1 * dx) / dx).astype(np.int32)

    for i in range(len(x)):
        if xi[i] == 0:
            return i

    print(f"Error in GetArray0Index in usefulsubs.f90: N0 = {len(x)}")
    raise ValueError("Zero index not found in array")


def GaussDelta(a, b):
    """
    Gaussian delta function.

    Computes normalized Gaussian function: (1/sqrt(pi)/b) * exp(-(a/b)^2)

    Parameters
    ----------
    a : float or ndarray
        Input value(s)
    b : float or ndarray
        Width parameter

    Returns
    -------
    float or ndarray
        Gaussian delta function value(s)

    Notes
    -----
    This is an elemental function that works with both scalars and arrays.
    """
    return 1.0 / np.sqrt(pi) / b * np.exp(-(a / b) ** 2)


def delta(x, dky=None):
    """
    Dirac delta function.

    Returns 1 if x/dky rounds to zero, otherwise 0.

    Parameters
    ----------
    x : float
        Input value
    dky : float, optional
        Step size for delta function. If None, uses a default small value.

    Returns
    -------
    int
        1 if nint(x/dky) == 0, else 0

    Notes
    -----
    Note: dky is a module variable in Fortran. Pass it as parameter if needed.
    """
    if dky is None:
        dky = 1e-6  # Default value if not provided
    if np.round(x / dky) == 0:
        return 1
    else:
        return 0


def kdel(x):
    """
    Kronecker delta function.

    Returns 1 if x == 0, otherwise 0.

    Parameters
    ----------
    x : int
        Input integer value

    Returns
    -------
    int
        1 if x == 0, else 0
    """
    if x == 0:
        return 1
    else:
        return 0


def delt(x):
    """
    Delta function variant.

    Computes: 1 - abs(x) / (abs(x) + 1e-100)

    Parameters
    ----------
    x : int
        Input integer value

    Returns
    -------
    float
        Delta function value
    """
    return 1.0 - abs(x) / (abs(x) + 1e-100)


def sgn(x):
    """
    Sign function.

    Returns the sign of a number: -1 if x < 0, +1 if x >= 0.

    Parameters
    ----------
    x : float
        Input value

    Returns
    -------
    int
        -1 if x < 0, +1 if x >= 0
    """
    if x < 0.0:
        return -1
    else:
        return +1


def sgn2(x):
    """
    Sign function for arrays.

    Applies sign function element-wise to an array.

    Parameters
    ----------
    x : ndarray
        Input array (real), 1D array

    Returns
    -------
    ndarray
        Array of signs (-1 or +1), 1D array of integers
    """
    result = np.zeros(len(x), dtype=np.int32)
    for i in range(len(x)):
        result[i] = sgn(x[i])
    return result


def TotalEnergy(n, E):
    """
    Calculate total energy.

    Computes total energy as real part of sum(n * E).

    Parameters
    ----------
    n : ndarray
        Occupation numbers (complex), 1D array
    E : ndarray
        Energy values (real), 1D array

    Returns
    -------
    float
        Total energy
    """
    return np.real(np.sum(n * E))


def AvgEnergy(n, E):
    """
    Calculate average energy.

    Computes average energy as real(sum(n*E) / (sum(n) + small)).

    Parameters
    ----------
    n : ndarray
        Occupation numbers (complex), 1D array
    E : ndarray
        Energy values (real), 1D array

    Returns
    -------
    float
        Average energy

    Notes
    -----
    Uses small constant to avoid division by zero.
    """
    return np.real(np.sum(n * E) / (np.sum(n) + small))


def Temperature(n, E):
    """
    Calculate temperature from energy distribution.

    Computes temperature as 2 * AvgEnergy(n, E) / kB.

    Parameters
    ----------
    n : ndarray
        Occupation numbers (complex), 1D array
    E : ndarray
        Energy values (real), 1D array

    Returns
    -------
    float
        Temperature (K)

    Notes
    -----
    Uses Boltzmann constant kB for temperature calculation.
    """
    return 2.0 * AvgEnergy(n, E) / kB


def Lrtz(a, b):
    """
    Lorentzian function.

    Computes normalized Lorentzian: (b/pi) / (a^2 + b^2)

    Parameters
    ----------
    a : float or ndarray
        Input value(s)
    b : float or ndarray
        Width parameter

    Returns
    -------
    float or ndarray
        Lorentzian function value(s)

    Notes
    -----
    This is an elemental function that works with both scalars and arrays.
    """
    return (b / pi) / (a ** 2 + b ** 2)


def theta(x):
    """
    Heaviside step function.

    Computes step function: (abs(x) + x) / 2 / (abs(x) + small)

    Parameters
    ----------
    x : float or ndarray
        Input value(s)

    Returns
    -------
    float or ndarray
        Step function value(s): 0 if x < 0, 1 if x > 0

    Notes
    -----
    This is an elemental function that works with both scalars and arrays.
    Uses small constant to avoid division by zero at x = 0.
    """
    return (np.abs(x) + x) / 2.0 / (np.abs(x) + small)


def softtheta(x, g):
    """
    Soft Heaviside step function.

    Computes smooth step function: 0.5 * (1.0 + 2.0/pi * atan(x/g))

    Parameters
    ----------
    x : float or ndarray
        Input value(s)
    g : float or ndarray
        Smoothness parameter

    Returns
    -------
    float or ndarray
        Soft step function value(s)

    Notes
    -----
    This is an elemental function that works with both scalars and arrays.
    Provides a smooth transition instead of a sharp step.
    """
    return 0.5 * (1.0 + 2.0 / pi * np.arctan(x / g))


def rad(degrees):
    """
    Convert degrees to radians.

    Parameters
    ----------
    degrees : float or ndarray
        Angle in degrees

    Returns
    -------
    float or ndarray
        Angle in radians

    Notes
    -----
    This is an elemental function that works with both scalars and arrays.
    """
    return degrees * pi / 180.0


@jit(nopython=True)
def _RotateField3D_jit(Ex0, Ey0, Ez0, R11, R12, R13, R21, R22, R23, R31, R32, R33):
    """JIT-compiled version of RotateField3D."""
    Nx = Ex0.shape[0]
    Ny = Ex0.shape[1]
    Nz = Ex0.shape[2]
    Ex = np.zeros_like(Ex0, dtype=np.complex128)
    Ey = np.zeros_like(Ey0, dtype=np.complex128)
    Ez = np.zeros_like(Ez0, dtype=np.complex128)

    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                Ex[i, j, k] = Ex0[i, j, k] * R11 + Ey0[i, j, k] * R12 + Ez0[i, j, k] * R13
                Ey[i, j, k] = Ex0[i, j, k] * R21 + Ey0[i, j, k] * R22 + Ez0[i, j, k] * R23
                Ez[i, j, k] = Ex0[i, j, k] * R31 + Ey0[i, j, k] * R32 + Ez0[i, j, k] * R33

    return Ex, Ey, Ez


def RotateField3D(theta, Ex, Ey, Ez):
    """
    Rotate 3D field by angle theta about y-axis.

    Performs right-handed rotation about y-axis using rotation matrix.

    Parameters
    ----------
    theta : float
        Rotation angle (radians)
    Ex : ndarray
        X component of field (complex), modified in-place, 3D array
    Ey : ndarray
        Y component of field (complex), modified in-place, 3D array
    Ez : ndarray
        Z component of field (complex), modified in-place, 3D array

    Returns
    -------
    None
        Ex, Ey, Ez arrays are modified in-place.

    Notes
    -----
    Rotation matrix for rotation about y-axis:
    R = [[cos(theta), 0, sin(theta)],
         [0,           1, 0          ],
         [-sin(theta), 0, cos(theta)]]
    """
    Ex0 = Ex.copy()
    Ey0 = Ey.copy()
    Ez0 = Ez.copy()

    R11 = np.cos(theta)
    R12 = 0.0
    R13 = np.sin(theta)
    R21 = 0.0
    R22 = 1.0
    R23 = 0.0
    R31 = -np.sin(theta)
    R32 = 0.0
    R33 = np.cos(theta)

    Ex[:, :, :], Ey[:, :, :], Ez[:, :, :] = _RotateField3D_jit(
        Ex0, Ey0, Ez0, R11, R12, R13, R21, R22, R23, R31, R32, R33
    )


@jit(nopython=True)
def _RotateField_jit(Ex0, Ey0, R11, R12, R21, R22):
    """JIT-compiled version of RotateField."""
    Nx = Ex0.shape[0]
    Ny = Ex0.shape[1]
    Ex = np.zeros_like(Ex0, dtype=np.complex128)
    Ey = np.zeros_like(Ey0, dtype=np.complex128)

    for j in range(Ny):
        for i in range(Nx):
            Ex[i, j] = Ex0[i, j] * R11 + Ey0[i, j] * R12
            Ey[i, j] = Ex0[i, j] * R21 + Ey0[i, j] * R22

    return Ex, Ey


def RotateField(theta, Ex, Ey):
    """
    Rotate 2D field by angle theta.

    Performs rotation in 2D plane using rotation matrix.

    Parameters
    ----------
    theta : float
        Rotation angle (radians)
    Ex : ndarray
        X component of field (complex), modified in-place, 2D array
    Ey : ndarray
        Y component of field (complex), modified in-place, 2D array

    Returns
    -------
    None
        Ex, Ey arrays are modified in-place.

    Notes
    -----
    Rotation matrix:
    R = [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]]
    """
    Ex0 = Ex.copy()
    Ey0 = Ey.copy()

    R11 = np.cos(theta)
    R12 = -np.sin(theta)
    R21 = np.sin(theta)
    R22 = np.cos(theta)

    Ex[:, :], Ey[:, :] = _RotateField_jit(Ex0, Ey0, R11, R12, R21, R22)


def ShiftField(Lx, Ly, dx, dy, Ex, Ey):
    """
    Shift field by specified distances.

    Shifts field arrays by integer number of grid points in x and y directions.

    Parameters
    ----------
    Lx : float
        Shift distance in x direction (m)
    Ly : float
        Shift distance in y direction (m)
    dx : float
        Grid spacing in x direction (m)
    dy : float
        Grid spacing in y direction (m)
    Ex : ndarray
        X component of field (complex), modified in-place, 2D array
    Ey : ndarray
        Y component of field (complex), modified in-place, 2D array

    Returns
    -------
    None
        Ex, Ey arrays are modified in-place.

    Notes
    -----
    Uses circular shift (cshift) to shift arrays by nx and ny grid points.
    """
    nx = int(np.round(Lx / dx))
    ny = int(np.round(Ly / dy))

    Ex[:, :] = np.roll(Ex, -nx, axis=0)
    Ex[:, :] = np.roll(Ex, -ny, axis=1)

    Ey[:, :] = np.roll(Ey, -nx, axis=0)
    Ey[:, :] = np.roll(Ey, -ny, axis=1)


def RotateShiftEField(theta, qx, qy, Ex, Ey, FFTC=None, IFFTC=None):
    """
    Rotate and shift E field in momentum space.

    Transforms field to momentum space, calculates center of mass,
    shifts by rotation angle, and transforms back.

    Parameters
    ----------
    theta : float
        Rotation angle (radians)
    qx : ndarray
        Momentum coordinates in x direction (1/m), 1D array
    qy : ndarray
        Momentum coordinates in y direction (1/m), 1D array
    Ex : ndarray
        X component of field (complex), modified in-place, 2D array
    Ey : ndarray
        Y component of field (complex), modified in-place, 2D array
    FFTC : callable, optional
        Forward FFT function. If None, uses pyfftw.
    IFFTC : callable, optional
        Inverse FFT function. If None, uses pyfftw.

    Returns
    -------
    None
        Ex, Ey arrays are modified in-place.

    Notes
    -----
    This function uses FFTC and IFFTC which need to be defined.
    Also uses module variables: dumb, qx0, qy0, q0x, q0y.
    These should be passed as parameters or defined as module variables.
    """
    if FFTC is None:
        def FFTC(f):
            f[:, :] = pyfftw.interfaces.numpy_fft.fft2(f)
    if IFFTC is None:
        def IFFTC(f):
            f[:, :] = pyfftw.interfaces.numpy_fft.ifft2(f)

    dqx = qx[1] - qx[0]
    dqy = qy[1] - qy[0]

    FFTC(Ex)
    FFTC(Ey)

    dumb = np.sum(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)

    qx0 = 0.0
    qy0 = 0.0
    for j in range(Ex.shape[1]):
        for i in range(Ex.shape[0]):
            qx0 = qx0 + qx[i] * (np.abs(Ex[i, j]) ** 2 + np.abs(Ey[i, j]) ** 2) / dumb
            # Note: Fortran bug at line 861 uses qy(i) instead of qy(j) - Python correctly uses qy[j]
            qy0 = qy0 + qy[j] * (np.abs(Ex[i, j]) ** 2 + np.abs(Ey[i, j]) ** 2) / dumb

    # Note: Fortran bug at lines 865-866 uses q0x/q0y (uninitialized) instead of qx0/qy0 (computed above)
    # Python correctly uses the computed qx0/qy0 values
    nqx = int(np.round(qx0 * np.sin(theta) / dqx))
    nqy = int(np.round(qy0 * np.cos(theta) / dqy))

    Ex[:, :] = np.roll(Ex, -nqx, axis=0)
    Ex[:, :] = np.roll(Ex, -nqy, axis=1)

    Ey[:, :] = np.roll(Ey, -nqx, axis=0)
    Ey[:, :] = np.roll(Ey, -nqy, axis=1)

    IFFTC(Ex)
    IFFTC(Ey)


# Constants for Bessel functions
_CBIK01_A = np.array([
    0.125, 7.03125e-2, 7.32421875e-2, 1.1215209960938e-1,
    2.2710800170898e-1, 5.7250142097473e-1, 1.7277275025845,
    6.0740420012735, 2.4380529699556e1, 1.1001714026925e2,
    5.5133589612202e2, 3.0380905109224e3
], dtype=np.float64)

_CBIK01_A1 = np.array([
    0.125, 0.2109375, 1.0986328125, 1.1775970458984e1,
    2.1461706161499e2, 5.9511522710323e3, 2.3347645606175e5,
    1.2312234987631e7, 8.401390346421e8, 7.2031420482627e10
], dtype=np.float64)

_CBIK01_B = np.array([
    -0.375, -1.171875e-1, -1.025390625e-1, -1.4419555664063e-1,
    -2.7757644653320e-1, -6.7659258842468e-1, -1.9935317337513,
    -6.8839142681099, -2.7248827311269e1, -1.2159789187654e2,
    -6.0384407670507e2, -3.3022722944809e3
], dtype=np.float64)


def cik01(z):
    """
    Compute modified Bessel functions I0(z), I1(z), K0(z) and K1(z) for complex argument.

    This procedure computes the modified Bessel functions I0(z), I1(z),
    K0(z), K1(z), and their derivatives for a complex argument.

    Parameters
    ----------
    z : complex
        The complex argument

    Returns
    -------
    tuple
        (cbi0, cdi0, cbi1, cdi1, cbk0, cdk0, cbk1, cdk1)
        Values of I0(z), I0'(z), I1(z), I1'(z), K0(z), K0'(z), K1(z), K1'(z)

    Notes
    -----
    This routine is copyrighted by Shanjie Zhang and Jianming Jin.
    Reference: Shanjie Zhang, Jianming Jin, Computation of Special Functions,
    Wiley, 1996, ISBN: 0-471-11963-6, LC: QA351.C45.
    """
    ci = 1j
    a0 = abs(z)
    z2 = z * z
    z1 = z

    if a0 == 0.0:
        cbi0 = 1.0 + 0.0j
        cbi1 = 0.0 + 0.0j
        cdi0 = 0.0 + 0.0j
        cdi1 = 0.5 + 0.0j
        cbk0 = 1e30 + 0.0j
        cbk1 = 1e30 + 0.0j
        cdk0 = -1e30 + 0.0j
        cdk1 = -1e30 + 0.0j
        return cbi0, cdi0, cbi1, cdi1, cbk0, cdk0, cbk1, cdk1

    if np.real(z) < 0.0:
        z1 = -z

    if a0 <= 18.0:
        cbi0 = 1.0 + 0.0j
        cr = 1.0 + 0.0j
        for k in range(1, 51):
            cr = 0.25 * cr * z2 / (k * k)
            cbi0 = cbi0 + cr
            if abs(cr / cbi0) < 1e-15:
                break

        cbi1 = 1.0 + 0.0j
        cr = 1.0 + 0.0j
        for k in range(1, 51):
            cr = 0.25 * cr * z2 / (k * (k + 1))
            cbi1 = cbi1 + cr
            if abs(cr / cbi1) < 1e-15:
                break

        cbi1 = 0.5 * z1 * cbi1

    else:
        if a0 < 35.0:
            k0 = 12
        elif a0 < 50.0:
            k0 = 9
        else:
            k0 = 7

        ca = np.exp(z1) / np.sqrt(2.0 * pi * z1)
        cbi0 = 1.0 + 0.0j
        zr = 1.0 / z1
        for k in range(1, k0 + 1):
            cbi0 = cbi0 + _CBIK01_A[k - 1] * (zr ** k)
        cbi0 = ca * cbi0
        cbi1 = 1.0 + 0.0j
        for k in range(1, k0 + 1):
            cbi1 = cbi1 + _CBIK01_B[k - 1] * (zr ** k)
        cbi1 = ca * cbi1

    if a0 <= 9.0:
        cs = 0.0 + 0.0j
        ct = -np.log(0.5 * z1) - 0.5772156649015329
        w0 = 0.0
        cr = 1.0 + 0.0j
        cw = 0.0 + 0.0j
        for k in range(1, 51):
            w0 = w0 + 1.0 / k
            cr = 0.25 * cr / (k * k) * z2
            cs = cs + cr * (w0 + ct)
            if abs((cs - cw) / cs) < 1e-15:
                break
            cw = cs

        cbk0 = ct + cs

    else:
        cb = 0.5 / z1
        zr2 = 1.0 / z2
        cbk0 = 1.0 + 0.0j
        for k in range(1, 11):
            cbk0 = cbk0 + _CBIK01_A1[k - 1] * (zr2 ** k)
        cbk0 = cb * cbk0 / cbi0

    cbk1 = (1.0 / z1 - cbi1 * cbk0) / cbi0

    if np.real(z) < 0.0:
        if np.imag(z) < 0.0:
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


def K03(x):
    """
    Modified Bessel function K0 for real argument.

    Computes K0(x) using the complex Bessel function routine.

    Parameters
    ----------
    x : float
        Real argument

    Returns
    -------
    float
        Value of K0(x), or 0.0 if x > 100

    Notes
    -----
    Returns 0.0 for large arguments (x > 100) to avoid overflow.
    """
    if x > 1e2:
        return 0.0

    z = x + 0.0j
    _, _, _, _, cbk0, _, _, _ = cik01(z)

    return np.real(cbk0)


def locator(x, x0):
    """
    Find index in sorted array where value should be inserted.

    Finds the index i such that x[i] <= x0 < x[i+1] (0-based indexing).

    Parameters
    ----------
    x : ndarray
        Sorted array (real), 1D array
    x0 : float
        Value to locate

    Returns
    -------
    int
        Index i such that x[i] <= x0 < x[i+1] (0-based)

    Notes
    -----
    Uses binary search to find the insertion point.
    Returns 0 if x0 < x[0], and len(x)-2 if x0 >= x[-1].
    """
    # Use numpy's searchsorted which finds the right insertion point
    i = np.searchsorted(x, x0, side='right') - 1
    # Ensure i is within valid range for interpolation
    i = max(0, min(i, len(x) - 2))
    return i


def WriteIT2D(V, file):
    """
    Write 2D array to file.

    Writes a 2D real array to a file in the 'dataQW' directory.

    Parameters
    ----------
    V : ndarray
        Array to write (real), 2D array
    file : str
        Base filename (without extension)

    Returns
    -------
    None

    Notes
    -----
    The file is written to 'dataQW/{file}.dat'.
    Each element is written on a separate line.
    """
    filename = f'dataQW/{file}.dat'
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                f.write(f"{V[i, j]}\n")


def ReadIT2D(V, file):
    """
    Read 2D array from file.

    Reads a 2D real array from a file in the 'dataQW' directory.

    Parameters
    ----------
    V : ndarray
        Array to read into (real), modified in-place, 2D array
    file : str
        Base filename (without extension)

    Returns
    -------
    None
        V array is modified in-place.

    Notes
    -----
    The file is read from 'dataQW/{file}.dat'.
    The array shape must match the file contents.
    """
    filename = f'dataQW/{file}.dat'
    with open(filename, 'r', encoding='utf-8') as f:
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                V[i, j] = float(f.readline().strip())


def WriteIT1D(V, file):
    """
    Write 1D array to file.

    Writes a 1D real array to a file in the 'dataQW' directory.

    Parameters
    ----------
    V : ndarray
        Array to write (real), 1D array
    file : str
        Base filename (without extension)

    Returns
    -------
    None

    Notes
    -----
    The file is written to 'dataQW/{file}.dat'.
    Each element is written on a separate line.
    """
    filename = f'dataQW/{file}.dat'
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(len(V)):
            f.write(f"{V[i]}\n")


def ReadIT1D(V, file):
    """
    Read 1D array from file.

    Reads a 1D real array from a file in the 'dataQW' directory.

    Parameters
    ----------
    V : ndarray
        Array to read into (real), modified in-place, 1D array
    file : str
        Base filename (without extension)

    Returns
    -------
    None
        V array is modified in-place.

    Notes
    -----
    The file is read from 'dataQW/{file}.dat'.
    The array length must match the file contents.
    """
    filename = f'dataQW/{file}.dat'
    with open(filename, 'r', encoding='utf-8') as f:
        for i in range(len(V)):
            V[i] = float(f.readline().strip())


def EAtX(f, x, x0):
    """
    Linear interpolation of 2D complex array at specified x position.

    Computes a linear interpolation of a complex 2D array at a specified
    x position. The interpolation is performed along the first dimension.

    Parameters
    ----------
    f : ndarray
        2D array to be interpolated (complex), 2D array
    x : ndarray
        1D position array corresponding to first dimension of f (real), 1D array
    x0 : float
        Position at which f is to be interpolated

    Returns
    -------
    ndarray
        Interpolated values (complex), 1D array of length f.shape[1]

    Notes
    -----
    Returns f[0, :] if x has only one element.
    Returns zero array if x0 is outside the valid range.
    """
    f0 = np.zeros(f.shape[1], dtype=np.complex128)

    if len(x) == 1:
        f0[:] = f[0, :]
        return f0

    i = locator(x, x0)

    if i < 0 or i >= len(x) - 1:
        return f0

    for j in range(f.shape[1]):
        f0[j] = (f[i, j] * (x[i + 1] - x0) + f[i + 1, j] * (x0 - x[i])) / (x[i + 1] - x[i])

    return f0


def EAtXYZ(f, x, y, z, x0, y0, z0):
    """
    Trilinear interpolation of 3D complex field at point (x0, y0, z0).

    Computes trilinear interpolation of a complex 3D field at a specified point.

    Parameters
    ----------
    f : ndarray
        3D field array (complex), 3D array
    x : ndarray
        1D array of grid coordinates along x axis (real), 1D array
    y : ndarray
        1D array of grid coordinates along y axis (real), 1D array
    z : ndarray
        1D array of grid coordinates along z axis (real), 1D array
    x0 : float
        Query coordinate in x direction
    y0 : float
        Query coordinate in y direction
    z0 : float
        Query coordinate in z direction

    Returns
    -------
    complex
        Interpolated field value at (x0, y0, z0)

    Notes
    -----
    Uses trilinear interpolation with weights computed for each axis.
    Handles degenerate cases where arrays have only one element.
    """
    # Handle each axis separately
    if len(x) == 1:
        i = 0
        alpha = 0.0
        wx = np.array([1.0, 0.0])
    else:
        i = locator(x, x0)
        alpha = (x0 - x[i]) / (x[i + 1] - x[i])
        wx = np.array([1.0 - alpha, alpha])

    if len(y) == 1:
        j = 0
        beta = 0.0
        wy = np.array([1.0, 0.0])
    else:
        j = locator(y, y0)
        beta = (y0 - y[j]) / (y[j + 1] - y[j])
        wy = np.array([1.0 - beta, beta])

    if len(z) == 1:
        k = 0
        gamma = 0.0
        wz = np.array([1.0, 0.0])
    else:
        k = locator(z, z0)
        gamma = (z0 - z[k]) / (z[k + 1] - z[k])
        wz = np.array([1.0 - gamma, gamma])

    f0 = 0.0 + 0.0j

    # Trilinear blend
    for r in range(2):
        for q in range(2):
            for p in range(2):
                f0 = f0 + wx[p] * wy[q] * wz[r] * f[i + p, j + q, k + r]

    return f0


def printIT(Dx, z, n, file):
    """
    Print field to file with index number.

    Writes a complex 1D field array to a file with index number in filename.

    Parameters
    ----------
    Dx : ndarray
        Field array (complex), 1D array
    z : ndarray
        Spatial coordinates (real), 1D array
    n : int
        Index number for filename
    file : str
        Base filename (without extension and index)

    Returns
    -------
    None

    Notes
    -----
    The file is written to 'dataQW/{file}{n:06d}.dat'.
    Each line contains: z(i), real(Dx(i)), imag(Dx(i))
    """
    filename = f'dataQW/{file}{n:06d}.dat'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(len(z)):
            f.write(f"{np.float32(z[i])} {np.float32(np.real(Dx[i]))} {np.float32(np.imag(Dx[i]))}\n")


def printITR(Dx, z, n, file):
    """
    Print real field to file with index number.

    Writes a real 1D field array to a file with index number in filename.

    Parameters
    ----------
    Dx : ndarray
        Field array (real), 1D array
    z : ndarray
        Spatial coordinates (real), 1D array
    n : int
        Index number for filename
    file : str
        Base filename (without extension and index)

    Returns
    -------
    None

    Notes
    -----
    The file is written to 'dataQW/{file}{n:06d}.dat'.
    Each line contains: z(i), Dx(i)
    """
    filename = f'dataQW/{file}{n:06d}.dat'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(len(z)):
            f.write(f"{np.float32(z[i])} {np.float32(Dx[i])}\n")


def printIT2D(Dx, z, n, file):  # noqa: ARG001
    """
    Print 2D complex field to file with index number.

    Writes a 2D complex field array to a file with index number in filename.

    Parameters
    ----------
    Dx : ndarray
        Field array (complex), 2D array
    z : ndarray
        Spatial coordinates (real), 1D array (not used in output)
    n : int
        Index number for filename
    file : str
        Base filename (without extension and index)

    Returns
    -------
    None

    Notes
    -----
    The file is written to 'dataQW/{file}{n:07d}.dat'.
    Each line contains: abs(Dx(i,j))
    The z parameter is kept for interface compatibility but not used.
    """
    filename = f'dataQW/{file}{n:07d}.dat'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for j in range(Dx.shape[1]):
            for i in range(Dx.shape[0]):
                f.write(f"{np.float32(np.abs(Dx[i, j]))}\n")


def gaussian(x, x0):
    """
    Gaussian function.

    Computes normalized Gaussian: exp(-x^2 / x0^2)

    Parameters
    ----------
    x : float or ndarray
        Input value(s)
    x0 : float or ndarray
        Width parameter

    Returns
    -------
    float or ndarray
        Gaussian function value(s)

    Notes
    -----
    This is an elemental function that works with both scalars and arrays.
    """
    return np.exp(-x ** 2 / x0 ** 2)


def convolve(x, h):
    """
    Convolution of two complex arrays.

    Computes convolution of signal array x with kernel/impulse array h.

    Parameters
    ----------
    x : ndarray
        Signal array (complex), 1D array
    h : ndarray
        Kernel/impulse array (complex), 1D array

    Returns
    -------
    ndarray
        Convolved array (complex), 1D array of same length as x

    Notes
    -----
    The convolution is computed using direct summation:
    - For i >= kernelsize: y(i) = sum(x(j)*h(k)) where j goes from i down to i-kernelsize+1
    - For i < kernelsize: y(i) = sum(x(j)*h(k)) where j goes from i down to 1
    """
    datasize = len(x)
    kernelsize = len(h)
    y = np.zeros(datasize, dtype=np.complex128)

    # Last part: i from kernelsize to datasize
    for i in range(kernelsize - 1, datasize):
        y[i] = 0.0 + 0.0j
        j = i
        for k in range(kernelsize):
            y[i] = y[i] + x[j] * h[k]
            j = j - 1

    # First part: i from 1 to kernelsize-1
    for i in range(kernelsize - 1):
        y[i] = 0.0 + 0.0j
        j = i
        k = 0
        while j >= 0:
            y[i] = y[i] + x[j] * h[k]
            j = j - 1
            k = k + 1

    return y


def FFTG(F):
    """
    FFT with Gaussian normalization.

    Performs FFT and applies Gaussian normalization: F = -FFT(F) / Nf

    Parameters
    ----------
    F : ndarray
        Input function (complex), modified in-place, 1D array

    Returns
    -------
    None
        F array is modified in-place.

    Notes
    -----
    Uses pyfftw for FFT computation.
    The function performs: FFTC(F) then F = -F / Nf
    """
    Nf = len(F)
    F[:] = pyfftw.interfaces.numpy_fft.fft(F)
    F[:] = -F[:] / (Nf * 1.0)


def iFFTG(F):
    """
    IFFT with Gaussian normalization.

    Performs IFFT and applies Gaussian normalization: F = -IFFT(F) * Nf

    Parameters
    ----------
    F : ndarray
        Input function (complex), modified in-place, 1D array

    Returns
    -------
    None
        F array is modified in-place.

    Notes
    -----
    Uses pyfftw for IFFT computation.
    The function performs: iFFTC(F) then F = -F * Nf
    """
    Nf = len(F)
    F[:] = pyfftw.interfaces.numpy_fft.ifft(F)
    F[:] = -F[:] * Nf


#######################################################
################ INTERFACE-STYLE WRAPPER FUNCTIONS #################
#######################################################
# These dispatch to 1D or 2D versions based on input array dimensions
#######################################################

def ReadIt(V, file):
    """
    Read array from file (interface-style wrapper).

    Automatically dispatches to ReadIT1D or ReadIT2D based on array dimensions.
    This provides a unified interface similar to Fortran's interface construct.

    Parameters
    ----------
    V : ndarray
        Array to read into (real), modified in-place, 1D or 2D array
    file : str
        Base filename (without extension)

    Returns
    -------
    None
        V array is modified in-place.

    Notes
    -----
    - 1D arrays: calls ReadIT1D
    - 2D arrays: calls ReadIT2D
    - Raises ValueError for unsupported array dimensions
    """
    if V.ndim == 1:
        ReadIT1D(V, file)
    elif V.ndim == 2:
        ReadIT2D(V, file)
    else:
        raise ValueError(f"ReadIt: Unsupported array dimension {V.ndim}. Only 1D and 2D arrays are supported.")


def WriteIt(V, file):
    """
    Write array to file (interface-style wrapper).

    Automatically dispatches to WriteIT1D or WriteIT2D based on array dimensions.
    This provides a unified interface similar to Fortran's interface construct.

    Parameters
    ----------
    V : ndarray
        Array to write (real), 1D or 2D array
    file : str
        Base filename (without extension)

    Returns
    -------
    None

    Notes
    -----
    - 1D arrays: calls WriteIT1D
    - 2D arrays: calls WriteIT2D
    - Raises ValueError for unsupported array dimensions
    """
    if V.ndim == 1:
        WriteIT1D(V, file)
    elif V.ndim == 2:
        WriteIT2D(V, file)
    else:
        raise ValueError(f"WriteIt: Unsupported array dimension {V.ndim}. Only 1D and 2D arrays are supported.")


def dfdy(f, qy):
    """
    Calculate derivative in y direction (interface-style wrapper).

    Automatically dispatches to dfdy1D or dfdy2D based on array dimensions.
    This provides a unified interface similar to Fortran's interface construct.

    Parameters
    ----------
    f : ndarray
        Input function (complex), 1D or 2D array
    qy : ndarray
        Momentum coordinates in y direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dy (complex), same shape as input array

    Notes
    -----
    - 1D arrays: calls dfdy1D
    - 2D arrays: calls dfdy2D
    - Raises ValueError for unsupported array dimensions
    """
    if f.ndim == 1:
        return dfdy1D(f, qy)
    elif f.ndim == 2:
        return dfdy2D(f, qy)
    else:
        raise ValueError(f"dfdy: Unsupported array dimension {f.ndim}. Only 1D and 2D arrays are supported.")


def dfdx(f, qx):
    """
    Calculate derivative in x direction (interface-style wrapper).

    Automatically dispatches to dfdx1D or dfdx2D based on array dimensions.
    This provides a unified interface similar to Fortran's interface construct.

    Parameters
    ----------
    f : ndarray
        Input function (complex), 1D or 2D array
    qx : ndarray
        Momentum coordinates in x direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dx (complex), same shape as input array

    Notes
    -----
    - 1D arrays: calls dfdx1D
    - 2D arrays: calls dfdx2D
    - Raises ValueError for unsupported array dimensions
    """
    if f.ndim == 1:
        return dfdx1D(f, qx)
    elif f.ndim == 2:
        return dfdx2D(f, qx)
    else:
        raise ValueError(f"dfdx: Unsupported array dimension {f.ndim}. Only 1D and 2D arrays are supported.")


def dfdy_q(f, qy):
    """
    Calculate derivative in y direction in q-space (interface-style wrapper).

    Automatically dispatches to dfdy1D_q or dfdy2D_q based on array dimensions.
    This provides a unified interface similar to Fortran's interface construct.

    Parameters
    ----------
    f : ndarray
        Input function in q-space (complex), 1D or 2D array
    qy : ndarray
        Momentum coordinates in y direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dy in q-space (complex), same shape as input array

    Notes
    -----
    - 1D arrays: calls dfdy1D_q
    - 2D arrays: calls dfdy2D_q
    - Raises ValueError for unsupported array dimensions
    """
    if f.ndim == 1:
        return dfdy1D_q(f, qy)
    elif f.ndim == 2:
        return dfdy2D_q(f, qy)
    else:
        raise ValueError(f"dfdy_q: Unsupported array dimension {f.ndim}. Only 1D and 2D arrays are supported.")


def dfdx_q(f, qx):
    """
    Calculate derivative in x direction in q-space (interface-style wrapper).

    Automatically dispatches to dfdx1D_q or dfdx2D_q based on array dimensions.
    This provides a unified interface similar to Fortran's interface construct.

    Parameters
    ----------
    f : ndarray
        Input function in q-space (complex), 1D or 2D array
    qx : ndarray
        Momentum coordinates in x direction (1/m), 1D array

    Returns
    -------
    ndarray
        Derivative df/dx in q-space (complex), same shape as input array

    Notes
    -----
    - 1D arrays: calls dfdx1D_q
    - 2D arrays: calls dfdx2D_q
    - Raises ValueError for unsupported array dimensions
    """
    if f.ndim == 1:
        return dfdx1D_q(f, qx)
    elif f.ndim == 2:
        return dfdx2D_q(f, qx)
    else:
        raise ValueError(f"dfdx_q: Unsupported array dimension {f.ndim}. Only 1D and 2D arrays are supported.")


def GFFT(f, dx, dy=None):
    """
    Gaussian FFT (interface-style wrapper).

    Automatically dispatches to GFFT_1D or GFFT_2D based on array dimensions.
    This provides a unified interface similar to Fortran's interface construct.

    Parameters
    ----------
    f : ndarray
        Input function (complex), modified in-place, 1D or 2D array
    dx : float
        Spatial step size in x direction (m)
    dy : float, optional
        Spatial step size in y direction (m). Required for 2D arrays.

    Returns
    -------
    None
        f array is modified in-place.

    Notes
    -----
    - 1D arrays: calls GFFT_1D(f, dx)
    - 2D arrays: calls GFFT_2D(f, dx, dy)
    - Raises ValueError for unsupported array dimensions or missing dy for 2D
    """
    if f.ndim == 1:
        GFFT_1D(f, dx)
    elif f.ndim == 2:
        if dy is None:
            raise ValueError("GFFT: dy parameter is required for 2D arrays")
        GFFT_2D(f, dx, dy)
    else:
        raise ValueError(f"GFFT: Unsupported array dimension {f.ndim}. Only 1D and 2D arrays are supported.")


def GIFFT(f, dq, dqy=None):
    """
    Gaussian IFFT (interface-style wrapper).

    Automatically dispatches to GIFFT_1D or GIFFT_2D based on array dimensions.
    This provides a unified interface similar to Fortran's interface construct.

    Parameters
    ----------
    f : ndarray
        Input function (complex), modified in-place, 1D or 2D array
    dq : float
        Momentum step size in x direction (1/m)
    dqy : float, optional
        Momentum step size in y direction (1/m). Required for 2D arrays.

    Returns
    -------
    None
        f array is modified in-place.

    Notes
    -----
    - 1D arrays: calls GIFFT_1D(f, dq)
    - 2D arrays: calls GIFFT_2D(f, dq, dqy)
    - Raises ValueError for unsupported array dimensions or missing dqy for 2D
    """
    if f.ndim == 1:
        GIFFT_1D(f, dq)
    elif f.ndim == 2:
        if dqy is None:
            raise ValueError("GIFFT: dqy parameter is required for 2D arrays")
        GIFFT_2D(f, dq, dqy)
    else:
        raise ValueError(f"GIFFT: Unsupported array dimension {f.ndim}. Only 1D and 2D arrays are supported.")


def fflip(f):
    """
    Flip array (interface-style wrapper).

    Automatically dispatches to fflip_dp or fflip_dpc based on array dtype.
    This provides a unified interface similar to Fortran's interface construct.

    Parameters
    ----------
    f : ndarray
        Input array (real or complex), 1D array

    Returns
    -------
    ndarray
        Flipped array, same dtype as input

    Notes
    -----
    - Real arrays: calls fflip_dp
    - Complex arrays: calls fflip_dpc
    - Raises ValueError for unsupported array dimensions
    """
    if f.ndim != 1:
        raise ValueError(f"fflip: Unsupported array dimension {f.ndim}. Only 1D arrays are supported.")

    if np.iscomplexobj(f):
        return fflip_dpc(f)
    else:
        return fflip_dp(f)

