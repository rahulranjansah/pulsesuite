"""
Quantum wire optics calculations for quantum wire simulations.

This module converts Maxwell electric fields from propagation space
to quantum wire (QW) electric fields, and vice versa. It also calculates
QW polarization and charge densities.

Author: Rahul R. Sah
"""

import numpy as np
import os
from numba import jit, prange
try:
    from numba import cuda
    _HAS_CUDA = cuda.is_available()
except (ImportError, RuntimeError):
    _HAS_CUDA = False
    cuda = None

from .usefulsubs import FFTG, iFFTG, printIT, printIT2D, GetArray0Index
from ..libpulsesuite.spliner import rescale_1D_dp, rescale_1D_dpc, rescale_1D
from scipy.constants import e as e0, c as c0_SI, hbar as hbar_SI, epsilon_0 as eps0_SI

# Physical constants
twopi = 2.0 * np.pi
c0 = c0_SI
hbar = hbar_SI
eps0 = eps0_SI
ii = 1j  # Imaginary unit

# Module-level state variables (matching Fortran module variables)
_small = 1e-100
_epsr = 9.1
_QWWindow = None
_Expikr = None
_Expikrc = None
_dcv0 = None
_Xcv0 = None
_Ycv0 = None
_Zcv0 = None
_Xvc0 = None
_Yvc0 = None
_Zvc0 = None
_firsttime = True
_Vol = 0.0


# def rescale_1D(x_old, f_old, x_new, f_new):
#     """
#     Rescale/interpolate 1D array from old grid to new grid.

#     Wrapper function that dispatches to appropriate spliner function
#     based on array type (real or complex).

#     Parameters
#     ----------
#     x_old : ndarray
#         Old grid points, 1D array
#     f_old : ndarray
#         Function values on old grid, 1D array (real or complex)
#     x_new : ndarray
#         New grid points, 1D array
#     f_new : ndarray
#         Output array (modified in-place), 1D array

#     Returns
#     -------
#     None
#         f_new is modified in-place

#     Notes
#     -----
#     Uses spline interpolation from spliner module.
#     """
#     if np.iscomplexobj(f_old) or np.iscomplexobj(f_new):
#         rescale_1D_dpc(x_old, f_old, x_new, f_new)
#     else:
#         rescale_1D_dp(x_old, f_old, x_new, f_new)


def Prop2QW(RR, Exx, Eyy, Ezz, Vrr, Edc, R, Ex, Ey, Ez, Vr, t, xxx):
    """
    Convert Maxwell electric fields from propagation space to QW space.

    Converts the Maxwell electric fields (Exx, Eyy, Ezz, Vrr) in
    the propagation space (RR) into the QW electric fields (Ex, Ey, Ez, Vr)
    in the QW electric field space (R). After interpolating from the RR-space
    to the R-space, the FFT from the R- to Qr-space is taken.

    Parameters
    ----------
    RR : ndarray
        Maxwell RR spatial array, 1D array
    Exx : ndarray
        Maxwell X electric field, 1D array, complex
    Eyy : ndarray
        Maxwell Y electric field, 1D array, complex
    Ezz : ndarray
        Maxwell Z electric field, 1D array, complex
    Vrr : ndarray
        Maxwell free charge potential, 1D array, complex
    Edc : float
        QW spatial array (modified in-place, currently unused)
    R : ndarray
        QW spatial array, 1D array
    Ex : ndarray
        QW X electric field (modified in-place), 1D array, complex
    Ey : ndarray
        QW Y electric field (modified in-place), 1D array, complex
    Ez : ndarray
        QW Z electric field (modified in-place), 1D array, complex
    Vr : ndarray
        QW free charge potential (modified in-place), 1D array, complex
    t : float
        Current time (s)
    xxx : int
        Time index

    Returns
    -------
    None
        Ex, Ey, Ez, Vr, Edc are modified in-place

    Notes
    -----
    Uses module-level variable _QWWindow for windowing.
    Fields are made real before FFT.
    """
    global _QWWindow

    Ny = len(Ey)

    # Initialize quantum wire fields
    Ex[:] = 0.0
    Ey[:] = 0.0
    Ez[:] = 0.0
    Vr[:] = 0.0

    # Take the fields from the propagation (Exx & Eyy)
    # and produce the fields in the QW only (Ex & Ey)
    rescale_1D(RR, Exx, R, Ex)
    rescale_1D(RR, Eyy, R, Ey)
    rescale_1D(RR, Ezz, R, Ez)
    rescale_1D(RR, Vrr, R, Vr)

    # Calculate the fields only between -L/2 to L/2
    if _QWWindow is not None:
        Ex[:] = Ex[:] * _QWWindow
        Ey[:] = Ey[:] * _QWWindow
        Ez[:] = Ez[:] * _QWWindow
        Vr[:] = Vr[:] * _QWWindow

    # Make sure the QW y-space fields are real
    Ex[:] = np.real(Ex)
    Ey[:] = np.real(Ey)
    Ez[:] = np.real(Ez)
    Vr[:] = np.real(Vr)

    # Perform a Centered Fast Fourier Transform on the fields... centered Gulley-style!
    FFTG(Ex)
    FFTG(Ey)
    FFTG(Ez)
    FFTG(Vr)


def QW2Prop(r, Qr, Ex, Ey, Ez, Vr, Px, Py, Pz, re, rh, RR, Pxx, Pyy, Pzz, RhoE, RhoH, w, xxx, WriteFields, Plasmonics):
    """
    Convert QW fields and polarizations back to propagation space.

    Converts QW electric fields, polarizations, and charge densities
    from QW space (r, Qr) back to propagation space (RR).

    Parameters
    ----------
    r : ndarray
        QW Y-spaces, 1D array
    Qr : ndarray
        QW momentum space, 1D array
    Ex : ndarray
        QW X electric field (modified in-place), 1D array, complex
    Ey : ndarray
        QW Y electric field (modified in-place), 1D array, complex
    Ez : ndarray
        QW Z electric field (modified in-place), 1D array, complex
    Vr : ndarray
        QW free charge potential (modified in-place), 1D array, complex
    Px : ndarray
        QW X polarization (modified in-place), 1D array, complex
    Py : ndarray
        QW Y polarization (modified in-place), 1D array, complex
    Pz : ndarray
        QW Z polarization (modified in-place), 1D array, complex
    re : ndarray
        QW electron charge density (modified in-place), 1D array, complex
    rh : ndarray
        QW hole charge density (modified in-place), 1D array, complex
    RR : ndarray
        Propagation Y-spaces, 1D array
    Pxx : ndarray
        Propagation X polarization (modified in-place), 1D array, complex
    Pyy : ndarray
        Propagation Y polarization (modified in-place), 1D array, complex
    Pzz : ndarray
        Propagation Z polarization (modified in-place), 1D array, complex
    RhoE : ndarray
        Propagation electron charge density (modified in-place), 1D array, complex
    RhoH : ndarray
        Propagation hole charge density (modified in-place), 1D array, complex
    w : int
        Wire index
    xxx : int
        Time index
    WriteFields : bool
        Record fields?
    Plasmonics : bool
        Calculate charge densities?

    Returns
    -------
    None
        All input arrays are modified in-place

    Notes
    -----
    Uses module-level variable _small for numerical stability.
    Charge densities are normalized if Plasmonics is True.
    """
    global _small

    # Constants needed for integration below
    dr = r[1] - r[0]
    dRR = RR[1] - RR[0]

    # Inverse Fourier Transform the fields back into the QW y-space
    iFFTG(Ex)
    iFFTG(Ey)
    iFFTG(Ez)
    iFFTG(Vr)
    iFFTG(Px)
    iFFTG(Py)
    iFFTG(Pz)
    iFFTG(re)
    iFFTG(rh)

    # Make sure the charge densities are well behaved (real, balanced, etc.)
    if Plasmonics:
        re[:] = np.abs(re)
        rh[:] = np.abs(rh)
        total = (np.sum(re) * dr + np.sum(rh) * dr + _small) / 2.0
        re[:] = re * total / (np.sum(re) * dr + _small)
        rh[:] = rh * total / (np.sum(rh) * dr + _small)

    # Record the QW Field arrays in the real Space
    if WriteFields:
        WriteQWFields(r, Ex, Ey, Ez, Vr, Px, Py, Pz, re, rh, 'r', w, xxx)

    # Rescale the fields to the propagation YY-space scale
    rescale_1D(r, Px, RR, Pxx)
    rescale_1D(r, Py, RR, Pyy)
    rescale_1D(r, Pz, RR, Pzz)
    rescale_1D(r, re, RR, RhoE)
    rescale_1D(r, rh, RR, RhoH)

    if Plasmonics:
        # Make sure the charge densities are well behaved (real, balanced, etc.)
        RhoE[:] = np.abs(RhoE)
        RhoH[:] = np.abs(RhoH)
        RhoE[:] = RhoE * total / (np.sum(RhoE) * dRR + _small)
        RhoH[:] = RhoH * total / (np.sum(RhoH) * dRR + _small)

def QWPolarization3(y, ky, p, ehint, area, L, Px, Py, Pz, xxx, w):
    """
    Calculate QW polarization in 3D.

    Computes the QW polarization components (Px, Py, Pz) from the
    density matrix p using the dipole matrix elements.

    Parameters
    ----------
    y : ndarray
        Spatial coordinate array, 1D array
    ky : ndarray
        Momentum coordinate array, 1D array
    p : ndarray
        Density matrix, shape (Nk, Nk), complex
    ehint : float
        Electron-hole space integral
    area : float
        Area of wire (m^2)
    L : float
        Length of wire (m)
    Px : ndarray
        QW X polarization (modified in-place), 1D array, complex
    Py : ndarray
        QW Y polarization (modified in-place), 1D array, complex
    Pz : ndarray
        QW Z polarization (modified in-place), 1D array, complex
    xxx : int
        Time index
    w : int
        Wire index

    Returns
    -------
    None
        Px, Py, Pz are modified in-place

    Notes
    -----
    Uses module-level variables _Xvc0, _Yvc0, _Zvc0, _Expikr, _Expikrc, _QWWindow.
    Uses JIT compilation for performance-critical loops.
    """
    global _Xvc0, _Yvc0, _Zvc0, _Expikr, _Expikrc, _QWWindow

    Px[:] = 0.0
    Py[:] = 0.0
    Pz[:] = 0.0

    if _Xvc0 is None or _Yvc0 is None or _Zvc0 is None:
        return
    if _Expikr is None or _Expikrc is None:
        return
    if _QWWindow is None:
        return

    # Try CUDA first, then JIT, then fallback
    if _HAS_CUDA:
        try:
            _QWPolarization3_cuda(y, ky, p, _Xvc0, _Yvc0, _Zvc0, _Expikr, _Expikrc, _QWWindow, yw(w), w, Px, Py, Pz)
        except Exception:
            # Fallback to JIT
            try:
                _QWPolarization3_jit(y, ky, p, _Xvc0, _Yvc0, _Zvc0, _Expikr, _Expikrc, _QWWindow, yw(w), w, Px, Py, Pz)
            except Exception:
                # Fallback to pure Python
                _QWPolarization3_fallback(y, ky, p, _Xvc0, _Yvc0, _Zvc0, _Expikr, _Expikrc, _QWWindow, yw(w), w, Px, Py, Pz)
    else:
        # No CUDA, use JIT
        try:
            _QWPolarization3_jit(y, ky, p, _Xvc0, _Yvc0, _Zvc0, _Expikr, _Expikrc, _QWWindow, yw(w), w, Px, Py, Pz)
        except Exception:
            # Fallback to pure Python
            _QWPolarization3_fallback(y, ky, p, _Xvc0, _Yvc0, _Zvc0, _Expikr, _Expikrc, _QWWindow, yw(w), w, Px, Py, Pz)

    Px[:] = Px * 2 * ehint / area / L
    Py[:] = Py * 2 * ehint / area / L
    Pz[:] = Pz * 2 * ehint / area / L

    FFTG(Px)
    FFTG(Py)
    FFTG(Pz)


@jit(nopython=True, parallel=True)
def _QWPolarization3_jit(y, ky, p, Xvc0, Yvc0, Zvc0, Expikr, Expikrc, QWWindow, yw_val, w, Px, Py, Pz):
    """JIT-compiled version of QWPolarization3 inner loops."""
    Nr = len(y)
    Nk = len(ky)

    for r in prange(Nr):
        for ke in range(Nk):
            for kh in range(Nk):
                # Px component
                prod = p[kh, ke] * (+Xvc0[kh, ke]) * Expikr[ke, r] * Expikrc[kh, r]
                Px[r] = Px[r] + prod.real * QWWindow[r]

    for r in prange(Nr):
        for ke in range(Nk):
            for kh in range(Nk):
                # Py component
                prod = p[kh, ke] * (+Yvc0[kh, ke]) * Expikr[ke, r] * Expikrc[kh, r]
                Py[r] = Py[r] + prod.real * QWWindow[r] * yw_val

    for r in prange(Nr):
        for ke in range(Nk):
            for kh in range(Nk):
                # Pz component
                prod = p[kh, ke] * (+Zvc0[kh, ke]) * Expikr[ke, r] * Expikrc[kh, r]
                Pz[r] = Pz[r] + prod.real * QWWindow[r] * ((-1)**w)


# CUDA implementation for QWPolarization3
if _HAS_CUDA:
    @cuda.jit
    def _QWPolarization3_cuda_kernel_px(p_real, p_imag, Xvc0_real, Xvc0_imag,
                                        Expikr_real, Expikr_imag, Expikrc_real, Expikrc_imag,
                                        QWWindow, Px, Nr, Nk):
        """CUDA kernel for Px component of QWPolarization3."""
        # Flatten 3D index: idx = r * Nk * Nk + ke * Nk + kh
        idx = cuda.grid(1)
        total_threads = Nr * Nk * Nk

        if idx < total_threads:
            r = idx // (Nk * Nk)
            ke_kh = idx % (Nk * Nk)
            ke = ke_kh // Nk
            kh = ke_kh % Nk

            if r < Nr and ke < Nk and kh < Nk:
                # Complex multiplication: p * Xvc0 * Expikr * Expikrc
                # p[kh, ke] * Xvc0[kh, ke] * Expikr[ke, r] * Expikrc[kh, r]
                p_re = p_real[kh, ke]
                p_im = p_imag[kh, ke]

                x_re = Xvc0_real[kh, ke]
                x_im = Xvc0_imag[kh, ke]

                er_re = Expikr_real[ke, r]
                er_im = Expikr_imag[ke, r]

                ec_re = Expikrc_real[kh, r]
                ec_im = Expikrc_imag[kh, r]

                # Multiply: (p * Xvc0) * (Expikr * Expikrc)
                # First: p * Xvc0
                px_re = p_re * x_re - p_im * x_im
                px_im = p_re * x_im + p_im * x_re

                # Second: Expikr * Expikrc
                ee_re = er_re * ec_re - er_im * ec_im
                ee_im = er_re * ec_im + er_im * ec_re

                # Final: (p * Xvc0) * (Expikr * Expikrc)
                prod_re = px_re * ee_re - px_im * ee_im
                prod_im = px_re * ee_im + px_im * ee_re

                # Add real part * QWWindow
                val = prod_re * QWWindow[r]
                cuda.atomic.add(Px, r, val)

    @cuda.jit
    def _QWPolarization3_cuda_kernel_py(p_real, p_imag, Yvc0_real, Yvc0_imag,
                                        Expikr_real, Expikr_imag, Expikrc_real, Expikrc_imag,
                                        QWWindow, yw_val, Py, Nr, Nk):
        """CUDA kernel for Py component of QWPolarization3."""
        idx = cuda.grid(1)
        total_threads = Nr * Nk * Nk

        if idx < total_threads:
            r = idx // (Nk * Nk)
            ke_kh = idx % (Nk * Nk)
            ke = ke_kh // Nk
            kh = ke_kh % Nk

            if r < Nr and ke < Nk and kh < Nk:
                p_re = p_real[kh, ke]
                p_im = p_imag[kh, ke]
                y_re = Yvc0_real[kh, ke]
                y_im = Yvc0_imag[kh, ke]
                er_re = Expikr_real[ke, r]
                er_im = Expikr_imag[ke, r]
                ec_re = Expikrc_real[kh, r]
                ec_im = Expikrc_imag[kh, r]

                py_re = p_re * y_re - p_im * y_im
                py_im = p_re * y_im + p_im * y_re
                ee_re = er_re * ec_re - er_im * ec_im
                ee_im = er_re * ec_im + er_im * ec_re
                prod_re = py_re * ee_re - py_im * ee_im

                val = prod_re * QWWindow[r] * yw_val
                cuda.atomic.add(Py, r, val)

    @cuda.jit
    def _QWPolarization3_cuda_kernel_pz(p_real, p_imag, Zvc0_real, Zvc0_imag,
                                        Expikr_real, Expikr_imag, Expikrc_real, Expikrc_imag,
                                        QWWindow, w, Pz, Nr, Nk):
        """CUDA kernel for Pz component of QWPolarization3."""
        idx = cuda.grid(1)
        total_threads = Nr * Nk * Nk

        if idx < total_threads:
            r = idx // (Nk * Nk)
            ke_kh = idx % (Nk * Nk)
            ke = ke_kh // Nk
            kh = ke_kh % Nk

            if r < Nr and ke < Nk and kh < Nk:
                p_re = p_real[kh, ke]
                p_im = p_imag[kh, ke]
                z_re = Zvc0_real[kh, ke]
                z_im = Zvc0_imag[kh, ke]
                er_re = Expikr_real[ke, r]
                er_im = Expikr_imag[ke, r]
                ec_re = Expikrc_real[kh, r]
                ec_im = Expikrc_imag[kh, r]

                pz_re = p_re * z_re - p_im * z_im
                pz_im = p_re * z_im + p_im * z_re
                ee_re = er_re * ec_re - er_im * ec_im
                ee_im = er_re * ec_im + er_im * ec_re
                prod_re = pz_re * ee_re - pz_im * ee_im

                w_sign = 1.0 if (w % 2 == 0) else -1.0
                val = prod_re * QWWindow[r] * w_sign
                cuda.atomic.add(Pz, r, val)

    def _QWPolarization3_cuda(y, ky, p, Xvc0, Yvc0, Zvc0, Expikr, Expikrc,
                              QWWindow, yw_val, w, Px, Py, Pz):
        """CUDA wrapper for QWPolarization3."""
        Nr = len(y)
        Nk = len(ky)

        # Split complex arrays into real and imaginary parts
        p_real = np.ascontiguousarray(p.real, dtype=np.float64)
        p_imag = np.ascontiguousarray(p.imag, dtype=np.float64)
        Xvc0_real = np.ascontiguousarray(Xvc0.real, dtype=np.float64)
        Xvc0_imag = np.ascontiguousarray(Xvc0.imag, dtype=np.float64)
        Yvc0_real = np.ascontiguousarray(Yvc0.real, dtype=np.float64)
        Yvc0_imag = np.ascontiguousarray(Yvc0.imag, dtype=np.float64)
        Zvc0_real = np.ascontiguousarray(Zvc0.real, dtype=np.float64)
        Zvc0_imag = np.ascontiguousarray(Zvc0.imag, dtype=np.float64)
        Expikr_real = np.ascontiguousarray(Expikr.real, dtype=np.float64)
        Expikr_imag = np.ascontiguousarray(Expikr.imag, dtype=np.float64)
        Expikrc_real = np.ascontiguousarray(Expikrc.real, dtype=np.float64)
        Expikrc_imag = np.ascontiguousarray(Expikrc.imag, dtype=np.float64)
        QWWindow_arr = np.ascontiguousarray(QWWindow, dtype=np.float64)

        # Allocate device arrays
        d_p_real = cuda.to_device(p_real)
        d_p_imag = cuda.to_device(p_imag)
        d_Xvc0_real = cuda.to_device(Xvc0_real)
        d_Xvc0_imag = cuda.to_device(Xvc0_imag)
        d_Yvc0_real = cuda.to_device(Yvc0_real)
        d_Yvc0_imag = cuda.to_device(Yvc0_imag)
        d_Zvc0_real = cuda.to_device(Zvc0_real)
        d_Zvc0_imag = cuda.to_device(Zvc0_imag)
        d_Expikr_real = cuda.to_device(Expikr_real)
        d_Expikr_imag = cuda.to_device(Expikr_imag)
        d_Expikrc_real = cuda.to_device(Expikrc_real)
        d_Expikrc_imag = cuda.to_device(Expikrc_imag)
        d_QWWindow = cuda.to_device(QWWindow_arr)

        # Allocate output arrays on device
        d_Px = cuda.device_array((Nr,), dtype=np.float64)
        d_Py = cuda.device_array((Nr,), dtype=np.float64)
        d_Pz = cuda.device_array((Nr,), dtype=np.float64)

        # Initialize to zero
        d_Px[:] = 0.0
        d_Py[:] = 0.0
        d_Pz[:] = 0.0

        # Configure kernel launch
        threads_per_block = 256
        total_elements = Nr * Nk * Nk
        blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block

        # Launch kernels
        _QWPolarization3_cuda_kernel_px[blocks_per_grid, threads_per_block](
            d_p_real, d_p_imag, d_Xvc0_real, d_Xvc0_imag,
            d_Expikr_real, d_Expikr_imag, d_Expikrc_real, d_Expikrc_imag,
            d_QWWindow, d_Px, Nr, Nk
        )

        _QWPolarization3_cuda_kernel_py[blocks_per_grid, threads_per_block](
            d_p_real, d_p_imag, d_Yvc0_real, d_Yvc0_imag,
            d_Expikr_real, d_Expikr_imag, d_Expikrc_real, d_Expikrc_imag,
            d_QWWindow, yw_val, d_Py, Nr, Nk
        )

        _QWPolarization3_cuda_kernel_pz[blocks_per_grid, threads_per_block](
            d_p_real, d_p_imag, d_Zvc0_real, d_Zvc0_imag,
            d_Expikr_real, d_Expikr_imag, d_Expikrc_real, d_Expikrc_imag,
            d_QWWindow, w, d_Pz, Nr, Nk
        )

        # Copy results back (convert to complex)
        Px[:] = d_Px.copy_to_host()
        Py[:] = d_Py.copy_to_host()
        Pz[:] = d_Pz.copy_to_host()
else:
    def _QWPolarization3_cuda(*args, **kwargs):
        """Dummy CUDA function when CUDA is not available."""
        raise RuntimeError("CUDA not available")


def _QWPolarization3_fallback(y, ky, p, Xvc0, Yvc0, Zvc0, Expikr, Expikrc, QWWindow, yw_val, w, Px, Py, Pz):
    """Fallback pure Python version of QWPolarization3 inner loops."""
    Nr = len(y)
    Nk = len(ky)

    for r in range(Nr):
        for ke in range(Nk):
            for kh in range(Nk):
                # Px component
                Px[r] = Px[r] + np.real(p[kh, ke] * (+Xvc0[kh, ke]) * Expikr[ke, r] * Expikrc[kh, r]) * QWWindow[r]

    for r in range(Nr):
        for ke in range(Nk):
            for kh in range(Nk):
                # Py component
                Py[r] = Py[r] + np.real(p[kh, ke] * (+Yvc0[kh, ke]) * Expikr[ke, r] * Expikrc[kh, r]) * QWWindow[r] * yw_val

    for r in range(Nr):
        for ke in range(Nk):
            for kh in range(Nk):
                # Pz component
                Pz[r] = Pz[r] + np.real(p[kh, ke] * (+Zvc0[kh, ke]) * Expikr[ke, r] * Expikrc[kh, r]) * QWWindow[r] * ((-1)**w)


def yw(w):
    """
    Calculate wire-dependent sign factor.

    Computes a sign factor based on the wire index.

    Parameters
    ----------
    w : int
        Wire index

    Returns
    -------
    int
        Sign factor: (-1)**floor((w-1)/2)

    Notes
    -----
    Used for alternating sign patterns in multi-wire systems.
    """
    return (-1)**int(np.floor((w - 1) / 2.0))


def printITReal2(Dx, z, n, file):
    """
    Print real part of complex field to file.

    Writes the real part of a complex field array to a file.
    On the first call, writes both the field and coordinate values.
    On subsequent calls, writes only the field values.

    Parameters
    ----------
    Dx : ndarray
        Complex field array, 1D array
    z : ndarray
        Coordinate array, 1D array
    n : int
        Time index
    file : str
        Base filename (without extension)

    Returns
    -------
    None

    Notes
    -----
    Uses module-level variable _firsttime to track first write.
    Files are written to 'dataQW/' directory.
    """
    global _firsttime
    import os

    filename = f'dataQW/{file}{n:06d}.dat'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        if _firsttime:
            for i in range(len(z)):
                f.write(f'{np.float32(np.real(Dx[i]))} {np.float32(z[i])}\n')
            _firsttime = False
        else:
            for i in range(len(z)):
                f.write(f'{np.float32(np.real(Dx[i]))}\n')


def WriteSBESolns(ky, ne, nh, C, D, P, Ee, Eh, w, xxx):
    """
    Write SBE (Semiconductor Bloch Equations) solutions to files.

    Writes quantum wire electron/hole occupation numbers, coherence matrices,
    and energies to output files.

    Parameters
    ----------
    ky : ndarray
        QW momentum array, 1D array
    ne : ndarray
        QW electron occupation numbers, 1D array, complex
    nh : ndarray
        QW hole occupation numbers, 1D array, complex
    C : ndarray
        QW electron/electron coherence matrix, 2D array, complex
    D : ndarray
        QW hole/hole coherence matrix, 2D array, complex
    P : ndarray
        QW electron/hole coherence matrix, 2D array, complex
    Ee : ndarray
        QW electron energies, 1D array, complex
    Eh : ndarray
        QW hole energies, 1D array, complex
    w : int
        Wire index
    xxx : int
        Time index

    Returns
    -------
    None

    Notes
    -----
    Uses printIT and printIT2D from usefulsubs to write data files.
    """
    wire = f'{w:02d}'

    printIT2D(C, ky, xxx, f'Wire/C/C.{wire}.k.kp.')
    printIT2D(D, ky, xxx, f'Wire/D/D.{wire}.k.kp.')
    printIT2D(P, ky, xxx, f'Wire/P/P.{wire}.k.kp.')

    printIT(ne, ky, xxx, f'Wire/ne/ne.{wire}.k.')
    printIT(nh, ky, xxx, f'Wire/nh/nh.{wire}.k.')

    printIT(Ee, ky, xxx, f'Wire/Ee/Ee.{wire}.k.')
    printIT(Eh, ky, xxx, f'Wire/Eh/Eh.{wire}.k.')


def WritePLSpectrum(hw, PLS, w, xxx):
    """
    Write photoluminescence spectrum to file.

    Writes the photoluminescence spectrum as a function of photon energy
    to an output file.

    Parameters
    ----------
    hw : ndarray
        Photon energy array (J), 1D array
    PLS : ndarray
        Photoluminescence spectrum, 1D array
    w : int
        Wire index
    xxx : int
        Time index

    Returns
    -------
    None

    Notes
    -----
    Converts photon energy from Joules to eV (divides by e0).
    Uses printIT from usefulsubs to write data file.
    """
    PLS0 = np.zeros(len(hw), dtype=complex)
    PLS0[:] = PLS

    wire = f'{w:02d}'

    printIT(PLS0, hw / e0, xxx, f'Wire/PL/pl.{wire}.hw.')


def WriteQWFields(QY, Ex, Ey, Ez, Vr, Px, Py, Pz, Re, Rh, sp, w, xxx):
    """
    Write QW field arrays to files.

    Writes QW electric fields, polarizations, and charge densities
    to output files.

    Parameters
    ----------
    QY : ndarray
        QY momentum/y-space array, 1D array
    Ex : ndarray
        QW X electric field, 1D array, complex
    Ey : ndarray
        QW Y electric field, 1D array, complex
    Ez : ndarray
        QW Z electric field, 1D array, complex
    Vr : ndarray
        QW free charge potential, 1D array, complex
    Px : ndarray
        QW X polarization, 1D array, complex
    Py : ndarray
        QW Y polarization, 1D array, complex
    Pz : ndarray
        QW Z polarization, 1D array, complex
    Re : ndarray
        QW electron charge density, 1D array, complex
    Rh : ndarray
        QW hole charge density, 1D array, complex
    sp : str
        Domain label ('k' or 'r') for file name
    w : int
        Wire index
    xxx : int
        Time index

    Returns
    -------
    None

    Notes
    -----
    Uses printIT from usefulsubs to write data files.
    """
    wire = f'{w:02d}'

    printIT(Ex, QY, xxx, f'Wire/Ex/Ex.{wire}.{sp}.')
    printIT(Ey, QY, xxx, f'Wire/Ey/Ey.{wire}.{sp}.')
    printIT(Ez, QY, xxx, f'Wire/Ez/Ez.{wire}.{sp}.')
    printIT(Vr, QY, xxx, f'Wire/Vr/Vr.{wire}.{sp}.')
    printIT(Px, QY, xxx, f'Wire/Px/Px.{wire}.{sp}.')
    printIT(Py, QY, xxx, f'Wire/Py/Py.{wire}.{sp}.')
    printIT(Pz, QY, xxx, f'Wire/Pz/Pz.{wire}.{sp}.')
    printIT(Re, QY, xxx, f'Wire/Re/Re.{wire}.{sp}.')
    printIT(Rh, QY, xxx, f'Wire/Rh/Rh.{wire}.{sp}.')
    printIT(Rh - Re, QY, xxx, f'Wire/Rho/Rho.{wire}.{sp}.')


def WritePropFields(y, Ex, Ey, Ez, Vr, Px, Py, Pz, Re, Rh, sp, w, xxx):
    """
    Write propagation field arrays to files.

    Writes propagation electric fields, polarizations, and charge densities
    to output files.

    Parameters
    ----------
    y : ndarray
        Spatial coordinate array, 1D array
    Ex : ndarray
        Propagation X electric field, 1D array, complex
    Ey : ndarray
        Propagation Y electric field, 1D array, complex
    Ez : ndarray
        Propagation Z electric field, 1D array, complex
    Vr : ndarray
        Propagation free charge potential, 1D array, complex
    Px : ndarray
        Propagation X polarization, 1D array, complex
    Py : ndarray
        Propagation Y polarization, 1D array, complex
    Pz : ndarray
        Propagation Z polarization, 1D array, complex
    Re : ndarray
        Propagation electron charge density, 1D array, complex
    Rh : ndarray
        Propagation hole charge density, 1D array, complex
    sp : str
        Space label for file name
    w : int
        Wire index
    xxx : int
        Time index

    Returns
    -------
    None

    Notes
    -----
    Uses printITReal2 to write real parts of complex fields.
    Uses printIT for charge density difference.
    """
    wire = f'{w:02d}'

    printITReal2(Ex, y, xxx, f'Prop/Ex/Ex.{wire}.{sp}.')
    printITReal2(Ey, y, xxx, f'Prop/Ey/Ey.{wire}.{sp}.')
    printITReal2(Ez, y, xxx, f'Prop/Ez/Ez.{wire}.{sp}.')
    printITReal2(Vr, y, xxx, f'Prop/Vr/Vr.{wire}.{sp}.')
    printITReal2(Px, y, xxx, f'Prop/Px/Px.{wire}.{sp}.')
    printITReal2(Py, y, xxx, f'Prop/Py/Py.{wire}.{sp}.')
    printITReal2(Pz, y, xxx, f'Prop/Pz/Pz.{wire}.{sp}.')
    printITReal2(Re, y, xxx, f'Prop/Re/Re.{wire}.{sp}.')
    printITReal2(Rh, y, xxx, f'Prop/Rh/Rh.{wire}.{sp}.')

    printIT(Rh - Re, y, xxx, f'Prop/Rho/Rho.{wire}.{sp}.')

####################################################################################################
### Subroutines and functions called from within this module ######################################
####################################################################################################


def QWRho5(Qr, kr, R, L, kkp, p, CC, DD, ne, nh, re, rh, xxx, jjj):
    """
    Calculate quantum wire charge densities.

    Computes electron and hole charge densities in real space from
    coherence matrices CC and DD using Fourier transforms.

    Parameters
    ----------
    Qr : ndarray
        QW momentum array, 1D array
    kr : ndarray
        QW momentum array, 1D array
    R : ndarray
        QW spatial coordinate array, 1D array
    L : float
        Length of wire (m)
    kkp : ndarray
        Momentum difference index matrix, 2D array, integer
    p : ndarray
        Density matrix (unused, kept for interface compatibility), 2D array, complex
    CC : ndarray
        Electron/electron coherence matrix, 2D array, complex
    DD : ndarray
        Hole/hole coherence matrix, 2D array, complex
    ne : ndarray
        Electron occupation numbers, 1D array, complex
    nh : ndarray
        Hole occupation numbers, 1D array, complex
    re : ndarray
        Electron charge density (modified in-place), 1D array, complex
    rh : ndarray
        Hole charge density (modified in-place), 1D array, complex
    xxx : int
        Time index
    jjj : int
        Additional index (unused, kept for interface compatibility)

    Returns
    -------
    None
        re and rh are modified in-place

    Notes
    -----
    Uses module-level variables _Expikr, _Expikrc, _QWWindow, _small.
    Normalizes charge densities to match total electron/hole numbers.
    Removes boundary effects by subtracting average of first/last 10 points.
    """
    global _Expikr, _Expikrc, _QWWindow, _small

    Nr = len(R)
    Nk = len(kr)
    NQ0 = GetArray0Index(Qr)
    Nk0 = GetArray0Index(kr)
    dkr = kr[1] - kr[0]
    dr = R[1] - R[0]

    NeTotal = np.sum(np.abs(ne)) + 1e-50
    NhTotal = np.sum(np.abs(nh)) + 1e-50

    re[:] = 0.0
    rh[:] = 0.0

    if _Expikr is None or _Expikrc is None or _QWWindow is None:
        return

    # Try CUDA first, then JIT, then fallback
    if _HAS_CUDA:
        try:
            _QWRho5_cuda(Nr, Nk, CC, DD, _Expikr, _Expikrc, _QWWindow, L, re, rh)
        except Exception:
            # Fallback to JIT
            try:
                _QWRho5_jit(Nr, Nk, CC, DD, _Expikr, _Expikrc, _QWWindow, L, re, rh)
            except Exception:
                # Fallback to pure Python
                _QWRho5_fallback(Nr, Nk, CC, DD, _Expikr, _Expikrc, _QWWindow, L, re, rh)
    else:
        # No CUDA, use JIT
        try:
            _QWRho5_jit(Nr, Nk, CC, DD, _Expikr, _Expikrc, _QWWindow, L, re, rh)
        except Exception:
            # Fallback to pure Python
            _QWRho5_fallback(Nr, Nk, CC, DD, _Expikr, _Expikrc, _QWWindow, L, re, rh)

    # Remove boundary effects
    re[:] = re - np.real(np.sum(re[0:10]) + np.sum(re[Nr-10:Nr])) / 20.0
    rh[:] = rh - np.real(np.sum(rh[0:10]) + np.sum(rh[Nr-10:Nr])) / 20.0

    # Normalize to match total numbers
    re[:] = re * NeTotal / (np.sum(np.abs(re)) * dr + _small)
    rh[:] = rh * NhTotal / (np.sum(np.abs(rh)) * dr + _small)

    FFTG(re)
    FFTG(rh)


@jit(nopython=True, parallel=True)
def _QWRho5_jit(Nr, Nk, CC, DD, Expikr, Expikrc, QWWindow, L, re, rh):
    """JIT-compiled version of QWRho5 inner loops."""
    for ri in prange(Nr):
        for k2 in range(Nk):
            for k1 in range(Nk):
                re[ri] = re[ri] + CC[k1, k2] * Expikrc[k1, ri] * Expikr[k2, ri] * QWWindow[ri] / (2.0 * L)
                # Complex conjugate: conj(z) = z.real - 1j*z.imag
                dd_val = DD[k1, k2]
                dd_conj = dd_val.real - 1j * dd_val.imag
                rh[ri] = rh[ri] + dd_conj * Expikrc[k1, ri] * Expikr[k2, ri] * QWWindow[ri] / (2.0 * L)


# CUDA implementation for QWRho5
if _HAS_CUDA:
    @cuda.jit
    def _QWRho5_cuda_kernel(CC_real, CC_imag, DD_real, DD_imag,
                            Expikr_real, Expikr_imag, Expikrc_real, Expikrc_imag,
                            QWWindow, L, re_real, re_imag, rh_real, rh_imag, Nr, Nk):
        """CUDA kernel for QWRho5 calculation."""
        # Flatten 3D index: idx = ri * Nk * Nk + k2 * Nk + k1
        idx = cuda.grid(1)
        total_threads = Nr * Nk * Nk

        if idx < total_threads:
            ri = idx // (Nk * Nk)
            k2_k1 = idx % (Nk * Nk)
            k2 = k2_k1 // Nk
            k1 = k2_k1 % Nk

            if ri < Nr and k2 < Nk and k1 < Nk:
                # Calculate re: CC[k1, k2] * Expikrc[k1, ri] * Expikr[k2, ri]
                cc_re = CC_real[k1, k2]
                cc_im = CC_imag[k1, k2]
                ec_re = Expikrc_real[k1, ri]
                ec_im = Expikrc_imag[k1, ri]
                er_re = Expikr_real[k2, ri]
                er_im = Expikr_imag[k2, ri]

                # Multiply: CC * Expikrc * Expikr
                # First: CC * Expikrc
                ccec_re = cc_re * ec_re - cc_im * ec_im
                ccec_im = cc_re * ec_im + cc_im * ec_re
                # Then: (CC * Expikrc) * Expikr
                prod_re = ccec_re * er_re - ccec_im * er_im
                prod_im = ccec_re * er_im + ccec_im * er_re

                # Add to re with factor QWWindow / (2*L)
                factor = QWWindow[ri] / (2.0 * L)
                cuda.atomic.add(re_real, ri, prod_re * factor)
                cuda.atomic.add(re_imag, ri, prod_im * factor)

                # Calculate rh: conj(DD[k1, k2]) * Expikrc[k1, ri] * Expikr[k2, ri]
                # conj(DD) = DD.real - 1j*DD.imag
                dd_re = DD_real[k1, k2]
                dd_im = DD_imag[k1, k2]
                dd_conj_re = dd_re  # Real part stays same
                dd_conj_im = -dd_im  # Imaginary part negated

                # Multiply: conj(DD) * Expikrc * Expikr
                # First: conj(DD) * Expikrc
                ddec_re = dd_conj_re * ec_re - dd_conj_im * ec_im
                ddec_im = dd_conj_re * ec_im + dd_conj_im * ec_re
                # Then: (conj(DD) * Expikrc) * Expikr
                prod_rh_re = ddec_re * er_re - ddec_im * er_im
                prod_rh_im = ddec_re * er_im + ddec_im * er_re

                # Add to rh with factor QWWindow / (2*L)
                cuda.atomic.add(rh_real, ri, prod_rh_re * factor)
                cuda.atomic.add(rh_imag, ri, prod_rh_im * factor)

    def _QWRho5_cuda(Nr, Nk, CC, DD, Expikr, Expikrc, QWWindow, L, re, rh):
        """CUDA wrapper for QWRho5."""
        # Split complex arrays into real and imaginary parts
        CC_real = np.ascontiguousarray(CC.real, dtype=np.float64)
        CC_imag = np.ascontiguousarray(CC.imag, dtype=np.float64)
        DD_real = np.ascontiguousarray(DD.real, dtype=np.float64)
        DD_imag = np.ascontiguousarray(DD.imag, dtype=np.float64)
        Expikr_real = np.ascontiguousarray(Expikr.real, dtype=np.float64)
        Expikr_imag = np.ascontiguousarray(Expikr.imag, dtype=np.float64)
        Expikrc_real = np.ascontiguousarray(Expikrc.real, dtype=np.float64)
        Expikrc_imag = np.ascontiguousarray(Expikrc.imag, dtype=np.float64)
        QWWindow_arr = np.ascontiguousarray(QWWindow, dtype=np.float64)

        # Allocate device arrays
        d_CC_real = cuda.to_device(CC_real)
        d_CC_imag = cuda.to_device(CC_imag)
        d_DD_real = cuda.to_device(DD_real)
        d_DD_imag = cuda.to_device(DD_imag)
        d_Expikr_real = cuda.to_device(Expikr_real)
        d_Expikr_imag = cuda.to_device(Expikr_imag)
        d_Expikrc_real = cuda.to_device(Expikrc_real)
        d_Expikrc_imag = cuda.to_device(Expikrc_imag)
        d_QWWindow = cuda.to_device(QWWindow_arr)

        # Allocate output arrays on device
        d_re_real = cuda.device_array((Nr,), dtype=np.float64)
        d_re_imag = cuda.device_array((Nr,), dtype=np.float64)
        d_rh_real = cuda.device_array((Nr,), dtype=np.float64)
        d_rh_imag = cuda.device_array((Nr,), dtype=np.float64)

        # Initialize to zero
        d_re_real[:] = 0.0
        d_re_imag[:] = 0.0
        d_rh_real[:] = 0.0
        d_rh_imag[:] = 0.0

        # Configure kernel launch
        threads_per_block = 256
        total_elements = Nr * Nk * Nk
        blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block

        # Launch kernel
        _QWRho5_cuda_kernel[blocks_per_grid, threads_per_block](
            d_CC_real, d_CC_imag, d_DD_real, d_DD_imag,
            d_Expikr_real, d_Expikr_imag, d_Expikrc_real, d_Expikrc_imag,
            d_QWWindow, L, d_re_real, d_re_imag, d_rh_real, d_rh_imag, Nr, Nk
        )

        # Copy results back and combine into complex arrays
        re_real_host = d_re_real.copy_to_host()
        re_imag_host = d_re_imag.copy_to_host()
        rh_real_host = d_rh_real.copy_to_host()
        rh_imag_host = d_rh_imag.copy_to_host()

        re[:] = re_real_host + 1j * re_imag_host
        rh[:] = rh_real_host + 1j * rh_imag_host
else:
    def _QWRho5_cuda(*args, **kwargs):
        """Dummy CUDA function when CUDA is not available."""
        raise RuntimeError("CUDA not available")


def _QWRho5_fallback(Nr, Nk, CC, DD, Expikr, Expikrc, QWWindow, L, re, rh):
    """Fallback pure Python version of QWRho5 inner loops."""
    for ri in range(Nr):
        for k2 in range(Nk):
            for k1 in range(Nk):
                re[ri] = re[ri] + CC[k1, k2] * Expikrc[k1, ri] * Expikr[k2, ri] * QWWindow[ri] / (2.0 * L)
                rh[ri] = rh[ri] + np.conj(DD[k1, k2]) * Expikrc[k1, ri] * Expikr[k2, ri] * QWWindow[ri] / (2.0 * L)


def printIT3D(Dx, z, n, file):
    """
    Print 3D complex array to file.

    Writes a 3D complex array to a file, storing both real and imaginary parts.

    Parameters
    ----------
    Dx : ndarray
        3D complex array, shape (N1, N2, N3)
    z : ndarray
        Coordinate array (unused, kept for interface compatibility), 1D array
    n : int
        Time index
    file : str
        Base filename (without extension)

    Returns
    -------
    None

    Notes
    -----
    Files are written to 'dataQW/' directory.
    Each line contains real and imaginary parts of one array element.
    """
    filename = f'dataQW/{file}{n:06d}.dat'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    Dx_ordered = Dx.transpose(2, 1, 0)

    # Flatten and save as binary or text
    data = np.column_stack([
        np.real(Dx.flatten()),
        np.imag(Dx.flatten())
    ])
    np.savetxt(filename, data, fmt='%.6e')

def printITReal(Dx, z, n, file):
    """
    Print real part of complex field to file.

    Writes the coordinate and real part of a complex field array to a file.

    Parameters
    ----------
    Dx : ndarray
        Complex field array, 1D array
    z : ndarray
        Coordinate array, 1D array
    n : int
        Time index
    file : str
        Base filename (without extension)

    Returns
    -------
    None

    Notes
    -----
    Files are written to 'dataQW/' directory.
    Each line contains coordinate and real part of field.
    """
    filename = f'dataQW/{file}{n:06d}.dat'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(len(z)):
            f.write(f'{np.float32(z[i])} {np.float32(np.real(Dx[i]))}\n')


def QWChi1(lam, dky, Ee, Eh, area, geh, dcv):
    """
    Calculate quantum wire linear susceptibility.

    Computes the frequency-dependent linear susceptibility of the quantum wire.

    Parameters
    ----------
    lam : float
        Wavelength (m)
    dky : float
        Momentum step (1/m)
    Ee : ndarray
        Electron energies, 1D array
    Eh : ndarray
        Hole energies, 1D array
    area : float
        Area of wire (m^2)
    geh : float
        Electron-hole dephasing rate (1/s)
    dcv : complex
        Carrier dipole moment (C m)

    Returns
    -------
    complex
        Linear susceptibility chi^(1)

    Notes
    -----
    Uses module-level constants twopi, c0, eps0, hbar, ii.
    Formula: chi^(1) = 4 * dcv^2 / (eps0 * area) * dky / (2*pi) *
             sum((Ee+Eh) / ((Ee+Eh - i*hbar*geh - hbar*w) * (Ee+Eh + i*hbar*geh + hbar*w)))
    """
    # Optical angular frequency
    ww = twopi * c0 / lam

    # Linear susceptibility chi^(1) of the quantum wire
    denominator1 = Ee + Eh - ii * hbar * geh - hbar * ww
    denominator2 = Ee + Eh + ii * hbar * geh + hbar * ww
    QWChi1 = 4 * dcv**2 / eps0 / area * dky / twopi * np.sum((Ee + Eh) / (denominator1 * denominator2))

    return QWChi1


def CalcQWWindow(YY, L):
    """
    Calculate quantum wire window function.

    Computes a window function that smoothly goes to zero outside
    the wire length L/2, using an exponential decay.

    Parameters
    ----------
    YY : ndarray
        Spatial coordinate array, 1D array
    L : float
        Length of wire (m)

    Returns
    -------
    None

    Notes
    -----
    Modifies module-level variable _QWWindow.
    Window function: exp(-(YY/(L/2))^150)
    Values outside |YY| > L/2 are set to 0 before applying exponential.
    Uses printIT to write envelope to file.
    """
    global _QWWindow

    Ny = len(YY)

    _QWWindow = np.ones(Ny)

    for k in range(Ny):
        if abs(YY[k]) > L / 2.0:
            _QWWindow[k] = 0.0

    _QWWindow = np.exp(-(YY / (L / 2.0))**150)

    printIT(_QWWindow * (ii)**0.0, YY, 0, 'Envl.y.')


def InitializeQWOptics(RR, L, dcv, kr, Qr, Ee, Eh, ehint, area, gap):
    """
    Initialize quantum wire optics module.

    Sets up all necessary arrays and parameters for quantum wire optics
    calculations, including window functions, phase factors, and dipole
    matrix elements.

    Parameters
    ----------
    RR : ndarray
        Spatial coordinate array, 1D array
    L : float
        Length of wire (m)
    dcv : complex
        Carrier dipole moment (C m)
    kr : ndarray
        Momentum coordinate array, 1D array
    Qr : ndarray
        Momentum coordinate array (unused, kept for interface compatibility), 1D array
    Ee : ndarray
        Electron energies, 1D array
    Eh : ndarray
        Hole energies, 1D array
    ehint : float
        Electron-hole space integral
    area : float
        Area of wire (m^2)
    gap : float
        Band gap (J) (unused, kept for interface compatibility)

    Returns
    -------
    None

    Notes
    -----
    Modifies module-level variables _QWWindow, _Expikr, _Expikrc, _dcv0, _Vol,
    _Xcv0, _Ycv0, _Zcv0, _Xvc0, _Yvc0, _Zvc0.
    Calculates dipole matrix elements Xcv0, Ycv0, Zcv0 and their conjugates.
    Uses module-level constants twopi.
    """
    global _QWWindow, _Expikr, _Expikrc, _dcv0, _Vol
    global _Xcv0, _Ycv0, _Zcv0, _Xvc0, _Yvc0, _Zvc0

    Nr = len(RR)
    Nk = len(kr)

    CalcQWWindow(RR, L)
    CalcExpikr(RR, kr)

    _dcv0 = dcv
    R0 = np.sqrt(area / twopi)
    _Vol = L * area / ehint

    _Xcv0 = np.zeros((Nk, Nk), dtype=complex)
    _Ycv0 = np.zeros((Nk, Nk), dtype=complex)
    _Zcv0 = np.zeros((Nk, Nk), dtype=complex)
    _Xvc0 = np.zeros((Nk, Nk), dtype=complex)
    _Yvc0 = np.zeros((Nk, Nk), dtype=complex)
    _Zvc0 = np.zeros((Nk, Nk), dtype=complex)

    for kh in range(Nk):
        for ke in range(Nk):
            _Xcv0[ke, kh] = dcv * ((-1)**kh)
            _Ycv0[ke, kh] = dcv
            _Zcv0[ke, kh] = -dcv

    _Xvc0 = np.conj(_Xcv0.T)
    _Yvc0 = np.conj(_Ycv0.T)
    _Zvc0 = np.conj(_Zcv0.T)


def CalcExpikr(y, ky):
    """Vectorized version (potentially faster than JIT for large arrays)."""
    global _Expikr, _Expikrc

    # Vectorized computation (NumPy handles this efficiently)
    _Expikr = np.exp(1j * np.outer(ky, y))  # Shape: (Nk, Nr)
    _Expikrc = np.conj(_Expikr)

def Xcv(k, kp):
    """
    Get X dipole matrix element.

    Returns the X component of the dipole matrix element at indices (k, kp).

    Parameters
    ----------
    k : int
        First index (0-based in Python)
    kp : int
        Second index (0-based in Python)

    Returns
    -------
    complex
        X dipole matrix element value

    Notes
    -----
    Uses module-level variable _Xcv0.
    """
    global _Xcv0
    return _Xcv0[k, kp]


def Ycv(k, kp):
    """
    Get Y dipole matrix element.

    Returns the Y component of the dipole matrix element at indices (k, kp).

    Parameters
    ----------
    k : int
        First index (0-based in Python)
    kp : int
        Second index (0-based in Python)

    Returns
    -------
    complex
        Y dipole matrix element value

    Notes
    -----
    Uses module-level variable _Ycv0.
    """
    global _Ycv0
    return _Ycv0[k, kp]


def Zcv(k, kp):
    """
    Get Z dipole matrix element.

    Returns the Z component of the dipole matrix element at indices (k, kp).

    Parameters
    ----------
    k : int
        First index (0-based in Python)
    kp : int
        Second index (0-based in Python)

    Returns
    -------
    complex
        Z dipole matrix element value

    Notes
    -----
    Uses module-level variable _Zcv0.
    """
    global _Zcv0
    return _Zcv0[k, kp]


def GetVn1n2(kr, rcv, Hcc, Hhh, Hcv, Vcc, Vvv, Vcv, Vvc):
    """
    Calculate interaction matrices Vcc, Vvv, Vcv, Vvc.

    Computes the interaction matrices from Hamiltonian matrices and
    dipole moment arrays using the Heisenberg equation of motion.

    Parameters
    ----------
    kr : ndarray
        Momentum coordinate array, 1D array
    rcv : ndarray
        Conduction-valence dipole moment array, 1D array, complex
    Hcc : ndarray
        Conduction-conduction Hamiltonian matrix, 2D array, complex
    Hhh : ndarray
        Hole-hole Hamiltonian matrix, 2D array, complex
    Hcv : ndarray
        Conduction-valence Hamiltonian matrix, 2D array, complex
    Vcc : ndarray
        Conduction-conduction interaction matrix (modified in-place), 2D array, complex
    Vvv : ndarray
        Valence-valence interaction matrix (modified in-place), 2D array, complex
    Vcv : ndarray
        Conduction-valence interaction matrix (modified in-place), 2D array, complex
    Vvc : ndarray
        Valence-conduction interaction matrix (modified in-place), 2D array, complex

    Returns
    -------
    None
        Vcc, Vvv, Vcv, Vvc are modified in-place

    Notes
    -----
    Uses module-level constants ii, hbar.
    Formula: V = (-i/hbar) * [rcv * H - H * rcv]
    Uses JIT compilation for performance-critical loops.
    """
    Nk = len(kr)

    Vcc[:] = 0.0
    Vvv[:] = 0.0
    Vcv[:] = 0.0
    Vvc[:] = 0.0

    Hvv = -Hhh.T
    Hvc = np.conj(Hcv.T)
    rvc = np.conj(rcv)

    try:
        _GetVn1n2_jit(Nk, rcv, Hcc, Hvv, Hcv, Hvc, rvc, Vcc, Vvv, Vcv, Vvc, ii, hbar)
    except Exception:
        # Fallback to pure Python
        _GetVn1n2_fallback(Nk, rcv, Hcc, Hvv, Hcv, Hvc, rvc, Vcc, Vvv, Vcv, Vvc)

    Vvc[:] = np.conj(Vcv.T)


@jit(nopython=True, parallel=True)
def _GetVn1n2_jit(Nk, rcv, Hcc, Hvv, Hcv, Hvc, rvc, Vcc, Vvv, Vcv, Vvc, ii_val, hbar_val):
    """JIT-compiled version of GetVn1n2 inner loops."""
    # Calculate Vcv
    for k2 in prange(Nk):
        for k1 in range(Nk):
            Vcv[k1, k2] = (-ii_val / hbar_val) * (rcv[k1] * Hvv[k1, k2] - Hcc[k1, k2] * rcv[k2])

    # Calculate Vcc
    for k2 in prange(Nk):
        for k1 in range(Nk):
            Vcc[k1, k2] = (-ii_val / hbar_val) * (rcv[k1] * Hvc[k1, k2] - Hcv[k1, k2] * rvc[k2])

    # Calculate Vvv
    for k2 in prange(Nk):
        for k1 in range(Nk):
            Vvv[k1, k2] = (-ii_val / hbar_val) * (rvc[k1] * Hcv[k1, k2] - Hvc[k1, k2] * rcv[k2])


def _GetVn1n2_fallback(Nk, rcv, Hcc, Hvv, Hcv, Hvc, rvc, Vcc, Vvv, Vcv, Vvc):
    """Fallback pure Python version of GetVn1n2 inner loops."""
    # Calculate Vcv
    for k2 in range(Nk):
        for k1 in range(Nk):
            Vcv[k1, k2] = (-ii / hbar) * (rcv[k1] * Hvv[k1, k2] - Hcc[k1, k2] * rcv[k2])

    # Calculate Vcc
    for k2 in range(Nk):
        for k1 in range(Nk):
            Vcc[k1, k2] = (-ii / hbar) * (rcv[k1] * Hvc[k1, k2] - Hcv[k1, k2] * rvc[k2])

    # Calculate Vvv
    for k2 in range(Nk):
        for k1 in range(Nk):
            Vvv[k1, k2] = (-ii / hbar) * (rvc[k1] * Hcv[k1, k2] - Hvc[k1, k2] * rcv[k2])
