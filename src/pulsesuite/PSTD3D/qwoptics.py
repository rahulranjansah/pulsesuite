"""
Quantum wire optics calculations for quantum wire simulations.

This module converts Maxwell electric fields from propagation space
to quantum wire (QW) electric fields, and vice versa. It also calculates
QW polarization and charge densities.

State is encapsulated in the :class:`QWOptics` class (one instance per
quantum wire configuration).  JIT/CUDA kernels and stateless helper
functions remain at module level.

Author: Rahul R. Sah
"""

import os

import numpy as np
from numba import jit, prange

try:
    from numba import cuda
    _HAS_CUDA = cuda.is_available()
except (ImportError, RuntimeError):
    _HAS_CUDA = False
    cuda = None

from scipy.constants import c as c0_SI, e as e0, epsilon_0 as eps0_SI, hbar as hbar_SI

from ..libpulsesuite.spliner import rescale_1D
from .usefulsubs import FFTG, GetArray0Index, iFFTG, printIT, printIT2D

# Physical constants
twopi = 2.0 * np.pi
c0 = c0_SI
hbar = hbar_SI
eps0 = eps0_SI
ii = 1j  # Imaginary unit

# Module-level I/O state
_firsttime = True


# ══════════════════════════════════════════════════════════════════════
# JIT / CUDA kernels  (must be module-level for Numba)
# ══════════════════════════════════════════════════════════════════════

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


# CUDA implementation for QWPolarization3
if _HAS_CUDA:
    @cuda.jit
    def _QWPolarization3_cuda_kernel_px(p_real, p_imag, Xvc0_real, Xvc0_imag,
                                        Expikr_real, Expikr_imag, Expikrc_real, Expikrc_imag,
                                        QWWindow, Px, Nr, Nk):
        """CUDA kernel for Px component of QWPolarization3."""
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
                x_re = Xvc0_real[kh, ke]
                x_im = Xvc0_imag[kh, ke]
                er_re = Expikr_real[ke, r]
                er_im = Expikr_imag[ke, r]
                ec_re = Expikrc_real[kh, r]
                ec_im = Expikrc_imag[kh, r]

                px_re = p_re * x_re - p_im * x_im
                px_im = p_re * x_im + p_im * x_re
                ee_re = er_re * ec_re - er_im * ec_im
                ee_im = er_re * ec_im + er_im * ec_re
                prod_re = px_re * ee_re - px_im * ee_im

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

        d_Px = cuda.device_array((Nr,), dtype=np.float64)
        d_Py = cuda.device_array((Nr,), dtype=np.float64)
        d_Pz = cuda.device_array((Nr,), dtype=np.float64)
        d_Px[:] = 0.0
        d_Py[:] = 0.0
        d_Pz[:] = 0.0

        threads_per_block = 256
        total_elements = Nr * Nk * Nk
        blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block

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

        Px[:] = d_Px.copy_to_host()
        Py[:] = d_Py.copy_to_host()
        Pz[:] = d_Pz.copy_to_host()
else:
    def _QWPolarization3_cuda(*args, **kwargs):
        """Dummy CUDA function when CUDA is not available."""
        raise RuntimeError("CUDA not available")


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


def _QWRho5_fallback(Nr, Nk, CC, DD, Expikr, Expikrc, QWWindow, L, re, rh):
    """Fallback pure Python version of QWRho5 inner loops."""
    for ri in range(Nr):
        for k2 in range(Nk):
            for k1 in range(Nk):
                re[ri] = re[ri] + CC[k1, k2] * Expikrc[k1, ri] * Expikr[k2, ri] * QWWindow[ri] / (2.0 * L)
                rh[ri] = rh[ri] + np.conj(DD[k1, k2]) * Expikrc[k1, ri] * Expikr[k2, ri] * QWWindow[ri] / (2.0 * L)


# CUDA implementation for QWRho5
if _HAS_CUDA:
    @cuda.jit
    def _QWRho5_cuda_kernel(CC_real, CC_imag, DD_real, DD_imag,
                            Expikr_real, Expikr_imag, Expikrc_real, Expikrc_imag,
                            QWWindow, L, re_real, re_imag, rh_real, rh_imag, Nr, Nk):
        """CUDA kernel for QWRho5 calculation."""
        idx = cuda.grid(1)
        total_threads = Nr * Nk * Nk

        if idx < total_threads:
            ri = idx // (Nk * Nk)
            k2_k1 = idx % (Nk * Nk)
            k2 = k2_k1 // Nk
            k1 = k2_k1 % Nk

            if ri < Nr and k2 < Nk and k1 < Nk:
                cc_re = CC_real[k1, k2]
                cc_im = CC_imag[k1, k2]
                ec_re = Expikrc_real[k1, ri]
                ec_im = Expikrc_imag[k1, ri]
                er_re = Expikr_real[k2, ri]
                er_im = Expikr_imag[k2, ri]

                ccec_re = cc_re * ec_re - cc_im * ec_im
                ccec_im = cc_re * ec_im + cc_im * ec_re
                prod_re = ccec_re * er_re - ccec_im * er_im
                prod_im = ccec_re * er_im + ccec_im * er_re

                factor = QWWindow[ri] / (2.0 * L)
                cuda.atomic.add(re_real, ri, prod_re * factor)
                cuda.atomic.add(re_imag, ri, prod_im * factor)

                dd_re = DD_real[k1, k2]
                dd_im = DD_imag[k1, k2]
                dd_conj_re = dd_re
                dd_conj_im = -dd_im

                ddec_re = dd_conj_re * ec_re - dd_conj_im * ec_im
                ddec_im = dd_conj_re * ec_im + dd_conj_im * ec_re
                prod_rh_re = ddec_re * er_re - ddec_im * er_im
                prod_rh_im = ddec_re * er_im + ddec_im * er_re

                cuda.atomic.add(rh_real, ri, prod_rh_re * factor)
                cuda.atomic.add(rh_imag, ri, prod_rh_im * factor)

    def _QWRho5_cuda(Nr, Nk, CC, DD, Expikr, Expikrc, QWWindow, L, re, rh):
        """CUDA wrapper for QWRho5."""
        CC_real = np.ascontiguousarray(CC.real, dtype=np.float64)
        CC_imag = np.ascontiguousarray(CC.imag, dtype=np.float64)
        DD_real = np.ascontiguousarray(DD.real, dtype=np.float64)
        DD_imag = np.ascontiguousarray(DD.imag, dtype=np.float64)
        Expikr_real = np.ascontiguousarray(Expikr.real, dtype=np.float64)
        Expikr_imag = np.ascontiguousarray(Expikr.imag, dtype=np.float64)
        Expikrc_real = np.ascontiguousarray(Expikrc.real, dtype=np.float64)
        Expikrc_imag = np.ascontiguousarray(Expikrc.imag, dtype=np.float64)
        QWWindow_arr = np.ascontiguousarray(QWWindow, dtype=np.float64)

        d_CC_real = cuda.to_device(CC_real)
        d_CC_imag = cuda.to_device(CC_imag)
        d_DD_real = cuda.to_device(DD_real)
        d_DD_imag = cuda.to_device(DD_imag)
        d_Expikr_real = cuda.to_device(Expikr_real)
        d_Expikr_imag = cuda.to_device(Expikr_imag)
        d_Expikrc_real = cuda.to_device(Expikrc_real)
        d_Expikrc_imag = cuda.to_device(Expikrc_imag)
        d_QWWindow = cuda.to_device(QWWindow_arr)

        d_re_real = cuda.device_array((Nr,), dtype=np.float64)
        d_re_imag = cuda.device_array((Nr,), dtype=np.float64)
        d_rh_real = cuda.device_array((Nr,), dtype=np.float64)
        d_rh_imag = cuda.device_array((Nr,), dtype=np.float64)
        d_re_real[:] = 0.0
        d_re_imag[:] = 0.0
        d_rh_real[:] = 0.0
        d_rh_imag[:] = 0.0

        threads_per_block = 256
        total_elements = Nr * Nk * Nk
        blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block

        _QWRho5_cuda_kernel[blocks_per_grid, threads_per_block](
            d_CC_real, d_CC_imag, d_DD_real, d_DD_imag,
            d_Expikr_real, d_Expikr_imag, d_Expikrc_real, d_Expikrc_imag,
            d_QWWindow, L, d_re_real, d_re_imag, d_rh_real, d_rh_imag, Nr, Nk
        )

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


@jit(nopython=True, parallel=True)
def _GetVn1n2_jit(Nk, rcv, Hcc, Hvv, Hcv, Hvc, rvc, Vcc, Vvv, Vcv, Vvc, ii_val, hbar_val):
    """JIT-compiled version of GetVn1n2 inner loops."""
    for k2 in prange(Nk):
        for k1 in range(Nk):
            Vcv[k1, k2] = (-ii_val / hbar_val) * (rcv[k1] * Hvv[k1, k2] - Hcc[k1, k2] * rcv[k2])

    for k2 in prange(Nk):
        for k1 in range(Nk):
            Vcc[k1, k2] = (-ii_val / hbar_val) * (rcv[k1] * Hvc[k1, k2] - Hcv[k1, k2] * rvc[k2])

    for k2 in prange(Nk):
        for k1 in range(Nk):
            Vvv[k1, k2] = (-ii_val / hbar_val) * (rvc[k1] * Hcv[k1, k2] - Hvc[k1, k2] * rcv[k2])


def _GetVn1n2_fallback(Nk, rcv, Hcc, Hvv, Hcv, Hvc, rvc, Vcc, Vvv, Vcv, Vvc):
    """Fallback pure Python version of GetVn1n2 inner loops."""
    for k2 in range(Nk):
        for k1 in range(Nk):
            Vcv[k1, k2] = (-ii / hbar) * (rcv[k1] * Hvv[k1, k2] - Hcc[k1, k2] * rcv[k2])

    for k2 in range(Nk):
        for k1 in range(Nk):
            Vcc[k1, k2] = (-ii / hbar) * (rcv[k1] * Hvc[k1, k2] - Hcv[k1, k2] * rvc[k2])

    for k2 in range(Nk):
        for k1 in range(Nk):
            Vvv[k1, k2] = (-ii / hbar) * (rvc[k1] * Hcv[k1, k2] - Hvc[k1, k2] * rcv[k2])


# ══════════════════════════════════════════════════════════════════════
# QWOptics class — encapsulates Fortran module state
# ══════════════════════════════════════════════════════════════════════

class QWOptics:
    """Quantum wire optics state and calculations.

    Wraps the Fortran-module state variables (_QWWindow, _Expikr, dipole
    matrices, etc.) into one object.  Method names are identical to the
    original Fortran / flat-Python API so that callers stay familiar.

    Parameters are the same as the old ``InitializeQWOptics`` function.
    """

    def __init__(self, RR, L, dcv, kr, Qr, Ee, Eh, ehint, area, gap):
        """Initialize quantum wire optics (replaces ``InitializeQWOptics``).

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
            Momentum coordinate array (unused, kept for interface compatibility)
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
        """
        self._small = 1e-100
        self._epsr = 9.1

        Nr = len(RR)
        Nk = len(kr)

        self.CalcQWWindow(RR, L)
        self.CalcExpikr(RR, kr)

        self._dcv0 = dcv
        R0 = np.sqrt(area / twopi)
        self._Vol = L * area / ehint

        self._Xcv0 = np.zeros((Nk, Nk), dtype=complex)
        self._Ycv0 = np.zeros((Nk, Nk), dtype=complex)
        self._Zcv0 = np.zeros((Nk, Nk), dtype=complex)

        for kh in range(Nk):
            for ke in range(Nk):
                self._Xcv0[ke, kh] = dcv * ((-1)**kh)
                self._Ycv0[ke, kh] = dcv
                self._Zcv0[ke, kh] = -dcv

        self._Xvc0 = np.conj(self._Xcv0.T)
        self._Yvc0 = np.conj(self._Ycv0.T)
        self._Zvc0 = np.conj(self._Zcv0.T)

    # ── helpers called during __init__ ────────────────────────────────

    def CalcQWWindow(self, YY, L):
        """Calculate quantum wire window function.

        Parameters
        ----------
        YY : ndarray
            Spatial coordinate array, 1D array
        L : float
            Length of wire (m)
        """
        Ny = len(YY)

        self._QWWindow = np.ones(Ny)

        for k in range(Ny):
            if abs(YY[k]) > L / 2.0:
                self._QWWindow[k] = 0.0

        self._QWWindow = np.exp(-(YY / (L / 2.0))**150)

        printIT(self._QWWindow * (ii)**0.0, YY, 0, 'Envl.y.')

    def CalcExpikr(self, y, ky):
        """Compute exp(i k r) phase arrays.

        Parameters
        ----------
        y : ndarray
            Spatial coordinate array, 1D array
        ky : ndarray
            Momentum coordinate array, 1D array
        """
        self._Expikr = np.exp(1j * np.outer(ky, y))   # Shape: (Nk, Nr)
        self._Expikrc = np.conj(self._Expikr)

    # ── field conversion ──────────────────────────────────────────────

    def Prop2QW(self, RR, Exx, Eyy, Ezz, Vrr, Edc, R, Ex, Ey, Ez, Vr, t, xxx):
        """Convert Maxwell electric fields from propagation space to QW space.

        Parameters
        ----------
        RR : ndarray
            Maxwell spatial array, 1D
        Exx, Eyy, Ezz : ndarray
            Maxwell electric field components, 1D, complex
        Vrr : ndarray
            Maxwell free charge potential, 1D, complex
        Edc : float
            DC field (currently unused)
        R : ndarray
            QW spatial array, 1D
        Ex, Ey, Ez : ndarray
            QW electric field components (modified in-place), 1D, complex
        Vr : ndarray
            QW free charge potential (modified in-place), 1D, complex
        t : float
            Current time (s)
        xxx : int
            Time index
        """
        Ex[:] = 0.0
        Ey[:] = 0.0
        Ez[:] = 0.0
        Vr[:] = 0.0

        rescale_1D(RR, Exx, R, Ex)
        rescale_1D(RR, Eyy, R, Ey)
        rescale_1D(RR, Ezz, R, Ez)
        rescale_1D(RR, Vrr, R, Vr)

        if self._QWWindow is not None:
            Ex[:] = Ex[:] * self._QWWindow
            Ey[:] = Ey[:] * self._QWWindow
            Ez[:] = Ez[:] * self._QWWindow
            Vr[:] = Vr[:] * self._QWWindow

        Ex[:] = np.real(Ex)
        Ey[:] = np.real(Ey)
        Ez[:] = np.real(Ez)
        Vr[:] = np.real(Vr)

        FFTG(Ex)
        FFTG(Ey)
        FFTG(Ez)
        FFTG(Vr)

    def QW2Prop(self, r, Qr, Ex, Ey, Ez, Vr, Px, Py, Pz, re, rh, RR, Pxx, Pyy, Pzz, RhoE, RhoH, w, xxx, WriteFields, Plasmonics):
        """Convert QW fields and polarizations back to propagation space.

        Parameters
        ----------
        r : ndarray
            QW Y-space, 1D
        Qr : ndarray
            QW momentum space, 1D
        Ex, Ey, Ez : ndarray
            QW electric field components (modified in-place), 1D, complex
        Vr : ndarray
            QW free charge potential (modified in-place), 1D, complex
        Px, Py, Pz : ndarray
            QW polarization components (modified in-place), 1D, complex
        re, rh : ndarray
            QW electron/hole charge density (modified in-place), 1D, complex
        RR : ndarray
            Propagation Y-space, 1D
        Pxx, Pyy, Pzz : ndarray
            Propagation polarizations (modified in-place), 1D, complex
        RhoE, RhoH : ndarray
            Propagation charge densities (modified in-place), 1D, complex
        w : int
            Wire index
        xxx : int
            Time index
        WriteFields : bool
            Record fields?
        Plasmonics : bool
            Calculate charge densities?
        """
        dr = r[1] - r[0]
        dRR = RR[1] - RR[0]

        iFFTG(Ex)
        iFFTG(Ey)
        iFFTG(Ez)
        iFFTG(Vr)
        iFFTG(Px)
        iFFTG(Py)
        iFFTG(Pz)
        iFFTG(re)
        iFFTG(rh)

        if Plasmonics:
            re[:] = np.abs(re)
            rh[:] = np.abs(rh)
            total = (np.sum(re) * dr + np.sum(rh) * dr + self._small) / 2.0
            re[:] = re * total / (np.sum(re) * dr + self._small)
            rh[:] = rh * total / (np.sum(rh) * dr + self._small)

        if WriteFields:
            WriteQWFields(r, Ex, Ey, Ez, Vr, Px, Py, Pz, re, rh, 'r', w, xxx)

        rescale_1D(r, Px, RR, Pxx)
        rescale_1D(r, Py, RR, Pyy)
        rescale_1D(r, Pz, RR, Pzz)
        rescale_1D(r, re, RR, RhoE)
        rescale_1D(r, rh, RR, RhoH)

        if Plasmonics:
            RhoE[:] = np.abs(RhoE)
            RhoH[:] = np.abs(RhoH)
            RhoE[:] = RhoE * total / (np.sum(RhoE) * dRR + self._small)
            RhoH[:] = RhoH * total / (np.sum(RhoH) * dRR + self._small)

    # ── polarization / charge density ─────────────────────────────────

    def QWPolarization3(self, y, ky, p, ehint, area, L, Px, Py, Pz, xxx, w):
        """Calculate QW polarization in 3D.

        Parameters
        ----------
        y : ndarray
            Spatial coordinate array, 1D
        ky : ndarray
            Momentum coordinate array, 1D
        p : ndarray
            Density matrix, shape (Nk, Nk), complex
        ehint : float
            Electron-hole space integral
        area : float
            Area of wire (m^2)
        L : float
            Length of wire (m)
        Px, Py, Pz : ndarray
            QW polarization components (modified in-place), 1D, complex
        xxx : int
            Time index
        w : int
            Wire index
        """
        Px[:] = 0.0
        Py[:] = 0.0
        Pz[:] = 0.0

        if self._Xvc0 is None or self._Yvc0 is None or self._Zvc0 is None:
            return
        if self._Expikr is None or self._Expikrc is None:
            return
        if self._QWWindow is None:
            return

        # Try CUDA first, then JIT, then fallback
        if _HAS_CUDA:
            try:
                _QWPolarization3_cuda(y, ky, p, self._Xvc0, self._Yvc0, self._Zvc0, self._Expikr, self._Expikrc, self._QWWindow, yw(w), w, Px, Py, Pz)
            except Exception:
                try:
                    _QWPolarization3_jit(y, ky, p, self._Xvc0, self._Yvc0, self._Zvc0, self._Expikr, self._Expikrc, self._QWWindow, yw(w), w, Px, Py, Pz)
                except Exception:
                    _QWPolarization3_fallback(y, ky, p, self._Xvc0, self._Yvc0, self._Zvc0, self._Expikr, self._Expikrc, self._QWWindow, yw(w), w, Px, Py, Pz)
        else:
            try:
                _QWPolarization3_jit(y, ky, p, self._Xvc0, self._Yvc0, self._Zvc0, self._Expikr, self._Expikrc, self._QWWindow, yw(w), w, Px, Py, Pz)
            except Exception:
                _QWPolarization3_fallback(y, ky, p, self._Xvc0, self._Yvc0, self._Zvc0, self._Expikr, self._Expikrc, self._QWWindow, yw(w), w, Px, Py, Pz)

        Px[:] = Px * 2 * ehint / area / L
        Py[:] = Py * 2 * ehint / area / L
        Pz[:] = Pz * 2 * ehint / area / L

        FFTG(Px)
        FFTG(Py)
        FFTG(Pz)

    def QWRho5(self, Qr, kr, R, L, kkp, p, CC, DD, ne, nh, re, rh, xxx, jjj):
        """Calculate quantum wire charge densities.

        Parameters
        ----------
        Qr : ndarray
            QW momentum array, 1D
        kr : ndarray
            QW momentum array, 1D
        R : ndarray
            QW spatial coordinate array, 1D
        L : float
            Length of wire (m)
        kkp : ndarray
            Momentum difference index matrix, 2D, integer
        p : ndarray
            Density matrix (unused, interface compat), 2D, complex
        CC : ndarray
            Electron/electron coherence matrix, 2D, complex
        DD : ndarray
            Hole/hole coherence matrix, 2D, complex
        ne : ndarray
            Electron occupation numbers, 1D, complex
        nh : ndarray
            Hole occupation numbers, 1D, complex
        re : ndarray
            Electron charge density (modified in-place), 1D, complex
        rh : ndarray
            Hole charge density (modified in-place), 1D, complex
        xxx : int
            Time index
        jjj : int
            Additional index (unused, interface compat)
        """
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

        if self._Expikr is None or self._Expikrc is None or self._QWWindow is None:
            return

        # Try CUDA first, then JIT, then fallback
        if _HAS_CUDA:
            try:
                _QWRho5_cuda(Nr, Nk, CC, DD, self._Expikr, self._Expikrc, self._QWWindow, L, re, rh)
            except Exception:
                try:
                    _QWRho5_jit(Nr, Nk, CC, DD, self._Expikr, self._Expikrc, self._QWWindow, L, re, rh)
                except Exception:
                    _QWRho5_fallback(Nr, Nk, CC, DD, self._Expikr, self._Expikrc, self._QWWindow, L, re, rh)
        else:
            try:
                _QWRho5_jit(Nr, Nk, CC, DD, self._Expikr, self._Expikrc, self._QWWindow, L, re, rh)
            except Exception:
                _QWRho5_fallback(Nr, Nk, CC, DD, self._Expikr, self._Expikrc, self._QWWindow, L, re, rh)

        # Remove boundary effects
        re[:] = re - np.real(np.sum(re[0:10]) + np.sum(re[Nr-10:Nr])) / 20.0
        rh[:] = rh - np.real(np.sum(rh[0:10]) + np.sum(rh[Nr-10:Nr])) / 20.0

        # Normalize to match total numbers
        re[:] = re * NeTotal / (np.sum(np.abs(re)) * dr + self._small)
        rh[:] = rh * NhTotal / (np.sum(np.abs(rh)) * dr + self._small)

        FFTG(re)
        FFTG(rh)

    # ── dipole matrix element getters ─────────────────────────────────

    def Xcv(self, k, kp):
        """Get X dipole matrix element at indices (k, kp)."""
        return self._Xcv0[k, kp]

    def Ycv(self, k, kp):
        """Get Y dipole matrix element at indices (k, kp)."""
        return self._Ycv0[k, kp]

    def Zcv(self, k, kp):
        """Get Z dipole matrix element at indices (k, kp)."""
        return self._Zcv0[k, kp]


# ══════════════════════════════════════════════════════════════════════
# Free functions — no instance state needed
# ══════════════════════════════════════════════════════════════════════

def yw(w):
    """Calculate wire-dependent sign factor.

    Parameters
    ----------
    w : int
        Wire index

    Returns
    -------
    int
        Sign factor: (-1)**floor((w-1)/2)
    """
    return (-1)**int(np.floor((w - 1) / 2.0))


def QWChi1(lam, dky, Ee, Eh, area, geh, dcv):
    """Calculate quantum wire linear susceptibility.

    Parameters
    ----------
    lam : float
        Wavelength (m)
    dky : float
        Momentum step (1/m)
    Ee : ndarray
        Electron energies, 1D
    Eh : ndarray
        Hole energies, 1D
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
    """
    ww = twopi * c0 / lam
    denominator1 = Ee + Eh - ii * hbar * geh - hbar * ww
    denominator2 = Ee + Eh + ii * hbar * geh + hbar * ww
    result = 4 * dcv**2 / eps0 / area * dky / twopi * np.sum((Ee + Eh) / (denominator1 * denominator2))
    return result


def GetVn1n2(kr, rcv, Hcc, Hhh, Hcv, Vcc, Vvv, Vcv, Vvc):
    """Calculate interaction matrices Vcc, Vvv, Vcv, Vvc.

    Parameters
    ----------
    kr : ndarray
        Momentum coordinate array, 1D
    rcv : ndarray
        Conduction-valence dipole moment array, 1D, complex
    Hcc : ndarray
        Conduction-conduction Hamiltonian matrix, 2D, complex
    Hhh : ndarray
        Hole-hole Hamiltonian matrix, 2D, complex
    Hcv : ndarray
        Conduction-valence Hamiltonian matrix, 2D, complex
    Vcc, Vvv, Vcv, Vvc : ndarray
        Interaction matrices (modified in-place), 2D, complex
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
        _GetVn1n2_fallback(Nk, rcv, Hcc, Hvv, Hcv, Hvc, rvc, Vcc, Vvv, Vcv, Vvc)

    Vvc[:] = np.conj(Vcv.T)


# ── I/O helpers ───────────────────────────────────────────────────────

def printITReal2(Dx, z, n, file):
    """Print real part of complex field to file.

    Parameters
    ----------
    Dx : ndarray
        Complex field array, 1D
    z : ndarray
        Coordinate array, 1D
    n : int
        Time index
    file : str
        Base filename (without extension)
    """
    global _firsttime

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


def printITReal(Dx, z, n, file):
    """Print real part of complex field to file.

    Parameters
    ----------
    Dx : ndarray
        Complex field array, 1D
    z : ndarray
        Coordinate array, 1D
    n : int
        Time index
    file : str
        Base filename (without extension)
    """
    filename = f'dataQW/{file}{n:06d}.dat'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(len(z)):
            f.write(f'{np.float32(z[i])} {np.float32(np.real(Dx[i]))}\n')


def printIT3D(Dx, z, n, file):
    """Print 3D complex array to file.

    Parameters
    ----------
    Dx : ndarray
        3D complex array, shape (N1, N2, N3)
    z : ndarray
        Coordinate array (unused, interface compat), 1D
    n : int
        Time index
    file : str
        Base filename (without extension)
    """
    filename = f'dataQW/{file}{n:06d}.dat'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    Dx_ordered = Dx.transpose(2, 1, 0)

    data = np.column_stack([
        np.real(Dx.flatten()),
        np.imag(Dx.flatten())
    ])
    np.savetxt(filename, data, fmt='%.6e')


def WriteSBESolns(ky, ne, nh, C, D, P, Ee, Eh, w, xxx):
    """Write SBE solutions to files.

    Parameters
    ----------
    ky : ndarray
        QW momentum array, 1D
    ne, nh : ndarray
        Electron/hole occupation numbers, 1D, complex
    C, D, P : ndarray
        Coherence matrices, 2D, complex
    Ee, Eh : ndarray
        Electron/hole energies, 1D, complex
    w : int
        Wire index
    xxx : int
        Time index
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
    """Write photoluminescence spectrum to file.

    Parameters
    ----------
    hw : ndarray
        Photon energy array (J), 1D
    PLS : ndarray
        Photoluminescence spectrum, 1D
    w : int
        Wire index
    xxx : int
        Time index
    """
    PLS0 = np.zeros(len(hw), dtype=complex)
    PLS0[:] = PLS

    wire = f'{w:02d}'

    printIT(PLS0, hw / e0, xxx, f'Wire/PL/pl.{wire}.hw.')


def WriteQWFields(QY, Ex, Ey, Ez, Vr, Px, Py, Pz, Re, Rh, sp, w, xxx):
    """Write QW field arrays to files.

    Parameters
    ----------
    QY : ndarray
        QY momentum/y-space array, 1D
    Ex, Ey, Ez : ndarray
        QW electric fields, 1D, complex
    Vr : ndarray
        QW free charge potential, 1D, complex
    Px, Py, Pz : ndarray
        QW polarizations, 1D, complex
    Re, Rh : ndarray
        QW charge densities, 1D, complex
    sp : str
        Domain label ('k' or 'r')
    w : int
        Wire index
    xxx : int
        Time index
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
    """Write propagation field arrays to files.

    Parameters
    ----------
    y : ndarray
        Spatial coordinate array, 1D
    Ex, Ey, Ez : ndarray
        Propagation electric fields, 1D, complex
    Vr : ndarray
        Propagation free charge potential, 1D, complex
    Px, Py, Pz : ndarray
        Propagation polarizations, 1D, complex
    Re, Rh : ndarray
        Propagation charge densities, 1D, complex
    sp : str
        Space label
    w : int
        Wire index
    xxx : int
        Time index
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
