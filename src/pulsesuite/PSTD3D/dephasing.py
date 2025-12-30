"""
Dephasing calculations for quantum wire simulations.

This module calculates dephasing rates for electrons and holes due to
carrier-carrier interactions in the Semiconductor Bloch equations.

Author: Rahul R. Sah

Bugs:
Why Lrtz jit here? -> tackle nopython=True in the future
"""

import numpy as np
from scipy.constants import hbar as hbar_SI
from numba import jit
from .usefulsubs import Lrtz
import os

# Physical constants
pi = np.pi
hbar = hbar_SI
ii = 1j  # Imaginary unit

# Module-level state variables (matching Fortran module variables)
_k_p_q = None
_k_m_q = None
_k1_m_q = None
_k1p_m_q = None
_k1 = None
_k1p = None
_xe = None
_xh = None
_fe_file = None
_fh_file = None


def InitializeDephasing(ky, me, mh):
    """
    Initialize the dephasing module.

    Sets up all module-level arrays required for dephasing calculations.
    Allocates and initializes momentum index arrays and opens output files.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    me : float
        Effective electron mass (kg)
    mh : float
        Effective hole mass (kg)

    Returns
    -------
    None
        All arrays are stored as module-level variables.
    """
    global _k_p_q, _k_m_q, _k1_m_q, _k1p_m_q, _k1, _k1p, _xe, _xh, _fe_file, _fh_file

    Nk = len(ky)
    dk = ky[1] - ky[0] if Nk > 1 else 1.0

    kmax = ky[Nk - 1] + dk
    kmin = ky[0] - dk

    Nk0 = (Nk - 1) // 2 + 1

    _k_p_q = np.zeros((Nk, Nk), dtype=np.int32)
    _k_m_q = np.zeros((Nk, Nk), dtype=np.int32)
    _k1_m_q = np.zeros((Nk, Nk), dtype=np.int32)
    _k1p_m_q = np.zeros((Nk, Nk), dtype=np.int32)
    _k1 = np.zeros((Nk, Nk), dtype=np.int32)
    _k1p = np.zeros((Nk, Nk), dtype=np.int32)

    _xe = me / hbar**2 * np.abs(ky) / (np.abs(ky) + dk * 1e-5)**2 / dk
    _xh = mh / hbar**2 * np.abs(ky) / (np.abs(ky) + dk * 1e-5)**2 / dk

    # Calculate k_p_q and k_m_q
    for q in range(Nk):
        for k in range(Nk):
            k_val = np.clip(ky[k] + ky[q], kmin, kmax)
            _k_p_q[k, q] = int(np.round(k_val / dk)) + Nk0 - 1  # Convert to 0-based

            k_val = np.clip(ky[k] - ky[q], kmin, kmax)
            _k_m_q[k, q] = int(np.round(k_val / dk)) + Nk0 - 1  # Convert to 0-based

    # Calculate k1p and k1p_m_q
    for q in range(Nk):
        for k in range(Nk):
            k1p_0 = ((me + mh) * ky[q] - 2 * mh * ky[k]) / 2.0 / me

            k_val = np.clip(k1p_0, kmin, kmax)
            _k1p[k, q] = int(np.round(k_val / dk)) + Nk0 - 1  # Convert to 0-based

            k_val = np.clip(k1p_0 - ky[q], kmin, kmax)
            _k1p_m_q[k, q] = int(np.round(k_val / dk)) + Nk0 - 1  # Convert to 0-based

    # Calculate k1 and k1_m_q
    for q in range(Nk):
        for kp in range(Nk):
            k1_0 = ((me + mh) * ky[q] - 2 * me * ky[kp]) / 2.0 / mh

            k_val = np.clip(k1_0, kmin, kmax)
            _k1[kp, q] = int(np.round(k_val / dk)) + Nk0 - 1  # Convert to 0-based

            k_val = np.clip(k1_0 - ky[q], kmin, kmax)
            _k1_m_q[kp, q] = int(np.round(k_val / dk)) + Nk0 - 1  # Convert to 0-based

    os.makedirs('dataQW/Wire/info', exist_ok=True)
    _fe_file = open('dataQW/Wire/info/MaxOffDiag.e.dat', 'w', encoding='utf-8')
    _fh_file = open('dataQW/Wire/info/MaxOffDiag.h.dat', 'w', encoding='utf-8')


def Vxx2(q, V):
    """
    Calculate squared interaction matrix elements.

    Computes V(1+iq, 1)^2 for each momentum q, where iq is the rounded
    index corresponding to abs(q/dq).

    Parameters
    ----------
    q : ndarray
        Momentum coordinates (1/m), 1D array
    V : ndarray
        Interaction matrix (J), 2D array

    Returns
    -------
    ndarray
        Squared interaction matrix elements, 1D array of length len(q)
    """
    dq = q[1] - q[0] if len(q) > 1 else 1.0

    iq = np.round(np.abs(q / dq)).astype(np.int32)

    Vxx2_arr = np.zeros(len(q))
    for i in range(len(q)):
        idx = iq[i]
        Vxx2_arr[i] = V[idx, 0]**2

    return Vxx2_arr


@jit(nopython=True)
def _CalcGammaE_jit(Vee2, Veh2, ne, nh, se, sh, k_p_q, k_m_q, k1p_m_q, k1p, xe, xh, pi_val, hbar_val):
    """JIT-compiled version of CalcGammaE."""
    Nk = len(ne) - 2
    GammaE = np.zeros(Nk)

    # Electron-Electron dephasing of electrons
    for q in range(Nk):
        for k in range(Nk):
            kpq_idx = k_p_q[k, q] + 1  # Convert to extended array index (0-based main array + 1)
            if 1 <= kpq_idx < len(ne) - 1:
                GammaE[k] = (GammaE[k] + pi_val / hbar_val * Vee2[q] *
                            ne[kpq_idx] * se[kpq_idx] * np.abs(xe[q]))

    # Electron-Hole dephasing of electrons
    for q in range(Nk):
        for k in range(Nk):
            k1pmq_idx = k1p_m_q[k, q] + 1  # Convert to extended array index
            k1p_idx = k1p[k, q] + 1  # Convert to extended array index
            kmq_idx = k_m_q[k, q] + 1  # Convert to extended array index

            if (1 <= k1pmq_idx < len(nh) - 1 and 1 <= k1p_idx < len(nh) - 1 and
                1 <= kmq_idx < len(ne) - 1):
                GammaE[k] = (GammaE[k] + pi_val / hbar_val * Veh2[q] *
                            (nh[k1pmq_idx] * sh[k1p_idx] * ne[kmq_idx] +
                             sh[k1pmq_idx] * nh[k1p_idx] * se[kmq_idx]) *
                            np.abs(xh[q]))

    return GammaE


def CalcGammaE(ky, ne0, nh0, VC, GammaE):
    """
    Calculate electron dephasing rate.

    Computes the dephasing rate for electrons due to electron-electron
    and electron-hole interactions.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    ne0 : ndarray
        Electron occupation numbers (complex), 1D array
    nh0 : ndarray
        Hole occupation numbers (complex), 1D array
    VC : ndarray
        Interaction matrix (J), 3D array VC[:,:,0] for e-h, VC[:,:,1] for e-e
    GammaE : ndarray
        Output electron dephasing rate (modified in-place), 1D array

    Returns
    -------
    None
        GammaE array is modified in-place.

    Notes
    -----
    The function computes dephasing from:
    1. Electron-electron interactions
    2. Electron-hole interactions
    Uses extended arrays (with boundary points) for indexing safety.
    """
    global _k_p_q, _k_m_q, _k1p_m_q, _k1p, _xe, _xh

    if _k_p_q is None:
        raise RuntimeError("InitializeDephasing must be called first")

    Veh2 = Vxx2(ky, VC[:, :, 0])
    Vee2 = Vxx2(ky, VC[:, :, 1])

    Nk = len(ky)

    # Extended arrays with boundary points (0:size+1 in Fortran)
    ne = np.zeros(Nk + 2)
    nh = np.zeros(Nk + 2)
    se = np.zeros(Nk + 2)
    sh = np.zeros(Nk + 2)

    ne[1:Nk+1] = np.real(ne0[:])
    nh[1:Nk+1] = np.real(nh0[:])

    se[1:Nk+1] = 1.0 - ne[1:Nk+1]
    sh[1:Nk+1] = 1.0 - nh[1:Nk+1]

    GammaE[:] = _CalcGammaE_jit(Vee2, Veh2, ne, nh, se, sh, _k_p_q, _k_m_q,
                                _k1p_m_q, _k1p, _xe, _xh, pi, hbar)


@jit(nopython=True)
def _CalcGammaH_jit(Vhh2, Veh2, ne, nh, se, sh, k_p_q, k_m_q, k1_m_q, k1, xe, xh, pi_val, hbar_val):
    """JIT-compiled version of CalcGammaH."""
    Nk = len(ne) - 2
    GammaH = np.zeros(Nk)

    # Hole-Hole dephasing of holes
    for q in range(Nk):
        for kp in range(Nk):
            kpq_idx = k_p_q[kp, q] + 1  # Convert to extended array index (0-based main array + 1)
            if 1 <= kpq_idx < len(nh) - 1:
                GammaH[kp] = (GammaH[kp] + pi_val / hbar_val * Vhh2[q] *
                             nh[kpq_idx] * sh[kpq_idx] * np.abs(xh[q]))

    # Electron-Hole dephasing of holes
    for q in range(Nk):
        for kp in range(Nk):
            k1mq_idx = k1_m_q[kp, q] + 1  # Convert to extended array index
            k1_idx = k1[kp, q] + 1  # Convert to extended array index
            kmq_idx = k_m_q[kp, q] + 1  # Convert to extended array index

            if (1 <= k1mq_idx < len(ne) - 1 and 1 <= k1_idx < len(ne) - 1 and
                1 <= kmq_idx < len(nh) - 1):
                GammaH[kp] = (GammaH[kp] + pi_val / hbar_val * Veh2[q] *
                             (ne[k1mq_idx] * se[k1_idx] * nh[kmq_idx] +
                              se[k1mq_idx] * ne[k1_idx] * sh[kmq_idx]) *
                             np.abs(xe[q]))

    return GammaH


def CalcGammaH(ky, ne0, nh0, VC, GammaH):
    """
    Calculate hole dephasing rate.

    Computes the dephasing rate for holes due to hole-hole
    and electron-hole interactions.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    ne0 : ndarray
        Electron occupation numbers (complex), 1D array
    nh0 : ndarray
        Hole occupation numbers (complex), 1D array
    VC : ndarray
        Interaction matrix (J), 3D array VC[:,:,0] for e-h, VC[:,:,2] for h-h
    GammaH : ndarray
        Output hole dephasing rate (modified in-place), 1D array

    Returns
    -------
    None
        GammaH array is modified in-place.

    Notes
    -----
    The function computes dephasing from:
    1. Hole-hole interactions
    2. Electron-hole interactions
    Uses extended arrays (with boundary points) for indexing safety.
    """
    global _k_p_q, _k_m_q, _k1_m_q, _k1, _xe, _xh

    if _k_p_q is None:
        raise RuntimeError("InitializeDephasing must be called first")

    Veh2 = Vxx2(ky, VC[:, :, 0])
    Vhh2 = Vxx2(ky, VC[:, :, 2])

    Nk = len(ky)

    # Extended arrays with boundary points (0:size+1 in Fortran)
    ne = np.zeros(Nk + 2)
    nh = np.zeros(Nk + 2)
    se = np.zeros(Nk + 2)
    sh = np.zeros(Nk + 2)

    ne[1:Nk+1] = np.real(ne0[:])
    nh[1:Nk+1] = np.real(nh0[:])

    se[1:Nk+1] = 1.0 - ne[1:Nk+1]
    sh[1:Nk+1] = 1.0 - nh[1:Nk+1]

    GammaH[:] = _CalcGammaH_jit(Vhh2, Veh2, ne, nh, se, sh, _k_p_q, _k_m_q,
                                _k1_m_q, _k1, _xe, _xh, pi, hbar)


@jit(nopython=True)
def _CalcOffDiagDeph_E_jit(ne, nh, Veh2, Vee2, Ee, Eh, k_p_q, k_m_q, gee, geh, pi_val, hbar_val):
    """JIT-compiled version of CalcOffDiagDeph_E."""
    Nk = len(ne) - 2
    D = np.zeros((Nk, Nk))

    # Electron-electron dephasing
    for q in range(Nk):
        for k in range(Nk):
            kpq_idx = k_p_q[k, q] + 1  # Convert to extended array index
            if 1 <= kpq_idx < len(Ee) - 1:
                for k1 in range(Nk):
                    k1pq_idx = k_p_q[k1, q] + 1  # Convert to extended array index
                    if 1 <= k1pq_idx < len(Ee) - 1:
                        E_diff = Ee[k1pq_idx] + Ee[k + 1] - Ee[k1 + 1] - Ee[kpq_idx]
                        D[k, q] = (D[k, q] + Vee2[k1 + 1, k1pq_idx] *
                                  _Lrtz_jit(E_diff, hbar_val * gee, pi_val) *
                                  (ne[k1pq_idx] * ne[k + 1] * (1.0 - ne[k1 + 1]) +
                                   (1.0 - ne[k1pq_idx]) * (1.0 - ne[k + 1]) * ne[k1 + 1]))

    # Electron-hole dephasing
    for q in range(Nk):
        for k in range(Nk):
            kpq_idx = k_p_q[k, q] + 1  # Convert to extended array index
            if 1 <= kpq_idx < len(Ee) - 1:
                for k1 in range(Nk):
                    k1mq_idx = k_m_q[k1, q] + 1  # Convert to extended array index
                    if 1 <= k1mq_idx < len(Eh) - 1:
                        E_diff = Eh[k1mq_idx] + Ee[k + 1] - Eh[k1 + 1] - Ee[kpq_idx]
                        D[k, q] = (D[k, q] + Veh2[k + 1, kpq_idx] *
                                  _Lrtz_jit(E_diff, hbar_val * geh, pi_val) *
                                  (nh[k1mq_idx] * (1.0 - nh[k1 + 1]) * ne[k + 1] +
                                   (1.0 - nh[k1mq_idx]) * nh[k1 + 1] * (1.0 - ne[k + 1])))

    return D * pi_val / hbar_val


def CalcOffDiagDeph_E(ne0, nh0, ky, Ee0, Eh0, gee, geh, VC):
    """
    Calculate off-diagonal dephasing matrix for electrons.

    Computes the off-diagonal dephasing matrix D(k,q) for electrons due to
    electron-electron and electron-hole interactions.

    Parameters
    ----------
    ne0 : ndarray
        Electron occupation numbers (complex), 1D array
    nh0 : ndarray
        Hole occupation numbers (complex), 1D array
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Ee0 : ndarray
        Electron energies (J), 1D array
    Eh0 : ndarray
        Hole energies (J), 1D array
    gee : float
        Electron-electron dephasing rate (Hz)
    geh : float
        Electron-hole dephasing rate (Hz)
    VC : ndarray
        Interaction matrix (J), 3D array VC[:,:,0] for e-h, VC[:,:,1] for e-e

    Returns
    -------
    ndarray
        Off-diagonal dephasing matrix (J), 2D array of shape (Nk, Nk)
    """
    global _k_p_q, _k_m_q

    if _k_p_q is None:
        raise RuntimeError("InitializeDephasing must be called first")

    Nk = len(ne0)

    # Extended arrays with boundary points (0:size+1 in Fortran)
    ne = np.zeros(Nk + 2)
    nh = np.zeros(Nk + 2)
    Ee = np.zeros(Nk + 2)
    Eh = np.zeros(Nk + 2)
    Veh2 = np.zeros((Nk + 2, Nk + 2))
    Vee2 = np.zeros((Nk + 2, Nk + 2))

    Veh2[1:Nk+1, 1:Nk+1] = VC[:, :, 0]**2
    Vee2[1:Nk+1, 1:Nk+1] = VC[:, :, 1]**2

    ne[1:Nk+1] = np.abs(ne0[:])
    nh[1:Nk+1] = np.abs(nh0[:])
    Ee[1:Nk+1] = Ee0[:]
    Eh[1:Nk+1] = Eh0[:]

    return _CalcOffDiagDeph_E_jit(ne, nh, Veh2, Vee2, Ee, Eh, _k_p_q, _k_m_q,
                                  gee, geh, pi, hbar)


@jit(nopython=True)
def _CalcOffDiagDeph_H_jit(ne, nh, Veh2, Vhh2, Ee, Eh, k_p_q, k_m_q, ghh, geh, pi_val, hbar_val):
    """JIT-compiled version of CalcOffDiagDeph_H."""
    Nk = len(ne) - 2
    D = np.zeros((Nk, Nk))

    # Hole-hole dephasing
    for q in range(Nk):
        for k in range(Nk):
            kpq_idx = k_p_q[k, q] + 1  # Convert to extended array index
            if 1 <= kpq_idx < len(Eh) - 1:
                for k1 in range(Nk):
                    k1pq_idx = k_p_q[k1, q] + 1  # Convert to extended array index
                    if 1 <= k1pq_idx < len(Eh) - 1:
                        E_diff = Eh[k1pq_idx] + Eh[k + 1] - Eh[k1 + 1] - Eh[kpq_idx]
                        D[k, q] = (D[k, q] + Vhh2[k1 + 1, k1pq_idx] *
                                  _Lrtz_jit(E_diff, hbar_val * ghh, pi_val) *
                                  (nh[k1pq_idx] * nh[k + 1] * (1.0 - nh[k1 + 1]) +
                                   (1.0 - nh[k1pq_idx]) * (1.0 - nh[k + 1]) * nh[k1 + 1]))

    # Electron-hole dephasing
    for q in range(Nk):
        for k in range(Nk):
            kpq_idx = k_p_q[k, q] + 1  # Convert to extended array index
            if 1 <= kpq_idx < len(Eh) - 1:
                for k1 in range(Nk):
                    k1mq_idx = k_m_q[k1, q] + 1  # Convert to extended array index
                    if 1 <= k1mq_idx < len(Ee) - 1:
                        E_diff = Ee[k1mq_idx] + Eh[k + 1] - Ee[k1 + 1] - Eh[kpq_idx]
                        D[k, q] = (D[k, q] + Veh2[k1 + 1, k1mq_idx] *
                                  _Lrtz_jit(E_diff, hbar_val * geh, pi_val) *
                                  (ne[k1mq_idx] * (1.0 - ne[k1 + 1]) * nh[k + 1] +
                                   (1.0 - ne[k1mq_idx]) * ne[k1 + 1] * (1.0 - nh[k + 1])))

    return D * pi_val / hbar_val


def CalcOffDiagDeph_H(ne0, nh0, ky, Ee0, Eh0, ghh, geh, VC):
    """
    Calculate off-diagonal dephasing matrix for holes.

    Computes the off-diagonal dephasing matrix D(k,q) for holes due to
    hole-hole and electron-hole interactions.

    Parameters
    ----------
    ne0 : ndarray
        Electron occupation numbers (complex), 1D array
    nh0 : ndarray
        Hole occupation numbers (complex), 1D array
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Ee0 : ndarray
        Electron energies (J), 1D array
    Eh0 : ndarray
        Hole energies (J), 1D array
    ghh : float
        Hole-hole dephasing rate (Hz)
    geh : float
        Electron-hole dephasing rate (Hz)
    VC : ndarray
        Interaction matrix (J), 3D array VC[:,:,0] for e-h, VC[:,:,2] for h-h

    Returns
    -------
    ndarray
        Off-diagonal dephasing matrix (J), 2D array of shape (Nk, Nk)
    """
    global _k_p_q, _k_m_q

    if _k_p_q is None:
        raise RuntimeError("InitializeDephasing must be called first")

    Nk = len(ne0)

    # Extended arrays with boundary points (0:size+1 in Fortran)
    ne = np.zeros(Nk + 2)
    nh = np.zeros(Nk + 2)
    Ee = np.zeros(Nk + 2)
    Eh = np.zeros(Nk + 2)
    Veh2 = np.zeros((Nk + 2, Nk + 2))
    Vhh2 = np.zeros((Nk + 2, Nk + 2))

    Veh2[1:Nk+1, 1:Nk+1] = VC[:, :, 0]**2
    Vhh2[1:Nk+1, 1:Nk+1] = VC[:, :, 2]**2

    ne[1:Nk+1] = np.abs(ne0[:])
    nh[1:Nk+1] = np.abs(nh0[:])
    Ee[1:Nk+1] = Ee0[:]
    Eh[1:Nk+1] = Eh0[:]

    return _CalcOffDiagDeph_H_jit(ne, nh, Veh2, Vhh2, Ee, Eh, _k_p_q, _k_m_q,
                                  ghh, geh, pi, hbar)


@jit(nopython=True)
def _OffDiagDephasing_jit(Dh, De, pt, pp, k_p_q, undel, ii_val, hbar_val):
    """JIT-compiled version of OffDiagDephasing inner loops."""
    Nk = Dh.shape[0]
    x = np.zeros((Nk, Nk), dtype=np.complex128)

    # First loop: x(kp,k) = x(kp,k) + Dh(qp,kp) * pt(k_p_q(qp,kp), k) * undel(qp)
    for k in range(Nk):
        for kp in range(Nk):
            for qp in range(Nk):
                kpq_idx = k_p_q[qp, kp] + 1  # Convert to extended array index
                if 1 <= kpq_idx < len(pt) - 1 and 0 <= k < Nk:
                    x[kp, k] = x[kp, k] + Dh[qp, kp] * pt[kpq_idx, k] * undel[qp]

    # Transpose x
    x = x.T

    # Second loop: x(k,kp) = x(k,kp) + De(q,k) * pp(k_p_q(q,k),kp) * undel(q)
    for kp in range(Nk):
        for k in range(Nk):
            for q in range(Nk):
                kpq_idx = k_p_q[q, k] + 1  # Convert to extended array index
                if 1 <= kpq_idx < len(pp) - 1 and 0 <= kp < Nk:
                    x[k, kp] = x[k, kp] + De[q, k] * pp[kpq_idx, kp] * undel[q]

    return x * ii_val * hbar_val


def OffDiagDephasing(ne, nh, p, ky, Ee, Eh, g, VC, x):
    """
    Calculate off-diagonal dephasing contribution.

    Computes the off-diagonal dephasing contribution to the polarization equation
    due to carrier-carrier interactions.

    Parameters
    ----------
    ne : ndarray
        Electron occupation numbers (complex), 1D array
    nh : ndarray
        Hole occupation numbers (complex), 1D array
    p : ndarray
        Polarization matrix (complex), 2D array
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Ee : ndarray
        Electron energies (J), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    g : ndarray
        Dephasing rates (Hz), 1D array [gee, ghh, geh]
    VC : ndarray
        Interaction matrix (J), 3D array
    x : ndarray
        Output dephasing contribution (modified in-place), 2D array

    Returns
    -------
    None
        x array is modified in-place.

    Notes
    -----
    The function computes dephasing from electron-electron, hole-hole,
    and electron-hole interactions. Note that the matrix k_p_q is symmetric.
    """
    global _k_p_q

    if _k_p_q is None:
        raise RuntimeError("InitializeDephasing must be called first")

    Nk = len(ky)
    x[:, :] = 0.0

    undel = np.abs(ky) / (np.abs(ky) + 1e-10)

    De = CalcOffDiagDeph_E(ne, nh, ky, Ee, Eh, g[0], g[2], VC)
    Dh = CalcOffDiagDeph_H(ne, nh, ky, Ee, Eh, g[1], g[2], VC)
    De = De.T
    Dh = Dh.T

    # Extended arrays for p (0:size+1 in Fortran)
    pp = np.zeros((Nk + 2, Nk + 2), dtype=np.complex128)
    pt = np.zeros((Nk + 2, Nk + 2), dtype=np.complex128)

    # Note: p is transposed compared to old code (Feb 08 2023)
    pt[1:Nk+1, 1:Nk+1] = p[:, :]
    pp[1:Nk+1, 1:Nk+1] = p.T[:, :]

    x[:, :] = _OffDiagDephasing_jit(Dh, De, pt, pp, _k_p_q, undel, ii, hbar)

############
# ATTEMPT #2 FOR OFF-DIAGONAL DEPHASING
############


@jit(nopython=True)
def _Lrtz_jit(a, b, pi_val):
    """JIT-compiled version of Lrtz."""
    return (b / pi_val) / (a**2 + b**2)


@jit(nopython=True)
def _CalcOffDiagDeph_E2_jit(ne, nh, Vsq, Ee, Eh, gee, geh, pi_val, hbar_val):
    """JIT-compiled version of CalcOffDiagDeph_E2."""
    Nk = len(ne)
    # D array with negative indexing: D(-Nk:Nk, Nk) -> D[0:2*Nk+1, Nk]
    # Index mapping: q -> q + Nk (so q=-Nk maps to 0, q=0 maps to Nk, q=Nk maps to 2*Nk)
    D = np.zeros((2 * Nk + 1, Nk))

    # Electron-electron dephasing
    # Fortran: do q=max(p-Nk,1-k), min(p-1,Nk-k)
    # Convert to 0-based: q from max(p-Nk, 0-k) to min(p-1, Nk-k-1), but q can be negative
    for k in range(Nk):
        for p in range(Nk):
            # Fortran bounds: max(p-Nk, 1-k) to min(p-1, Nk-k)
            # In 0-based, we allow negative q, so: max(p-Nk, -k) to min(p-1, Nk-k-1)
            q_min = max(p - Nk, -k)
            q_max = min(p - 1, Nk - k - 1)
            for q in range(q_min, q_max + 1):
                if -Nk <= q <= Nk:
                    q_arr_idx = q + Nk  # Map to array index (q=-Nk -> 0, q=0 -> Nk, q=Nk -> 2*Nk)
                    k_plus_q = k + q
                    p_minus_q = p - q
                    if 0 <= k_plus_q < Nk and 0 <= p_minus_q < Nk:
                        E_diff = Ee[k_plus_q] + Ee[p_minus_q] - Ee[p] - Ee[k]
                        D[q_arr_idx, k] = (D[q_arr_idx, k] +
                                          Vsq[q_arr_idx, 1] *  # Index 1 for Vee2 (0-based)
                                          _Lrtz_jit(E_diff, hbar_val * gee, pi_val) *
                                          (ne[p_minus_q] * (1.0 - ne[p]) * (1.0 - ne[k]) +
                                           (1.0 - ne[p_minus_q]) * ne[p] * ne[k]))

    # Electron-hole dephasing
    # Fortran: do q=max(1-p,1-k), min(Nk-p,Nk-k)
    # Convert to 0-based: q from max(-p, -k) to min(Nk-p-1, Nk-k-1)
    for k in range(Nk):
        for p in range(Nk):
            # Fortran bounds: max(1-p, 1-k) to min(Nk-p, Nk-k)
            # In 0-based: max(-p, -k) to min(Nk-p-1, Nk-k-1)
            q_min = max(-p, -k)
            q_max = min(Nk - p - 1, Nk - k - 1)
            for q in range(q_min, q_max + 1):
                if -Nk <= q <= Nk:
                    q_arr_idx = q + Nk  # Map to array index
                    k_plus_q = k + q
                    p_plus_q = p + q
                    if 0 <= k_plus_q < Nk and 0 <= p_plus_q < Nk:
                        E_diff = Ee[k_plus_q] + Eh[p_plus_q] - Eh[p] - Ee[k]
                        D[q_arr_idx, k] = (D[q_arr_idx, k] +
                                          Vsq[q_arr_idx, 0] *  # Index 0 for Veh2 (0-based)
                                          _Lrtz_jit(E_diff, hbar_val * geh, pi_val) *
                                          (nh[p_plus_q] * (1.0 - nh[p]) * (1.0 - ne[k]) +
                                           (1.0 - nh[p_plus_q]) * nh[p] * ne[k]))

    return D * pi_val / hbar_val


def CalcOffDiagDeph_E2(ne, nh, ky, Ee, Eh, gee, geh, VC, Nk):  # noqa: ARG001
    """
    Calculate off-diagonal dephasing matrix for electrons (version 2).

    Computes the off-diagonal dephasing matrix D(q,k) for electrons using
    a different algorithm with negative indexing support.

    Parameters
    ----------
    ne : ndarray
        Electron occupation numbers (complex), 1D array
    nh : ndarray
        Hole occupation numbers (complex), 1D array
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Ee : ndarray
        Electron energies (J), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    gee : float
        Electron-electron dephasing rate (Hz)
    geh : float
        Electron-hole dephasing rate (Hz)
    VC : ndarray
        Interaction matrix (J), 3D array VC[:,:,0] for e-h, VC[:,:,1] for e-e
    Nk : int
        Number of momentum points

    Returns
    -------
    ndarray
        Off-diagonal dephasing matrix (J), 2D array of shape (2*Nk+1, Nk)
        First dimension uses shifted indexing: D[q+Nk, k] corresponds to D(q,k) where q can be negative
    """
    # Vsq array with negative indexing: Vsq(-Nk:Nk, 3) -> Vsq[0:2*Nk+1, 3]
    Vsq = np.zeros((2 * Nk + 1, 3))

    # Set up Vsq: Vsq(+q,:) = VC(1+q,1,:)**2 and Vsq(-q,:) = VC(1+q,1,:)**2
    # Note: q=0 is left as zero
    for q in range(1, Nk):
        # Positive q: Vsq[q+Nk, :] = VC[q, 0, :]**2 (0-based indexing)
        Vsq[q + Nk, :] = VC[q, 0, :]**2
        # Negative q: Vsq[-q+Nk, :] = VC[q, 0, :]**2
        Vsq[-q + Nk, :] = VC[q, 0, :]**2

    # Convert complex arrays to real
    ne_real = np.abs(ne)
    nh_real = np.abs(nh)

    return _CalcOffDiagDeph_E2_jit(ne_real, nh_real, Vsq, Ee, Eh, gee, geh, pi, hbar)


@jit(nopython=True)
def _CalcOffDiagDeph_H2_jit(ne, nh, Vsq, Ee, Eh, ghh, geh, pi_val, hbar_val):
    """JIT-compiled version of CalcOffDiagDeph_H2."""
    Nk = len(ne)
    # D array with negative indexing: D(-Nk:Nk, Nk) -> D[0:2*Nk+1, Nk]
    D = np.zeros((2 * Nk + 1, Nk))

    # Hole-hole dephasing
    # Fortran: do q=max(p-Nk,1-k), min(p-1,Nk-k)
    for k in range(Nk):
        for p in range(Nk):
            q_min = max(p - Nk, -k)
            q_max = min(p - 1, Nk - k - 1)
            for q in range(q_min, q_max + 1):
                if -Nk <= q <= Nk:
                    q_arr_idx = q + Nk
                    k_plus_q = k + q
                    p_minus_q = p - q
                    if 0 <= k_plus_q < Nk and 0 <= p_minus_q < Nk:
                        E_diff = Eh[k_plus_q] + Eh[p_minus_q] - Eh[p] - Eh[k]
                        D[q_arr_idx, k] = (D[q_arr_idx, k] +
                                          Vsq[q_arr_idx, 2] *  # Index 2 for Vhh2 (0-based)
                                          _Lrtz_jit(E_diff, hbar_val * ghh, pi_val) *
                                          (nh[p_minus_q] * (1.0 - nh[p]) * (1.0 - nh[k]) +
                                           (1.0 - nh[p_minus_q]) * nh[p] * nh[k]))

    # Electron-hole dephasing
    # Fortran: do q=max(1-p,1-k), min(Nk-p,Nk-k)
    for k in range(Nk):
        for p in range(Nk):
            q_min = max(-p, -k)
            q_max = min(Nk - p - 1, Nk - k - 1)
            for q in range(q_min, q_max + 1):
                if -Nk <= q <= Nk:
                    q_arr_idx = q + Nk
                    k_plus_q = k + q
                    p_plus_q = p + q
                    if 0 <= k_plus_q < Nk and 0 <= p_plus_q < Nk:
                        E_diff = Eh[k_plus_q] + Ee[p_plus_q] - Ee[p] - Eh[k]
                        D[q_arr_idx, k] = (D[q_arr_idx, k] +
                                          Vsq[q_arr_idx, 0] *  # Index 0 for Veh2 (0-based)
                                          _Lrtz_jit(E_diff, hbar_val * geh, pi_val) *
                                          (ne[p_plus_q] * (1.0 - ne[p]) * (1.0 - nh[k]) +
                                           (1.0 - ne[p_plus_q]) * ne[p] * nh[k]))

    return D * pi_val / hbar_val


def CalcOffDiagDeph_H2(ne, nh, ky, Ee, Eh, ghh, geh, VC, Nk):  # noqa: ARG001
    """
    Calculate off-diagonal dephasing matrix for holes (version 2).

    Computes the off-diagonal dephasing matrix D(q,k) for holes using
    a different algorithm with negative indexing support.

    Parameters
    ----------
    ne : ndarray
        Electron occupation numbers (complex), 1D array
    nh : ndarray
        Hole occupation numbers (complex), 1D array
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Ee : ndarray
        Electron energies (J), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    ghh : float
        Hole-hole dephasing rate (Hz)
    geh : float
        Electron-hole dephasing rate (Hz)
    VC : ndarray
        Interaction matrix (J), 3D array VC[:,:,0] for e-h, VC[:,:,2] for h-h
    Nk : int
        Number of momentum points

    Returns
    -------
    ndarray
        Off-diagonal dephasing matrix (J), 2D array of shape (2*Nk+1, Nk)
        First dimension uses shifted indexing: D[q+Nk, k] corresponds to D(q,k) where q can be negative
    """
    # Vsq array with negative indexing: Vsq(-Nk:Nk, 3) -> Vsq[0:2*Nk+1, 3]
    Vsq = np.zeros((2 * Nk + 1, 3))

    # Set up Vsq: Vsq(+q,:) = VC(1+q,1,:)**2 and Vsq(-q,:) = VC(1+q,1,:)**2
    for q in range(1, Nk):
        Vsq[q + Nk, :] = VC[q, 0, :]**2
        Vsq[-q + Nk, :] = VC[q, 0, :]**2

    # Convert complex arrays to real
    ne_real = np.abs(ne)
    nh_real = np.abs(nh)

    return _CalcOffDiagDeph_H2_jit(ne_real, nh_real, Vsq, Ee, Eh, ghh, geh, pi, hbar)


@jit(nopython=True)
def _OffDiagDephasing2_jit(Dh, De, p, pt, undel, ii_val, hbar_val):
    """JIT-compiled version of OffDiagDephasing2 inner loops."""
    Nk = p.shape[0]
    x = np.zeros((Nk, Nk), dtype=np.complex128)

    # First loop: x(kh,ke) = x(kh,ke) + Dh(q,kh) * p(kh+q, ke) * undel(q)
    # Fortran: do q=1-kh, Nk-kh
    # Convert to 0-based: q from -kh to Nk-kh-1
    for ke in range(Nk):
        for kh in range(Nk):
            q_min = -kh
            q_max = Nk - kh - 1
            for q in range(q_min, q_max + 1):
                if -Nk <= q <= Nk:
                    q_arr_idx = q + Nk  # Map to D array index
                    kh_plus_q = kh + q
                    if 0 <= kh_plus_q < Nk and 0 <= q_arr_idx < Dh.shape[0]:
                        x[kh, ke] = x[kh, ke] + Dh[q_arr_idx, kh] * p[kh_plus_q, ke] * undel[q_arr_idx]

    # Second loop: x(kh,ke) = x(kh,ke) + De(q,ke) * pt(ke+q,kh) * undel(q)
    # Fortran: do q=1-ke, Nk-ke
    # Convert to 0-based: q from -ke to Nk-ke-1
    for ke in range(Nk):
        for kh in range(Nk):
            q_min = -ke
            q_max = Nk - ke - 1
            for q in range(q_min, q_max + 1):
                if -Nk <= q <= Nk:
                    q_arr_idx = q + Nk  # Map to D array index
                    ke_plus_q = ke + q
                    if 0 <= ke_plus_q < Nk and 0 <= q_arr_idx < De.shape[0]:
                        x[kh, ke] = x[kh, ke] + De[q_arr_idx, ke] * pt[ke_plus_q, kh] * undel[q_arr_idx]

    return x * ii_val * hbar_val


def OffDiagDephasing2(ne, nh, p, ky, Ee, Eh, g, VC, t, x):
    """
    Calculate off-diagonal dephasing contribution (version 2).

    Computes the off-diagonal dephasing contribution to the polarization equation
    using version 2 algorithm with negative indexing support.

    Parameters
    ----------
    ne : ndarray
        Electron occupation numbers (complex), 1D array
    nh : ndarray
        Hole occupation numbers (complex), 1D array
    p : ndarray
        Polarization matrix (complex), 2D array
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Ee : ndarray
        Electron energies (J), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    g : ndarray
        Dephasing rates (Hz), 1D array [gee, ghh, geh]
    VC : ndarray
        Interaction matrix (J), 3D array
    t : float
        Time (s)
    x : ndarray
        Output dephasing contribution (modified in-place), 2D array

    Returns
    -------
    None
        x array is modified in-place.

    Notes
    -----
    The function computes dephasing from electron-electron, hole-hole,
    and electron-hole interactions. Writes max/min values to output files.
    """
    global _fe_file, _fh_file

    Nk = len(ky)
    x[:, :] = 0.0

    # undel array with negative indexing: undel(-Nk:Nk) -> undel[0:2*Nk+1]
    undel = np.ones(2 * Nk + 1)
    undel[Nk] = 0.0  # undel(0) = 0

    De = CalcOffDiagDeph_E2(ne, nh, ky, Ee, Eh, g[0], g[2], VC, Nk)
    Dh = CalcOffDiagDeph_H2(ne, nh, ky, Ee, Eh, g[1], g[2], VC, Nk)

    pt = p.T

    x[:, :] = _OffDiagDephasing2_jit(Dh, De, p, pt, undel, ii, hbar)

    # Write max/min values to files
    if _fe_file is not None:
        _fe_file.write(f"{np.real(t)} {np.max(np.real(De))} {np.min(np.real(De))}\n")
        _fe_file.flush()
    if _fh_file is not None:
        _fh_file.write(f"{np.real(t)} {np.max(np.real(Dh))} {np.min(np.real(Dh))}\n")
        _fh_file.flush()

#######################################################################
# printing statements for sbes.f90
#######################################################################

def printGam(Dx, z, n, file):
    """
    Print dephasing rate to file with index number.

    Writes a real 1Dvxx dephasing rate array to a file with index number in filename.

    Parameters
    ----------
    Dx : ndarray
        Dephasing rate array (real), 1D array
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
    The file is written to 'dataQW/{file}{n:05d}.dat'.
    Each line contains: z(i), Dx(i)
    """
    filename = f'dataQW/{file}{n:05d}.dat'
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(len(z)):
            f.write(f"{np.float32(z[i])} {np.float32(Dx[i])}\n")


def WriteDephasing(ky, gamE, gamH, w, xxx):
    """
    Write dephasing rates to files.

    Writes electron and hole dephasing rates to files with wire index.

    Parameters
    ----------
    ky : ndarray
        Quantum wire momentum coordinates (1/m), 1D array
    gamE : ndarray
        Electron diagonal dephasing rate (Hz), 1D array
    gamH : ndarray
        Hole diagonal dephasing rate (Hz), 1D array
    w : int
        Wire index
    xxx : int
        Time index

    Returns
    -------
    None

    Notes
    -----
    Writes files to:
    - 'dataQW/Wire/Ge/Ge.{w:02d}.k.{xxx:05d}.dat' for electrons
    - 'dataQW/Wire/Gh/Gh.{w:02d}.k.{xxx:05d}.dat' for holes
    """
    wire = f"{w:02d}"

    printGam(gamE, ky, xxx, f'Wire/Ge/Ge.{wire}.k.')
    printGam(gamH, ky, xxx, f'Wire/Gh/Gh.{wire}.k.')

