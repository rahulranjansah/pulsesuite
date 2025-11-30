"""
Host material polarization calculations for quantum wire simulations.

This module calculates the host material polarization response
for propagation simulations.

Converted from phost.f90 (lines 33-653).
"""

import numpy as np
from scipy.constants import e as e0, epsilon_0 as eps0, m_e as me0, c as c0_SI
from numba import jit
import os
import pyfftw
pyfftw.interfaces.cache.enable()

# Physical constants
eps0_val = eps0
c0 = c0_SI
me0_val = me0
pi = np.pi
twopi = 2.0 * np.pi
ii = 1j  # Imaginary unit

# Module-level state variables (matching Fortran module variables)
_osc = 2
_q = -e0
_w = None
_gam = None
_B = None
_C = None
_chi1 = None
_Nf = None
_A0 = 0.0
_chi3 = 0.0 + 0.0j
_epsr_0 = 10.0
_epsr_infty = 8.2
_w1 = 0.0
_w2 = 1e20
_w0 = 0.0
_lambda0 = 0.0
_N1 = 0
_N2 = 0
_material = 'AlAs'

# Allocatable arrays
_Px_before = None
_Py_before = None
_Px_now = None
_Py_now = None
_Px_after = None
_Py_after = None
_omega_q = None
_EpsrWq = None




def CalcPHost(Ex, Ey, dt, m, epsb, Px, Py):
    """
    Calculate host material polarization.

    Computes the host material polarization response using a time-stepping
    scheme with previous, current, and next polarization values.

    Parameters
    ----------
    Ex : ndarray
        Electric field x-component, shape (N1, N2), complex
    Ey : ndarray
        Electric field y-component, shape (N1, N2), complex
    dt : float
        Time step (s)
    m : int
        Time step index (unused, kept for interface compatibility)
    epsb : float
        Dielectric constant (modified in-place)
    Px : ndarray
        Polarization x-component (modified in-place), shape (N1, N2), complex
    Py : ndarray
        Polarization y-component (modified in-place), shape (N1, N2), complex

    Returns
    -------
    None

    Notes
    -----
    Modifies epsb, Px, Py in-place.
    Uses module-level variables _Px_before, _Px_now, _Px_after, etc.
    The polarization is summed over the oscillator dimension (axis 2).
    """
    global _Px_before, _Py_before, _Px_now, _Py_now, _Px_after, _Py_after, _A0

    _Px_before = _Px_now.copy() if _Px_now is not None else np.zeros((Ex.shape[0], Ex.shape[1], _osc), dtype=complex)
    _Py_before = _Py_now.copy() if _Py_now is not None else np.zeros((Ey.shape[0], Ey.shape[1], _osc), dtype=complex)

    _Px_now = _Px_after.copy() if _Px_after is not None else np.zeros((Ex.shape[0], Ex.shape[1], _osc), dtype=complex)
    _Py_now = _Py_after.copy() if _Py_after is not None else np.zeros((Ey.shape[0], Ey.shape[1], _osc), dtype=complex)

    _Px_after = CalcNextP(_Px_before, _Px_now, Ex, dt)
    _Py_after = CalcNextP(_Py_before, _Py_now, Ey, dt)

    epsb = _A0
    Px[:, :] = np.sum(_Px_after, axis=2)
    Py[:, :] = np.sum(_Py_after, axis=2)


def CalcPHostOld(Ex, Ey, dt, m, epsb, Px, Py):
    """
    Calculate host material polarization (old version).

    Computes the host material polarization response using a time-stepping
    scheme with different initialization depending on the time step index m.

    Parameters
    ----------
    Ex : ndarray
        Electric field x-component, shape (N1, N2), complex
    Ey : ndarray
        Electric field y-component, shape (N1, N2), complex
    dt : float
        Time step (s)
    m : int
        Time step index
    epsb : float
        Dielectric constant (modified in-place)
    Px : ndarray
        Polarization x-component (modified in-place), shape (N1, N2), complex
    Py : ndarray
        Polarization y-component (modified in-place), shape (N1, N2), complex

    Returns
    -------
    None

    Notes
    -----
    Modifies epsb, Px, Py in-place.
    Uses module-level variables _Px_before, _Px_now, _Px_after, etc.
    The behavior depends on m:
    - m > 2: Use previous values and calculate next
    - m >= 2: Initialize with monochromatic polarization, then calculate next
    - m < 2: Initialize with monochromatic polarization, set next to zero
    """
    global _Px_before, _Py_before, _Px_now, _Py_now, _Px_after, _Py_after, _A0, _w0

    _Px_before = _Px_now.copy() if _Px_now is not None else np.zeros((Ex.shape[0], Ex.shape[1], _osc), dtype=complex)
    _Py_before = _Py_now.copy() if _Py_now is not None else np.zeros((Ey.shape[0], Ey.shape[1], _osc), dtype=complex)

    if m > 2:
        _Px_now = _Px_after.copy() if _Px_after is not None else np.zeros((Ex.shape[0], Ex.shape[1], _osc), dtype=complex)
        _Px_after = CalcNextP(_Px_before, _Px_now, Ex, dt)

        _Py_now = _Py_after.copy() if _Py_after is not None else np.zeros((Ey.shape[0], Ey.shape[1], _osc), dtype=complex)
        _Py_after = CalcNextP(_Py_before, _Py_now, Ey, dt)

        epsb = _A0
    elif m >= 2:
        _Px_now = CalcMonoP(Ex)
        _Px_after = CalcNextP(_Px_before, _Px_now, Ex, dt)

        _Py_now = CalcMonoP(Ey)
        _Py_after = CalcNextP(_Py_before, _Py_now, Ey, dt)

        epsb = _A0
    else:
        _Px_now = CalcMonoP(Ex)
        _Px_after = np.zeros_like(_Px_now)

        _Py_now = CalcMonoP(Ey)
        _Py_after = np.zeros_like(_Py_now)

        epsb = np.real(nw2_no_gam(_w0))

    Px[:, :] = np.sum(_Px_after, axis=2)
    Py[:, :] = np.sum(_Py_after, axis=2)




@jit(nopython=True)
def _CalcNextP_jit(E_size1, E_size2, osc, P1, P2, E, dt, gam, w, B, eps0_val):
    """JIT-compiled version of CalcNextP."""
    CalcNextP = np.zeros((E_size1, E_size2, osc), dtype=np.complex128)
    f1 = np.zeros(osc)
    f2 = np.zeros(osc)
    f3 = np.zeros(osc)

    for n in range(osc):
        f1[n] = -(1.0 - gam[n] * dt) / (gam[n] * dt + 1.0)
        f2[n] = (2.0 - w[n]**2 * dt**2) / (gam[n] * dt + 1.0)
        f3[n] = (B[n] * w[n]**2 * dt**2) / (gam[n] * dt + 1.0) * eps0_val

    for n in range(osc):
        for j in range(E_size2):
            for i in range(E_size1):
                CalcNextP[i, j, n] = f1[n] * P1[i, j, n] + f2[n] * P2[i, j, n] + f3[n] * E[i, j]

    return CalcNextP


def CalcNextP(P1, P2, E, dt):
    """
    Calculate next polarization value.

    Computes the next polarization value using a finite difference scheme
    for the oscillator model.

    Parameters
    ----------
    P1 : ndarray
        Previous polarization values, shape (N1, N2, osc), complex
    P2 : ndarray
        Current polarization values, shape (N1, N2, osc), complex
    E : ndarray
        Electric field, shape (N1, N2), complex
    dt : float
        Time step (s)

    Returns
    -------
    ndarray
        Next polarization values, shape (N1, N2, osc), complex

    Notes
    -----
    Uses module-level variables _gam, _w, _B, _osc.
    The function computes:
    CalcNextP = f1 * P1 + f2 * P2 + f3 * E
    where f1, f2, f3 are frequency-dependent coefficients.
    """
    global _gam, _w, _B, _osc

    E_size1, E_size2 = E.shape

    try:
        return _CalcNextP_jit(E_size1, E_size2, _osc, P1, P2, E, dt, _gam, _w, _B, eps0_val)
    except Exception:
        # Fallback to pure Python
        CalcNextP_result = np.zeros((E_size1, E_size2, _osc), dtype=complex)
        f1 = np.zeros(_osc)
        f2 = np.zeros(_osc)
        f3 = np.zeros(_osc)

        for n in range(_osc):
            f1[n] = -(1.0 - _gam[n] * dt) / (_gam[n] * dt + 1.0)
            f2[n] = (2.0 - _w[n]**2 * dt**2) / (_gam[n] * dt + 1.0)
            f3[n] = (_B[n] * _w[n]**2 * dt**2) / (_gam[n] * dt + 1.0) * eps0_val

        for n in range(_osc):
            for j in range(E_size2):
                for i in range(E_size1):
                    CalcNextP_result[i, j, n] = f1[n] * P1[i, j, n] + f2[n] * P2[i, j, n] + f3[n] * E[i, j]

        return CalcNextP_result


@jit(nopython=True)
def _CalcMonoP_jit(E_size1, E_size2, osc, E, chi1_real, eps0_val):
    """JIT-compiled version of CalcMonoP."""
    CalcMonoP = np.zeros((E_size1, E_size2, osc), dtype=np.complex128)

    for n in range(osc):
        for j in range(E_size2):
            for i in range(E_size1):
                CalcMonoP[i, j, n] = eps0_val * E[i, j] * chi1_real[n]

    return CalcMonoP


def CalcMonoP(E):
    """
    Calculate monochromatic polarization.

    Computes the polarization for a monochromatic electric field
    using the linear susceptibility.

    Parameters
    ----------
    E : ndarray
        Electric field, shape (N1, N2), complex

    Returns
    -------
    ndarray
        Polarization values, shape (N1, N2, osc), complex

    Notes
    -----
    Uses module-level variables _chi1, _osc.
    The function computes:
    CalcMonoP = eps0 * E * real(chi1)
    """
    global _chi1, _osc

    E_size1, E_size2 = E.shape

    try:
        chi1_real = np.real(_chi1)
        return _CalcMonoP_jit(E_size1, E_size2, _osc, E, chi1_real, eps0_val)
    except Exception:
        # Fallback to pure Python
        CalcMonoP_result = np.zeros((E_size1, E_size2, _osc), dtype=complex)

        for n in range(_osc):
            for j in range(E_size2):
                for i in range(E_size1):
                    CalcMonoP_result[i, j, n] = eps0_val * E[i, j] * np.real(_chi1[n])

        return CalcMonoP_result

def SetHostMaterial(host, mat, lam, epsr, n0):
    """
    Set host material parameters.

    Initializes the host material parameters based on the material type
    and wavelength.

    Parameters
    ----------
    host : bool
        Whether to use host material dispersion
    mat : str
        Material name ('AlAs', 'fsil', 'GaAs', 'none')
    lam : float
        Wavelength (m)
    epsr : float
        Dielectric constant (modified in-place)
    n0 : float
        Refractive index (modified in-place)

    Returns
    -------
    None

    Notes
    -----
    Modifies epsr and n0 in-place.
    Sets module-level variables based on material type.
    Calls SetParamsAlAs, SetParamsSilica, SetParamsGaAs, or SetParamsNone
    based on material type, and WriteHostDispersion if host=True.
    """
    global _lambda0, _material, _chi1, _w0, _osc, _B, _C

    _lambda0 = lam

    mat_trimmed = mat.strip()
    if mat_trimmed == 'AlAs':
        SetParamsAlAs()
    elif mat_trimmed == 'fsil':
        SetParamsSilica()
    elif mat_trimmed == 'GaAs':
        SetParamsGaAs()
    elif mat_trimmed == 'none':
        SetParamsNone()
    else:
        print(f"ERROR: Host Material = {mat} Is Not Included In phost.f90 Code")
        raise ValueError(f"Unknown material: {mat}")

    _material = mat_trimmed

    if _B is not None and _C is not None:
        _chi1 = np.zeros(_osc, dtype=complex)
        _chi1 = _B * lam**2 / (lam**2 - _C)
    else:
        _chi1 = np.zeros(_osc, dtype=complex)

    _w0 = twopi * c0 / lam

    if host:
        epsr_val = np.real(nw2_no_gam(_w0))
        n0_val = np.real(np.sqrt(epsr_val))
        epsr = epsr_val
        n0 = n0_val
        WriteHostDispersion()

    # Note: Fortran always executes epsr = n0**2 at the end
    # This matches Fortran behavior (always normalize to n0**2)
    epsr = n0**2


def InitializeHost(Nx, Ny, n0, qsq, host):
    """
    Initialize host material arrays.

    Allocates and initializes arrays for host material calculations.

    Parameters
    ----------
    Nx : int
        Number of points in x-direction
    Ny : int
        Number of points in y-direction
    n0 : float
        Refractive index
    qsq : ndarray
        Squared momentum array, shape (Nx, Ny), complex
    host : bool
        Whether to use host material dispersion

    Returns
    -------
    None

    Notes
    -----
    Sets module-level arrays _omega_q, _EpsrWq, and polarization arrays.
    When host=False, _EpsrWq is set to an array with all elements equal to n0**2.
    """
    global _omega_q, _EpsrWq, _Px_before, _Py_before, _Px_now, _Py_now, _Px_after, _Py_after, _osc

    _omega_q = np.zeros((Nx, Ny), dtype=complex)
    _EpsrWq = np.zeros((Nx, Ny), dtype=complex)

    if host:
        _Px_before = np.zeros((Nx, Ny, _osc), dtype=complex)
        _Px_now = np.zeros((Nx, Ny, _osc), dtype=complex)
        _Px_after = np.zeros((Nx, Ny, _osc), dtype=complex)
        _Py_before = np.zeros((Nx, Ny, _osc), dtype=complex)
        _Py_now = np.zeros((Nx, Ny, _osc), dtype=complex)
        _Py_after = np.zeros((Nx, Ny, _osc), dtype=complex)

        _omega_q = np.zeros((Nx, Ny), dtype=complex)
        CalcWq(np.sqrt(qsq))
        CalcEpsrWq(np.sqrt(qsq))
    else:
        _omega_q = np.sqrt(np.real(qsq)) * c0 / n0
        _EpsrWq = np.full((Nx, Ny), n0**2, dtype=complex)  # Array assignment (broadcasts scalar to array)


def CalcWq(q):
    """
    Calculate frequency from momentum.

    Computes the frequency omega_q from momentum q using the
    dispersion relation.

    Parameters
    ----------
    q : ndarray
        Momentum array, shape (N1, N2), complex

    Returns
    -------
    None

    Notes
    -----
    Modifies module-level variable _omega_q.
    Writes output to 'fields/host/w.q.dat'.
    """
    global _omega_q, _w0

    n0 = np.sqrt(nw2_no_gam(_w0))
    np0 = nwp_no_gam(_w0)
    x = n0 - np0 * _w0

    _omega_q = (-x + np.sqrt(x**2 + 4 * np0 * q * c0)) / (2 * np0)

    os.makedirs('fields/host', exist_ok=True)
    with open('fields/host/w.q.dat', 'w', encoding='utf-8') as f:
        j_max = max(q.shape[1] // 2, 1)
        i_max = max(q.shape[0] // 2, 1)
        for j in range(j_max):
            for i in range(i_max):
                f.write(f"{np.real(q[i, j]) * 1e-7} {np.real(_omega_q[i, j]) * 1e-15} {np.imag(_omega_q[i, j]) * 1e-15}\n")


def CalcEpsrWq(q):
    """
    Calculate dielectric constant as function of frequency.

    Computes the dielectric constant for each frequency in omega_q.

    Parameters
    ----------
    q : ndarray
        Momentum array, shape (N1, N2), complex (unused, kept for interface compatibility)

    Returns
    -------
    None

    Notes
    -----
    Modifies module-level variable _EpsrWq.
    Uses DetermineCoeffs and CalcEpsrWq_ij.
    """
    global _EpsrWq, _omega_q

    aw = np.zeros(2)
    bw = np.zeros(2)
    DetermineCoeffs(aw, bw)

    for j in range(_omega_q.shape[1]):
        for i in range(_omega_q.shape[0]):
            _EpsrWq[i, j] = CalcEpsrWq_ij(np.abs(_omega_q[i, j]), aw, bw)


def CalcEpsrWq_ij(w_ij, aw, bw):
    """
    Calculate dielectric constant for a single frequency.

    Computes the dielectric constant using piecewise interpolation
    or the full model depending on frequency range.

    Parameters
    ----------
    w_ij : float
        Frequency (Hz)
    aw : ndarray
        Low-frequency expansion coefficients, shape (2)
    bw : ndarray
        High-frequency expansion coefficients, shape (2)

    Returns
    -------
    complex
        Dielectric constant value

    Notes
    -----
    Uses module-level variables _epsr_0, _epsr_infty, _w1, _w2.
    Note: In Fortran, this is a subroutine with inout parameter;
    in Python, it's a function that returns the value (functionally equivalent).
    """
    global _epsr_0, _epsr_infty, _w1, _w2

    if w_ij < _w1:
        return _epsr_0 + aw[0] * w_ij**2 + aw[1] * w_ij**3
    elif w_ij > _w2:
        return _epsr_infty + bw[0] / w_ij**2 + bw[1] / w_ij**3
    else:
        return nw2_no_gam(w_ij)


def DetermineCoeffs(aw, bw):
    """
    Determine expansion coefficients.

    Computes the expansion coefficients for low and high frequency
    approximations of the dielectric constant.

    Parameters
    ----------
    aw : ndarray
        Low-frequency expansion coefficients (modified in-place), shape (2)
    bw : ndarray
        High-frequency expansion coefficients (modified in-place), shape (2)

    Returns
    -------
    None

    Notes
    -----
    Modifies aw and bw in-place.
    Uses module-level variables _w1, _w2, _epsr_0, _epsr_infty.
    """
    global _w1, _w2, _epsr_0, _epsr_infty

    e1 = nw2_no_gam(_w1)
    e2 = nw2_no_gam(_w2)

    ep1 = epsrwp_no_gam(_w1)
    ep2 = epsrwp_no_gam(_w2)

    aw[0] = + 3.0 / _w1**2 * (e1 - _epsr_0) - ep1 / _w1
    aw[1] = - 2.0 / _w1**3 * (e1 - _epsr_0) + ep1 / _w1**2

    bw[0] = + 3.0 * _w2**2 * (e2 - _epsr_infty) + ep2 * _w2**3
    bw[1] = - 2.0 * _w2**3 * (e2 - _epsr_infty) - ep2 * _w2**4


def Epsr_q(q):
    """
    Get dielectric constant array.

    Returns the dielectric constant array as a function of momentum.

    Parameters
    ----------
    q : ndarray
        Momentum array, shape (N1, N2), complex (unused, kept for interface compatibility)

    Returns
    -------
    ndarray
        Dielectric constant array, shape (N1, N2), complex

    Notes
    -----
    Returns module-level variable _EpsrWq.
    """
    global _EpsrWq
    return _EpsrWq.copy()


def Epsr_qij(i, j):
    """
    Get dielectric constant at specific indices.

    Returns the dielectric constant at indices (i, j).

    Parameters
    ----------
    i : int
        First index
    j : int
        Second index

    Returns
    -------
    complex
        Dielectric constant value at (i, j)

    Notes
    -----
    Returns value from module-level variable _EpsrWq.
    """
    global _EpsrWq
    return _EpsrWq[i, j]


def FDTD_Dispersion(qx, qy, dx, dy, dt, n0):
    """
    Calculate FDTD dispersion relation.

    Computes the frequency from momentum using the FDTD dispersion relation.

    Parameters
    ----------
    qx : ndarray
        x-component of momentum (1/m), 1D array
    qy : ndarray
        y-component of momentum (1/m), 1D array
    dx : float
        Spatial step in x-direction (m)
    dy : float
        Spatial step in y-direction (m)
    dt : float
        Time step (s)
    n0 : float
        Refractive index

    Returns
    -------
    None

    Notes
    -----
    Modifies module-level variable _omega_q.
    Uses Sin and ASin functions (np.sin and np.arcsin).
    """
    global _omega_q

    _omega_q = np.zeros((len(qx), len(qy)), dtype=complex)

    for j in range(len(qy)):
        for i in range(len(qx)):
            _omega_q[i, j] = np.sqrt(np.sin(qx[i] * dx / 2.0)**2 / dx**2 +
                                     np.sin(qy[j] * dy / 2.0)**2 / dy**2)
            _omega_q[i, j] = 2.0 / dt * np.arcsin((c0 / n0) * dt * np.real(_omega_q[i, j]))


def wq(i, j):
    """
    Get frequency at specific indices.

    Returns the frequency at indices (i, j).

    Parameters
    ----------
    i : int
        First index
    j : int
        Second index

    Returns
    -------
    complex
        Frequency value at (i, j)

    Notes
    -----
    Returns value from module-level variable _omega_q.
    """
    global _omega_q
    return _omega_q[i, j]


def SetInitialP(Ex, Ey, qx, qy, qsq, dt, Px, Py, epsb):
    """
    Set initial polarization values.

    Initializes the polarization arrays for the first time step.

    Parameters
    ----------
    Ex : ndarray
        Electric field x-component (modified in-place), shape (N1, N2), complex
    Ey : ndarray
        Electric field y-component (modified in-place), shape (N1, N2), complex
    qx : ndarray
        x-component of momentum (1/m), 1D array
    qy : ndarray
        y-component of momentum (1/m), 1D array
    qsq : ndarray
        Squared momentum array, shape (N1, N2), complex
    dt : float
        Time step (s)
    Px : ndarray
        Polarization x-component (modified in-place), shape (N1, N2), complex
    Py : ndarray
        Polarization y-component (modified in-place), shape (N1, N2), complex
    epsb : float
        Dielectric constant (modified in-place)

    Returns
    -------
    None

    Notes
    -----
    Modifies Ex, Ey, Px, Py, epsb in-place.
    Uses module-level variables _Px_after, _Px_now, _Py_after, _Py_now, _omega_q, _A0, _B, _w, _osc.
    """
    global _Px_after, _Py_after, _Px_now, _Py_now, _omega_q, _A0, _B, _w, _osc

    for n in range(_osc):
        _Px_after[:, :, n] = eps0_val * Ex[:, :] * _B[n] * _w[n]**2 / (_w[n]**2 - _omega_q[:, :]**2)
        _Py_after[:, :, n] = eps0_val * Ey[:, :] * _B[n] * _w[n]**2 / (_w[n]**2 - _omega_q[:, :]**2)

    for n in range(_osc):
        _Px_now[:, :, n] = _Px_after[:, :, n] * np.exp(-ii * _omega_q[:, :] * (-dt))
        _Py_now[:, :, n] = _Py_after[:, :, n] * np.exp(-ii * _omega_q[:, :] * (-dt))

    MakeTransverse(Ex, Ey, qx, qy, qsq)
    for n in range(_osc):
        MakeTransverse(_Px_now[:, :, n], _Py_now[:, :, n], qx, qy, qsq)
        MakeTransverse(_Px_after[:, :, n], _Py_after[:, :, n], qx, qy, qsq)

    for n in range(_osc):
        IFFT(_Px_now[:, :, n])
        IFFT(_Py_now[:, :, n])
        IFFT(_Px_after[:, :, n])
        IFFT(_Py_after[:, :, n])

    _Px_now = np.real(_Px_now)
    _Py_now = np.real(_Py_now)
    _Px_after = np.real(_Px_after)
    _Py_after = np.real(_Py_after)

    epsb = _A0
    Px[:, :] = np.sum(_Px_after, axis=2)
    Py[:, :] = np.sum(_Py_after, axis=2)


def MakeTransverse(Ex, Ey, qx, qy, qsq):
    """
    Make electric field transverse.

    Projects the electric field onto the transverse direction
    perpendicular to the momentum vector.

    Parameters
    ----------
    Ex : ndarray
        Electric field x-component (modified in-place), shape (N1, N2), complex
    Ey : ndarray
        Electric field y-component (modified in-place), shape (N1, N2), complex
    qx : ndarray
        x-component of momentum (1/m), 1D array
    qy : ndarray
        y-component of momentum (1/m), 1D array
    qsq : ndarray
        Squared momentum array, shape (N1, N2), complex

    Returns
    -------
    None

    Notes
    -----
    Modifies Ex and Ey in-place.
    Projects out the longitudinal component: E = E - q * (q · E) / |q|^2
    """
    Ny = Ex.shape[1]

    for j in range(Ny):
        dot_product = qx * Ex[:, j] + qy[j] * Ey[:, j]
        Ex[:, j] = Ex[:, j] - qx * dot_product / qsq[:, j]
        Ey[:, j] = Ey[:, j] - qy[j] * dot_product / qsq[:, j]


def SetParamsSilica():
    """
    Set material parameters for silica.

    Initializes oscillator parameters for silica material model.

    Returns
    -------
    None

    Notes
    -----
    Modifies module-level variables _osc, _B, _C, _w, _gam, _Nf, _A0, _epsr_0, _epsr_infty.
    Uses Sellmeier coefficients for silica.
    """
    global _osc, _B, _C, _w, _gam, _Nf, _A0, _epsr_0, _epsr_infty, _lambda0

    _osc = 3
    _B = np.zeros(_osc)
    _C = np.zeros(_osc)
    _w = np.zeros(_osc)
    _gam = np.zeros(_osc)
    _Nf = np.zeros(_osc)

    _A0 = 1.0
    _B[0] = 0.696166300
    _B[1] = 0.407942600
    _B[2] = 0.897479400
    _C[0] = 4.67914826e-3 * 1e-12
    _C[1] = 1.35120631e-2 * 1e-12
    _C[2] = 97.9340025 * 1e-12

    _w = twopi * c0 / np.sqrt(_C)
    _gam = _w / 25.0

    _Nf = _B * _w**2 * eps0_val * me0_val / _q**2

    _epsr_0 = _A0 + np.sum(_B * _lambda0**2 / (_lambda0**2 - _C))
    _epsr_infty = _A0 + np.sum(_B * _lambda0**2 / (_lambda0**2 - _C))


def SetParamsGaAs():
    """
    Set material parameters for GaAs.

    Initializes oscillator parameters for GaAs material model.

    Returns
    -------
    None

    Notes
    -----
    Modifies module-level variables _osc, _B, _C, _w, _gam, _Nf, _A0, _epsr_0, _epsr_infty.
    Uses Sellmeier coefficients for GaAs.
    """
    global _osc, _B, _C, _w, _gam, _Nf, _A0, _epsr_0, _epsr_infty, _lambda0

    _osc = 3
    _B = np.zeros(_osc)
    _C = np.zeros(_osc)
    _w = np.zeros(_osc)
    _gam = np.zeros(_osc)
    _Nf = np.zeros(_osc)

    _A0 = 4.37251400
    _B[0] = 5.46674200
    _B[1] = 0.02429960
    _B[2] = 1.95752200
    _C[0] = 0.4431307**2 * 1e-12
    _C[1] = 0.8746453**2 * 1e-12
    _C[2] = 36.916600**2 * 1e-12

    _w = twopi * c0 / np.sqrt(_C)
    _gam = _w / 10.0

    _Nf = _B * _w**2 * eps0_val * me0_val / _q**2

    _epsr_0 = _A0 + np.sum(_B * _lambda0**2 / (_lambda0**2 - _C))
    _epsr_infty = _A0 + np.sum(_B * _lambda0**2 / (_lambda0**2 - _C))


def SetParamsAlAs():
    """
    Set material parameters for AlAs.

    Initializes oscillator parameters for AlAs material model.

    Returns
    -------
    None

    Notes
    -----
    Modifies module-level variables _osc, _B, _C, _w, _gam, _Nf, _A0, _epsr_0, _epsr_infty, _w1, _w2.
    Uses Sellmeier coefficients for AlAs.
    """
    global _osc, _B, _C, _w, _gam, _Nf, _A0, _epsr_0, _epsr_infty, _w1, _w2

    _osc = 2
    _B = np.zeros(_osc)
    _C = np.zeros(_osc)
    _w = np.zeros(_osc)
    _gam = np.zeros(_osc)
    _Nf = np.zeros(_osc)

    _A0 = 2.0792
    _B[0] = 6.0840
    _B[1] = 1.9000
    _C[0] = 0.2822**2 * 1e-12
    _C[1] = 27.620**2 * 1e-12

    _w = twopi * c0 / np.sqrt(_C)
    _gam[:] = 0.0  # Assign to all elements of array (not scalar)

    _Nf = _B * _w**2 * eps0_val * me0_val / _q**2

    _w1 = twopi * c0 / (2.2e-6)
    _w2 = twopi * c0 / (0.56e-6)

    _epsr_0 = 10.0
    _epsr_infty = 8.2


def SetParamsNone():
    """
    Set material parameters for vacuum (no material).

    Initializes oscillator parameters for vacuum (no dispersion).

    Returns
    -------
    None

    Notes
    -----
    Modifies module-level variables _osc, _B, _C, _w, _gam, _Nf, _A0, _epsr_0, _epsr_infty.
    Sets all parameters to vacuum values (no dispersion).
    """
    global _osc, _B, _C, _w, _gam, _Nf, _A0, _epsr_0, _epsr_infty

    _osc = 1
    _B = np.zeros(_osc)
    _C = np.zeros(_osc)
    _w = np.zeros(_osc)
    _gam = np.zeros(_osc)
    _Nf = np.zeros(_osc)

    _A0 = 1.0
    _B[0] = 0.0
    _C[0] = 1.0

    _w = twopi * c0 / np.sqrt(_C)
    _gam[:] = 0.0  # Assign to all elements of array (not scalar)
    _Nf[:] = 0.0   # Assign to all elements of array (not scalar)

    _epsr_0 = 1.0
    _epsr_infty = 1.0

#######################################################################
# Local Utitlities ####################################
#####################################################



def nw2_no_gam(wL):
    """
    Calculate dielectric constant without damping.
    (Somewhere below in phost.f90)

    Computes the dielectric constant using the oscillator model
    without damping: n^2 = A0 + sum(B * w^2 / (w^2 - wL^2))

    Parameters
    ----------
    wL : float or complex
        Frequency (Hz)

    Returns
    -------
    complex
        Dielectric constant value (n^2)

    Notes
    -----
    Uses module-level variables _A0, _B, _w, _osc.
    """
    global _A0, _B, _w, _osc

    if _B is None or _w is None:
        return _epsr_infty

    result = _A0
    for n in range(_osc):
        result = result + _B[n] * _w[n]**2 / (_w[n]**2 - wL**2)

    return result

def nw2(wL):
    """
    Calculate dielectric constant with damping.

    Computes the dielectric constant using the oscillator model
    with damping: n^2 = A0 + sum(B * w^2 / (w^2 - i*2*gam*wL - wL^2))

    Parameters
    ----------
    wL : float or complex
        Frequency (Hz)

    Returns
    -------
    complex
        Dielectric constant value (n^2)

    Notes
    -----
    Uses module-level variables _A0, _B, _w, _gam, _osc.
    """
    global _A0, _B, _w, _gam, _osc

    if _B is None or _w is None or _gam is None:
        return _epsr_infty

    result = _A0
    for n in range(_osc):
        result = result + _B[n] * _w[n]**2 / (_w[n]**2 - ii * 2.0 * _gam[n] * wL - wL**2)

    return result


def IFFT(f):
    """
    Inverse Fast Fourier Transform wrapper. (Not the right place should be imported)

    Performs 2D IFFT on the input array.

    Parameters
    ----------
    f : ndarray
        Input array, shape (N1, N2), complex

    Returns
    -------
    None
        f is modified in-place

    Notes
    -----
    Uses pyfftw for FFT operations.
    """
    f[:, :] = pyfftw.interfaces.numpy_fft.ifft2(f)


def nwp_no_gam(wL):
    """
    Calculate derivative of refractive index without damping.

    Computes dn/dw using the oscillator model without damping.

    Parameters
    ----------
    wL : float or complex
        Frequency (Hz)

    Returns
    -------
    complex
        Derivative of refractive index (dn/dw)

    Notes
    -----
    Uses module-level variables _B, _w, _osc.
    """
    global _B, _w, _osc

    if _B is None or _w is None:
        return 0.0

    nw = np.sqrt(nw2_no_gam(wL))
    result = 0.0
    for n in range(_osc):
        result = result + _B[n] * _w[n]**2 * wL / ((_w[n]**2 - wL**2)**2)

    return result / nw


def epsrwp_no_gam(wL):
    """
    Calculate derivative of dielectric constant without damping.

    Computes depsr/dw using the oscillator model without damping.

    Parameters
    ----------
    wL : float or complex
        Frequency (Hz)

    Returns
    -------
    complex
        Derivative of dielectric constant (depsr/dw)

    Notes
    -----
    Uses module-level variables _B, _w, _osc.
    """
    global _B, _w, _osc

    if _B is None or _w is None:
        return 0.0

    result = 0.0
    for n in range(_osc):
        result = result + _B[n] * _w[n]**2 * (2 * wL) / ((_w[n]**2 - wL**2)**2)

    return result


def nwp(wL):
    """
    Calculate derivative of refractive index with damping.

    Computes dn/dw using the oscillator model with damping.

    Parameters
    ----------
    wL : float or complex
        Frequency (Hz)

    Returns
    -------
    complex
        Derivative of refractive index (dn/dw)

    Notes
    -----
    Uses module-level variables _B, _w, _gam, _osc.
    """
    global _B, _w, _gam, _osc

    if _B is None or _w is None or _gam is None:
        return 0.0

    nw = np.sqrt(nw2(wL))
    result = 0.0
    for n in range(_osc):
        result = result + _B[n] * _w[n]**2 * (ii * _gam[n] + wL) / ((_w[n]**2 - ii * 2.0 * _gam[n] * wL - wL**2)**2)

    return result / nw


def nl2_no_gam(lam):
    """
    Calculate dielectric constant from wavelength without damping.

    Computes the dielectric constant using the oscillator model
    without damping: n^2 = A0 + sum(B * lam^2 / (lam^2 - C))

    Parameters
    ----------
    lam : float
        Wavelength (m)

    Returns
    -------
    complex
        Dielectric constant value (n^2)

    Notes
    -----
    Uses module-level variables _A0, _B, _C, _osc.
    """
    global _A0, _B, _C, _osc

    if _B is None or _C is None:
        return _epsr_infty

    result = _A0
    for n in range(_osc):
        result = result + _B[n] * lam**2 / (lam**2 - _C[n])

    return result


def nl2(lam):
    """
    Calculate dielectric constant from wavelength with damping.

    Computes the dielectric constant using the oscillator model
    with damping, converting wavelength to frequency first.

    Parameters
    ----------
    lam : float
        Wavelength (m)

    Returns
    -------
    complex
        Dielectric constant value (n^2)

    Notes
    -----
    Uses module-level variables _A0, _B, _w, _gam, _osc.
    Converts wavelength to frequency: wL = 2*pi*c0 / lam
    """
    global _A0, _B, _w, _gam, _osc

    if _B is None or _w is None or _gam is None:
        return _epsr_infty

    wL = twopi * c0 / (lam + 1e-100)
    result = _A0
    for n in range(_osc):
        result = result + _B[n] * _w[n]**2 / (_w[n]**2 - ii * 2.0 * _gam[n] * wL - wL**2)

    return result


def WriteHostDispersion():
    """
    Write host material dispersion data to files.

    Writes refractive index and dielectric constant data
    as functions of frequency and wavelength to output files.

    Returns
    -------
    None

    Notes
    -----
    Creates output files in 'fields/host/' and 'fields/host/nogam/' directories.
    Writes both real and imaginary parts of n and epsr.
    Uses module-level variables _w, _B, _osc.
    """
    global _w, _B, _osc

    if _w is None or _B is None:
        return

    Nxx = 10000

    # Frequency domain
    x0 = 0.0
    xf = np.max(_w) * 3.0
    dx = (xf - x0) / Nxx

    # Create output directories
    os.makedirs('fields/host', exist_ok=True)
    os.makedirs('fields/host/nogam', exist_ok=True)

    # Open files for frequency domain
    f_nw_real = open('fields/host/n.w.real.dat', 'w', encoding='utf-8')
    f_nw_imag = open('fields/host/n.w.imag.dat', 'w', encoding='utf-8')
    f_epsrw_real = open('fields/host/epsr.w.real.dat', 'w', encoding='utf-8')
    f_epsrw_imag = open('fields/host/epsr.w.imag.dat', 'w', encoding='utf-8')
    f_nw_real_ng = open('fields/host/nogam/n.w.real.dat', 'w', encoding='utf-8')
    f_nw_imag_ng = open('fields/host/nogam/n.w.imag.dat', 'w', encoding='utf-8')
    f_epsrw_real_ng = open('fields/host/nogam/epsr.w.real.dat', 'w', encoding='utf-8')
    f_epsrw_imag_ng = open('fields/host/nogam/epsr.w.imag.dat', 'w', encoding='utf-8')
    f_q2w_real = open('fields/host/q2.w.real.dat', 'w', encoding='utf-8')
    f_q2w_imag = open('fields/host/q2.w.imag.dat', 'w', encoding='utf-8')

    for l in range(1, Nxx + 1):
        x = x0 + l * dx
        n2 = nw2(x)
        n2X = nw2_no_gam(x)
        n = np.sqrt(n2)
        nX = np.sqrt(n2X)

        f_nw_real.write(f'{x} {np.real(n)}\n')
        f_nw_imag.write(f'{x} {np.imag(n)}\n')
        f_epsrw_real.write(f'{x} {np.real(n2)}\n')
        f_epsrw_imag.write(f'{x} {np.imag(n2)}\n')
        exp_factor = np.exp(-((np.abs(n2X) - 9) / 8)**12)
        f_nw_real_ng.write(f'{x} {np.real(nX) * exp_factor}\n')
        f_nw_imag_ng.write(f'{x} {np.imag(nX) * exp_factor}\n')
        f_epsrw_real_ng.write(f'{x} {np.real(n2X) * exp_factor}\n')
        f_epsrw_imag_ng.write(f'{x} {np.imag(n2X) * exp_factor}\n')

    f_nw_real.close()
    f_nw_imag.close()
    f_epsrw_real.close()
    f_epsrw_imag.close()
    f_nw_real_ng.close()
    f_nw_imag_ng.close()
    f_epsrw_real_ng.close()
    f_epsrw_imag_ng.close()
    f_q2w_real.close()
    f_q2w_imag.close()

    # Wavelength domain
    x0 = 0.0
    xf = twopi * c0 / (np.min(_w) + 1e-100) * 3.0
    dx = (xf - x0) / Nxx

    f_nl_real = open('fields/host/n.l.real.dat', 'w', encoding='utf-8')
    f_nl_imag = open('fields/host/n.l.imag.dat', 'w', encoding='utf-8')
    f_epsrl_real = open('fields/host/epsr.l.real.dat', 'w', encoding='utf-8')
    f_epsrl_imag = open('fields/host/epsr.l.imag.dat', 'w', encoding='utf-8')
    f_nl_real_ng = open('fields/host/nogam/n.l.real.dat', 'w', encoding='utf-8')
    f_nl_imag_ng = open('fields/host/nogam/n.l.imag.dat', 'w', encoding='utf-8')
    f_epsrl_real_ng = open('fields/host/nogam/epsr.l.real.dat', 'w', encoding='utf-8')
    f_epsrl_imag_ng = open('fields/host/nogam/epsr.l.imag.dat', 'w', encoding='utf-8')

    for l in range(1, Nxx + 1):
        x = x0 + (l - 1) * dx
        n2 = nl2(x)
        n2X = nl2_no_gam(x)
        n = np.sqrt(n2)
        nX = np.sqrt(n2X)

        # Convert to micrometers
        x_um = x * 1e6

        f_nl_real.write(f'{x_um} {np.real(n)}\n')
        f_nl_imag.write(f'{x_um} {np.imag(n)}\n')
        f_epsrl_real.write(f'{x_um} {np.real(n2)}\n')
        f_epsrl_imag.write(f'{x_um} {np.imag(n2)}\n')
        exp_factor = np.exp(-((np.abs(n2X) - 9) / 8)**12)
        f_nl_real_ng.write(f'{x_um} {np.real(nX) * exp_factor}\n')
        f_nl_imag_ng.write(f'{x_um} {np.imag(nX) * exp_factor}\n')
        f_epsrl_real_ng.write(f'{x_um} {np.real(n2X) * exp_factor}\n')
        f_epsrl_imag_ng.write(f'{x_um} {np.imag(n2X) * exp_factor}\n')

    f_nl_real.close()
    f_nl_imag.close()
    f_epsrl_real.close()
    f_epsrl_imag.close()
    f_nl_real_ng.close()
    f_nl_imag_ng.close()
    f_epsrl_real_ng.close()
    f_epsrl_imag_ng.close()
