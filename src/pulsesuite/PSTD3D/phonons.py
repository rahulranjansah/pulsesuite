"""
Phonon interaction calculations for quantum wire simulations.

This module calculates the many-body electron-hole phonon contribution
to the Semiconductor Bloch equations in support of propagation simulations
for a quantum wire.

Author: Rahul R. Sah
"""

import numpy as np
from numba import jit
from scipy.constants import hbar as hbar_SI, k as kB_SI

# Physical constants
hbar = hbar_SI
kB = kB_SI
ii = 1j  # Imaginary unit

# Module-level state variables (matching Fortran module variables)
_small = 1e-200  # Smallest number worthy of consideration
_Temp = 77.0  # Temperature of QW solid (K)
_epsr0 = 10.0  # Dielectric constant in host AlAs at w=0
_epsrINF = 8.2  # Dielectric constant in host AlAs at w=INFINITY
_Vscale = 0.0  # Scaling constant e-e and h-h Coulomb arrays
_NO = 0.0  # Bose function for thermal equilibrium longitudinal-optical phonons in host

# Allocatable arrays
_EP = None
_EPT = None
_HP = None
_HPT = None
_idel = None


@jit(nopython=True)
def _InitializePhonons_loop_jit(Nk, Ee, Eh, NO_val, hbar_val, Gph, Oph):
    """JIT-compiled loop for InitializePhonons."""
    EP = np.zeros((Nk, Nk))
    HP = np.zeros((Nk, Nk))

    for k1 in range(Nk):
        for k in range(Nk):
            denom1 = (Ee[k] - Ee[k1] - hbar_val * Oph)**2 + (hbar_val * Gph)**2
            denom2 = (Ee[k] - Ee[k1] + hbar_val * Oph)**2 + (hbar_val * Gph)**2
            EP[k, k1] = NO_val / denom1 + (NO_val + 1.0) / denom2

            denom1 = (Eh[k] - Eh[k1] - hbar_val * Oph)**2 + (hbar_val * Gph)**2
            denom2 = (Eh[k] - Eh[k1] + hbar_val * Oph)**2 + (hbar_val * Gph)**2
            HP[k, k1] = NO_val / denom1 + (NO_val + 1.0) / denom2

    return EP, HP


def InitializePhonons(ky, Ee, Eh, L, epsr, Gph, Oph):
    """
    Initialize phonon interaction matrices (Unecessary the jit version?).

    Sets up the electron and hole phonon interaction matrices EP, HP
    and their transposes EPT, HPT. Also calculates the scaling factor Vscale.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Ee : ndarray
        Electron energies (J), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    L : float
        Length parameter (unused, kept for interface compatibility)
    epsr : float
        Dielectric constant
    Gph : float
        Phonon damping rate (Hz)
    Oph : float
        Phonon frequency (Hz)

    Returns
    -------
    None

    Notes
    -----
    Sets module-level variables:
    - _NO: Bose distribution for phonons
    - _idel: Identity exclusion matrix (1 where k != k1, 0 where k == k1)
    - _EP, _EPT: Electron phonon interaction matrix and its transpose
    - _HP, _HPT: Hole phonon interaction matrix and its transpose
    - _Vscale: Scaling constant for Coulomb interactions
    """
    global _NO, _EP, _EPT, _HP, _HPT, _idel, _Vscale, _Temp, _epsr0, _epsrINF

    _NO = 1.0 / (np.exp(hbar * Oph / kB / _Temp) - 1.0)
    Nk = len(ky)

    # Create identity exclusion matrix
    # idel = 1 where k != k1, 0 where k == k1
    _idel = np.ones((Nk, Nk), dtype=np.float64)
    for k in range(Nk):
        _idel[k, k] = 0.0

    # Allocate arrays
    _EP = np.zeros((Nk, Nk))
    _EPT = np.zeros((Nk, Nk))
    _HP = np.zeros((Nk, Nk))
    _HPT = np.zeros((Nk, Nk))

    try:
        _EP, _HP = _InitializePhonons_loop_jit(Nk, Ee, Eh, _NO, hbar, Gph, Oph)
    except Exception:
        # Fallback to pure Python
        for k1 in range(Nk):
            for k in range(Nk):
                _EP[k, k1] = (_NO / ((Ee[k] - Ee[k1] - hbar * Oph)**2 + (hbar * Gph)**2) +
                              (_NO + 1.0) / ((Ee[k] - Ee[k1] + hbar * Oph)**2 + (hbar * Gph)**2))

                _HP[k, k1] = (_NO / ((Eh[k] - Eh[k1] - hbar * Oph)**2 + (hbar * Gph)**2) +
                              (_NO + 1.0) / ((Eh[k] - Eh[k1] + hbar * Oph)**2 + (hbar * Gph)**2))

    # Multiply by scaling factor and idel matrix
    # Multiplication by idel ensures that k != k1
    _EP = _EP * 2.0 * Gph * _idel
    _HP = _HP * 2.0 * Gph * _idel

    # Transpose
    _EPT = _EP.T
    _HPT = _HP.T

    _Vscale = hbar * Oph * epsr * (1.0 / _epsrINF - 1.0 / _epsr0)


@jit(nopython=True)
def _MBPE_jit(Nk, Vep, ne, EPT, EP):
    """JIT-compiled version of MBPE."""
    Win = np.zeros(Nk)
    Wout = np.zeros(Nk)

    for k in range(Nk):
        sum_win = 0.0
        sum_wout = 0.0
        for i in range(Nk):
            sum_win += Vep[k, i] * ne[i] * EPT[i, k]
            sum_wout += Vep[k, i] * (1.0 - ne[i]) * EP[i, k]
        Win[k] = sum_win
        Wout[k] = sum_wout

    return Win, Wout


def MBPE(ne, VC, E1D, Win, Wout):
    """
    Many-body phonon-electron interaction.

    Calculates the in-scattering and out-scattering rates for electrons
    due to phonon interactions.

    Parameters
    ----------
    ne : ndarray
        Electron carrier populations, 1D array
    VC : ndarray
        Coulomb interaction array, shape (Nk, Nk, 3)
        Uses VC[:, :, 1] for electron-electron interaction (Fortran uses VC(:,:,2) which is 1-based)
    E1D : ndarray
        1D energy array, shape (Nk, Nk)
    Win : ndarray
        In-scattering rates (modified in-place), 1D array
    Wout : ndarray
        Out-scattering rates (modified in-place), 1D array

    Returns
    -------
    None

    Notes
    -----
    Modifies Win and Wout in-place.
    Uses module-level variables _EPT, _EP, _Vscale.
    """
    global _EPT, _EP, _Vscale

    Nk = len(ne)
    Vep = VC[:, :, 1] / E1D * _Vscale  # Fortran uses VC(:,:,2) which is 1-based -> Python uses index 1

    try:
        Win_add, Wout_add = _MBPE_jit(Nk, Vep, ne, _EPT, _EP)
        Win[:] = Win + Win_add
        Wout[:] = Wout + Wout_add
    except Exception:
        # Fallback to pure Python
        for k in range(Nk):
            Win[k] = Win[k] + np.sum(Vep[k, :] * ne * _EPT[:, k])
            Wout[k] = Wout[k] + np.sum(Vep[k, :] * (1.0 - ne) * _EP[:, k])


@jit(nopython=True)
def _MBPH_jit(Nk, Vhp, nh, HPT, HP):
    """JIT-compiled version of MBPH."""
    Win = np.zeros(Nk)
    Wout = np.zeros(Nk)

    for kp in range(Nk):
        sum_win = 0.0
        sum_wout = 0.0
        for i in range(Nk):
            sum_win += Vhp[i, kp] * nh[i] * HPT[i, kp]
            sum_wout += Vhp[i, kp] * (1.0 - nh[i]) * HP[i, kp]
        Win[kp] = sum_win
        Wout[kp] = sum_wout

    return Win, Wout


def MBPH(nh, VC, E1D, Win, Wout):
    """
    Many-body phonon-hole interaction.

    Calculates the in-scattering and out-scattering rates for holes
    due to phonon interactions.

    Parameters
    ----------
    nh : ndarray
        Hole carrier populations, 1D array
    VC : ndarray
        Coulomb interaction array, shape (Nk, Nk, 3)
        Uses VC[:, :, 2] for hole-hole interaction (Fortran uses VC(:,:,3) which is 1-based)
    E1D : ndarray
        1D energy array, shape (Nk, Nk)
    Win : ndarray
        In-scattering rates (modified in-place), 1D array
    Wout : ndarray
        Out-scattering rates (modified in-place), 1D array

    Returns
    -------
    None

    Notes
    -----
    Modifies Win and Wout in-place.
    Uses module-level variables _HPT, _HP, _Vscale.
    """
    global _HPT, _HP, _Vscale

    Nk = len(nh)
    Vhp = VC[:, :, 2] / E1D * _Vscale  # Fortran uses VC(:,:,3) which is 1-based -> Python uses index 2

    try:
        Win_add, Wout_add = _MBPH_jit(Nk, Vhp, nh, _HPT, _HP)
        Win[:] = Win + Win_add
        Wout[:] = Wout + Wout_add
    except Exception:
        # Fallback to pure Python
        for kp in range(Nk):
            Win[kp] = Win[kp] + np.sum(Vhp[:, kp] * nh * _HPT[:, kp])
            Wout[kp] = Wout[kp] + np.sum(Vhp[:, kp] * (1.0 - nh) * _HP[:, kp])


def Cq2(q, V, E1D):
    """
    Calculate Cq for use in the DC Field module.

    Computes the phonon coupling constant Cq for a given momentum array.

    Parameters
    ----------
    q : ndarray
        Momentum coordinates (1/m), 1D array
    V : ndarray
        Interaction potential array, shape (Nk, Nk)
    E1D : ndarray
        1D energy array, shape (Nk, Nk)

    Returns
    -------
    ndarray
        Cq values, 1D array of same length as q

    Notes
    -----
    Uses module-level variable _Vscale for scaling.
    The function maps q values to indices in V and E1D arrays.
    """
    global _Vscale

    if len(q) < 2:
        dq = 1.0
    else:
        dq = q[1] - q[0]

    iq = np.round(np.abs(q / dq)).astype(int)

    Cq2_result = np.zeros(len(q))
    for i in range(len(q)):
        idx = iq[i]  # iq is already calculated with nint, Fortran uses 1+iq for 1-based indexing
        # Convert to 0-based: Fortran V(1+iq, 1) -> Python V[iq, 0]
        if 0 <= idx < V.shape[0] and 0 <= idx < E1D.shape[0]:
            Cq2_result[i] = V[idx, 0] / E1D[idx, 0] * _Vscale

    return Cq2_result


def FermiDistr(En):
    """
    Calculate Fermi-Dirac distribution.

    Computes the Fermi-Dirac distribution assuming host temperature
    and Fermi Energy = 0.

    Parameters
    ----------
    En : float or ndarray
        Energy (J)

    Returns
    -------
    float or ndarray
        Fermi-Dirac distribution value: 1 / (exp(En / (kB * T)) + 1)

    Notes
    -----
    Uses module-level variable _Temp for temperature.
    Note: Fortran returns complex type, but the value is always real, so Python returns float.
    """
    global _Temp
    return 1.0 / (np.exp(En / kB / _Temp) + 1.0)


def BoseDistr(En):
    """
    Calculate Bose-Einstein distribution.

    Computes the Bose-Einstein distribution for phonons.

    Parameters
    ----------
    En : float or ndarray
        Energy (J)

    Returns
    -------
    float or ndarray
        Bose-Einstein distribution value: 1 / (exp(En / (kB * T)) - 1)

    Notes
    -----
    Uses module-level variable _Temp for temperature.
    """
    global _Temp
    return 1.0 / (np.exp(En / kB / _Temp) - 1.0)


def N00():
    """
    Get the Bose distribution value for phonons.

    Returns the module-level variable _NO which is the Bose function
    for thermal equilibrium longitudinal-optical phonons in host.

    Returns
    -------
    float
        Bose distribution value NO
    """
    global _NO
    return _NO

