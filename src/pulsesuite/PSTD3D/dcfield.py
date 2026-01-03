"""
DC field carrier transport calculations for quantum wire simulations.

This module calculates the dc field carrier transport contributions to the
Semiconductor Bloch equations in support of propagation simulations for a
quantum wire.

Converted from dcfield.f90 (lines 30-253).
"""

import numpy as np
from scipy.constants import e as e0, hbar as hbar_SI
from numba import jit
import pyfftw
pyfftw.interfaces.cache.enable()
from ..libpulsesuite.spliner import rescale_1D
import os

# Physical constants
pi = np.pi
hbar = hbar_SI
ii = 1j  # Imaginary unit

# Module-level state variables (matching Fortran module variables)
_Y = None
_xe = None
_xh = None
_qinv = None
_ERate = 0.0
_HRate = 0.0
_VEDrift = 0.0
_VHDrift = 0.0
_dkk = 0.0
_kmin = 0.0
_kmax = 0.0
_WithPhns = True  # Couple the damping rate with phonons or no
_fe_file = None
_fh_file = None


def GetKArray(Nk, L):
    """
    (typespace.py should be there)
    Generate k-space array for Fourier transforms.

    Parameters
    ----------
    Nk : int
        Number of k-space points
    L : float
        Length of the spatial domain (m)

    Returns
    -------
    ndarray
        Array of k values (1/m), 1D array of length Nk
    """
    dk = 2.0 * pi / L if L > 0 else 1.0
    k = np.arange(Nk, dtype=float) * dk
    k = k - k[Nk // 2]
    return k


def InitializeDC(ky, me, mh):
    """
    Initialize the DC field module.

    Sets up all module-level arrays required for DC field calculations.
    Allocates and initializes Y, xe, xh, qinv arrays and opens output files.

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
    global _Y, _xe, _xh, _qinv, _ERate, _HRate, _dkk, _kmin, _kmax, _fe_file, _fh_file

    Nk = len(ky)
    dky = ky[2] - ky[1] if Nk > 2 else ky[1] - ky[0] if Nk > 1 else 1.0
    _dkk = dky

    _ERate = 0.0
    _HRate = 0.0

    _Y = GetKArray(Nk, (Nk - 1) * dky)

    _xe = me / hbar**2 * np.abs(ky) / (np.abs(ky) + dky * 1e-5)**2 / dky
    _xh = mh / hbar**2 * np.abs(ky) / (np.abs(ky) + dky * 1e-5)**2 / dky

    _qinv = np.zeros(Nk + 2)
    _qinv[1:Nk+1] = ky / (np.abs(ky) + dky * 1e-5)**2

    _kmin = ky[0] - 2 * dky
    _kmax = ky[Nk - 1] + 2 * dky

    os.makedirs('dataQW', exist_ok=True)
    _fe_file = open('dataQW/Fe.dat', 'w', encoding='utf-8')
    _fh_file = open('dataQW/Fh.dat', 'w', encoding='utf-8')


def CalcDCE2(DCTrans, ky, Cq2, Edc, me, ge, Ephn, N0, ne, Ee, Vee, n, j, DC):
    """
    Calculate DC field contribution for electrons (version 2).

    Computes the DC field transport contribution to the electron distribution
    evolution. This version uses a finite difference derivative instead of FFT.

    Parameters
    ----------
    DCTrans : bool
        Whether to include DC transport terms
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Cq2 : ndarray
        Coupling constant squared, 1D array
    Edc : float
        DC electric field (V/m)
    me : float
        Effective electron mass (kg)
    ge : float
        Inverse electron lifetime (Hz)
    Ephn : float
        Average phonon energy (J)
    N0 : float
        Phonon occupation number
    ne : ndarray
        Electron occupation numbers (complex), 1D array
    Ee : ndarray
        Electron energies (J), 1D array
    Vee : ndarray
        Electron-electron interaction matrix (J), 2D array
    n : int
        Time step index
    j : int
        Iteration index
    DC : ndarray
        Output DC field contribution (modified in-place), 1D array

    Returns
    -------
    None
        DC array is modified in-place.
    """
    global _ERate, _VEDrift

    DC[:] = 0.0
    gate = np.ones(len(ky))  # gate = 1.0
    dk = ky[1] - ky[0] if len(ky) > 1 else 1.0

    Eec = EkReNorm(np.real(ne[:]), Ee[:], Vee[:, :])

    v = DriftVt(np.real(ne[:]), Eec[:])

    DC0 = np.zeros(len(ky), dtype=complex)

    if _WithPhns:
        Fd = FDrift2(Ephn, me, ge, ky, np.real(ne), Cq2, v, N0, _xe)
    else:
        Fd = np.zeros(len(ky))

    if j == 25:
        # printIT(Fd*(1d0,0d0), ky, n, "Wire/Fe/Fe.k.")
        pass

    _ERate = np.sum(Fd[:] / hbar / (np.abs(ky) + 1e-5)**2 * ky[:] * ne[:]) / (np.sum(ne[:]) + 1e-20)

    Fd = np.sum(Fd) / (np.sum(np.abs(ne)) + 1e-20) * 2.0

    if _fe_file is not None:
        _fe_file.write(f"{n} {Fd}\n")
        _fe_file.flush()

    if not DCTrans:
        return

    DC0 = -(-e0 * Edc - Fd) * gate / hbar * ne[:]

    _VEDrift = v

    DC0_shift_p1 = np.roll(DC0, -1)
    DC0_shift_m1 = np.roll(DC0, 1)
    DC0 = (DC0_shift_p1 - DC0_shift_m1) / 2.0 / dk

    DC[:] = np.real(DC0)

    if j == 25:
        # printIT(DC0, ky, n, "Wire/Fe/De.k.")
        pass


def CalcDCH2(DCTrans, ky, Cq2, Edc, mh, gh, Ephn, N0, nh, Eh, Vhh, n, j, DC):
    """
    Calculate DC field contribution for holes (version 2).

    Computes the DC field transport contribution to the hole distribution
    evolution. This version uses a finite difference derivative instead of FFT.

    Parameters
    ----------
    DCTrans : bool
        Whether to include DC transport terms
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Cq2 : ndarray
        Coupling constant squared, 1D array
    Edc : float
        DC electric field (V/m)
    mh : float
        Effective hole mass (kg)
    gh : float
        Inverse hole lifetime (Hz)
    Ephn : float
        Average phonon energy (J)
    N0 : float
        Phonon occupation number
    nh : ndarray
        Hole occupation numbers (complex), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    Vhh : ndarray
        Hole-hole interaction matrix (J), 2D array
    n : int
        Time step index
    j : int
        Iteration index
    DC : ndarray
        Output DC field contribution (modified in-place), 1D array

    Returns
    -------
    None
        DC array is modified in-place.
    """
    global _HRate, _VHDrift

    DC[:] = 0.0
    gate = np.ones(len(ky))  # gate = 1.0
    dk = ky[1] - ky[0] if len(ky) > 1 else 1.0

    Ehc = EkReNorm(np.real(nh[:]), Eh[:], Vhh[:, :])

    v = DriftVt(np.real(nh[:]), Ehc[:])

    DC0 = np.zeros(len(ky), dtype=complex)

    if _WithPhns:
        Fd = FDrift2(Ephn, mh, gh, ky, np.real(nh), Cq2, v, N0, _xh)
    else:
        Fd = np.zeros(len(ky))

    _HRate = np.sum(Fd[:] / hbar / (np.abs(ky) + 1e-5)**2 * ky[:] * nh[:]) / (np.sum(nh[:]) + 1e-20)

    if j == 25:
        # printIT(Fd*(1d0,0d0), ky, n, "Wire/Fh/Fh.k.")
        pass

    Fd = np.sum(Fd) / (np.sum(np.abs(nh)) + 1e-20) * 2.0

    if _fh_file is not None:
        _fh_file.write(f"{n} {Fd}\n")
        _fh_file.flush()

    if not DCTrans:
        return

    DC0 = -(-e0 * Edc - Fd) * gate / hbar * nh[:]

    _VHDrift = v

    DC0_shift_p1 = np.roll(DC0, -1)
    DC0_shift_m1 = np.roll(DC0, 1)
    DC0 = (DC0_shift_p1 - DC0_shift_m1) / 2.0 / dk

    DC[:] = np.real(DC0)

    if j == 25:
        # printIT(DC0, ky, n, "Wire/Fh/Dh.k.")
        pass


def CalcDCE(DCTrans, ky, Cq2, Edc, me, ge, Ephn, N0, ne, Ee, Vee, DC):
    """
    Calculate DC field contribution for electrons (original version).

    Computes the DC field transport contribution to the electron distribution
    evolution. This version uses FFT-based derivative.

    Parameters
    ----------
    DCTrans : bool
        Whether to include DC transport terms
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Cq2 : ndarray
        Coupling constant squared, 1D array
    Edc : float
        DC electric field (V/m)
    me : float
        Effective electron mass (kg)
    ge : float
        Inverse electron lifetime (Hz)
    Ephn : float
        Average phonon energy (J)
    N0 : float
        Phonon occupation number
    ne : ndarray
        Electron occupation numbers (complex), 1D array
    Ee : ndarray
        Electron energies (J), 1D array
    Vee : ndarray
        Electron-electron interaction matrix (J), 2D array
    DC : ndarray
        Output DC field contribution (modified in-place), 1D array

    Returns
    -------
    None
        DC array is modified in-place.
    """
    global _ERate, _VEDrift

    DC[:] = 0.0

    Eec = EkReNorm(np.real(ne[:]), Ee[:], Vee[:, :])

    if not DCTrans:
        return

    v = DriftVt(np.real(ne[:]), Eec[:])

    dndk = np.real(ne[:]).astype(complex)

    # FFT-based derivative using pyfftw
    dndk[:] = pyfftw.interfaces.numpy_fft.fft(dndk)
    dndk[:] = dndk[:] * (ii * _Y[:])
    dndk[:] = pyfftw.interfaces.numpy_fft.ifft(dndk)

    if _WithPhns:
        Fd = FDrift2(Ephn, me, ge, ky, np.real(ne), Cq2, v, N0, _xe)
    else:
        Fd = np.zeros(len(ky))

    DC[:] = -(-e0 * Edc - Fd) / hbar * np.real(dndk)

    _ERate = np.sum(Fd[:] / hbar / (np.abs(ky) + 1e-5)**2 * ky[:] * ne[:]) / (np.sum(ne[:]) + 1e-20)

    _VEDrift = v


def CalcDCH(DCTrans, ky, Cq2, Edc, mh, gh, Ephn, N0, nh, Eh, Vhh, DC):
    """
    Calculate DC field contribution for holes (original version).

    Computes the DC field transport contribution to the hole distribution
    evolution. This version uses FFT-based derivative.

    Parameters
    ----------
    DCTrans : bool
        Whether to include DC transport terms
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Cq2 : ndarray
        Coupling constant squared, 1D array
    Edc : float
        DC electric field (V/m)
    mh : float
        Effective hole mass (kg)
    gh : float
        Inverse hole lifetime (Hz)
    Ephn : float
        Average phonon energy (J)
    N0 : float
        Phonon occupation number
    nh : ndarray
        Hole occupation numbers (complex), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    Vhh : ndarray
        Hole-hole interaction matrix (J), 2D array
    DC : ndarray
        Output DC field contribution (modified in-place), 1D array

    Returns
    -------
    None
        DC array is modified in-place.
    """
    global _HRate, _VHDrift

    DC[:] = 0.0

    Ehc = EkReNorm(np.real(nh[:]), Eh[:], Vhh[:, :])

    if not DCTrans:
        return

    v = DriftVt(np.real(nh[:]), Ehc[:])

    dndk = np.real(nh[:]).astype(complex)

    # FFT-based derivative using pyfftw
    dndk[:] = pyfftw.interfaces.numpy_fft.fft(dndk)
    dndk[:] = dndk[:] * (ii * _Y[:])
    dndk[:] = pyfftw.interfaces.numpy_fft.ifft(dndk)

    if _WithPhns:
        Fd = FDrift2(Ephn, mh, gh, ky, np.real(nh), Cq2, v, N0, _xh)
    else:
        Fd = np.zeros(len(ky))

    DC[:] = -(+e0 * Edc - Fd) / hbar * np.real(dndk)

    _HRate = np.sum(Fd[:] / hbar / (np.abs(ky) + 1e-5)**2 * ky[:] * nh[:]) / (np.sum(nh[:]) + 1e-20)

    _VHDrift = v


def CalcI0n(ne, me, ky):
    """
    Calculate electron current.

    Computes the electron current from the electron distribution and momentum.

    Parameters
    ----------
    ne : ndarray
        Electron occupation numbers (complex), 1D array
    me : float
        Effective electron mass (kg)
    ky : ndarray
        Momentum coordinates (1/m), 1D array

    Returns
    -------
    float
        Electron current (A)
    """
    dk = ky[1] - ky[0] if len(ky) > 1 else 1.0

    Ie = -e0 * np.sum(ne[:] * ky[:] * hbar / me) * 2.0 * dk

    return Ie


def CalcI0(ne, nh, Ee, Eh, VC, dk, ky, I0):
    """
    Calculate total current from electron and hole distributions.

    Computes the total current by calculating drift velocities for both
    electrons and holes, then combining them.

    Parameters
    ----------
    ne : ndarray
        Electron occupation numbers (complex), 1D array
    nh : ndarray
        Hole occupation numbers (complex), 1D array
    Ee : ndarray
        Electron energies (J), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    VC : ndarray
        Interaction matrix (J), 3D array VC[:,:,2] for electrons, VC[:,:,3] for holes
    dk : float
        Momentum step size (1/m)
    ky : ndarray
        Momentum coordinates (1/m), 1D array (unused, kept for interface compatibility)
    I0 : float
        Input value (unused, kept for interface compatibility)

    Returns
    -------
    float
        Total current (A)
    """
    global _dkk

    _dkk = dk

    _ = ky  # Unused, kept for interface compatibility
    _ = I0  # Unused, kept for interface compatibility

    Ec = EkReNorm(np.real(ne[:]), Ee[:], VC[:, :, 1])
    ve = DriftVt(np.real(ne[:]), Ec[:])

    Ec = EkReNorm(np.real(nh[:]), Eh[:], VC[:, :, 2])
    vh = DriftVt(np.real(nh[:]), Ec[:])

    v = ve + vh

    return -e0 * v * np.sum(ne[:]) * dk * 2.0


# JIT-compiled functions for EkReNorm and DriftVt
@jit(nopython=True)
def _EkReNorm_jit(n, En, V):
    Nk = len(n)
    Ec = np.zeros(Nk)
    for k in range(Nk):
        Ec[k] = En[k] + np.sum(n[:] * (V[k, k] - V[k, :])) / 2.0
    return Ec

@jit(nopython=True)
def _DriftVt_jit(n, Ec, dkk, hbar_val):
    Nk = len(n)
    dEdk = np.zeros(Nk, dtype=np.complex128)

    for i in range(1, Nk - 1):
        dEdk[i] = (Ec[i + 1] - Ec[i - 1]) / 2.0 / dkk

    dEdk[0] = 2.0 * dEdk[1] - dEdk[2]
    dEdk[Nk - 1] = 2.0 * dEdk[Nk - 2] - dEdk[Nk - 3]

    v = np.sum(dEdk[:] * n[:]) / (1e-100 + np.sum(n[:])) / hbar_val
    return np.real(v)

def EkReNorm(n, En, V):
    """
    Renormalize energy with many-body corrections.

    Computes the renormalized energy including Hartree-Fock corrections:
    Ec(k) = En(k) + sum(n(:) * (V(k,k) - V(k,:))) / 2

    Parameters
    ----------
    n : ndarray
        Carrier occupation numbers, 1D array
    En : ndarray
        Single-particle energies (J), 1D array
    V : ndarray
        Interaction matrix (J), 2D array V[k1, k2]

    Returns
    -------
    ndarray
        Renormalized energies (J), 1D array
    """
    return _EkReNorm_jit(n, En, V)

def DriftVt(n, Ec):
    """
    Calculate drift velocity.

    Computes the average drift velocity from the energy gradient and
    carrier distribution: v = sum(dEcdk * n) / sum(n) / hbar

    Parameters
    ----------
    n : ndarray
        Carrier occupation numbers, 1D array
    Ec : ndarray
        Renormalized energies (J), 1D array

    Returns
    -------
    float
        Drift velocity (m/s)

    Notes
    -----
    Uses finite difference to compute dEcdk, with boundary extrapolation.
    """
    dkk = _dkk
    return _DriftVt_jit(n, Ec, dkk, hbar)


def Lrtz(a, b):
    """
    Lorentzian function.

    Computes the Lorentzian line shape function: (b/pi) / (a^2 + b^2)

    Parameters
    ----------
    a : float or ndarray
        Frequency offset or energy difference
    b : float or ndarray
        Half-width at half-maximum (HWHM)

    Returns
    -------
    float or ndarray
        Lorentzian function value
    """
    return (b / pi) / (a**2 + b**2)


def theta(x):
    """
    Heaviside step function (theta function).

    Computes the step function: (abs(x) + x) / 2 / (abs(x) + small)
    Returns 1 for x > 0, 0 for x <= 0.

    Parameters
    ----------
    x : float or ndarray
        Input value

    Returns
    -------
    float or ndarray
        Step function value (0 or 1, 0 at x=0)
    """
    small = 1e-200
    return (np.abs(x) + x) / 2.0 / (np.abs(x) + small)


def FDrift2(Ephn, m, g, ky, n, Cq2, v, N0, x):
    """
    Calculate drift force from phonon interactions.

    Computes the drift force due to phonon emission and absorption processes.

    Parameters
    ----------
    Ephn : float
        Average phonon energy (J)
    m : float
        Carrier mass (kg)
    g : float
        Inverse lifetime (Hz)
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    n : ndarray
        Carrier occupation numbers, 1D array
    Cq2 : ndarray
        Coupling constant squared, 1D array
    v : float
        Drift velocity (m/s)
    N0 : float
        Phonon occupation number
    x : ndarray
        k-dependent delta-function coefficients, 1D array (unused, kept for interface compatibility)

    Returns
    -------
    ndarray
        Drift force (N), 1D array
    """
    _ = x  # Unused parameter, kept for interface compatibility
    Nk = len(ky)
    EM = np.zeros((Nk, Nk))
    ABSB = np.zeros((Nk, Nk))
    FDrift2_arr = np.zeros(Nk)

    # Compute emission and absorption matrices
    for k in range(1, Nk + 1):  # 1-based indexing for Theta functions
        for q in range(1, Nk + 1):
            EM[q - 1, k - 1] = ThetaEM(Ephn, m, g, ky, n, Cq2, v, N0, q, k)
            ABSB[q - 1, k - 1] = ThetaABS(Ephn, m, g, ky, n, Cq2, v, N0, q, k)

    # Compute drift force
    for k in range(Nk):
        FDrift2_arr[k] = np.sum(hbar * ky[:] * (EM[:, k] - ABSB[:, k]))

    return FDrift2_arr


def FDrift(Ephn, m, q, dndk, Cq2, v, N0, x):
    """
    Calculate drift force (alternative implementation).

    Computes the drift force using a different approach with dndk derivative.

    Parameters
    ----------
    Ephn : float
        Average phonon energy (J)
    m : float
        Carrier mass (kg)
    q : ndarray
        Momentum coordinates (1/m), 1D array
    dndk : ndarray
        Derivative of carrier occupation with respect to momentum, 1D array
    Cq2 : ndarray
        Coupling constant squared, 1D array
    v : float
        Drift velocity (m/s)
    N0 : float
        Phonon occupation number
    x : ndarray
        k-dependent delta-function coefficients, 1D array

    Returns
    -------
    float
        Drift force (N)
    """
    ThetaEMABS_val = ThetaEMABS(Ephn, m, q[:], dndk[:], Cq2[:], v)

    FDrift = np.sum(hbar * q[:] * ThetaEMABS_val * (2.0 * N0 + 1.0) * x[:]) / 2.0

    return FDrift


# JIT-compiled functions for ThetaEM and ThetaABS
@jit(nopython=True)
def _Lrtz_jit(a, b):
    pi_val = 3.141592653589793
    return (b / pi_val) / (a**2 + b**2)

@jit(nopython=True)
def _theta_jit(x):
    small = 1e-200
    return (np.abs(x) + x) / 2.0 / (np.abs(x) + small)

@jit(nopython=True)
def _ThetaEM_jit(Ephn, m, g, ky, n, Cq2, v, N0, q, k, hbar_val, pi_val):
    q_idx = q - 1 if q > 0 else 0
    k_idx = k - 1 if k > 0 else 0

    Nk = len(ky)
    Nk0 = int(np.ceil(Nk / 2.0))

    dk = ky[1] - ky[0] if Nk > 1 else 1.0

    kmq = Nk0 + int(np.round(ky[k_idx] / dk)) - int(np.round(ky[q_idx] / dk))

    if kmq < 0 or kmq >= Nk:
        return 0.0

    xq = Ephn - hbar_val * ky[q_idx] * v
    Ek = hbar_val**2 * ky[k_idx]**2 / 2.0 / m
    Ekmq = hbar_val**2 * ky[kmq]**2 / 2.0 / m

    lrtz_val = _Lrtz_jit(Ekmq - Ek + xq, hbar_val * g)
    theta_val = _theta_jit(xq)

    ThetaEM_val = (4.0 * pi_val / hbar_val * Cq2[q_idx] * n[k_idx] *
                   (1.0 - n[kmq]) * (N0 + 1.0) *
                   lrtz_val * theta_val)

    return ThetaEM_val

@jit(nopython=True)
def _ThetaABS_jit(Ephn, m, g, ky, n, Cq2, v, N0, q, k, hbar_val, pi_val):
    q_idx = q - 1 if q > 0 else 0
    k_idx = k - 1 if k > 0 else 0

    Nk = len(ky)
    Nk0 = int(np.ceil(Nk / 2.0))

    dk = ky[1] - ky[0] if Nk > 1 else 1.0

    kmq = Nk0 + int(np.round(ky[k_idx] / dk)) - int(np.round(ky[q_idx] / dk))

    if kmq < 0 or kmq >= Nk:
        return 0.0

    xq = Ephn - hbar_val * ky[q_idx] * v
    Ek = hbar_val**2 * ky[k_idx]**2 / 2.0 / m
    Ekmq = hbar_val**2 * ky[kmq]**2 / 2.0 / m

    lrtz_val = _Lrtz_jit(Ek - Ekmq - xq, hbar_val * g)
    theta_val = _theta_jit(xq)

    ThetaABS_val = (4.0 * pi_val / hbar_val * Cq2[q_idx] * n[kmq] *
                    (1.0 - n[k_idx]) * N0 *
                    lrtz_val * theta_val)

    return ThetaABS_val

def ThetaEM(Ephn, m, g, ky, n, Cq2, v, N0, q, k):
    """
    Calculate emission matrix element.

    Computes the emission rate matrix element for phonon-assisted transitions.

    Parameters
    ----------
    Ephn : float
        Average phonon energy (J)
    m : float
        Carrier mass (kg)
    g : float
        Inverse lifetime (Hz)
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    n : ndarray
        Carrier occupation numbers, 1D array
    Cq2 : ndarray
        Coupling constant squared, 1D array
    v : float
        Drift velocity (m/s)
    N0 : float
        Phonon occupation number
    q : int
        Phonon momentum index (1-based, will be converted to 0-based)
    k : int
        Carrier momentum index (1-based, will be converted to 0-based)

    Returns
    -------
    float
        Emission matrix element
    """
    return _ThetaEM_jit(Ephn, m, g, ky, n, Cq2, v, N0, q, k, hbar, pi)

def ThetaABS(Ephn, m, g, ky, n, Cq2, v, N0, q, k):
    """
    Calculate absorption matrix element.

    Computes the absorption rate matrix element for phonon-assisted transitions.

    Parameters
    ----------
    Ephn : float
        Average phonon energy (J)
    m : float
        Carrier mass (kg)
    g : float
        Inverse lifetime (Hz)
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    n : ndarray
        Carrier occupation numbers, 1D array
    Cq2 : ndarray
        Coupling constant squared, 1D array
    v : float
        Drift velocity (m/s)
    N0 : float
        Phonon occupation number
    q : int
        Phonon momentum index (1-based, will be converted to 0-based)
    k : int
        Carrier momentum index (1-based, will be converted to 0-based)

    Returns
    -------
    float
        Absorption matrix element
    """
    return _ThetaABS_jit(Ephn, m, g, ky, n, Cq2, v, N0, q, k, hbar, pi)


def CalcAvgCoeff(ky, dk, k1, k2, i1, i2, x1, x2, x3, x4):
    """
    Calculate average coefficients for interpolation.

    Computes interpolation coefficients for bilinear interpolation in k-space.
    The function extends the ky array with boundary points and calculates
    coefficients based on the positions k1, k2 and indices i1, i2.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    dk : float
        Momentum step size (1/m)
    k1 : float
        First momentum value for interpolation
    k2 : float
        Second momentum value for interpolation
    i1 : int
        First index (1-based, will be converted to 0-based)
    i2 : int
        Second index (1-based, will be converted to 0-based)
    x1 : float
        Output coefficient 1 (modified in-place, but Python can't modify floats in-place)
    x2 : float
        Output coefficient 2 (modified in-place, but Python can't modify floats in-place)
    x3 : float
        Output coefficient 3 (modified in-place, but Python can't modify floats in-place)
    x4 : float
        Output coefficient 4 (modified in-place, but Python can't modify floats in-place)

    Returns
    -------
    tuple
        Tuple containing (x1, x2, x3, x4) coefficients
    """
    Nk = len(ky)

    # Create extended k array with boundary points
    k = np.zeros(Nk + 4)
    k[2:Nk+2] = ky[:]

    k[1] = ky[0] - dk
    k[0] = ky[0] - 2.0 * dk
    k[Nk + 2] = ky[Nk - 1] + dk
    k[Nk + 3] = ky[Nk - 1] + 2.0 * dk

    # Convert to 0-based indexing (Fortran uses 1-based)
    i1_idx = i1 - 1 if i1 > 0 else 0
    i2_idx = i2 - 1 if i2 > 0 else 0

    # Adjust indices for extended array (shift by 2 due to boundary points)
    i1_ext = i1_idx + 2
    i2_ext = i2_idx + 2

    x1 = (k[i1_ext + 1] - k1) * (k[i2_ext + 1] - k2) / dk**2
    x2 = (k1 - k[i1_ext]) * (k[i2_ext + 1] - k2) / dk**2
    x3 = (k[i1_ext + 1] - k1) * (k2 - k[i2_ext]) / dk**2
    x4 = (k1 - k[i1_ext]) * (k2 - k[i2_ext]) / dk**2

    return x1, x2, x3, x4


def CalcVD(ky, m, n):
    """
    Calculate drift velocity from distribution.

    Computes the average drift velocity from the carrier distribution
    and momentum: v = sum(real(n) * hbar * ky / m) / sum(real(n) + small)

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    m : float
        Carrier mass (kg)
    n : ndarray
        Carrier occupation numbers (complex), 1D array

    Returns
    -------
    float
        Drift velocity (m/s)
    """
    small = 1e-200
    CalcVd = np.sum(np.real(n[:]) * hbar * ky[:] / m) / (np.sum(np.real(n[:]) + small))

    return CalcVd


def CalcPD(ky, m, n):
    """
    Calculate momentum from distribution.

    Computes the average momentum from the carrier distribution:
    p = sum(abs(n) * hbar * ky) / sum(abs(n) + small)

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    m : float
        Carrier mass (kg) (unused, kept for interface compatibility)
    n : ndarray
        Carrier occupation numbers (complex), 1D array

    Returns
    -------
    float
        Average momentum (kg·m/s)
    """
    _ = m  # Unused parameter, kept for interface compatibility
    small = 1e-200
    CalcPd = np.sum(np.abs(n[:]) * hbar * ky[:]) / (np.sum(np.abs(n[:]) + small))

    return CalcPd


def GetEDrift():
    """
    Get electron drift rate.

    Returns the electron temperature damping rate.

    Returns
    -------
    float
        Electron temperature damping rate (Hz)
    """
    return _ERate


def GetHDrift():
    """
    Get hole drift rate.

    Returns the hole temperature damping rate.

    Returns
    -------
    float
        Hole temperature damping rate (Hz)
    """
    return _HRate


def GetVEDrift():
    """
    Get electron drift velocity.

    Returns the electron drift velocity.

    Returns
    -------
    float
        Electron drift velocity (m/s)
    """
    return _VEDrift


def GetVHDrift():
    """
    Get hole drift velocity.

    Returns the hole drift velocity.

    Returns
    -------
    float
        Hole drift velocity (m/s)
    """
    return _VHDrift


# JIT-compiled functions for dndEk and ThetaEMABS
@jit(nopython=True)
def _dndEk_jit(Ephn, m, q, dndq, hbar_val):
    x0 = Ephn * 2.0 * m / hbar_val**2
    N = len(q)
    dndEk = np.zeros(N)
    for i in range(N):
        dndEk[i] = dndq[i] * 4.0 * q[i]**3 / (q[i]**4 - x0**2) * m / hbar_val**2
    return dndEk

@jit(nopython=True)
def _ThetaEMABS_jit(Ephn, m, q, dndk, Cq2, v, hbar_val, pi_val):
    x0 = Ephn * 2.0 * m / hbar_val**2
    N = len(q)
    dndEk_val = np.zeros(N)
    for i in range(N):
        dndEk_val[i] = dndk[i] * 4.0 * q[i]**3 / (q[i]**4 - x0**2) * m / hbar_val**2

    ThetaEMABS = np.zeros(N)
    for i in range(N):
        ThetaEMABS[i] = 4.0 * pi_val / hbar_val * Cq2[i] * hbar_val * q[i] * v * (-dndEk_val[i])
    return ThetaEMABS

def dndEk(Ephn, m, q, dndq):
    """
    Calculate derivative of occupation with respect to energy.

    Computes the derivative dndEk from the derivative dndq with respect to
    momentum, using the relationship between energy and momentum.

    Parameters
    ----------
    Ephn : float
        Average phonon energy (J)
    m : float
        Carrier mass (kg)
    q : ndarray
        Momentum coordinates (1/m), 1D array
    dndq : ndarray
        Derivative of occupation with respect to momentum, 1D array

    Returns
    -------
    ndarray
        Derivative of occupation with respect to energy, 1D array
    """
    return _dndEk_jit(Ephn, m, q, dndq, hbar)

def ThetaEMABS(Ephn, m, q, dndk, Cq2, v):
    """
    Calculate emission-absorption matrix element.

    Computes the combined emission-absorption rate matrix element
    for phonon-assisted transitions using the energy derivative.

    Parameters
    ----------
    Ephn : float
        Average phonon energy (J)
    m : float
        Carrier mass (kg)
    q : ndarray
        Momentum coordinates (1/m), 1D array
    dndk : ndarray
        Derivative of occupation with respect to momentum, 1D array
    Cq2 : ndarray
        Coupling constant squared, 1D array
    v : float
        Drift velocity (m/s)

    Returns
    -------
    ndarray
        Emission-absorption matrix element, 1D array
    """
    return _ThetaEMABS_jit(Ephn, m, q, dndk, Cq2, v, hbar, pi)


##########################################################
# Legacy code functions (lines 608-721)
##########################################################


def DC_Step_Scale(ne, nh, ky, Edc, dt):
    """
    DC step using scaling method (legacy code).

    Performs a DC field step by scaling the distribution functions.
    Uses rescale_1D to interpolate the distributions to shifted momentum grids.

    Parameters
    ----------
    ne : ndarray
        Electron occupation numbers (complex), modified in-place, 1D array
    nh : ndarray
        Hole occupation numbers (complex), modified in-place, 1D array
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Edc : float
        DC electric field (V/m)
    dt : float
        Time step (s)

    Returns
    -------
    None
        ne and nh arrays are modified in-place.

    Notes
    -----
    The function shifts the momentum grid by -e0*Edc/hbar*dt for electrons
    and +e0*Edc/hbar*dt for holes. This corresponds to the DC field accelerating
    electrons in the negative direction and holes in the positive direction.
    """
    # Create copies of the distributions
    ned = ne.copy()
    nhd = nh.copy()

    # Calculate shifted momentum grids
    # Electrons: shift grid left (negative direction) by e0*Edc/hbar*dt
    ky_shift_e = ky - e0 * Edc / hbar * dt
    # Holes: shift grid right (positive direction) by e0*Edc/hbar*dt
    ky_shift_h = ky + e0 * Edc / hbar * dt

    # Rescale electron distribution from shifted grid to original grid
    # rescale_1D(x0, z0, x1, z1): interpolates z0 from grid x0 to grid x1, stores in z1
    rescale_1D(ky_shift_e, ned, ky, ne)

    # Rescale hole distribution from shifted grid to original grid
    rescale_1D(ky_shift_h, nhd, ky, nh)


def DC_Step_FD(ne, nh, nemid, nhmid, ky, Edc, dt, me, mh):
    """
    DC step using finite difference method (legacy code).

    Performs a DC field step using finite difference derivatives.
    Updates the electron and hole distributions based on the DC field
    and momentum-dependent terms.

    Parameters
    ----------
    ne : ndarray
        Electron occupation numbers (complex), modified in-place, 1D array
    nh : ndarray
        Hole occupation numbers (complex), modified in-place, 1D array
    nemid : ndarray
        Midpoint electron occupation numbers (complex), 1D array
    nhmid : ndarray
        Midpoint hole occupation numbers (complex), 1D array
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Edc : float
        DC electric field (V/m)
    dt : float
        Time step (s)
    me : float
        Effective electron mass (kg)
    mh : float
        Effective hole mass (kg)

    Returns
    -------
    None
        ne and nh arrays are modified in-place.
    """
    Nk = len(ky)
    dky = ky[2] - ky[1] if Nk > 2 else ky[1] - ky[0] if Nk > 1 else 1.0
    small = 1e-200
    s = int(np.round(Edc / (Edc + small)))

    # Circular shift: cshift(array, 1) shifts right, cshift(array, -1) shifts left
    ned = np.roll(nemid, -1)
    nhd = np.roll(nhmid, -1)

    neu = np.roll(nemid, 1)
    nhu = np.roll(nhmid, 1)

    ne[:] = ne[:] + (-e0 * Edc - CalcPD(ky, me, ne) * 1e13) / hbar * (neu - ned) / dky * dt
    nh[:] = nh[:] + (+e0 * Edc - CalcPD(ky, mh, nh) * 1e13) / hbar * (nhu - nhd) / dky * dt


def ShiftN1D(ne, dk):
    """
    Shift 1D distribution in momentum space (legacy code).

    Shifts the distribution function in momentum space by dk using FFT.
    This is equivalent to multiplying by exp(-ii * Y * dk) in Fourier space.

    Parameters
    ----------
    ne : ndarray
        Carrier occupation numbers (complex), modified in-place, 1D array
    dk : float
        Momentum shift (1/m)

    Returns
    -------
    None
        ne array is modified in-place.

    Notes
    -----
    Uses pyfftw for FFT operations.
    The function performs: FFT -> multiply by exp(-ii*Y*dk) -> IFFT
    """
    global _Y

    if _Y is not None:
        # FFT to Fourier space
        ne[:] = pyfftw.interfaces.numpy_fft.fft(ne)
        # Multiply by phase factor
        ne[:] = ne[:] * np.exp(-ii * _Y[:] * dk)
        # IFFT back to real space
        ne[:] = pyfftw.interfaces.numpy_fft.ifft(ne)


def ShiftN2D(C, dk):
    """
    Shift 2D distribution in momentum space (legacy code).

    Shifts the 2D distribution function in momentum space by dk using FFT.
    This is equivalent to multiplying by exp(-ii * (Y(k1) + Y2(k2)) * dk) in Fourier space.

    Parameters
    ----------
    C : ndarray
        Distribution matrix (complex), modified in-place, 2D array
    dk : float
        Momentum shift (1/m)

    Returns
    -------
    None
        C array is modified in-place.

    Notes
    -----
    Uses pyfftw for FFT operations.
    The function performs: FFT -> multiply by exp(-ii*(Y+Y2)*dk) -> IFFT
    """
    global _Y

    if _Y is not None:
        Nk = C.shape[0]
        Y2 = _Y.copy()

        # FFT to Fourier space (2D)
        C[:, :] = pyfftw.interfaces.numpy_fft.fft2(C)
        # Multiply by phase factor
        Y_expanded = _Y[:, np.newaxis] + Y2[np.newaxis, :]
        C[:, :] = C[:, :] * np.exp(-ii * Y_expanded * dk)
        # IFFT back to real space
        C[:, :] = pyfftw.interfaces.numpy_fft.ifft2(C)


def Transport(C, Edc, Eac, dt, DCTrans, k1nek2):
    """
    Transport step for distribution matrix (legacy code).

    Performs a transport step on the distribution matrix C by shifting
    in momentum space. Can operate on the full 2D matrix or just the diagonal.

    Parameters
    ----------
    C : ndarray
        Distribution matrix (complex), modified in-place, 2D array
    Edc : float
        DC electric field (V/m)
    Eac : float
        AC electric field (V/m)
    dt : float
        Time step (s)
    DCTrans : bool
        Whether to include DC transport terms
    k1nek2 : bool
        If True, shift the full 2D matrix; if False, shift only the diagonal

    Returns
    -------
    None
        C array is modified in-place.
    """
    if not DCTrans:
        return

    dk = -e0 * (Edc + Eac) / hbar * dt

    if k1nek2:
        ShiftN2D(C, dk)
    else:
        nk = np.diag(C).copy()
        ShiftN1D(nk, dk)
        np.fill_diagonal(C, nk)
