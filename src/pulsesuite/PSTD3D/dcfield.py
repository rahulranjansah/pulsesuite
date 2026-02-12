"""
DC field carrier transport calculations for quantum wire simulations.

This module calculates the dc field carrier transport contributions to the
Semiconductor Bloch equations in support of propagation simulations for a
quantum wire.

Converted from dcfield.f90 (lines 30-253).

Structure
---------
1. Pure functions (no module state) -- module-level
2. JIT-compiled helpers            -- module-level (Numba requirement)
3. DCFieldModule class             -- mutable state + methods
4. Backward-compatibility layer    -- singleton + wrapper functions
"""

import numpy as np
import pyfftw
from numba import jit
from scipy.constants import e as e0, hbar as hbar_SI

pyfftw.interfaces.cache.enable()
import os

from ..libpulsesuite.spliner import rescale_1D

# Physical constants
pi = np.pi
hbar = hbar_SI
ii = 1j  # Imaginary unit


# ============================================================================
# Pure functions (no module state)
# ============================================================================


def GetKArray(Nk, L):
    """
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
        Average momentum (kg*m/s)
    """
    _ = m  # Unused parameter, kept for interface compatibility
    small = 1e-200
    CalcPd = np.sum(np.abs(n[:]) * hbar * ky[:]) / (np.sum(np.abs(n[:]) + small))
    return CalcPd


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
        Output coefficient 1
    x2 : float
        Output coefficient 2
    x3 : float
        Output coefficient 3
    x4 : float
        Output coefficient 4

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
    """
    # Create copies of the distributions
    ned = ne.copy()
    nhd = nh.copy()

    # Calculate shifted momentum grids
    ky_shift_e = ky - e0 * Edc / hbar * dt
    ky_shift_h = ky + e0 * Edc / hbar * dt

    # Rescale distributions from shifted grid to original grid
    rescale_1D(ky_shift_e, ned, ky, ne)
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

    ned = np.roll(nemid, -1)
    nhd = np.roll(nhmid, -1)

    neu = np.roll(nemid, 1)
    nhu = np.roll(nhmid, 1)

    ne[:] = ne[:] + (-e0 * Edc - CalcPD(ky, me, ne) * 1e13) / hbar * (neu - ned) / dky * dt
    nh[:] = nh[:] + (+e0 * Edc - CalcPD(ky, mh, nh) * 1e13) / hbar * (nhu - nhd) / dky * dt


# ============================================================================
# JIT-compiled helpers (must stay module-level for Numba)
# ============================================================================


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


# ============================================================================
# Pure function wrappers around JIT helpers
# ============================================================================


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


def DriftVt(n, Ec, dkk):
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
    dkk : float
        Momentum step size (1/m)

    Returns
    -------
    float
        Drift velocity (m/s)
    """
    return _DriftVt_jit(n, Ec, dkk, hbar)


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
        Phonon momentum index (1-based)
    k : int
        Carrier momentum index (1-based)

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
        Phonon momentum index (1-based)
    k : int
        Carrier momentum index (1-based)

    Returns
    -------
    float
        Absorption matrix element
    """
    return _ThetaABS_jit(Ephn, m, g, ky, n, Cq2, v, N0, q, k, hbar, pi)


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
        k-dependent delta-function coefficients, 1D array (unused)

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

    FDrift_val = np.sum(hbar * q[:] * ThetaEMABS_val * (2.0 * N0 + 1.0) * x[:]) / 2.0

    return FDrift_val


def dndEk(Ephn, m, q, dndq):
    """
    Calculate derivative of occupation with respect to energy.

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


# ============================================================================
# DCFieldModule class
# ============================================================================


class DCFieldModule:
    """
    DC field carrier transport module for quantum wire simulations.

    Encapsulates the mutable state and stateful methods that were previously
    held as module-level globals. Mirrors the Fortran ``module dcfield``
    structure: private allocatable arrays + public subroutines.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    me : float
        Effective electron mass (kg)
    mh : float
        Effective hole mass (kg)
    WithPhns : bool, optional
        Whether to couple damping rate with phonons (default True)
    datadir : str or None, optional
        Directory for output files.  Pass ``None`` to skip file creation
        (useful for testing).  Default ``'dataQW'``.
    """

    def __init__(self, ky, me, mh, WithPhns=True, datadir='dataQW'):
        Nk = len(ky)
        dky = ky[2] - ky[1] if Nk > 2 else ky[1] - ky[0] if Nk > 1 else 1.0
        self.dkk = dky

        self.ERate = 0.0
        self.HRate = 0.0
        self.VEDrift = 0.0
        self.VHDrift = 0.0
        self.WithPhns = WithPhns

        self.Y = GetKArray(Nk, (Nk - 1) * dky)

        self.xe = me / hbar**2 * np.abs(ky) / (np.abs(ky) + dky * 1e-5)**2 / dky
        self.xh = mh / hbar**2 * np.abs(ky) / (np.abs(ky) + dky * 1e-5)**2 / dky

        self.qinv = np.zeros(Nk + 2)
        self.qinv[1:Nk+1] = ky / (np.abs(ky) + dky * 1e-5)**2

        self.kmin = ky[0] - 2 * dky
        self.kmax = ky[Nk - 1] + 2 * dky

        # Output files (optional)
        if datadir is not None:
            os.makedirs(datadir, exist_ok=True)
            self.fe_file = open(os.path.join(datadir, 'Fe.dat'), 'w',
                                encoding='utf-8')
            self.fh_file = open(os.path.join(datadir, 'Fh.dat'), 'w',
                                encoding='utf-8')
        else:
            self.fe_file = None
            self.fh_file = None

    def close(self):
        """Close output file handles."""
        if self.fe_file is not None:
            self.fe_file.close()
            self.fe_file = None
        if self.fh_file is not None:
            self.fh_file.close()
            self.fh_file = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # DC field calculations (version 2: finite-difference derivative)
    # ------------------------------------------------------------------

    def CalcDCE2(self, DCTrans, ky, Cq2, Edc, me, ge, Ephn, N0, ne, Ee,
                 Vee, n, j, DC):
        """
        Calculate DC field contribution for electrons (version 2).

        Computes the DC field transport contribution to the electron
        distribution evolution using a finite difference derivative.

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
        DC[:] = 0.0
        gate = np.ones(len(ky))
        dk = ky[1] - ky[0] if len(ky) > 1 else 1.0

        Eec = EkReNorm(np.real(ne[:]), Ee[:], Vee[:, :])
        v = DriftVt(np.real(ne[:]), Eec[:], self.dkk)

        if self.WithPhns:
            Fd = FDrift2(Ephn, me, ge, ky, np.real(ne), Cq2, v, N0, self.xe)
        else:
            Fd = np.zeros(len(ky))

        self.ERate = np.sum(
            Fd[:] / hbar / (np.abs(ky) + 1e-5)**2 * ky[:] * ne[:]
        ) / (np.sum(ne[:]) + 1e-20)

        Fd = np.sum(Fd) / (np.sum(np.abs(ne)) + 1e-20) * 2.0

        if self.fe_file is not None:
            self.fe_file.write(f"{n} {Fd}\n")
            self.fe_file.flush()

        if not DCTrans:
            return

        DC0 = -(-e0 * Edc - Fd) * gate / hbar * ne[:]
        self.VEDrift = v

        DC0_shift_p1 = np.roll(DC0, -1)
        DC0_shift_m1 = np.roll(DC0, 1)
        DC0 = (DC0_shift_p1 - DC0_shift_m1) / 2.0 / dk

        DC[:] = np.real(DC0)

    def CalcDCH2(self, DCTrans, ky, Cq2, Edc, mh, gh, Ephn, N0, nh, Eh,
                 Vhh, n, j, DC):
        """
        Calculate DC field contribution for holes (version 2).

        Computes the DC field transport contribution to the hole
        distribution evolution using a finite difference derivative.

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
        DC[:] = 0.0
        gate = np.ones(len(ky))
        dk = ky[1] - ky[0] if len(ky) > 1 else 1.0

        Ehc = EkReNorm(np.real(nh[:]), Eh[:], Vhh[:, :])
        v = DriftVt(np.real(nh[:]), Ehc[:], self.dkk)

        if self.WithPhns:
            Fd = FDrift2(Ephn, mh, gh, ky, np.real(nh), Cq2, v, N0, self.xh)
        else:
            Fd = np.zeros(len(ky))

        self.HRate = np.sum(
            Fd[:] / hbar / (np.abs(ky) + 1e-5)**2 * ky[:] * nh[:]
        ) / (np.sum(nh[:]) + 1e-20)

        Fd = np.sum(Fd) / (np.sum(np.abs(nh)) + 1e-20) * 2.0

        if self.fh_file is not None:
            self.fh_file.write(f"{n} {Fd}\n")
            self.fh_file.flush()

        if not DCTrans:
            return

        DC0 = -(-e0 * Edc - Fd) * gate / hbar * nh[:]
        self.VHDrift = v

        DC0_shift_p1 = np.roll(DC0, -1)
        DC0_shift_m1 = np.roll(DC0, 1)
        DC0 = (DC0_shift_p1 - DC0_shift_m1) / 2.0 / dk

        DC[:] = np.real(DC0)

    # ------------------------------------------------------------------
    # DC field calculations (original: FFT-based derivative)
    # ------------------------------------------------------------------

    def CalcDCE(self, DCTrans, ky, Cq2, Edc, me, ge, Ephn, N0, ne, Ee,
                Vee, DC):
        """
        Calculate DC field contribution for electrons (original version).

        Uses FFT-based derivative.

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
        DC[:] = 0.0

        Eec = EkReNorm(np.real(ne[:]), Ee[:], Vee[:, :])

        if not DCTrans:
            return

        v = DriftVt(np.real(ne[:]), Eec[:], self.dkk)

        dndk = np.real(ne[:]).astype(complex)

        # FFT-based derivative using pyfftw
        dndk[:] = pyfftw.interfaces.numpy_fft.fft(dndk)
        dndk[:] = dndk[:] * (ii * self.Y[:])
        dndk[:] = pyfftw.interfaces.numpy_fft.ifft(dndk)

        if self.WithPhns:
            Fd = FDrift2(Ephn, me, ge, ky, np.real(ne), Cq2, v, N0, self.xe)
        else:
            Fd = np.zeros(len(ky))

        DC[:] = -(-e0 * Edc - Fd) / hbar * np.real(dndk)

        self.ERate = np.sum(
            Fd[:] / hbar / (np.abs(ky) + 1e-5)**2 * ky[:] * ne[:]
        ) / (np.sum(ne[:]) + 1e-20)

        self.VEDrift = v

    def CalcDCH(self, DCTrans, ky, Cq2, Edc, mh, gh, Ephn, N0, nh, Eh,
                Vhh, DC):
        """
        Calculate DC field contribution for holes (original version).

        Uses FFT-based derivative.

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
        DC[:] = 0.0

        Ehc = EkReNorm(np.real(nh[:]), Eh[:], Vhh[:, :])

        if not DCTrans:
            return

        v = DriftVt(np.real(nh[:]), Ehc[:], self.dkk)

        dndk = np.real(nh[:]).astype(complex)

        # FFT-based derivative using pyfftw
        dndk[:] = pyfftw.interfaces.numpy_fft.fft(dndk)
        dndk[:] = dndk[:] * (ii * self.Y[:])
        dndk[:] = pyfftw.interfaces.numpy_fft.ifft(dndk)

        if self.WithPhns:
            Fd = FDrift2(Ephn, mh, gh, ky, np.real(nh), Cq2, v, N0, self.xh)
        else:
            Fd = np.zeros(len(ky))

        DC[:] = -(+e0 * Edc - Fd) / hbar * np.real(dndk)

        self.HRate = np.sum(
            Fd[:] / hbar / (np.abs(ky) + 1e-5)**2 * ky[:] * nh[:]
        ) / (np.sum(nh[:]) + 1e-20)

        self.VHDrift = v

    # ------------------------------------------------------------------
    # Current calculation
    # ------------------------------------------------------------------

    def CalcI0(self, ne, nh, Ee, Eh, VC, dk, ky, I0):
        """
        Calculate total current from electron and hole distributions.

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
            Interaction matrix (J), 3D array
        dk : float
            Momentum step size (1/m)
        ky : ndarray
            Momentum coordinates (1/m), 1D array (unused)
        I0 : float
            Input value (unused)

        Returns
        -------
        float
            Total current (A)
        """
        self.dkk = dk

        Ec = EkReNorm(np.real(ne[:]), Ee[:], VC[:, :, 1])
        ve = DriftVt(np.real(ne[:]), Ec[:], self.dkk)

        Ec = EkReNorm(np.real(nh[:]), Eh[:], VC[:, :, 2])
        vh = DriftVt(np.real(nh[:]), Ec[:], self.dkk)

        v = ve + vh

        return -e0 * v * np.sum(ne[:]) * dk * 2.0

    # ------------------------------------------------------------------
    # FFT-based shift operations
    # ------------------------------------------------------------------

    def ShiftN1D(self, ne, dk):
        """
        Shift 1D distribution in momentum space.

        Shifts the distribution function in momentum space by dk using FFT.

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
        """
        ne[:] = pyfftw.interfaces.numpy_fft.fft(ne)
        ne[:] = ne[:] * np.exp(-ii * self.Y[:] * dk)
        ne[:] = pyfftw.interfaces.numpy_fft.ifft(ne)

    def ShiftN2D(self, C, dk):
        """
        Shift 2D distribution in momentum space.

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
        """
        Y2 = self.Y.copy()
        C[:, :] = pyfftw.interfaces.numpy_fft.fft2(C)
        Y_expanded = self.Y[:, np.newaxis] + Y2[np.newaxis, :]
        C[:, :] = C[:, :] * np.exp(-ii * Y_expanded * dk)
        C[:, :] = pyfftw.interfaces.numpy_fft.ifft2(C)

    def Transport(self, C, Edc, Eac, dt, DCTrans, k1nek2):
        """
        Transport step for distribution matrix.

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
            If True, shift the full 2D matrix; if False, shift only diagonal

        Returns
        -------
        None
            C array is modified in-place.
        """
        if not DCTrans:
            return

        dk = -e0 * (Edc + Eac) / hbar * dt

        if k1nek2:
            self.ShiftN2D(C, dk)
        else:
            nk = np.diag(C).copy()
            self.ShiftN1D(nk, dk)
            np.fill_diagonal(C, nk)

    # ------------------------------------------------------------------
    # Getter methods
    # ------------------------------------------------------------------

    def GetEDrift(self):
        """Return electron drift rate (Hz)."""
        return self.ERate

    def GetHDrift(self):
        """Return hole drift rate (Hz)."""
        return self.HRate

    def GetVEDrift(self):
        """Return electron drift velocity (m/s)."""
        return self.VEDrift

    def GetVHDrift(self):
        """Return hole drift velocity (m/s)."""
        return self.VHDrift


# ============================================================================
# Backward-compatibility layer
# ============================================================================

_instance = None
_WithPhns = True  # Module-level flag, read by InitializeDC


def _require_instance():
    """Return the singleton or raise if not yet initialized."""
    if _instance is None:
        raise RuntimeError(
            "DCFieldModule not initialized. Call InitializeDC() first."
        )
    return _instance


def InitializeDC(ky, me, mh):
    """
    Initialize the DC field module (backward-compatible wrapper).

    Creates a new :class:`DCFieldModule` singleton from *ky*, *me*, *mh*.
    Uses the module-level ``_WithPhns`` flag.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    me : float
        Effective electron mass (kg)
    mh : float
        Effective hole mass (kg)
    """
    global _instance
    _instance = DCFieldModule(ky, me, mh, WithPhns=_WithPhns)


# --- stateful wrappers (delegate to singleton) ---

def CalcDCE2(DCTrans, ky, Cq2, Edc, me, ge, Ephn, N0, ne, Ee, Vee, n, j,
             DC):
    """Backward-compatible wrapper for :meth:`DCFieldModule.CalcDCE2`."""
    _require_instance().CalcDCE2(DCTrans, ky, Cq2, Edc, me, ge, Ephn, N0,
                                 ne, Ee, Vee, n, j, DC)


def CalcDCH2(DCTrans, ky, Cq2, Edc, mh, gh, Ephn, N0, nh, Eh, Vhh, n, j,
             DC):
    """Backward-compatible wrapper for :meth:`DCFieldModule.CalcDCH2`."""
    _require_instance().CalcDCH2(DCTrans, ky, Cq2, Edc, mh, gh, Ephn, N0,
                                 nh, Eh, Vhh, n, j, DC)


def CalcDCE(DCTrans, ky, Cq2, Edc, me, ge, Ephn, N0, ne, Ee, Vee, DC):
    """Backward-compatible wrapper for :meth:`DCFieldModule.CalcDCE`."""
    _require_instance().CalcDCE(DCTrans, ky, Cq2, Edc, me, ge, Ephn, N0,
                                ne, Ee, Vee, DC)


def CalcDCH(DCTrans, ky, Cq2, Edc, mh, gh, Ephn, N0, nh, Eh, Vhh, DC):
    """Backward-compatible wrapper for :meth:`DCFieldModule.CalcDCH`."""
    _require_instance().CalcDCH(DCTrans, ky, Cq2, Edc, mh, gh, Ephn, N0,
                                nh, Eh, Vhh, DC)


def CalcI0(ne, nh, Ee, Eh, VC, dk, ky, I0):
    """Backward-compatible wrapper for :meth:`DCFieldModule.CalcI0`."""
    return _require_instance().CalcI0(ne, nh, Ee, Eh, VC, dk, ky, I0)


def GetEDrift():
    """Backward-compatible wrapper for :meth:`DCFieldModule.GetEDrift`."""
    return _require_instance().GetEDrift()


def GetHDrift():
    """Backward-compatible wrapper for :meth:`DCFieldModule.GetHDrift`."""
    return _require_instance().GetHDrift()


def GetVEDrift():
    """Backward-compatible wrapper for :meth:`DCFieldModule.GetVEDrift`."""
    return _require_instance().GetVEDrift()


def GetVHDrift():
    """Backward-compatible wrapper for :meth:`DCFieldModule.GetVHDrift`."""
    return _require_instance().GetVHDrift()


def ShiftN1D(ne, dk):
    """Backward-compatible wrapper for :meth:`DCFieldModule.ShiftN1D`."""
    _require_instance().ShiftN1D(ne, dk)


def ShiftN2D(C, dk):
    """Backward-compatible wrapper for :meth:`DCFieldModule.ShiftN2D`."""
    _require_instance().ShiftN2D(C, dk)


def Transport(C, Edc, Eac, dt, DCTrans, k1nek2):
    """Backward-compatible wrapper for :meth:`DCFieldModule.Transport`."""
    _require_instance().Transport(C, Edc, Eac, dt, DCTrans, k1nek2)


def __getattr__(name):
    """Module-level ``__getattr__`` for backward-compat attribute reads.

    Maps old ``dcfield._Y``, ``dcfield._xe``, etc. to the singleton's
    attributes.  Returns ``None`` when the singleton has not been created yet.
    """
    _attr_map = {
        '_Y': 'Y',
        '_xe': 'xe',
        '_xh': 'xh',
        '_qinv': 'qinv',
        '_ERate': 'ERate',
        '_HRate': 'HRate',
        '_VEDrift': 'VEDrift',
        '_VHDrift': 'VHDrift',
        '_dkk': 'dkk',
        '_kmin': 'kmin',
        '_kmax': 'kmax',
        '_fe_file': 'fe_file',
        '_fh_file': 'fh_file',
    }
    if name in _attr_map:
        if _instance is None:
            return None
        return getattr(_instance, _attr_map[name])
    raise AttributeError(f"module 'dcfield' has no attribute {name!r}")
