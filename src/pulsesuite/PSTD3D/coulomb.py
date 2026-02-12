"""
Coulomb interaction calculations for quantum wire simulations.

This module calculates the electron-hole, electron-electron, and hole-hole
collision integrals and other carrier-optical related calculations required
for the Semiconductor Bloch Equations in support of simulations of pulse
propagation through a quantum wire.

The primary API is the CoulombModule class, which encapsulates all pre-computed
arrays and methods (mirroring the Fortran ``module coulomb`` with private state).
Module-level wrapper functions are provided for backward compatibility.

Author: Rahul R. Sah

"""

import numpy as np
from numba import jit
from scipy.constants import e as e0, epsilon_0 as eps0, hbar as hbar_SI

from .usefulsubs import K03

# Physical constants
pi = np.pi
twopi = 2.0 * np.pi
hbar = hbar_SI
eV = 1.602176634e-19  # Electron volt in Joules
ii = 1j  # Imaginary unit


###############################################################################
# Module-level pure functions (no mutable state)
###############################################################################

# JIT-compatible K0 approximation for modified Bessel function
# nopython friendly version of K03 which is in usefulsubs.py
@jit(nopython=True, cache=True)
def _K03_jit(x):
    """
    JIT-compatible approximation of modified Bessel function K0(x).

    Uses asymptotic expansion for large x and series expansion for small x.
    This is a simplified version that works with numba nopython mode.
    """
    if x > 100.0:
        return 0.0

    if x < 1e-10:
        # K0(x) ~ -ln(x/2) - gamma for small x
        return 23.0  # Approximate large value for very small x

    if x < 2.0:
        # Series expansion for small x
        x2 = x * x
        result = -np.log(0.5 * x) - 0.5772156649015329
        term = 1.0
        for k in range(1, 20):
            term = term * 0.25 * x2 / (k * k)
            result = result + term * (1.0 / k - 0.5772156649015329)
            if abs(term) < 1e-15:
                break
        return result
    else:
        # Asymptotic expansion for large x: K0(x) ~ sqrt(pi/(2x)) * exp(-x)
        return np.sqrt(np.pi / (2.0 * x)) * np.exp(-x)


@jit(nopython=True, cache=True)
def _Vint_jit(Qyk, y, alphae, alphah, Delta0, N1, N2, Ny):
    """JIT-compiled version of Vint calculation."""
    Vint_val = 0.0

    # Pre-compute arrays (need to pass as parameters for JIT)
    aey2 = np.zeros(Ny)
    ahy2 = np.zeros(Ny)
    for i in range(Ny):
        aey2[i] = (alphae * y[i]) ** 2
        ahy2[i] = (alphah * y[i]) ** 2

    # Minimum momentum and actual momentum difference
    kmin = (alphae + alphah) / 4.0
    dk = max(abs(Qyk), kmin)

    # Multiplication constant
    multconst = alphae * alphah / np.pi * (y[1] - y[0]) ** 2

    # Double loop over integration region
    for i in range(N1, N2 + 1):
        for j in range(N1, N2 + 1):
            # Distance in y-direction with thickness contribution
            r_sq = (y[i] - y[j]) ** 2 + Delta0 ** 2
            r = np.sqrt(r_sq)

            # Exponential factor from wavefunction overlap
            exp_factor = np.exp(-aey2[i] - ahy2[j])

            # Modified Bessel function K0 for Coulomb interaction
            k0_arg = dk * r
            k0_val = _K03_jit(k0_arg)

            Vint_val += exp_factor * multconst * k0_val

    return Vint_val


def Vint(Qyk, y, alphae, alphah, Delta0):
    """
    Calculate the interaction integral for Coulomb potential.

    Computes the integral over spatial coordinates for the Coulomb interaction
    between particles with level separations alphae and alphah.

    Parameters
    ----------
    Qyk : float
        Momentum difference (1/m)
    y : ndarray
        Length coordinates of quantum wire (m), 1D array
    alphae : float
        Level separation for electrons (1/m)
    alphah : float
        Level separation for holes (1/m)
    Delta0 : float
        Thickness of the quantum wire (m)

    Returns
    -------
    float
        Interaction integral value (dimensionless)
    """
    Ny = len(y)
    N1 = Ny // 4
    N2 = 3 * Ny // 4

    # Try JIT-compiled version first
    try:
        result = _Vint_jit(Qyk, y, alphae, alphah, Delta0, N1, N2, Ny)
        if not hasattr(Vint, '_jit_used_printed'):
            print("Vint: Using JIT-compiled version")
            Vint._jit_used_printed = True
        return result
    except Exception:
        # Fallback to original implementation
        if not hasattr(Vint, '_fallback_used_printed'):
            print("Vint: JIT compilation failed, using fallback (slower)")
            Vint._fallback_used_printed = True

        Vint_val = 0.0
        # Pre-compute arrays
        aey2 = (alphae * y) ** 2
        ahy2 = (alphah * y) ** 2

        # Minimum momentum and actual momentum difference
        kmin = (alphae + alphah) / 4.0
        dk = max(abs(Qyk), kmin)

        # Multiplication constant
        multconst = alphae * alphah / pi * (y[1] - y[0]) ** 2

        # Double loop over integration region
        for i in range(N1, N2 + 1):
            for j in range(N1, N2 + 1):
                # Distance in y-direction with thickness contribution
                r_sq = (y[i] - y[j]) ** 2 + Delta0 ** 2
                r = np.sqrt(r_sq)

                # Exponential factor from wavefunction overlap
                exp_factor = np.exp(-aey2[i] - ahy2[j])

                # Modified Bessel function K0 for Coulomb interaction
                k0_arg = dk * r
                k0_val = K03(k0_arg)

                Vint_val += exp_factor * multconst * k0_val

        return Vint_val


def Vehint(k, q, y, ky, alphae, alphah, Delta0):
    """
    Calculate electron-hole interaction integral.

    Parameters
    ----------
    k : int
        Electron momentum index (1-based, matching Fortran)
    q : int
        Hole momentum index (1-based, matching Fortran)
    y : ndarray
        Length coordinates of quantum wire (m), 1D array
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    alphae : float
        Level separation for electrons (1/m)
    alphah : float
        Level separation for holes (1/m)
    Delta0 : float
        Thickness of the quantum wire (m)

    Returns
    -------
    float
        Electron-hole interaction integral value (dimensionless)
    """
    Vehint_val = 0.0
    Ny = len(y)

    # Pre-compute arrays
    aey2 = (alphae * y) ** 2
    ahy2 = (alphah * y) ** 2

    # Minimum momentum and actual momentum difference
    k_idx = k - 1 if k > 0 else 0
    q_idx = q - 1 if q > 0 else 0
    kmin = (alphae + alphah) / 4.0
    dk = max(abs(ky[k_idx] - ky[q_idx]), kmin)

    # Multiplication constant
    multconst = alphae * alphah / pi * (y[1] - y[0]) ** 2

    # Integration region: central portion of the array
    N1 = Ny // 4
    N2 = 3 * Ny // 4

    # Double loop over integration region
    for i in range(N1, N2 + 1):
        for j in range(N1, N2 + 1):
            # Distance in y-direction with thickness contribution
            r_sq = (y[i] - y[j]) ** 2 + Delta0 ** 2
            r = np.sqrt(r_sq)

            # Exponential factor from wavefunction overlap
            exp_factor = np.exp(-aey2[i] - ahy2[j])

            # Modified Bessel function K0 for Coulomb interaction
            k0_arg = dk * r
            k0_val = K03(k0_arg)

            Vehint_val += exp_factor * multconst * k0_val

    return Vehint_val


def GaussDelta(a, b):
    """
    Gaussian delta function approximation.

    Parameters
    ----------
    a : float
        Energy difference argument (J)
    b : float
        Broadening parameter (J)

    Returns
    -------
    float
        Gaussian delta function value (1/J)
    """
    if abs(b) < 1e-300:
        return 0.0
    return 1.0 / (np.sqrt(pi) * b) * np.exp(-(a / b) ** 2)


###############################################################################
# Array builder functions (pure — no module state)
###############################################################################

def MakeK3(ky):
    """
    Construct the k3 indexing array for momentum conservation.

    Creates a 3D array where k3(k1, k2, k4) = k1 + k2 - k4 (1-based).
    Invalid combinations (out of [1, N]) are stored as 0.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array

    Returns
    -------
    ndarray
        3D integer array of shape (N, N, N), values 1-based or 0 (invalid).
    """
    N = len(ky)
    k3 = np.zeros((N, N, N), dtype=np.int32)

    for k4 in range(N):
        for k2 in range(N):
            for k1 in range(N):
                k3i = (k1 + 1) + (k2 + 1) - (k4 + 1)
                if k3i < 1 or k3i > N:
                    k3i = 0
                k3[k1, k2, k4] = k3i

    return k3


def MakeQs(ky, ae, ah):
    """
    Construct the qe and qh momentum difference arrays.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    ae : float
        Level separation parameter for electrons (1/m)
    ah : float
        Level separation parameter for holes (1/m)

    Returns
    -------
    tuple of ndarray
        (qe, qh), each of shape (N, N)
    """
    Nk = len(ky)

    qe = np.zeros((Nk, Nk))
    qh = np.zeros((Nk, Nk))

    for k2 in range(Nk):
        for k1 in range(Nk):
            qe[k1, k2] = max(abs(ky[k2] - ky[k1]), ae / 2.0)

    for k2 in range(Nk):
        for k1 in range(Nk):
            qh[k1, k2] = max(abs(ky[k2] - ky[k1]), ah / 2.0)

    return qe, qh


def MakeUnDel(ky):
    """
    Construct the UnDel (1 - delta) array.

    Creates an array of shape (N+1, N+1) with:
    - Row/column 0 all zeros
    - Diagonal elements (i, i) for i=1..N are zero
    - All other elements are 1.0

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array

    Returns
    -------
    ndarray
        2D array of shape (N+1, N+1)
    """
    N = len(ky)
    UnDel = np.ones((N + 1, N + 1))

    UnDel[0, :] = 0.0
    UnDel[:, 0] = 0.0

    for i in range(1, N + 1):
        UnDel[i, i] = 0.0

    return UnDel


def CalcMBArrays(ky, Ee, Eh, ge, gh, k3, UnDel, LorentzDelta=False):
    """
    Calculate the many-body interaction arrays Ceh, Cee, Chh.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Ee : ndarray
        Electron energies (J), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    ge : float
        Electron inverse lifetime (Hz)
    gh : float
        Hole inverse lifetime (Hz)
    k3 : ndarray
        3D indexing array from MakeK3
    UnDel : ndarray
        2D array from MakeUnDel
    LorentzDelta : bool
        If True, use Lorentzian broadening; if False, use Gaussian delta.

    Returns
    -------
    tuple of ndarray
        (Ceh, Cee, Chh), each of shape (N+1, N+1, N+1)
    """
    N = len(ky)

    geh = (ge + gh) / 2.0
    hge2 = (hbar * ge) ** 2
    hgh2 = (hbar * gh) ** 2
    hgeh2 = (hbar * geh) ** 2

    Ceh = np.zeros((N + 1, N + 1, N + 1))
    Cee = np.zeros((N + 1, N + 1, N + 1))
    Chh = np.zeros((N + 1, N + 1, N + 1))

    if LorentzDelta:
        for k1 in range(N):
            for k2 in range(N):
                for k4 in range(N):
                    k1_1b = k1 + 1
                    k2_1b = k2 + 1
                    k4_1b = k4 + 1

                    # Veh(k1,k2,k30,k4)
                    k30_1b = k3[k4, k2, k1]
                    if k30_1b > 0:
                        k30 = k30_1b - 1
                        E_diff = Ee[k1] + Eh[k2] - Eh[k30] - Ee[k4]
                        Ceh[k1_1b, k2_1b, k4_1b] = (2.0 * geh * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] /
                                                   (E_diff ** 2 + hgeh2))

                    # Vee(k1,k2,k30,k4)
                    k30_1b = k3[k1, k2, k4]
                    if k30_1b > 0:
                        k30 = k30_1b - 1
                        E_diff = Ee[k1] + Ee[k2] - Ee[k30] - Ee[k4]
                        Cee[k1_1b, k2_1b, k4_1b] = (2.0 * ge * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] /
                                                   (E_diff ** 2 + hge2))

                    # Vhh(k1,k2,k30,k4)
                    k30_1b = k3[k1, k2, k4]
                    if k30_1b > 0:
                        k30 = k30_1b - 1
                        E_diff = Eh[k1] + Eh[k2] - Eh[k30] - Eh[k4]
                        Chh[k1_1b, k2_1b, k4_1b] = (2.0 * gh * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] /
                                                   (E_diff ** 2 + hgh2))
    else:
        for k1 in range(N):
            for k2 in range(N):
                for k4 in range(N):
                    k1_1b = k1 + 1
                    k2_1b = k2 + 1
                    k4_1b = k4 + 1

                    # Veh(k1,k2,k30,k4)
                    k30_1b = k3[k4, k2, k1]
                    if k30_1b > 0:
                        k30 = k30_1b - 1
                        E_diff = Ee[k1] + Eh[k2] - Eh[k30] - Ee[k4]
                        Ceh[k1_1b, k2_1b, k4_1b] = (twopi / hbar * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] *
                                                   GaussDelta(E_diff, hbar * geh))

                    # Vee(k1,k2,k30,k4)
                    k30_1b = k3[k1, k2, k4]
                    if k30_1b > 0:
                        k30 = k30_1b - 1
                        E_diff = Ee[k1] + Ee[k2] - Ee[k30] - Ee[k4]
                        Cee[k1_1b, k2_1b, k4_1b] = (twopi / hbar * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] *
                                                   GaussDelta(E_diff, hbar * ge))

                    # Vhh(k1,k2,k30,k4)
                    k30_1b = k3[k1, k2, k4]
                    if k30_1b > 0:
                        k30 = k30_1b - 1
                        E_diff = Eh[k1] + Eh[k2] - Eh[k30] - Eh[k4]
                        Chh[k1_1b, k2_1b, k4_1b] = (twopi / hbar * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] *
                                                   GaussDelta(E_diff, hbar * gh))

    return Ceh, Cee, Chh


def CalcCoulombArrays(y, ky, er, alphae, alphah, L, Delta0, Qy, kkp,
                      ReadArrays=False, ScrewThis=False):
    """
    Construct the unscreened Coulomb collision arrays Veh0, Vee0, Vhh0.

    Parameters
    ----------
    y : ndarray
        Length coordinates (m), 1D array
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    er : float
        Background dielectric constant
    alphae : float
        Level separation for electrons (1/m)
    alphah : float
        Level separation for holes (1/m)
    L : float
        Length of quantum wire (m)
    Delta0 : float
        Thickness of quantum wire (m)
    Qy : ndarray
        Momentum difference array (1/m)
    kkp : ndarray
        Index mapping array, 2D integer
    ReadArrays : bool, optional
        If True, read from files (not implemented). Default False.
    ScrewThis : bool, optional
        If True, return zero arrays. Default False.

    Returns
    -------
    tuple of ndarray
        (Veh0, Vee0, Vhh0), each of shape (N, N)
    """
    N = len(ky)
    NQ = len(Qy)

    Veh0 = np.zeros((N, N))
    Vee0 = np.zeros((N, N))
    Vhh0 = np.zeros((N, N))

    if ReadArrays:
        print("Warning: ReadArrays=True not implemented, returning zero arrays")
        return Veh0, Vee0, Vhh0

    if ScrewThis:
        return Veh0, Vee0, Vhh0

    print("Calculating Coulomb Arrays")

    eh = np.zeros(NQ)
    ee = np.zeros(NQ)
    hh = np.zeros(NQ)

    prefactor = e0 ** 2 / (twopi * eps0 * er * L)

    for k in range(NQ):
        if k % 10 == 0:
            print(f"  Progress: {k}/{NQ} ({100*k/NQ:.1f}%)")
        eh[k] = prefactor * Vint(Qy[k], y, alphae, alphah, Delta0)
        ee[k] = prefactor * Vint(Qy[k], y, alphae, alphae, Delta0)
        hh[k] = prefactor * Vint(Qy[k], y, alphah, alphah, Delta0)

    for k in range(N):
        for q in range(N):
            kkp_idx = kkp[k, q]
            if kkp_idx >= 0 and kkp_idx < NQ:
                Veh0[k, q] = eh[kkp_idx]
                Vee0[k, q] = ee[kkp_idx]
                Vhh0[k, q] = hh[kkp_idx]

    print("Finished Calculating Unscreened Coulomb Arrays")

    return Veh0, Vee0, Vhh0


def CalcChi1D(ky, alphae, alphah, Delta0, epsr, me, mh, qe, qh):
    """
    Calculate the 1D susceptibility arrays Chi1De and Chi1Dh.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    alphae : float
        Level separation for electrons (1/m)
    alphah : float
        Level separation for holes (1/m)
    Delta0 : float
        Thickness of quantum wire (m)
    epsr : float
        Background dielectric constant
    me : float
        Electron effective mass (kg)
    mh : float
        Hole effective mass (kg)
    qe : ndarray
        Electron momentum difference array from MakeQs
    qh : ndarray
        Hole momentum difference array from MakeQs

    Returns
    -------
    tuple of ndarray
        (Chi1De, Chi1Dh), each of shape (N, N)
    """
    N = len(ky)

    Chi1De = np.zeros((N, N))
    Chi1Dh = np.zeros((N, N))

    Re = np.sqrt((2.0 / alphae) ** 2 + Delta0 ** 2)
    Rh = np.sqrt((2.0 / alphah) ** 2 + Delta0 ** 2)

    for k2 in range(N):
        for k1 in range(N):
            Chi1De[k1, k2] = me * K03(qe[k1, k2] * Re) / qe[k1, k2]

    for k2 in range(N):
        for k1 in range(N):
            Chi1Dh[k1, k2] = mh * K03(qh[k1, k2] * Rh) / qh[k1, k2]

    scale = e0 ** 2 / (twopi * eps0 * epsr * hbar ** 2)
    Chi1De = Chi1De * scale
    Chi1Dh = Chi1Dh * scale

    return Chi1De, Chi1Dh


###############################################################################
# Standalone diagnostic functions (no module state needed)
###############################################################################

def GetChi1Dqw(alphae, alphah, Delta0, epsr, game, gamh, ky, Ee, Eh, ne, nh, qq, w):
    """
    Calculate the 1D quantum wire susceptibility chi(q, w).

    Parameters
    ----------
    alphae, alphah : float
        Level separation parameters (1/m)
    Delta0 : float
        Thickness of quantum wire (m)
    epsr : float
        Background dielectric constant
    game, gamh : ndarray
        Inverse lifetime arrays (Hz)
    ky : ndarray
        Momentum coordinates (1/m)
    Ee, Eh : ndarray
        Electron/hole energies (J)
    ne, nh : ndarray
        Electron/hole populations
    qq : float
        Momentum value (1/m)
    w : float
        Frequency (rad/s)

    Returns
    -------
    tuple of float
        (chir, chii) — real and imaginary parts of susceptibility
    """
    Nk = len(ky)
    dk = ky[1] - ky[0]

    hge = hbar * game
    hgh = hbar * gamh

    hge = np.maximum(hge, 1e-4 * eV)
    hgh = np.maximum(hgh, 1e-4 * eV)

    qmine = alphae / 2.0
    qminh = alphah / 2.0

    Re = np.sqrt((2.0 / alphae) ** 2 + Delta0 ** 2)
    Rh = np.sqrt((2.0 / alphah) ** 2 + Delta0 ** 2)

    beta = e0 ** 2 / (4 * pi * eps0 * epsr)

    ql = abs(qq)
    Ke = K03(max(ql, qminh) * Re) * beta * 2
    Kh = K03(max(ql, qmine) * Rh) * beta * 2

    chi = 0.0 + 0.0j
    q = int(round(qq / dk))

    k_start = max(0, -q)
    k_end = min(Nk, Nk - q)

    for k in range(k_start, k_end):
        k_plus_q = k + q
        if 0 <= k_plus_q < Nk:
            denom_e = hbar * w - Ee[k_plus_q] + Ee[k] + ii * (hge[k_plus_q] + hge[k])
            if abs(denom_e) > 1e-300:
                chi -= Ke / pi * (ne[k] - ne[k_plus_q]) / denom_e * dk

            denom_h = hbar * w - Eh[k_plus_q] + Eh[k] + ii * (hgh[k_plus_q] + hgh[k])
            if abs(denom_h) > 1e-300:
                chi -= Kh / pi * (nh[k] - nh[k_plus_q]) / denom_h * dk

    chir = np.real(chi)
    chii = np.imag(chi)

    return chir, chii


def GetEps1Dqw(alphae, alphah, Delta0, epsr, me, mh, n1D, q, w):
    """
    Calculate the 1D quantum wire dielectric function epsilon(q, w).

    Parameters
    ----------
    alphae, alphah : float
        Level separation parameters (1/m)
    Delta0 : float
        Thickness of quantum wire (m)
    epsr : float
        Background dielectric constant
    me, mh : float
        Effective masses (kg)
    n1D : float
        1D carrier density (1/m)
    q : float
        Momentum (1/m)
    w : float
        Frequency (rad/s)

    Returns
    -------
    tuple of float
        (epr, epi) — real and imaginary parts of dielectric function
    """
    qmine = alphae / 2.0
    qminh = alphah / 2.0

    Re = np.sqrt((2.0 / alphae) ** 2 + Delta0 ** 2)
    Rh = np.sqrt((2.0 / alphah) ** 2 + Delta0 ** 2)

    beta = e0 ** 2 / (4 * pi * eps0 * epsr)

    if abs(q) < 1.0:
        q = 1.0
    ql = abs(q)

    Ce = 2 * beta * me / pi / hbar ** 2 / ql
    Ch = 2 * beta * mh / pi / hbar ** 2 / ql

    Ke = K03(max(ql, qminh) * Re)
    Kh = K03(max(ql, qmine) * Rh)

    OhmEp = hbar * ql / 2.0 / me * abs(ql + pi * n1D)
    OhmEm = hbar * ql / 2.0 / me * abs(ql - pi * n1D)
    OhmHp = hbar * ql / 2.0 / mh * abs(ql + pi * n1D)
    OhmHm = hbar * ql / 2.0 / mh * abs(ql - pi * n1D)

    epr = 1.0 - Ce * Ke * np.log(abs((w ** 2 - OhmEm ** 2) / (w ** 2 - OhmEp ** 2))) - \
          Ch * Kh * np.log(abs((w ** 2 - OhmHm ** 2) / (w ** 2 - OhmHp ** 2)))

    epi = 0.0
    if min(OhmEm, OhmEp) < w < max(OhmEm, OhmEp):
        if min(OhmHm, OhmHp) < w < max(OhmHm, OhmHp):
            epi = -Ce * Ke - Ch * Kh

    if np.isnan(epr) or np.isnan(epi):
        print(f"NAN in EpsL(q,w) at (q,w) = {q}, {w}")
        print(f"Re[EpsL(q,w)] = {epr}")
        print(f"Im[EpsL(q,w)] = {epi}")

    return epr, epi


###############################################################################
# JIT core functions (module-level for numba compatibility)
###############################################################################

@jit(nopython=True, parallel=True)
def _CalcMVeh_core(p, Veh, MVeh, k3, UnDel):
    """Core JIT-compiled computation for CalcMVeh."""
    N = p.shape[0]
    Nf = p.shape[2]

    for f in range(Nf):
        for kp in range(N):
            for k in range(N):
                for q in range(N):
                    qp_1b = k3[kp, q, k]
                    if qp_1b > 0:
                        qp = qp_1b - 1
                        if 0 <= qp < N:
                            k_1b = k + 1
                            kp_1b = kp + 1
                            q_1b = q + 1
                            qp_1b_val = qp + 1

                            MVeh[k, kp, f] += (p[q, qp, f] * Veh[k, q] *
                                             UnDel[k_1b, q_1b] *
                                             UnDel[kp_1b, qp_1b_val])


@jit(nopython=True, parallel=True)
def _BGRenorm_core(ne, nh, Vee, Vhh, BGR, UnDel):
    """Core JIT-compiled computation for BGRenorm."""
    N = len(ne)

    for kp in range(N):
        for k in range(N):
            k_1b = k + 1
            kp_1b = kp + 1

            sum_hh = 0.0
            sum_ee = 0.0

            for i in range(N):
                i_1b = i + 1
                sum_hh += nh[i] * Vhh[i, kp] * UnDel[i_1b, kp_1b]
                sum_ee += ne[i] * Vee[i, k] * UnDel[i_1b, k_1b]

            BGR[k, kp] = -sum_hh - sum_ee


@jit(nopython=True, parallel=True)
def _EeRenorm_core(ne, Vee, BGR, UnDel):
    """Core JIT-compiled computation for EeRenorm."""
    N = len(ne)

    for kp in range(N):
        for k in range(N):
            k_1b = k + 1
            kp_1b = kp + 1

            sum_vkp_kp = 0.0
            sum_vkp_ud = 0.0
            sum_vk_k = 0.0
            sum_vk_ud = 0.0

            for i in range(N):
                i_1b = i + 1
                sum_vkp_kp += ne[i] * Vee[kp, kp]
                sum_vkp_ud += ne[i] * Vee[i, kp] * UnDel[i_1b, kp_1b]
                sum_vk_k += ne[i] * Vee[k, k]
                sum_vk_ud += ne[i] * Vee[i, k] * UnDel[i_1b, k_1b]

            BGR[k, kp] = (2.0 * sum_vkp_kp - sum_vkp_ud +
                          2.0 * sum_vk_k - sum_vk_ud -
                          2.0 * sum_vkp_kp - sum_vk_k)


@jit(nopython=True, parallel=True)
def _EhRenorm_core(nh, Vhh, BGR, UnDel):
    """Core JIT-compiled computation for EhRenorm."""
    N = len(nh)

    for kp in range(N):
        for k in range(N):
            k_1b = k + 1
            kp_1b = kp + 1

            sum_vkp_kp = 0.0
            sum_vkp_ud = 0.0
            sum_vk_k = 0.0
            sum_vk_ud = 0.0

            for i in range(N):
                i_1b = i + 1
                sum_vkp_kp += nh[i] * Vhh[kp, kp]
                sum_vkp_ud += nh[i] * Vhh[i, kp] * UnDel[i_1b, kp_1b]
                sum_vk_k += nh[i] * Vhh[k, k]
                sum_vk_ud += nh[i] * Vhh[i, k] * UnDel[i_1b, k_1b]

            BGR[k, kp] = (2.0 * sum_vkp_kp - sum_vkp_ud +
                          2.0 * sum_vk_k - sum_vk_ud -
                          2.0 * sum_vkp_kp - sum_vk_k)


@jit(nopython=True, parallel=True)
def _MBCE2_core(ne, nh, Veh2, Vee2, Win, Wout, k3, Ceh, Cee):
    """Core JIT-compiled computation for MBCE2."""
    Nk = len(ne) - 1

    for k in range(1, Nk + 1):
        for q1 in range(1, Nk + 1):
            for q2 in range(1, Nk + 1):
                kp = q1
                k1 = q2

                k1p_1b = k3[kp - 1, k1 - 1, k - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b
                    Win[k - 1] += (Veh2[k - 1, k1 - 1] * (1.0 - nh[kp]) * nh[k1p] *
                                   ne[k1] * Ceh[k - 1, kp - 1, k1 - 1])

                k1p_1b = k3[kp - 1, k - 1, k1 - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b
                    Wout[k - 1] += (Veh2[k1 - 1, k - 1] * (1.0 - ne[k1]) * (1.0 - nh[kp]) *
                                    nh[k1p] * Ceh[k1 - 1, kp - 1, k - 1])

                k2 = q1
                k4 = q2

                k30_1b = k3[k - 1, k2 - 1, k4 - 1]
                if k30_1b > 0:
                    k30 = k30_1b
                    Win[k - 1] += (Vee2[k - 1, k4 - 1] * (1.0 - ne[k2]) * ne[k30] *
                                   ne[k4] * Cee[k - 1, k2 - 1, k4 - 1])

                k30_1b = k3[k4 - 1, k2 - 1, k - 1]
                if k30_1b > 0:
                    k30 = k30_1b
                    Wout[k - 1] += (Vee2[k4 - 1, k - 1] * (1.0 - ne[k4]) * (1.0 - ne[k2]) *
                                    ne[k30] * Cee[k4 - 1, k2 - 1, k - 1])


@jit(nopython=True, parallel=True)
def _MBCE_core(ne, nh, Veh2, Vee2, Win, Wout, k3, Ceh, Cee):
    """Core JIT-compiled computation for MBCE (identical to MBCE2)."""
    Nk = len(ne) - 1

    for k in range(1, Nk + 1):
        for q1 in range(1, Nk + 1):
            for q2 in range(1, Nk + 1):
                kp = q1
                k1 = q2

                k1p_1b = k3[kp - 1, k1 - 1, k - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b
                    Win[k - 1] += (Veh2[k - 1, k1 - 1] * (1.0 - nh[kp]) * nh[k1p] *
                                   ne[k1] * Ceh[k - 1, kp - 1, k1 - 1])

                k1p_1b = k3[kp - 1, k - 1, k1 - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b
                    Wout[k - 1] += (Veh2[k1 - 1, k - 1] * (1.0 - ne[k1]) * (1.0 - nh[kp]) *
                                    nh[k1p] * Ceh[k1 - 1, kp - 1, k - 1])

                k2 = q1
                k4 = q2

                k30_1b = k3[k - 1, k2 - 1, k4 - 1]
                if k30_1b > 0:
                    k30 = k30_1b
                    Win[k - 1] += (Vee2[k - 1, k4 - 1] * (1.0 - ne[k2]) * ne[k30] *
                                   ne[k4] * Cee[k - 1, k2 - 1, k4 - 1])

                k30_1b = k3[k4 - 1, k2 - 1, k - 1]
                if k30_1b > 0:
                    k30 = k30_1b
                    Wout[k - 1] += (Vee2[k4 - 1, k - 1] * (1.0 - ne[k4]) * (1.0 - ne[k2]) *
                                    ne[k30] * Cee[k4 - 1, k2 - 1, k - 1])


@jit(nopython=True, parallel=True)
def _MBCH_core(ne, nh, Veh2, Vhh2, Win, Wout, k3, Ceh, Chh):
    """Core JIT-compiled computation for MBCH."""
    Nk = len(ne) - 1

    for kp in range(1, Nk + 1):
        for q1 in range(1, Nk + 1):
            for q2 in range(1, Nk + 1):
                k = q1
                k1 = q2

                k1p_1b = k3[kp - 1, k1 - 1, k - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b
                    Win[kp - 1] += (Veh2[k - 1, k1 - 1] * (1.0 - ne[k]) * nh[k1p] *
                                    ne[k1] * Ceh[k - 1, kp - 1, k1 - 1])

                k1p_1b = k3[kp - 1, k - 1, k1 - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b
                    Wout[kp - 1] += (Veh2[k1 - 1, k - 1] * (1.0 - ne[k]) * (1.0 - nh[k1p]) *
                                     ne[k1] * Ceh[k - 1, k1p - 1, k1 - 1])

                k2p = q1
                k4p = q2

                k3p_1b = k3[kp - 1, k2p - 1, k4p - 1]
                if k3p_1b > 0:
                    k3p = k3p_1b
                    Win[kp - 1] += (Vhh2[kp - 1, k4p - 1] * (1.0 - nh[k2p]) * nh[k3p] *
                                    nh[k4p] * Chh[kp - 1, k2p - 1, k4p - 1])

                k3p_1b = k3[k4p - 1, k2p - 1, kp - 1]
                if k3p_1b > 0:
                    k3p = k3p_1b
                    Wout[kp - 1] += (Vhh2[k4p - 1, kp - 1] * (1.0 - nh[k4p]) * (1.0 - nh[k2p]) *
                                     nh[k3p] * Chh[k4p - 1, k2p - 1, kp - 1])


###############################################################################
# CoulombModule class — encapsulates Fortran module state
###############################################################################

class CoulombModule:
    """
    Coulomb interaction module for quantum wire simulations.

    Encapsulates all pre-computed arrays required for Coulomb interaction
    calculations in the Semiconductor Bloch Equations.  Mirrors the Fortran
    ``module coulomb`` where private allocatable arrays hold persistent state.

    Parameters
    ----------
    y : ndarray
        Length coordinates of quantum wire (m)
    ky : ndarray
        Momentum coordinates (1/m)
    L : float
        Length of quantum wire (m)
    Delta0 : float
        Thickness of quantum wire (m)
    me, mh : float
        Effective electron/hole masses (kg)
    Ee, Eh : ndarray
        Electron/hole energies (J)
    ge, gh : float
        Inverse electron/hole lifetimes (Hz)
    alphae, alphah : float
        Level separation parameters (1/m)
    er : float
        Background dielectric constant
    Qy : ndarray
        Momentum difference array (1/m)
    kkp : ndarray
        Index mapping array (int)
    screened : bool
        Whether to use screened interactions
    LorentzDelta : bool
        If True use Lorentzian broadening, else Gaussian delta.

    Attributes
    ----------
    UnDel, k3, qe, qh, Ceh, Cee, Chh, Veh0, Vee0, Vhh0, Chi1De, Chi1Dh
        Pre-computed arrays (see builder functions for details).
    LorentzDelta : bool
        Current broadening mode flag.
    """

    def __init__(self, y, ky, L, Delta0, me, mh, Ee, Eh, ge, gh,
                 alphae, alphah, er, Qy, kkp, screened,
                 LorentzDelta=False):
        self.LorentzDelta = LorentzDelta

        # Utility arrays
        self.UnDel = MakeUnDel(ky)
        self.k3 = MakeK3(ky)
        self.qe, self.qh = MakeQs(ky, alphae, alphah)

        # Many-body interaction arrays
        self.Ceh, self.Cee, self.Chh = CalcMBArrays(
            ky, Ee, Eh, ge, gh, self.k3, self.UnDel, self.LorentzDelta)

        # Unscreened Coulomb arrays
        self.Veh0, self.Vee0, self.Vhh0 = CalcCoulombArrays(
            y, ky, er, alphae, alphah, L, Delta0, Qy, kkp)

        # Susceptibility arrays
        self.Chi1De, self.Chi1Dh = CalcChi1D(
            ky, alphae, alphah, Delta0, er, me, mh, self.qe, self.qh)

    def SetLorentzDelta(self, boolean):
        """Set the LorentzDelta flag."""
        self.LorentzDelta = bool(boolean)

    # ----- Screening --------------------------------------------------------

    def Eps1D(self, n1D, Nk=None):
        """
        Calculate the 1D dielectric function matrix.

        Parameters
        ----------
        n1D : float
            1D carrier density (1/m)
        Nk : int, optional
            Momentum grid size (ignored — derived from stored arrays).

        Returns
        -------
        ndarray
            2D dielectric function matrix.
        """
        eps1d = (np.ones_like(self.Chi1De) -
                 self.Chi1De * 2 * np.log(np.abs((self.qe - pi * n1D) / (self.qe + n1D))) -
                 self.Chi1Dh * 2 * np.log(np.abs((self.qh - pi * n1D) / (self.qh + n1D))))
        return eps1d

    def CalcScreenedArrays(self, screened, L, ne, nh, VC, E1D):
        """
        Calculate (optionally screened) Coulomb interaction arrays.

        Fills *VC* and *E1D* in-place.

        Parameters
        ----------
        screened : bool
            Apply screening?
        L : float
            Quantum wire length (m)
        ne, nh : ndarray
            Carrier populations
        VC : ndarray, shape (N, N, 3)
            Output interaction matrices (modified in-place)
        E1D : ndarray, shape (N, N)
            Output dielectric function (modified in-place)
        """
        N = len(ne)

        VC[:, :, 0] = self.Veh0
        VC[:, :, 1] = self.Vee0
        VC[:, :, 2] = self.Vhh0
        E1D[:, :] = 1.0

        if screened:
            density_1D = np.sum(np.real(ne) + np.real(nh)) / 2.0 / L
            density_max = min(self.qe[1, 1], self.qh[1, 1]) / pi * 0.99
            density_1D = min(density_1D, density_max)

            E1D[:, :] = self.Eps1D(density_1D)

            VC[:, :, 0] = VC[:, :, 0] / E1D
            VC[:, :, 1] = VC[:, :, 1] / E1D
            VC[:, :, 2] = VC[:, :, 2] / E1D

    # ----- SBE terms --------------------------------------------------------

    def CalcMVeh(self, p, VC, MVeh):
        """
        Calculate the many-body electron-hole interaction term MVeh.

        Modifies *MVeh* in-place.
        """
        Veh = VC[:, :, 0]
        MVeh[:, :, :] = 0.0
        _CalcMVeh_core(p, Veh, MVeh, self.k3, self.UnDel)

    def undell(self, k, q):
        """Access UnDel(k, q) (1-based indices)."""
        return self.UnDel[k, q]

    def BGRenorm(self, C, D, VC, BGR):
        """
        Calculate band gap renormalization.

        Modifies *BGR* in-place.
        """
        Vee = VC[:, :, 1]
        Vhh = VC[:, :, 2]
        ne = np.diag(C)
        nh = np.diag(D)
        BGR[:, :] = 0.0
        _BGRenorm_core(ne, nh, Vee, Vhh, BGR, self.UnDel)

    def EeRenorm(self, ne, VC, BGR):
        """
        Calculate electron energy renormalization.

        Modifies *BGR* in-place.
        """
        Vee = VC[:, :, 1]
        BGR[:, :] = 0.0
        _EeRenorm_core(ne, Vee, BGR, self.UnDel)

    def EhRenorm(self, nh, VC, BGR):
        """
        Calculate hole energy renormalization.

        Modifies *BGR* in-place.
        """
        Vhh = VC[:, :, 2]
        BGR[:, :] = 0.0
        _EhRenorm_core(nh, Vhh, BGR, self.UnDel)

    # ----- Many-body relaxation (non-Hartree-Fock) --------------------------

    def MBCE2(self, ne0, nh0, ky, Ee, Eh, VC, geh, ge, Win, Wout):
        """
        Many-body Coulomb in/out rates for electrons (version 2).

        Modifies *Win* and *Wout* in-place.
        """
        Nk = len(ne0)
        Veh2 = VC[:, :, 0] ** 2
        Vee2 = VC[:, :, 1] ** 2
        ne = np.zeros(Nk + 1)
        nh = np.zeros(Nk + 1)
        ne[1:Nk + 1] = np.abs(ne0)
        nh[1:Nk + 1] = np.abs(nh0)
        _MBCE2_core(ne, nh, Veh2, Vee2, Win, Wout, self.k3, self.Ceh, self.Cee)

    def MBCE(self, ne0, nh0, ky, Ee, Eh, VC, geh, ge, Win, Wout):
        """
        Many-body Coulomb in/out rates for electrons.

        Modifies *Win* and *Wout* in-place.
        """
        Nk = len(ne0)
        Veh2 = VC[:, :, 0] ** 2
        Vee2 = VC[:, :, 1] ** 2
        ne = np.zeros(Nk + 1)
        nh = np.zeros(Nk + 1)
        ne[1:Nk + 1] = np.abs(ne0)
        nh[1:Nk + 1] = np.abs(nh0)
        _MBCE_core(ne, nh, Veh2, Vee2, Win, Wout, self.k3, self.Ceh, self.Cee)

    def MBCH(self, ne0, nh0, ky, Ee, Eh, VC, geh, gh, Win, Wout):
        """
        Many-body Coulomb in/out rates for holes.

        Modifies *Win* and *Wout* in-place.
        """
        Nk = len(ne0)
        Veh2 = VC[:, :, 0] ** 2
        Vhh2 = VC[:, :, 2] ** 2
        ne = np.zeros(Nk + 1)
        nh = np.zeros(Nk + 1)
        ne[1:Nk + 1] = np.abs(ne0)
        nh[1:Nk + 1] = np.abs(nh0)
        _MBCH_core(ne, nh, Veh2, Vhh2, Win, Wout, self.k3, self.Ceh, self.Chh)


###############################################################################
# Backward-compatibility layer (singleton + module-level wrappers)
#
# SBEs.py does:
#   from .coulomb import (CalcScreenedArrays, SetLorentzDelta,
#                          GetEps1Dqw, GetChi1Dqw, InitializeCoulomb,
#                          MBCE, MBCH)
# These wrappers keep that import working without changes.
###############################################################################

_instance = None        # Module-level CoulombModule singleton
_LorentzDelta = False   # Needed because SetLorentzDelta is called before init


def SetLorentzDelta(boolean):
    """Set the LorentzDelta flag (module-level wrapper)."""
    global _LorentzDelta
    _LorentzDelta = bool(boolean)
    if _instance is not None:
        _instance.SetLorentzDelta(boolean)


def InitializeCoulomb(y, ky, L, Delta0, me, mh, Ee, Eh, ge, gh,
                      alphae, alphah, er, Qy, kkp, screened):
    """Create / replace the module-level CoulombModule singleton."""
    global _instance
    _instance = CoulombModule(
        y, ky, L, Delta0, me, mh, Ee, Eh, ge, gh,
        alphae, alphah, er, Qy, kkp, screened,
        LorentzDelta=_LorentzDelta)


def _require_instance():
    if _instance is None:
        raise ValueError("CoulombModule not initialized. Call InitializeCoulomb first.")
    return _instance


# Wrappers that delegate to the singleton (used by SBEs.py)

def CalcScreenedArrays(screened, L, ne, nh, VC, E1D):
    _require_instance().CalcScreenedArrays(screened, L, ne, nh, VC, E1D)

def Eps1D(n1D, Nk):
    return _require_instance().Eps1D(n1D, Nk)

def CalcMVeh(p, VC, MVeh, k3=None, UnDel=None):
    _require_instance().CalcMVeh(p, VC, MVeh)

def undell(k, q):
    return _require_instance().undell(k, q)

def BGRenorm(C, D, VC, BGR, UnDel=None):
    _require_instance().BGRenorm(C, D, VC, BGR)

def EeRenorm(ne, VC, BGR, UnDel=None):
    _require_instance().EeRenorm(ne, VC, BGR)

def EhRenorm(nh, VC, BGR, UnDel=None):
    _require_instance().EhRenorm(nh, VC, BGR)

def MBCE2(ne0, nh0, ky, Ee, Eh, VC, geh, ge, Win, Wout, k3=None, Ceh=None, Cee=None):
    _require_instance().MBCE2(ne0, nh0, ky, Ee, Eh, VC, geh, ge, Win, Wout)

def MBCE(ne0, nh0, ky, Ee, Eh, VC, geh, ge, Win, Wout, k3=None, Ceh=None, Cee=None):
    _require_instance().MBCE(ne0, nh0, ky, Ee, Eh, VC, geh, ge, Win, Wout)

def MBCH(ne0, nh0, ky, Ee, Eh, VC, geh, gh, Win, Wout, k3=None, Ceh=None, Chh=None):
    _require_instance().MBCH(ne0, nh0, ky, Ee, Eh, VC, geh, gh, Win, Wout)


# Module-level __getattr__ for backward-compat reads of coulomb._k3, etc.
def __getattr__(name):
    _attr_map = {
        '_k3': 'k3', '_UnDel': 'UnDel', '_qe': 'qe', '_qh': 'qh',
        '_Ceh': 'Ceh', '_Cee': 'Cee', '_Chh': 'Chh',
        '_Veh0': 'Veh0', '_Vee0': 'Vee0', '_Vhh0': 'Vhh0',
        '_Chi1De': 'Chi1De', '_Chi1Dh': 'Chi1Dh',
        '_LorentzDelta': 'LorentzDelta',
    }
    if name in _attr_map:
        if _instance is None:
            return None
        return getattr(_instance, _attr_map[name])
    raise AttributeError(f"module 'coulomb' has no attribute {name!r}")
