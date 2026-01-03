"""
Emission calculations for quantum wire simulations.

This module calculates spontaneous emission and photoluminescence spectrum
for quantum wire systems in support of propagation simulations.
Author: Rahul R. Sah
"""

import numpy as np
from scipy.constants import hbar as hbar_SI, k as kB_SI, epsilon_0 as eps0_SI, c as c0_SI
from ..libpulsesuite.helpers import LinearInterp_dp, LinearInterp_dpc
from .usefulsubs import softtheta, Lrtz, Temperature

# Physical constants
pi = np.pi
hbar = hbar_SI
kB = kB_SI
eps0 = eps0_SI
c0 = c0_SI

# Module-level state variables (matching Fortran module variables)
_RScale = 0.0  # Scaling factor for emission calculations
_Temp = 77.0  # Temperature of QW solid (K)
_HOmega = None  # Energy array for integration
_square = None  # Pre-calculated square array
_idel = None  # Identity-like matrix (1 - identity)


def InitializeEmission(ky, Ee, Eh, dcv, epsr, geh, ehint):
    """
    Initialize the emission module.

    Sets up all module-level arrays required for emission calculations.
    Allocates and initializes the idel matrix, calculates RScale,
    and pre-calculates HOmega and square arrays.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    Ee : ndarray
        Electron energies (J), 1D array (not used, kept for signature consistency)
    Eh : ndarray
        Hole energies (J), 1D array (not used, kept for signature consistency)
    dcv : float
        Dipole matrix element (C·m)
    epsr : float
        Relative dielectric constant
    geh : float
        Electron-hole dephasing rate (Hz)
    ehint : float
        Electron-hole interaction strength

    Returns
    -------
    None
        All arrays are stored as module-level variables.
    """
    global _RScale, _HOmega, _square, _idel

    Nk = len(ky)

    # Initialize idel matrix: 1 - identity matrix
    _idel = np.ones((Nk, Nk), dtype=np.float64)
    np.fill_diagonal(_idel, 0.0)

    # Calculate RScale
    _RScale = 3.0 * dcv**2 / eps0 / np.sqrt(epsr) * ehint**2

    # Calculate HOmega array
    CalcHOmega(kB * _Temp, hbar * geh)

    # Pre-calculate square array
    _square = np.zeros(len(_HOmega), dtype=np.float64)
    _square[:] = (3.0 * dcv**2 * ehint / eps0 / np.sqrt(epsr) * ehint / hbar *
                   Lrtz(_HOmega[:], hbar * geh) * np.exp(-_HOmega[:] / kB / _Temp))


def SpontEmission(ne, nh, Ee, Eh, gap, geh, VC, Rsp):
    """
    Calculate spontaneous emission rates.

    Computes the spontaneous emission rate Rsp for each momentum state
    based on electron and hole occupation numbers and energies.

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
    gap : float
        Band gap energy (J)
    geh : float
        Electron-hole dephasing rate (Hz) (not used, kept for signature consistency)
    VC : ndarray
        Coulomb interaction array, shape (Nk, Nk, 3)
    Rsp : ndarray
        Spontaneous emission rates (modified in-place), 1D array

    Returns
    -------
    None
        Rsp is modified in-place.
    """
    Rsp[:] = 0.0

    Ek = gap + Ee + Eh + Ec(np.real(ne), np.real(nh), VC)

    for k in range(len(ne)):
        Rsp[k] = SpontIntegral(Ek[k])


def Ec(ne, nh, VC):
    """
    Calculate Coulomb energy correction.

    Computes the many-body Coulomb energy correction for each momentum state
    due to electron-electron, hole-hole, and electron-hole interactions.

    Parameters
    ----------
    ne : ndarray
        Electron carrier populations, 1D array
    nh : ndarray
        Hole carrier populations, 1D array
    VC : ndarray
        Coulomb interaction array, shape (Nk, Nk, 3)
        VC[:, :, 0] = Veh (electron-hole)
        VC[:, :, 1] = Vee (electron-electron)
        VC[:, :, 2] = Vhh (hole-hole)

    Returns
    -------
    ndarray
        Coulomb energy correction for each momentum state, 1D array
    """
    Nk = len(ne)
    Ec_result = np.zeros(Nk, dtype=np.float64)

    Veh = VC[:, :, 0]
    Vee = VC[:, :, 1]
    Vhh = VC[:, :, 2]

    for k in range(Nk):
        Ec_result[k] = (np.sum(ne[:] * (Vee[k, k] - Vee[:, k]) + nh[:] * (Vhh[k, k] - Vhh[:, k])) +
                        np.sum(ne[:] * Veh[k, k] * _idel[:, k] - nh[:] * Veh[k, k] * _idel[:, k]) -
                        Veh[k, k])

    return Ec_result


def SpontIntegral(Ek):
    """
    Calculate spontaneous emission integral numerically.

    Computes the integral over photon energy for spontaneous emission
    using pre-calculated HOmega and square arrays.

    Parameters
    ----------
    Ek : float
        Transition energy (J)

    Returns
    -------
    float
        Spontaneous emission integral value
    """
    if _HOmega is None or len(_HOmega) < 2:
        raise ValueError("HOmega not initialized. Call InitializeEmission first.")

    dhw = _HOmega[1] - _HOmega[0]

    Integrand = (_HOmega[:] + Ek) * rho0(_HOmega[:] + Ek) * _square[:]

    return np.sum(Integrand) * dhw


def rho0(hw):
    """
    Photon density of states as a function of photon energy.

    Calculates the photon density of states rho0(hw) = hw^2 / (c0^3 * pi^2 * hbar^3).

    Parameters
    ----------
    hw : float or ndarray
        Photon energy hbar*omega (J)

    Returns
    -------
    float or ndarray
        Photon density of states (1/(J·m^3))
    """
    return hw**2 / (c0**3 * pi**2 * hbar**3)


def CalcHOmega(kBT, hg):
    """
    Calculate HOmega array for integration.

    Sets up the energy array HOmega used for numerical integration
    in spontaneous emission calculations.

    Parameters
    ----------
    kBT : float
        Thermal energy k_B * T (J)
    hg : float
        Electron-hole dephasing energy hbar * geh (J)

    Returns
    -------
    None
        HOmega is stored as module-level variable.

    Raises
    ------
    ValueError
        If calculated Nw < 10 (temperature too low).
    """
    global _HOmega

    hwmax = (kBT + hg) * 4.0
    dhw = min(kBT, hg) / 20.0
    Nw = int(np.ceil(hwmax / dhw))

    if Nw < 10:
        raise ValueError("Error: temperature is too low in emission.py")

    _HOmega = np.zeros(Nw, dtype=np.float64)
    for n in range(Nw):
        _HOmega[n] = 0.0 + (n + 0.5) * dhw


def Calchw(hw, PLS, Estart, Emax):
    """
    Calculate photon energy array for PL spectrum.

    Sets up the photon energy array hw for photoluminescence spectrum
    calculations.

    Parameters
    ----------
    hw : ndarray
        Photon energy array (modified in-place), 1D array
    PLS : ndarray
        PL spectrum array (initialized to zero), 1D array
    Estart : float
        Starting energy (J)
    Emax : float
        Maximum energy (J)

    Returns
    -------
    None
        hw and PLS are modified in-place.
    """
    Nw = len(hw)
    hw[:] = 0.0
    PLS[:] = 0.0

    dhw = (Emax - Estart) / float(Nw)

    for w in range(Nw):
        hw[w] = Estart + w * dhw


def PLSpectrum(ne, nh, Ee, Eh, gap, geh, VC, hw, t, PLS):
    """
    Calculate photoluminescence spectrum.

    Computes the photoluminescence spectrum PLS as a function of photon energy hw
    based on electron and hole occupation numbers and energies.

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
    gap : float
        Band gap energy (J)
    geh : float
        Electron-hole dephasing rate (Hz)
    VC : ndarray
        Coulomb interaction array, shape (Nk, Nk, 3)
    hw : ndarray
        Photon energy array (J), 1D array
    t : float
        Time (s)
    PLS : ndarray
        PL spectrum (modified in-place), 1D array

    Returns
    -------
    None
        PLS is modified in-place.
    """
    Nk = len(ne)
    X = 100

    Ek = gap + Ee + Eh + Ec(np.abs(ne), np.abs(nh), VC)

    Te = Temperature(ne, Ee)
    Th = Temperature(nh, Eh)

    tempavg = (Te + Th) / 2.0

    E = np.zeros(X * Nk, dtype=np.float64)
    nenh = np.zeros(X * Nk, dtype=np.float64)
    qy = np.zeros(X * Nk, dtype=np.float64)
    ky = np.zeros(Nk, dtype=np.float64)

    for k in range(Nk):
        ky[k] = float(k)
    for k in range(X * Nk):
        qy[k] = float(k) / float(X)

    for k in range(X * Nk):
        E[k] = LinearInterp_dp(Ek[:], ky[:], qy[k])
        nenh[k] = np.abs(LinearInterp_dpc(ne * nh, ky[:], qy[k]))

    # Parallel loop over photon energies
    for w in range(len(hw)):
        PLS[w] = (PLS[w] + _RScale * np.sum(hw[w] * rho0(hw[w]) * nenh[:] *
                                             np.exp(-np.abs(hw[w] - E[:]) / kB / tempavg) *
                                             Lrtz(hw[w] - E[:], hbar * geh) *
                                             softtheta(hw[w] - E[X * Nk // 2], hbar * geh)))

    PLS[:] = PLS[:] * softtheta(t, geh)
