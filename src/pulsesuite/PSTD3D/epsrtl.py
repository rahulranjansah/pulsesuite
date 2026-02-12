"""
Dielectric permittivity calculations for quantum wire simulations.

This module calculates longitudinal and transverse permittivity
for quantum wire systems.

Author: Rahul R. Sah
"""

import os

import numpy as np
from numba import jit
from scipy.constants import e as e0, epsilon_0 as eps0, hbar as hbar_SI, k as kB_SI

from .usefulsubs import K03, theta

# Physical constants
pi = np.pi
twopi = 2.0 * np.pi
hbar = hbar_SI
kB = kB_SI
ii = 1j  # Imaginary unit
c0 = 299792458.0  # Speed of light (m/s)

# Module-level state variables (matching Fortran module variables)
_Nw = 500
_dw = 2.35e15 / 500.0 * 2
_R0 = 1e-9
_g = 1.0 / 1e-12
_epsb = 3.011**2
_dcv = 0.0
_n00 = 0.0
_kf = 0.0


def Eng(m, k):
    """
    Calculate energy from momentum. (Placed below in epsrtl.f90)

    Computes kinetic energy: E = hbar^2 * k^2 / (2 * m)

    Parameters
    ----------
    m : float or ndarray
        Mass (kg)
    k : float or ndarray
        Momentum (1/m)

    Returns
    -------
    float or ndarray
        Energy (J)
    """
    return hbar**2 * k**2 / 2.0 / m


def ff0(E, T, m):
    """
    Fermi function at finite temperature.

    Computes: n00 * sqrt(hbar^2 / (2*pi * m * kB * T)) * exp(-E/(kB*T))

    Parameters
    ----------
    E : float or ndarray
        Energy (J)
    T : float or ndarray
        Temperature (K)
    m : float or ndarray
        Mass (kg)

    Returns
    -------
    float or ndarray
        Fermi function value

    Notes
    -----
    Uses module-level variable _n00 for carrier density.
    """
    # Read-only access to global _n00
    return _n00 * np.sqrt(hbar**2 / twopi / m / kB / T) * np.exp(-E / kB / T)


def fT0(k, kf):
    """
    Fermi function at zero temperature.

    Computes: 1 - theta(|k| - kf)

    Parameters
    ----------
    k : float or ndarray
        Momentum (1/m)
    kf : float
        Fermi momentum (1/m)

    Returns
    -------
    float or ndarray
        Fermi function value at zero temperature
    """
    return 1.0 - theta(np.abs(k) - kf)


def atanJG(z):
    """
    Arctangent function.

    Computes: log((ii - z) / (ii + z)) / (2 * ii)

    Parameters
    ----------
    z : complex or ndarray
        Complex argument

    Returns
    -------
    complex or ndarray
        Arctangent value
    """
    return np.log((ii - z) / (ii + z)) / (2.0 * ii)


def atanhc(x):
    """
    Hyperbolic arctangent function.

    Computes: 0.5 * log((1 + x) / (1 - x))

    Parameters
    ----------
    x : complex or ndarray
        Complex argument

    Returns
    -------
    complex or ndarray
        Hyperbolic arctangent value
    """
    return 0.5 * np.log((1.0 + x) / (1.0 - x))


@jit(nopython=True)
def _PiT_jit(w, me, mh, Te, Th, dk, Ek, Ekq, n00_val, hbar_val, kB_val, twopi_val, pi_val):
    """JIT-compiled version of PiT. (Do I even need this?)"""
    a = 2.0 / pi_val * dk
    g = 2.35e15 / 1000.0

    N = len(Ek)
    result_real = 0.0
    result_imag = 0.0

    for i in range(N):
        # ff0 calculation
        ff0_Ek = n00_val * np.sqrt(hbar_val**2 / twopi_val / mh / kB_val / Th) * np.exp(-Ek[i] / kB_val / Th)
        ff0_Ekq = n00_val * np.sqrt(hbar_val**2 / twopi_val / me / kB_val / Te) * np.exp(-Ekq[i] / kB_val / Te)

        denom1_real = hbar_val * w - Ekq[i] - Ek[i]
        denom1_imag = hbar_val * g
        denom1_mag2 = denom1_real**2 + denom1_imag**2

        # Note: second term has 0d0 in numerator, so it's zero
        result_real += (1.0 - ff0_Ek - ff0_Ekq) * (denom1_real / denom1_mag2)
        result_imag += (1.0 - ff0_Ek - ff0_Ekq) * (-denom1_imag / denom1_mag2)

    return a * result_real, a * result_imag


def PiT(q, w, me, mh, Te, Th, dk, Ek, Ekq):
    """
    Calculate transverse polarization function.

    Computes the transverse polarization function PiT for permittivity calculations.

    Parameters
    ----------
    q : float
        Momentum transfer (1/m) (unused, kept for interface compatibility)
    w : float
        Frequency (Hz)
    me : float
        Electron mass (kg)
    mh : float
        Hole mass (kg)
    Te : float
        Electron temperature (K)
    Th : float
        Hole temperature (K)
    dk : float
        Momentum step size (1/m)
    Ek : ndarray
        Energy array for holes (J)
    Ekq : ndarray
        Energy array for electrons (J)

    Returns
    -------
    complex
        Transverse polarization function value

    Notes
    -----
    The function computes:
    PiT = (2/pi) * dk * sum((1 - ff0(Ek,Th,mh) - ff0(Ekq,Te,me)) *
          (1 / (hbar*w - Ekq - Ek + i*hbar*g)))
    """
    try:
        result_real, result_imag = _PiT_jit(w, me, mh, Te, Th, dk, Ek, Ekq, _n00, hbar, kB, twopi, pi)
        return result_real + 1j * result_imag
    except Exception:
        # Fallback to pure Python
        a = 2.0 / pi * dk
        g = 2.35e15 / 1000.0

        ff0_Ek = ff0(Ek, Th, mh)
        ff0_Ekq = ff0(Ekq, Te, me)

        result = a * np.sum((1.0 - ff0_Ek - ff0_Ekq) *
                           (1.0 / (hbar * w - Ekq - Ek + ii * hbar * g)))

        return result


def GetEpsrLEpsrT(n1D, dcv0, Te, Th, me, mh, Eg, ky):
    """
    Get longitudinal and transverse permittivity.

    Sets up momentum arrays and calls functions to calculate permittivity
    at zero temperature.

    Parameters
    ----------
    n1D : float
        1D carrier density (1/m)
    dcv0 : float
        Dipole matrix element (C*m)
    Te : float
        Electron temperature (K) (unused, kept for interface compatibility)
    Th : float
        Hole temperature (K) (unused, kept for interface compatibility)
    me : float
        Effective electron mass (kg)
    mh : float
        Effective hole mass (kg)
    Eg : float
        Band gap (J) (unused, kept for interface compatibility)
    ky : ndarray
        Momentum coordinates (1/m), 1D array

    Returns
    -------
    None

    Notes
    -----
    This function sets up module-level variables and calls ZeroT_L and ZeroT_T.
    Writes output to 'dataQW/Wire/qw.dat'.
    """
    global _n00, _kf, _dcv

    Nq = (len(ky) - 1) * 100 + 1
    qy = np.zeros(Nq)
    dq = (ky[1] - ky[0]) / 50.0

    for n in range(Nq):
        qy[n] = -(Nq - 1) / 2.0 * dq + n * dq

    _n00 = n1D
    _kf = _n00 / 2.0
    _dcv = dcv0

    ZeroT_L("E", me, qy, _kf)
    ZeroT_L("H", mh, qy, _kf)
    ZeroT_T(me, mh, Eg, dcv0, qy, _kf)

    os.makedirs('dataQW/Wire', exist_ok=True)
    with open('dataQW/Wire/qw.dat', 'w', encoding='utf-8') as f:
        f.write(f"Nq {Nq}\n")
        f.write(f"Nw {_Nw * 2 + 1}\n")
        f.write(f"ky(1) {qy[0]}\n")
        f.write(f"dky {qy[1] - qy[0]}\n")
        f.write(f"w(1) {-_Nw * _dw}\n")
        f.write(f"dw {_dw}\n")

    # Note: stop statement removed in Python version


@jit(nopython=True)
def _RecordEpsrT_loop_jit(ky_size, Nw, dw, a, me, mh, Te, Th, dk, ky, Ek, Eg,
                          n00_val, hbar_val, kB_val, twopi_val, pi_val):
    """JIT-compiled loop for RecordEpsrT."""
    epsR = np.zeros((ky_size, 2 * Nw + 1))
    epsI = np.zeros((ky_size, 2 * Nw + 1))

    g_pi = 2.35e15 / 1000.0

    for w_idx in range(2 * Nw + 1):
        w = w_idx - Nw
        ww = w * dw

        for q in range(ky_size):
            # Calculate Ekq = Eng(me, ky + ky[q]) + Eg
            # ky + ky[q] means each element of ky plus ky[q]
            Ekq_arr = np.zeros(ky_size)
            for i in range(ky_size):
                k_val = ky[i] + ky[q]
                Ekq_arr[i] = hbar_val**2 * k_val**2 / 2.0 / me + Eg

            # Calculate PiT
            a_pi = 2.0 / pi_val * dk

            tmp_real = 0.0
            tmp_imag = 0.0

            for i in range(ky_size):
                # ff0 calculation
                ff0_Ek = n00_val * np.sqrt(hbar_val**2 / twopi_val / mh / kB_val / Th) * np.exp(-Ek[i] / kB_val / Th)
                ff0_Ekq = n00_val * np.sqrt(hbar_val**2 / twopi_val / me / kB_val / Te) * np.exp(-Ekq_arr[i] / kB_val / Te)

                denom1_real = hbar_val * ww - Ekq_arr[i] - Ek[i]
                denom1_imag = hbar_val * g_pi
                denom1_mag2 = denom1_real**2 + denom1_imag**2

                tmp_real += (1.0 - ff0_Ek - ff0_Ekq) * (denom1_real / denom1_mag2)
                tmp_imag += (1.0 - ff0_Ek - ff0_Ekq) * (-denom1_imag / denom1_mag2)

            tmp_real_final = a_pi * tmp_real
            tmp_imag_final = a_pi * tmp_imag

            epsR[q, w_idx] = 1.0 - a * tmp_real_final
            epsI[q, w_idx] = -a * tmp_imag_final

    return epsR, epsI


def RecordEpsrT(Te, Th, me, mh, Eg, ky):
    """
    Record transverse permittivity.

    Calculates and writes the transverse permittivity components
    for all momentum and frequency values.

    Parameters
    ----------
    Te : float
        Electron temperature (K)
    Th : float
        Hole temperature (K)
    me : float
        Electron mass (kg)
    mh : float
        Hole mass (kg)
    Eg : float
        Band gap (J)
    ky : ndarray
        Momentum coordinates (1/m), 1D array

    Returns
    -------
    None

    Notes
    -----
    Writes output to 'dataQW/Wire/EpsT.dat' and 'chi.0.w.dat'.
    The permittivity arrays use negative indexing mapping: w from -Nw to Nw
    maps to array indices 0 to 2*Nw.
    """
    global _dcv, _epsb, _R0, _Nw, _dw, _g

    a = 1.0 / eps0 / _epsb * _dcv**2 / pi / _R0**2

    ky_size = len(ky)
    epsR = np.zeros((ky_size, 2 * _Nw + 1))
    epsI = np.zeros((ky_size, 2 * _Nw + 1))
    dk = ky[1] - ky[0]
    _g = 1e12

    Ek = Eng(mh, ky)

    try:
        epsR, epsI = _RecordEpsrT_loop_jit(ky_size, _Nw, _dw, a, me, mh, Te, Th, dk, ky, Ek, Eg,
                                           _n00, hbar, kB, twopi, pi)
    except Exception:
        # Fallback to pure Python
        for w_idx in range(2 * _Nw + 1):
            w = w_idx - _Nw
            print(f"T w {w}")
            ww = w * _dw

            for q in range(ky_size):
                Ekq = Eng(me, ky + ky[q]) + Eg
                tmp = PiT(ky[q], ww, me, mh, Te, Th, dk, Ek, Ekq)

                epsR[q, w_idx] = 1.0 - a * tmp.real
                epsI[q, w_idx] = -a * tmp.imag

    print("min val EpsT real", "max val EpsT real")
    print(np.min(epsR), np.max(epsR))
    print("min val EpsT imag", "max val EpsT imag")
    print(np.min(epsI), np.max(epsI))

    os.makedirs('dataQW/Wire', exist_ok=True)
    with open('dataQW/Wire/EpsT.dat', 'w', encoding='utf-8') as f:
        for w_idx in range(2 * _Nw + 1):
            for q in range(ky_size):
                f.write(f"{epsR[q, w_idx]} {epsI[q, w_idx]}\n")

    with open("chi.0.w.dat", 'w', encoding='utf-8') as f:
        idx = int(np.floor(1141.0 / 2.0 + 2))
        if idx >= ky_size:
            idx = ky_size - 1
        for w_idx in range(2 * _Nw + 1):
            w = w_idx - _Nw
            f.write(f"{w} {epsR[idx, w_idx]}\n")


def PiL(q, w, m, T, dk, Ek, Ekq):
    """
    Calculate longitudinal polarization function.

    Computes the longitudinal polarization function PiL for permittivity calculations.

    Parameters
    ----------
    q : float
        Momentum transfer (1/m) (unused, kept for interface compatibility)
    w : float
        Frequency (Hz)
    m : float
        Mass (kg)
    T : float
        Temperature (K)
    dk : float
        Momentum step size (1/m)
    Ek : ndarray
        Energy array (J)
    Ekq : ndarray
        Energy array shifted by q (J)

    Returns
    -------
    complex
        Longitudinal polarization function value

    Notes
    -----
    The function computes:
    PiL = (2/pi) * dk * sum((ff0(Ek,T,m) - ff0(Ekq,T,m)) *
          (1 / (hbar*w - (Ekq - Ek) + i*hbar*g)))
    """
    g = 0.01 * e0 * 1e-3 / hbar

    ff0_Ek = ff0(Ek, T, m)
    ff0_Ekq = ff0(Ekq, T, m)

    result = 2.0 / pi * dk * np.sum((ff0_Ek - ff0_Ekq) *
                                   (1.0 / (hbar * w - (Ekq - Ek) + ii * hbar * g)))

    return result


def PiL_T0(q, w, m, T, dk, Ek, Ekq):
    """
    Calculate longitudinal polarization function at zero temperature.

    Computes the longitudinal polarization function PiL_T0 for permittivity calculations
    at zero temperature.

    Parameters
    ----------
    q : float
        Momentum transfer (1/m)
    w : float
        Frequency (Hz)
    m : float
        Mass (kg)
    T : float
        Temperature (K) (unused, kept for interface compatibility)
    dk : float
        Momentum step size (1/m)
    Ek : ndarray
        Energy array (J) (unused, kept for interface compatibility)
    Ekq : ndarray
        Energy array shifted by q (J) (unused, kept for interface compatibility)

    Returns
    -------
    complex
        Longitudinal polarization function value at zero temperature

    Notes
    -----
    Uses module-level variable _kf for Fermi momentum.
    The function computes:
    PiL_T0 = -m / (pi * hbar^2) * q_inv * (atanhc(...) + atanhc(...))
    """
    global _kf

    q_inv = q / (q**2 + dk**2)
    g = 0.01 * e0 * 1e-3 / hbar

    term1 = atanhc(2.0 * hbar * _kf * q / (ii * g * m + hbar * q**2 - m * w))
    term2 = atanhc(2.0 * hbar * _kf * q / (ii * g * m + hbar * q**2 + m * w))

    result = -m / pi / hbar**2 * q_inv * (term1 + term2)

    return result


def RecordEpsrL_T0(me, ky):
    """
    Record longitudinal permittivity at zero temperature.

    Calculates and writes the longitudinal permittivity components
    at zero temperature for all momentum and frequency values.

    Parameters
    ----------
    me : float
        Electron mass (kg) (unused, kept for interface compatibility)
    ky : ndarray
        Momentum coordinates (1/m), 1D array

    Returns
    -------
    None

    Notes
    -----
    Writes output to 'dataQW/Wire/EpsL.dat'.
    The permittivity arrays use negative indexing mapping: w from -Nw to Nw
    maps to array indices 0 to 2*Nw.
    Note: The function has empty loops in the original Fortran code.
    """
    global _Nw, _dw, _epsb, _R0

    Nk = len(ky)
    eps = np.ones((Nk, 2 * _Nw + 1), dtype=complex)
    PiE = np.zeros((Nk, 2 * _Nw + 1))
    PiH = np.zeros((Nk, 2 * _Nw + 1))

    dk = ky[1] - ky[0]

    Vc = np.zeros(Nk)
    for q in range(Nk):
        Vc[q] = (e0**2 / twopi / eps0 / _epsb) * K03(max(abs(ky[q] * _R0), 1e-10))

    # Empty loops from original Fortran code
    # for w_idx in range(2 * _Nw + 1):
    #     w = w_idx - _Nw
    #     for q_idx in range(2 * Nk + 1):
    #         q = q_idx - Nk
    #         for k_idx in range(2 * Nk + 1):
    #             k = k_idx - Nk
    #             pass

    # Calculate eps
    for w_idx in range(2 * _Nw + 1):
        for q in range(Nk):
            eps[q, w_idx] = 1.0 - Vc[q] * PiE[q, w_idx] - Vc[q] * PiH[q, w_idx]

    print("min val EpsL real", "max val EpsL real")
    print(np.min(eps.real), np.max(eps.real))
    print("min val EpsL imag", "max val EpsL imag")
    print(np.min(eps.imag), np.max(eps.imag))

    os.makedirs('dataQW/Wire', exist_ok=True)
    with open('dataQW/Wire/EpsL.dat', 'w', encoding='utf-8') as f:
        for w_idx in range(2 * _Nw + 1):
            w = w_idx - _Nw
            print(f"writing {w}")
            for q in range(Nk):
                f.write(f"{eps[q, w_idx].real} {eps[q, w_idx].imag}\n")


def QqGq(ky, Nk, dk, dw, EpsR, EpsI, eh):
    """
    Calculate Omega and Gam from permittivity.

    Computes the quasiparticle frequency Omega and damping Gam
    from the permittivity components.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates (1/m), 1D array
    Nk : int
        Number of momentum points
    dk : float
        Momentum step size (1/m)
    dw : float
        Frequency step size (Hz)
    EpsR : ndarray
        Real part of permittivity, shape (Nk, 2*Nw+1)
    EpsI : ndarray
        Imaginary part of permittivity, shape (Nk, 2*Nw+1)
    eh : str
        Character identifier for output file

    Returns
    -------
    None

    Notes
    -----
    Writes output to 'dataQW/Wire/Omega_qp.{eh}.dat'.
    The permittivity arrays use negative indexing mapping: w from -Nw to Nw
    maps to array indices 0 to 2*Nw.
    """
    global _Nw

    Omega = np.zeros(Nk)
    Gam = np.zeros(Nk)
    dEpsRdw = np.zeros(2 * _Nw + 1)

    for q in range(Nk):
        # Calculate derivative using cshift equivalent (np.roll)
        dEpsRdw = (np.roll(EpsR[q, :], 1) - np.roll(EpsR[q, :], -1)) / 2.0 / dw

        tmp = 0.0

        for w_idx in range(2 * _Nw + 1):
            w = w_idx - _Nw

            if 1.0 / (np.abs(EpsR[q, w_idx]) + 1e-15) > tmp:
                tmp = 1.0 / (np.abs(EpsR[q, w_idx]) + 1e-15)
                Omega[q] = w * dw
                Gam[q] = EpsI[q, w_idx] / dEpsRdw[w_idx]

    os.makedirs('dataQW/Wire', exist_ok=True)
    with open(f'dataQW/Wire/Omega_qp.{eh}.dat', 'w', encoding='utf-8') as f:
        for q in range(Nk):
            f.write(f"{ky[q]} {Omega[q]} {Gam[q]}\n")


@jit(nopython=True)
def _ZeroT_L_loop_jit(qy_size, Nw, dw, m, kf, qy, dq, Vc, hbar_val, e0_val, pi_val, ii_real, ii_imag):
    """JIT-compiled loop for ZeroT_L."""
    Pi1 = np.zeros((qy_size, 2 * Nw + 1))
    Pi2 = np.zeros((qy_size, 2 * Nw + 1))
    eps_real = np.zeros((qy_size, 2 * Nw + 1))
    eps_imag = np.zeros((qy_size, 2 * Nw + 1))

    for ww_idx in range(2 * Nw + 1):
        ww = ww_idx - Nw
        hw_real = hbar_val * ww * dw
        hw_imag = 1e-4 * e0_val

        for qq in range(qy_size):
            q = qy[qq]
            Aq = m / hbar_val**2 * q / (q**2 + dq**2)

            # Calculate xqw for first term
            Eng_kf_q = hbar_val**2 * (kf - q)**2 / 2.0 / m
            Eng_kf = hbar_val**2 * kf**2 / 2.0 / m
            Eng_kf_pq = hbar_val**2 * (kf + q)**2 / 2.0 / m
            Eng_q = hbar_val**2 * q**2 / 2.0 / m

            num1 = (Eng_kf_q - Eng_kf - hw_real)**2 + hw_imag**2
            num2 = (Eng_kf_q - Eng_kf + hw_real)**2 + hw_imag**2
            den1 = (Eng_kf_pq - Eng_kf - hw_real)**2 + hw_imag**2
            den2 = (Eng_kf_pq - Eng_kf + hw_real)**2 + hw_imag**2

            xqw_mag = np.sqrt((num1 * num2) / (den1 * den2))
            xqw_phase = np.arctan2(2.0 * hw_imag * (Eng_kf_q - Eng_kf), (Eng_kf_q - Eng_kf)**2 - hw_real**2 - hw_imag**2) - \
                       np.arctan2(2.0 * hw_imag * (Eng_kf_pq - Eng_kf), (Eng_kf_pq - Eng_kf)**2 - hw_real**2 - hw_imag**2)

            Pi1[qq, ww_idx] = Aq / pi_val * np.log(xqw_mag)

            # Second term
            num1 = (-Eng_kf_q + Eng_kf - hw_real)**2 + hw_imag**2
            num2 = (-Eng_kf_q + Eng_kf + hw_real)**2 + hw_imag**2
            den1 = (-Eng_kf_pq + Eng_kf - hw_real)**2 + hw_imag**2
            den2 = (-Eng_kf_pq + Eng_kf + hw_real)**2 + hw_imag**2

            xqw_mag = np.sqrt((num1 * num2) / (den1 * den2))
            Pi1[qq, ww_idx] += Aq / pi_val * np.log(xqw_mag)

            # Calculate Pi2
            k = Aq * (hw_real - Eng_q)
            # fT0 calculation: 1 - theta(|k| - kf)
            fT0_k = 1.0 if np.abs(k) < kf else 0.0
            fT0_kq = 1.0 if np.abs(k + q) < kf else 0.0

            Pi2[qq, ww_idx] = -Aq * (fT0_k - fT0_kq)

            # Calculate eps
            eps_real[qq, ww_idx] = -Vc[qq] * Pi1[qq, ww_idx]
            eps_imag[qq, ww_idx] = -Vc[qq] * Pi2[qq, ww_idx]

    return eps_real, eps_imag


def ZeroT_L(B, m, qy, kf):
    """
    Calculate zero temperature longitudinal permittivity.

    Computes the longitudinal permittivity at zero temperature.

    Parameters
    ----------
    B : str
        Character identifier ('E' for electrons, 'H' for holes)
    m : float
        Mass (kg)
    qy : ndarray
        Momentum coordinates (1/m), 1D array
    kf : float
        Fermi momentum (1/m)

    Returns
    -------
    None

    Notes
    -----
    Writes output to 'dataQW/Wire/ChiL.{B}.dat'.
    The permittivity arrays use negative indexing mapping: w from -Nw to Nw
    maps to array indices 0 to 2*Nw.
    """
    global _Nw, _dw, _epsb, _R0

    dq = qy[1] - qy[0]

    Vc = np.zeros(len(qy))
    for qq in range(len(qy)):
        Vc[qq] = (e0**2 / twopi / eps0 / _epsb) * K03(max(abs(qy[qq] * _R0), dq * _R0))

    try:
        eps_real, eps_imag = _ZeroT_L_loop_jit(len(qy), _Nw, _dw, m, kf, qy, dq, Vc, hbar, e0, pi, ii.real, ii.imag)
        eps = eps_real + 1j * eps_imag
    except Exception:
        # Fallback to pure Python
        Pi1 = np.zeros((len(qy), 2 * _Nw + 1))
        Pi2 = np.zeros((len(qy), 2 * _Nw + 1))
        eps = np.zeros((len(qy), 2 * _Nw + 1), dtype=complex)

        for ww_idx in range(2 * _Nw + 1):
            ww = ww_idx - _Nw
            hw = hbar * ww * _dw + ii * 1e-4 * e0

            for qq in range(len(qy)):
                q = qy[qq]
                Aq = m / hbar**2 * q / (q**2 + dq**2)

                xqw = ((Eng(m, kf - q) - Eng(m, kf) - hw) *
                       (Eng(m, kf - q) - Eng(m, kf) + hw) /
                       (Eng(m, kf + q) - Eng(m, kf) - hw) /
                       (Eng(m, kf + q) - Eng(m, kf) + hw))
                k = np.real(Aq * (hw - Eng(m, q)))

                Pi1[qq, ww_idx] = Aq / pi * np.real(np.log(xqw))

                # Second term
                Aq = m / hbar**2 * q / (q**2 + dq**2)
                xqw = ((-Eng(m, kf - q) + Eng(m, kf) - hw) *
                       (-Eng(m, kf - q) + Eng(m, kf) + hw) /
                       (-Eng(m, kf + q) + Eng(m, kf) - hw) /
                       (-Eng(m, kf + q) + Eng(m, kf) + hw))
                k = np.real(Aq * (hw + Eng(m, q)))

                Pi1[qq, ww_idx] += Aq / pi * np.real(np.log(xqw))

                Pi2[qq, ww_idx] = -Aq * (fT0(k, kf) - fT0(k + q, kf))

                eps[qq, ww_idx] = -Vc[qq] * (Pi1[qq, ww_idx] + ii * Pi2[qq, ww_idx])

    os.makedirs('dataQW/Wire', exist_ok=True)
    with open(f'dataQW/Wire/ChiL.{B}.dat', 'w', encoding='utf-8') as f:
        for ww_idx in range(2 * _Nw + 1):
            ww = ww_idx - _Nw
            print(f"writing {ww}")
            for qq in range(len(qy)):
                f.write(f"{eps[qq, ww_idx].real} {eps[qq, ww_idx].imag}\n")


@jit(nopython=True)
def _ZeroT_T_loop_jit(qy_size, Nw, dw, me, mh, Egap, qy, dq, Vc, kf, hbar_val, e0_val, pi_val, ii_real, ii_imag, c0_val, epsb_val):
    """JIT-compiled loop for ZeroT_T."""
    Pi1 = np.zeros((qy_size, 2 * Nw + 1))
    Pi2 = np.zeros((qy_size, 2 * Nw + 1))
    Pi3_real = np.zeros((qy_size, 2 * Nw + 1))
    Pi3_imag = np.zeros((qy_size, 2 * Nw + 1))

    b = hbar_val**2 / 2.0 / me
    c = hbar_val**2 / 2.0 / mh

    for ww_idx in range(2 * Nw + 1):
        ww = ww_idx - Nw
        a_real = hbar_val * ww * dw - Egap
        a_imag = 1e-3 * e0_val

        for qq in range(qy_size):
            q = qy[qq]

            # Calculate d = sqrt(a*(b+c) + b*c*q^2)
            d_real = a_real * (b + c) + b * c * q**2
            d_imag = a_imag * (b + c)
            d_mag = np.sqrt(d_real**2 + d_imag**2)
            d_phase = np.arctan2(d_imag, d_real)

            # Calculate atanJG terms (simplified)
            # atanJG(z) = log((ii-z)/(ii+z)) / (2*ii)
            # For real arguments, this simplifies
            term1 = (+kf * (b + c) + b * q) / d_mag
            term2 = (-kf * (b + c) + b * q) / d_mag
            term3 = (+kf * (b + c) - c * q) / d_mag
            term4 = (-kf * (b + c) - c * q) / d_mag

            # Simplified atanJG calculation
            Pi1[qq, ww_idx] = -Vc * (np.arctan(term1) + np.arctan(term2) - np.arctan(term3) + np.arctan(term4)) / d_mag

            Pi2[qq, ww_idx] = q**2 * c0_val**2 / epsb_val / ((ww * dw)**2 + 9 * dw**2)

            # Calculate Pi3
            for kk in range(qy_size):
                k = qy[kk]
                # fT0 calculation
                fT0_k = 1.0 if np.abs(k) < kf else 0.0
                fT0_kq = 1.0 if np.abs(k + q) < kf else 0.0

                Eng_me_kq = hbar_val**2 * (k + q)**2 / 2.0 / me
                Eng_mh_k = hbar_val**2 * k**2 / 2.0 / mh

                denom1_real = hbar_val * ww * dw - Egap - Eng_me_kq - Eng_mh_k
                denom1_imag = e0_val * 5e-3
                denom1_mag2 = denom1_real**2 + denom1_imag**2

                denom2_real = hbar_val * ww * dw + Egap + Eng_me_kq + Eng_mh_k
                denom2_imag = e0_val * 5e-3
                denom2_mag2 = denom2_real**2 + denom2_imag**2

                Pi3_real[qq, ww_idx] += Vc * dq * (1.0 - fT0_k - fT0_kq) * (denom1_real / denom1_mag2 - denom2_real / denom2_mag2)
                Pi3_imag[qq, ww_idx] += Vc * dq * (1.0 - fT0_k - fT0_kq) * (-denom1_imag / denom1_mag2 + denom2_imag / denom2_mag2)

    return Pi1, Pi2, Pi3_real, Pi3_imag


def ZeroT_T(me, mh, Egap, dcv, qy, kf):
    """
    Calculate zero temperature transverse permittivity.

    Computes the transverse permittivity at zero temperature.

    Parameters
    ----------
    me : float
        Electron mass (kg)
    mh : float
        Hole mass (kg)
    Egap : float
        Band gap (J)
    dcv : float
        Dipole matrix element (C*m)
    qy : ndarray
        Momentum coordinates (1/m), 1D array
    kf : float
        Fermi momentum (1/m)

    Returns
    -------
    None

    Notes
    -----
    Writes output to 'dataQW/Wire/ChiT.dat'.
    The permittivity arrays use negative indexing mapping: w from -Nw to Nw
    maps to array indices 0 to 2*Nw.
    """
    global _Nw, _dw, _epsb, _R0

    dq = qy[1] - qy[0]

    Vc = dcv**2 / eps0 / _epsb / pi / _R0**2
    b = hbar**2 / 2.0 / me
    c = hbar**2 / 2.0 / mh

    Pi1 = np.zeros((len(qy), 2 * _Nw + 1))
    Pi2 = np.zeros((len(qy), 2 * _Nw + 1))
    Pi3 = np.zeros((len(qy), 2 * _Nw + 1), dtype=complex)

    try:
        Pi1, Pi2, Pi3_real, Pi3_imag = _ZeroT_T_loop_jit(len(qy), _Nw, _dw, me, mh, Egap, qy, dq, Vc, kf,
                                                          hbar, e0, pi, ii.real, ii.imag, c0, _epsb)
        Pi3 = Pi3_real + 1j * Pi3_imag
    except Exception:
        # Fallback to pure Python
        for ww_idx in range(2 * _Nw + 1):
            ww = ww_idx - _Nw
            print(ww)

            for qq in range(len(qy)):
                q = qy[qq]
                a = hbar * ww * _dw - Egap + ii * 1e-3 * e0
                d = np.sqrt(a * (b + c) + b * c * q**2)

                Pi1[qq, ww_idx] = np.real(-Vc * atanJG((+kf * (b + c) + b * q) / d) / d +
                                          Vc * atanJG((-kf * (b + c) + b * q) / d) / d -
                                          Vc * atanJG((+kf * (b + c) - c * q) / d) / d +
                                          Vc * atanJG((-kf * (b + c) - c * q) / d) / d)

                Pi2[qq, ww_idx] = q**2 * c0**2 / _epsb / ((ww * _dw)**2 + 9 * _dw**2)

                for kk in range(len(qy)):
                    k = qy[kk]
                    Pi3[qq, ww_idx] += Vc * dq * (1.0 - fT0(k, kf) - fT0(k + q, kf)) * \
                                       (+1.0 / (hbar * ww * _dw - Egap - Eng(me, k + q) - Eng(mh, k) + ii * e0 * 5e-3) -
                                        1.0 / (hbar * ww * _dw + Egap + Eng(me, k + q) + Eng(mh, k) + ii * e0 * 5e-3))

    os.makedirs('dataQW/Wire', exist_ok=True)
    with open('dataQW/Wire/ChiT.dat', 'w', encoding='utf-8') as f:
        for ww_idx in range(2 * _Nw + 1):
            ww = ww_idx - _Nw
            print(f"writing {ww}")
            for qq in range(len(qy)):
                f.write(f"{Pi2[qq, ww_idx]} {-Pi3[qq, ww_idx].real}\n")


@jit(nopython=True)
def _RecordEpsrL_loop_jit(ky_size, Nw, dw, me, mh, Te, Th, dk, ky, Vc,
                          n00_val, hbar_val, kB_val, twopi_val, pi_val, e0_val):
    """JIT-compiled loop for RecordEpsrL."""
    eps_real = np.zeros((ky_size, 2 * Nw + 1))
    eps_imag = np.zeros((ky_size, 2 * Nw + 1))
    PiE_real = np.zeros((ky_size, 2 * Nw + 1))
    PiE_imag = np.zeros((ky_size, 2 * Nw + 1))
    PiH_real = np.zeros((ky_size, 2 * Nw + 1))
    PiH_imag = np.zeros((ky_size, 2 * Nw + 1))

    g = 0.01 * e0_val * 1e-3 / hbar_val

    # For electrons
    T = Te
    m = me
    Ek = np.zeros(ky_size)
    for i in range(ky_size):
        Ek[i] = hbar_val**2 * ky[i]**2 / 2.0 / m

    for w_idx in range(2 * Nw + 1):
        w = w_idx - Nw
        ww = w * dw

        for q in range(ky_size):
            # Calculate Ekq = Eng(m, ky + ky[q])
            Ekq_arr = np.zeros(ky_size)
            for i in range(ky_size):
                k_val = ky[i] + ky[q]
                Ekq_arr[i] = hbar_val**2 * k_val**2 / 2.0 / m

            # Calculate PiL
            a_pi = 2.0 / pi_val * dk
            tmp_real = 0.0
            tmp_imag = 0.0

            for i in range(ky_size):
                ff0_Ek = n00_val * np.sqrt(hbar_val**2 / twopi_val / m / kB_val / T) * np.exp(-Ek[i] / kB_val / T)
                ff0_Ekq = n00_val * np.sqrt(hbar_val**2 / twopi_val / m / kB_val / T) * np.exp(-Ekq_arr[i] / kB_val / T)

                denom1_real = hbar_val * ww - (Ekq_arr[i] - Ek[i])
                denom1_imag = hbar_val * g
                denom1_mag2 = denom1_real**2 + denom1_imag**2

                tmp_real += (ff0_Ek - ff0_Ekq) * (denom1_real / denom1_mag2)
                tmp_imag += (ff0_Ek - ff0_Ekq) * (-denom1_imag / denom1_mag2)

            PiE_real[q, w_idx] = a_pi * tmp_real
            PiE_imag[q, w_idx] = a_pi * tmp_imag

    # For holes
    T = Th
    m = mh
    for i in range(ky_size):
        Ek[i] = hbar_val**2 * ky[i]**2 / 2.0 / m

    for w_idx in range(2 * Nw + 1):
        w = w_idx - Nw
        ww = w * dw

        for q in range(ky_size):
            # Calculate Ekq = Eng(m, ky + ky[q])
            Ekq_arr = np.zeros(ky_size)
            for i in range(ky_size):
                k_val = ky[i] + ky[q]
                Ekq_arr[i] = hbar_val**2 * k_val**2 / 2.0 / m

            # Calculate PiL
            a_pi = 2.0 / pi_val * dk
            tmp_real = 0.0
            tmp_imag = 0.0

            for i in range(ky_size):
                ff0_Ek = n00_val * np.sqrt(hbar_val**2 / twopi_val / m / kB_val / T) * np.exp(-Ek[i] / kB_val / T)
                ff0_Ekq = n00_val * np.sqrt(hbar_val**2 / twopi_val / m / kB_val / T) * np.exp(-Ekq_arr[i] / kB_val / T)

                denom1_real = hbar_val * ww - (Ekq_arr[i] - Ek[i])
                denom1_imag = hbar_val * g
                denom1_mag2 = denom1_real**2 + denom1_imag**2

                tmp_real += (ff0_Ek - ff0_Ekq) * (denom1_real / denom1_mag2)
                tmp_imag += (ff0_Ek - ff0_Ekq) * (-denom1_imag / denom1_mag2)

            PiH_real[q, w_idx] = a_pi * tmp_real
            PiH_imag[q, w_idx] = a_pi * tmp_imag

    # Calculate eps
    for w_idx in range(2 * Nw + 1):
        for q in range(ky_size):
            # eps = 1 - Vc * PiE - Vc * PiH
            # where PiE and PiH are complex
            eps_real[q, w_idx] = 1.0 - Vc[q] * PiE_real[q, w_idx] - Vc[q] * PiH_real[q, w_idx]
            eps_imag[q, w_idx] = -Vc[q] * PiE_imag[q, w_idx] - Vc[q] * PiH_imag[q, w_idx]

    return eps_real, eps_imag


def RecordEpsrL(Te, Th, me, mh, ky):
    """
    Record longitudinal permittivity.

    Calculates and writes the longitudinal permittivity components
    for all momentum and frequency values.

    Parameters
    ----------
    Te : float
        Electron temperature (K)
    Th : float
        Hole temperature (K)
    me : float
        Electron mass (kg)
    mh : float
        Hole mass (kg)
    ky : ndarray
        Momentum coordinates (1/m), 1D array

    Returns
    -------
    None

    Notes
    -----
    Writes output to 'dataQW/Wire/EpsL.dat'.
    The permittivity arrays use negative indexing mapping: w from -Nw to Nw
    maps to array indices 0 to 2*Nw.
    """
    global _Nw, _dw, _epsb, _R0

    ky_size = len(ky)
    eps = np.ones((ky_size, 2 * _Nw + 1), dtype=complex)
    PiE = np.zeros((ky_size, 2 * _Nw + 1), dtype=complex)
    PiH = np.zeros((ky_size, 2 * _Nw + 1), dtype=complex)

    dk = ky[1] - ky[0]

    Vc = np.zeros(ky_size)
    for q in range(ky_size):
        Vc[q] = (e0**2 / twopi / eps0 / _epsb) * K03(max(abs(ky[q] * _R0), 1e-10))

    # For electrons
    T = Te
    m = me
    Ek = Eng(m, ky)

    try:
        eps_real, eps_imag = _RecordEpsrL_loop_jit(ky_size, _Nw, _dw, me, mh, Te, Th, dk, ky, Vc,
                                                    _n00, hbar, kB, twopi, pi, e0)
        eps = eps_real + 1j * eps_imag
    except Exception:
        # Fallback to pure Python
        for w_idx in range(2 * _Nw + 1):
            w = w_idx - _Nw
            print(f"L e w {w}")
            ww = w * _dw

            for q in range(ky_size):
                Ekq = Eng(m, ky + ky[q])
                PiE[q, w_idx] = PiL(ky[q], ww, m, T, dk, Ek, Ekq)

        # For holes
        T = Th
        m = mh
        Ek = Eng(m, ky)

        for w_idx in range(2 * _Nw + 1):
            w = w_idx - _Nw
            ww = w * _dw
            print(f"L h w {w}")

            for q in range(ky_size):
                Ekq = Eng(m, ky + ky[q])
                PiH[q, w_idx] = PiL(ky[q], ww, m, T, dk, Ek, Ekq)

        # Calculate eps
        for w_idx in range(2 * _Nw + 1):
            for q in range(ky_size):
                eps[q, w_idx] = 1.0 - Vc[q] * PiE[q, w_idx] - Vc[q] * PiH[q, w_idx]

    print("min val EpsL real", "max val EpsL real")
    print(np.min(eps.real), np.max(eps.real))
    print("min val EpsL imag", "max val EpsL imag")
    print(np.min(eps.imag), np.max(eps.imag))

    os.makedirs('dataQW/Wire', exist_ok=True)
    with open('dataQW/Wire/EpsL.dat', 'w', encoding='utf-8') as f:
        for w_idx in range(2 * _Nw + 1):
            for q in range(ky_size):
                f.write(f"{eps[q, w_idx].real} {eps[q, w_idx].imag}\n")
