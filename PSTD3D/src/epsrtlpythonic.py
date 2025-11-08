"""
Pythonic implementation of the Fortran epsrtl module.

This module calculates dielectric functions for quantum wire systems,
including longitudinal and transverse components with finite and zero
temperature effects.

Author: Rahul Sah
Date: 2025
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any
import os

# Try to import Numba for JIT compilation
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    # Create dummy decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

# Import dependencies
from constants import pi, hbar, c0, eps0, e0, me0, twopi, ii
from usefulsubspythonic import K03, theta, cshift

# Type aliases
FloatArray = np.ndarray
ComplexArray = np.ndarray
IntArray = np.ndarray

# Set up logging
logger = logging.getLogger(__name__)

# Physical constants
kB = 1.38064852e-23  # Boltzmann constant


@dataclass
class EpsrtlParameters:
    """Physical parameters for dielectric function calculations."""

    # Material parameters
    n1D: float = 1e6  # 1D carrier density
    dcv: float = 1e-28  # Dipole matrix element
    Te: float = 300.0  # Electron temperature (K)
    Th: float = 300.0  # Hole temperature (K)
    me: float = 0.067 * me0  # Electron effective mass
    mh: float = 0.45 * me0  # Hole effective mass
    Eg: float = 1.42  # Band gap (eV)

    # System parameters
    R0: float = 1e-9  # Quantum wire radius
    epsb: float = 3.011**2  # Background dielectric constant
    g: float = 1e12  # Broadening parameter (1/s)

    # Grid parameters
    Nw: int = 500  # Number of frequency points
    dw: float = 2.35e15 / 500.0 * 2  # Frequency step

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.n1D <= 0:
            raise ValueError("n1D must be positive")
        if self.R0 <= 0:
            raise ValueError("R0 must be positive")
        if self.epsb <= 0:
            raise ValueError("epsb must be positive")
        if self.Nw <= 0:
            raise ValueError("Nw must be positive")


@dataclass
class FrequencyGrid:
    """Frequency grid for dielectric function calculations."""

    Nw: int
    dw: float

    @property
    def frequencies(self) -> FloatArray:
        """Get frequency array."""
        return np.arange(-self.Nw, self.Nw + 1) * self.dw

    @property
    def size(self) -> int:
        """Get total number of frequency points."""
        return 2 * self.Nw + 1


@dataclass
class MomentumGrid:
    """Momentum grid for calculations."""

    ky: FloatArray

    @property
    def size(self) -> int:
        """Get number of momentum points."""
        return len(self.ky)

    @property
    def dk(self) -> float:
        """Get momentum step."""
        return self.ky[1] - self.ky[0]


class MathematicalFunctions:
    """Mathematical utility functions."""

    @staticmethod
    @njit(cache=True)
    def eng(m: float, k: FloatArray) -> FloatArray:
        """Energy dispersion: E = hbar²k²/2m."""
        return hbar**2 * k**2 / (2.0 * m)

    @staticmethod
    @njit(cache=True)
    def fT0(k: FloatArray, kf: float) -> FloatArray:
        """Zero temperature Fermi function."""
        return 1.0 - theta(np.abs(k) - kf)

    @staticmethod
    def ff0(E: FloatArray, T: float, m: float, n00: float) -> FloatArray:
        """Finite temperature distribution function."""
        if T <= 0:
            return np.zeros_like(E)
        return n00 * np.sqrt(hbar**2 / (twopi * m * kB * T)) * np.exp(-E / (kB * T))

    @staticmethod
    @njit(cache=True)
    def atanhc(x: ComplexArray) -> ComplexArray:
        """Inverse hyperbolic tangent."""
        return 0.5 * np.log((1.0 + x) / (1.0 - x))

    @staticmethod
    @njit(cache=True)
    def atanJG(z: ComplexArray) -> ComplexArray:
        """Complex arctangent function."""
        return np.log((1j - z) / (1j + z)) / (2j)


class PolarizationCalculator:
    """Calculates polarization functions for dielectric response."""

    def __init__(self, params: EpsrtlParameters):
        self.params = params
        self.math_funcs = MathematicalFunctions()

    def calculate_piT(self, q: float, w: float, me: float, mh: float,
                     Te: float, Th: float, dk: float, Ek: FloatArray,
                     Ekq: FloatArray) -> complex:
        """Calculate transverse polarization function."""
        a = 2.0 / pi * dk
        g = 2.35e15 / 1000.0

        # Calculate distribution functions
        fh = self.math_funcs.ff0(Ek, Th, mh, self.params.n1D)
        fe = self.math_funcs.ff0(Ekq, Te, me, self.params.n1D)

        # Calculate polarization
        numerator = (1.0 - fh - fe)
        denominator1 = hbar * w - Ekq - Ek + 1j * hbar * g
        denominator2 = hbar * w + Ekq + Ek - 1j * hbar * g

        piT = a * np.sum(numerator * (1.0 / denominator1 - 0.0 / denominator2))

        return piT

    def calculate_piL(self, q: float, w: float, m: float, T: float,
                     dk: float, Ek: FloatArray, Ekq: FloatArray) -> complex:
        """Calculate longitudinal polarization function."""
        g = 0.01 * e0 * 1e-3 / hbar

        # Calculate distribution functions
        fk = self.math_funcs.ff0(Ek, T, m, self.params.n1D)
        fkq = self.math_funcs.ff0(Ekq, T, m, self.params.n1D)

        # Calculate polarization
        numerator = fk - fkq
        denominator1 = hbar * w - (Ekq - Ek) + 1j * hbar * g
        denominator2 = hbar * w + (Ekq - Ek) - 1j * hbar * g

        piL = 2.0 / pi * dk * np.sum(numerator * (1.0 / denominator1 - 0.0 / denominator2))

        return piL

    def calculate_piL_T0(self, q: float, w: float, m: float, T: float,
                        dk: float, Ek: FloatArray, Ekq: FloatArray) -> complex:
        """Calculate longitudinal polarization function at T=0."""
        q_inv = q / (q**2 + dk**2)
        g = 0.01 * e0 * 1e-3 / hbar
        kf = self.params.n1D / 2.0

        # Calculate complex argument
        arg1 = 2 * hbar * kf * q / (1j * g * m + hbar * q**2 - m * w)
        arg2 = 2 * hbar * kf * q / (1j * g * m + hbar * q**2 + m * w)

        piL_T0 = -m / pi / hbar**2 * q_inv * (
            self.math_funcs.atanhc(arg1) + self.math_funcs.atanhc(arg2)
        )

        return piL_T0


class DielectricCalculator:
    """Calculates dielectric functions."""

    def __init__(self, params: EpsrtlParameters):
        self.params = params
        self.polarization_calc = PolarizationCalculator(params)
        self.math_funcs = MathematicalFunctions()

    def calculate_coulomb_potential(self, q: FloatArray) -> FloatArray:
        """Calculate Coulomb potential."""
        return (e0**2 / twopi / eps0 / self.params.epsb) * K03(
            np.maximum(np.abs(q * self.params.R0), 1e-10)
        )

    def calculate_transverse_dielectric(self, ky: FloatArray, Te: float, Th: float,
                                      me: float, mh: float, Eg: float) -> Tuple[FloatArray, FloatArray]:
        """Calculate transverse dielectric function."""
        Nk = len(ky)
        Nw = self.params.Nw
        dw = self.params.dw

        epsR = np.zeros((Nk, 2*Nw+1), dtype=np.float64)
        epsI = np.zeros((Nk, 2*Nw+1), dtype=np.float64)

        a = 1.0 / eps0 / self.params.epsb * self.params.dcv**2 / pi / self.params.R0**2
        dk = ky[1] - ky[0]

        # Calculate hole energy dispersion
        Ek = self.math_funcs.eng(mh, ky)

        for w_idx, w in enumerate(range(-Nw, Nw+1)):
            ww = w * dw
            print(f"T w {w}")

            for q_idx, q in enumerate(ky):
                Ekq = self.math_funcs.eng(me, ky + q) + Eg
                tmp = self.polarization_calc.calculate_piT(q, ww, me, mh, Te, Th, dk, Ek, Ekq)

                epsR[q_idx, w_idx] = 1.0 - a * np.real(tmp)
                epsI[q_idx, w_idx] = -a * np.imag(tmp)

        return epsR, epsI

    def calculate_longitudinal_dielectric(self, ky: FloatArray, Te: float, Th: float,
                                        me: float, mh: float) -> ComplexArray:
        """Calculate longitudinal dielectric function."""
        Nk = len(ky)
        Nw = self.params.Nw
        dw = self.params.dw

        eps = np.ones((Nk, 2*Nw+1), dtype=np.complex128)
        PiE = np.zeros((Nk, 2*Nw+1), dtype=np.complex128)
        PiH = np.zeros((Nk, 2*Nw+1), dtype=np.complex128)

        dk = ky[1] - ky[0]

        # Calculate Coulomb potential
        Vc = self.calculate_coulomb_potential(ky)

        # Calculate electron polarization
        Ek = self.math_funcs.eng(me, ky)
        for w_idx, w in enumerate(range(-Nw, Nw+1)):
            ww = w * dw
            print(f"L e w {w}")

            for q_idx, q in enumerate(ky):
                Ekq = self.math_funcs.eng(me, ky + q)
                PiE[q_idx, w_idx] = self.polarization_calc.calculate_piL(
                    q, ww, me, Te, dk, Ek, Ekq
                )

        # Calculate hole polarization
        Ek = self.math_funcs.eng(mh, ky)
        for w_idx, w in enumerate(range(-Nw, Nw+1)):
            ww = w * dw
            print(f"L h w {w}")

            for q_idx, q in enumerate(ky):
                Ekq = self.math_funcs.eng(mh, ky + q)
                PiH[q_idx, w_idx] = self.polarization_calc.calculate_piL(
                    q, ww, mh, Th, dk, Ek, Ekq
                )

        # Calculate dielectric function
        for w_idx in range(2*Nw+1):
            for q_idx in range(Nk):
                eps[q_idx, w_idx] = 1.0 - Vc[q_idx] * PiE[q_idx, w_idx] - Vc[q_idx] * PiH[q_idx, w_idx]

        return eps

    def calculate_zero_temperature_longitudinal(self, ky: FloatArray, me: float, mh: float) -> ComplexArray:
        """Calculate zero temperature longitudinal dielectric function."""
        Nk = len(ky)
        Nw = self.params.Nw
        dw = self.params.dw

        eps = np.ones((Nk, 2*Nw+1), dtype=np.complex128)
        PiE = np.zeros((Nk, 2*Nw+1), dtype=np.complex128)
        PiH = np.zeros((Nk, 2*Nw+1), dtype=np.complex128)

        dk = ky[1] - ky[0]
        kf = self.params.n1D / 2.0

        # Calculate Coulomb potential
        Vc = self.calculate_coulomb_potential(ky)

        # Calculate electron polarization at T=0
        for w_idx, w in enumerate(range(-Nw, Nw+1)):
            ww = w * dw
            print(f"L e w {w}")

            for q_idx, q in enumerate(ky):
                Ek = self.math_funcs.eng(me, ky)
                Ekq = self.math_funcs.eng(me, ky + q)
                PiE[q_idx, w_idx] = self.polarization_calc.calculate_piL_T0(
                    q, ww, me, 0.0, dk, Ek, Ekq
                )

        # Calculate hole polarization at T=0
        for w_idx, w in enumerate(range(-Nw, Nw+1)):
            ww = w * dw
            print(f"L h w {w}")

            for q_idx, q in enumerate(ky):
                Ek = self.math_funcs.eng(mh, ky)
                Ekq = self.math_funcs.eng(mh, ky + q)
                PiH[q_idx, w_idx] = self.polarization_calc.calculate_piL_T0(
                    q, ww, mh, 0.0, dk, Ek, Ekq
                )

        # Calculate dielectric function
        for w_idx in range(2*Nw+1):
            for q_idx in range(Nk):
                eps[q_idx, w_idx] = 1.0 - Vc[q_idx] * PiE[q_idx, w_idx] - Vc[q_idx] * PiH[q_idx, w_idx]

        return eps


class EpsrtlSolver:
    """Main solver for dielectric function calculations."""

    def __init__(self, params: EpsrtlParameters):
        self.params = params
        self.dielectric_calc = DielectricCalculator(params)
        self.math_funcs = MathematicalFunctions()

    def get_epsrL_epsrT(self, n1D: float, dcv0: float, Te: float, Th: float,
                       me: float, mh: float, Eg: float, ky: FloatArray) -> None:
        """Main subroutine for dielectric function calculations."""
        # Update parameters
        self.params.n1D = n1D
        self.params.dcv = dcv0
        self.params.Te = Te
        self.params.Th = Th
        self.params.me = me
        self.params.mh = mh
        self.params.Eg = Eg

        # Create extended momentum grid
        Nq = (len(ky) - 1) * 100 + 1
        dq = (ky[1] - ky[0]) / 50.0
        qy = np.linspace(-(Nq-1)/2.0*dq, (Nq-1)/2.0*dq, Nq)

        n00 = n1D
        kf = n00 / 2.0

        # Calculate zero temperature functions
        self.zeroT_L("E", me, qy, kf)
        self.zeroT_L("H", mh, qy, kf)
        self.zeroT_T(me, mh, Eg, dcv0, qy, kf)

        # Write grid information
        self._write_grid_info(Nq, qy)

        # Stop execution (as in original Fortran)
        raise SystemExit("Calculation complete")

    def record_epsrT(self, Te: float, Th: float, me: float, mh: float,
                    Eg: float, ky: FloatArray) -> None:
        """Record transverse dielectric function."""
        epsR, epsI = self.dielectric_calc.calculate_transverse_dielectric(
            ky, Te, Th, me, mh, Eg
        )

        print("min val EpsT real", "max val EpsT real")
        print(np.min(epsR), np.max(epsR))
        print("min val EpsT imag", "max val EpsT imag")
        print(np.min(epsI), np.max(epsI))

        # Write to file
        self._write_dielectric_data(epsR, epsI, "EpsT.dat")

        # Write specific point data
        self._write_chi_data(epsR, 1141//2+2, "chi.0.w.dat")

    def record_epsrL(self, Te: float, Th: float, me: float, mh: float, ky: FloatArray) -> None:
        """Record longitudinal dielectric function."""
        eps = self.dielectric_calc.calculate_longitudinal_dielectric(ky, Te, Th, me, mh)

        print("min val EpsL real", "max val EpsL real")
        print(np.min(np.real(eps)), np.max(np.real(eps)))
        print("min val EpsL imag", "max val EpsL imag")
        print(np.min(np.imag(eps)), np.max(np.imag(eps)))

        # Write to file
        self._write_complex_dielectric_data(eps, "EpsL.dat")

    def record_epsrL_T0(self, me: float, mh: float, ky: FloatArray) -> None:
        """Record longitudinal dielectric function at T=0."""
        eps = self.dielectric_calc.calculate_zero_temperature_longitudinal(ky, me, mh)

        print("min val EpsL real", "max val EpsL real")
        print(np.min(np.real(eps)), np.max(np.real(eps)))
        print("min val EpsL imag", "max val EpsL imag")
        print(np.min(np.imag(eps)), np.max(np.imag(eps)))

        # Write to file
        self._write_complex_dielectric_data(eps, "EpsL.dat")

    def zeroT_L(self, B: str, m: float, qy: FloatArray, kf: float) -> None:
        """Calculate zero temperature longitudinal response."""
        Nw = self.params.Nw
        dw = self.params.dw
        dq = qy[1] - qy[0]

        # Calculate Coulomb potential
        Vc = (e0**2 / twopi / eps0 / self.params.epsb) * K03(
            np.maximum(np.abs(qy * self.params.R0), dq * self.params.R0)
        )

        # Initialize arrays
        Pi1 = np.zeros((len(qy), 2*Nw+1), dtype=np.float64)
        Pi2 = np.zeros((len(qy), 2*Nw+1), dtype=np.float64)
        eps = np.zeros((len(qy), 2*Nw+1), dtype=np.complex128)

        for ww_idx, ww in enumerate(range(-Nw, Nw+1)):
            for qq_idx, q in enumerate(qy):
                hw = hbar * ww * dw + 1j * 1e-4 * e0
                Aq = m / hbar**2 * q / (q**2 + dq**2)

                # Calculate complex argument
                xqw = ((self.math_funcs.eng(m, kf-q) - self.math_funcs.eng(m, kf) - hw) *
                       (self.math_funcs.eng(m, kf-q) - self.math_funcs.eng(m, kf) + hw) /
                       (self.math_funcs.eng(m, kf+q) - self.math_funcs.eng(m, kf) - hw) /
                       (self.math_funcs.eng(m, kf+q) - self.math_funcs.eng(m, kf) + hw))

                k = np.real(Aq * (hw - self.math_funcs.eng(m, q)))
                Pi1[qq_idx, ww_idx] = Aq / pi * np.real(np.log(xqw))

                # Second term
                xqw2 = ((-self.math_funcs.eng(m, kf-q) + self.math_funcs.eng(m, kf) - hw) *
                        (-self.math_funcs.eng(m, kf-q) + self.math_funcs.eng(m, kf) + hw) /
                        (-self.math_funcs.eng(m, kf+q) + self.math_funcs.eng(m, kf) - hw) /
                        (-self.math_funcs.eng(m, kf+q) + self.math_funcs.eng(m, kf) + hw))

                k2 = np.real(Aq * (hw + self.math_funcs.eng(m, q)))
                Pi1[qq_idx, ww_idx] += Aq / pi * np.real(np.log(xqw2))

                # Calculate Pi2
                Pi2[qq_idx, ww_idx] = -Aq * (self.math_funcs.fT0(k, kf) - self.math_funcs.fT0(k+q, kf))

                # Calculate dielectric function
                eps[qq_idx, ww_idx] = -Vc[qq_idx] * (Pi1[qq_idx, ww_idx] + 1j * Pi2[qq_idx, ww_idx])

        # Write to file
        self._write_complex_dielectric_data(eps, f"ChiL.{B}.dat")

    def zeroT_T(self, me: float, mh: float, Egap: float, dcv: float,
                qy: FloatArray, kf: float) -> None:
        """Calculate zero temperature transverse response."""
        Nw = self.params.Nw
        dw = self.params.dw
        dq = qy[1] - qy[0]

        Vc = dcv**2 / eps0 / self.params.epsb / pi / self.params.R0**2
        b = hbar**2 / 2.0 / me
        c = hbar**2 / 2.0 / mh

        Pi1 = np.zeros((len(qy), 2*Nw+1), dtype=np.float64)
        Pi2 = np.zeros((len(qy), 2*Nw+1), dtype=np.float64)
        Pi3 = np.zeros((len(qy), 2*Nw+1), dtype=np.complex128)

        for ww_idx, ww in enumerate(range(-Nw, Nw+1)):
            print(ww)
            for qq_idx, q in enumerate(qy):
                a = hbar * ww * dw - Egap + 1j * 1e-3 * e0
                d = np.sqrt(a * (b + c) + b * c * q**2)

                # Calculate Pi1
                Pi1[qq_idx, ww_idx] = np.real(
                    -Vc * self.math_funcs.atanJG((+kf * (b + c) + b * q) / d) / d +
                    Vc * self.math_funcs.atanJG((-kf * (b + c) + b * q) / d) / d -
                    Vc * self.math_funcs.atanJG((+kf * (b + c) - c * q) / d) / d +
                    Vc * self.math_funcs.atanJG((-kf * (b + c) - c * q) / d) / d
                )

                # Calculate Pi2
                Pi2[qq_idx, ww_idx] = q**2 * c0**2 / self.params.epsb / ((ww * dw)**2 + 9 * dw**2)

                # Calculate Pi3
                for kk_idx, k in enumerate(qy):
                    Pi3[qq_idx, ww_idx] += Vc * dq * (1.0 - self.math_funcs.fT0(k, kf) - self.math_funcs.fT0(k+q, kf)) * (
                        +1.0 / (hbar * ww * dw - Egap - self.math_funcs.eng(me, k+q) - self.math_funcs.eng(mh, k) + 1j * e0 * 5e-3) -
                        1.0 / (hbar * ww * dw + Egap + self.math_funcs.eng(me, k+q) + self.math_funcs.eng(mh, k) + 1j * e0 * 5e-3)
                    )

        # Write to file
        self._write_transverse_data(Pi2, Pi3, "ChiT.dat")

    def qqGq(self, ky: FloatArray, Nk: int, dk: float, dw: float,
             EpsR: FloatArray, EpsI: FloatArray, eh: str) -> None:
        """Calculate plasmon frequencies and damping."""
        Nw = self.params.Nw

        Omega = np.zeros(Nk, dtype=np.float64)
        Gam = np.zeros(Nk, dtype=np.float64)
        depsRdw = np.zeros(2*Nw+1, dtype=np.float64)

        for q in range(Nk):
            # Calculate derivative
            depsRdw[:] = (cshift(EpsR[q, :], 1) - cshift(EpsR[q, :], -1)) / (2.0 * dw)

            tmp = 0.0

            for w_idx, w in enumerate(range(-Nw, Nw+1)):
                if 1.0 / (np.abs(EpsR[q, w_idx]) + 1e-3) > tmp:
                    tmp = 1.0 / (np.abs(EpsR[q, w_idx]) + 1e-3)
                    Omega[q] = w * dw
                    Gam[q] = EpsI[q, w_idx] / depsRdw[w_idx]

        # Write to file
        self._write_plasmon_data(ky, Omega, Gam, f"Omega_qp.{eh}.dat")

    def _write_grid_info(self, Nq: int, qy: FloatArray) -> None:
        """Write grid information to file."""
        os.makedirs("dataQW/Wire", exist_ok=True)
        with open("dataQW/Wire/qw.dat", "w") as f:
            f.write(f"Nq {Nq}\n")
            f.write(f"Nw {self.params.Nw*2+1}\n")
            f.write(f"ky(1) {qy[0]}\n")
            f.write(f"dky {qy[1]-qy[0]}\n")
            f.write(f"w(1) {-self.params.Nw*self.params.dw}\n")
            f.write(f"dw {self.params.dw}\n")

    def _write_dielectric_data(self, epsR: FloatArray, epsI: FloatArray, filename: str) -> None:
        """Write dielectric function data to file."""
        os.makedirs("dataQW/Wire", exist_ok=True)
        with open(f"dataQW/Wire/{filename}", "w") as f:
            for w_idx in range(epsR.shape[1]):
                for q_idx in range(epsR.shape[0]):
                    f.write(f"{epsR[q_idx, w_idx]} {epsI[q_idx, w_idx]}\n")

    def _write_complex_dielectric_data(self, eps: ComplexArray, filename: str) -> None:
        """Write complex dielectric function data to file."""
        os.makedirs("dataQW/Wire", exist_ok=True)
        with open(f"dataQW/Wire/{filename}", "w") as f:
            for w_idx in range(eps.shape[1]):
                print(f"writing {w_idx}")
                for q_idx in range(eps.shape[0]):
                    f.write(f"{np.real(eps[q_idx, w_idx])} {np.imag(eps[q_idx, w_idx])}\n")

    def _write_transverse_data(self, Pi2: FloatArray, Pi3: ComplexArray, filename: str) -> None:
        """Write transverse response data to file."""
        os.makedirs("dataQW/Wire", exist_ok=True)
        with open(f"dataQW/Wire/{filename}", "w") as f:
            for w_idx in range(Pi2.shape[1]):
                print(f"writing {w_idx}")
                for q_idx in range(Pi2.shape[0]):
                    f.write(f"{Pi2[q_idx, w_idx]} {-np.real(Pi3[q_idx, w_idx])}\n")

    def _write_chi_data(self, epsR: FloatArray, q_idx: int, filename: str) -> None:
        """Write specific chi data to file."""
        os.makedirs("dataQW/Wire", exist_ok=True)
        with open(f"dataQW/Wire/{filename}", "w") as f:
            for w_idx in range(epsR.shape[1]):
                f.write(f"{w_idx} {epsR[q_idx, w_idx]}\n")

    def _write_plasmon_data(self, ky: FloatArray, Omega: FloatArray, Gam: FloatArray, filename: str) -> None:
        """Write plasmon data to file."""
        os.makedirs("dataQW/Wire", exist_ok=True)
        with open(f"dataQW/Wire/{filename}", "w") as f:
            for q in range(len(ky)):
                f.write(f"{ky[q]} {Omega[q]} {Gam[q]}\n")


# ============================================================================
# FORTRAN-COMPATIBLE INTERFACE FUNCTIONS
# ============================================================================

def GetEpsrLEpsrT(n1D: float, dcv0: float, Te: float, Th: float,
                  me: float, mh: float, Eg: float, ky: FloatArray) -> None:
    """Get longitudinal and transverse dielectric functions (Fortran-compatible interface)."""
    params = EpsrtlParameters()
    solver = EpsrtlSolver(params)
    solver.get_epsrL_epsrT(n1D, dcv0, Te, Th, me, mh, Eg, ky)


def RecordEpsrT(Te: float, Th: float, me: float, mh: float, Eg: float, ky: FloatArray) -> None:
    """Record transverse dielectric function (Fortran-compatible interface)."""
    params = EpsrtlParameters()
    solver = EpsrtlSolver(params)
    solver.record_epsrT(Te, Th, me, mh, Eg, ky)


def RecordEpsrL(Te: float, Th: float, me: float, mh: float, ky: FloatArray) -> None:
    """Record longitudinal dielectric function (Fortran-compatible interface)."""
    params = EpsrtlParameters()
    solver = EpsrtlSolver(params)
    solver.record_epsrL(Te, Th, me, mh, ky)


def RecordEpsrL_T0(me: float, mh: float, ky: FloatArray) -> None:
    """Record longitudinal dielectric function at T=0 (Fortran-compatible interface)."""
    params = EpsrtlParameters()
    solver = EpsrtlSolver(params)
    solver.record_epsrL_T0(me, mh, ky)


def PiT(q: float, w: float, me: float, mh: float, Te: float, Th: float,
        dk: float, Ek: FloatArray, Ekq: FloatArray) -> complex:
    """Transverse polarization function (Fortran-compatible interface)."""
    params = EpsrtlParameters()
    calc = PolarizationCalculator(params)
    return calc.calculate_piT(q, w, me, mh, Te, Th, dk, Ek, Ekq)


def PiL(q: float, w: float, m: float, T: float, dk: float, Ek: FloatArray, Ekq: FloatArray) -> complex:
    """Longitudinal polarization function (Fortran-compatible interface)."""
    params = EpsrtlParameters()
    calc = PolarizationCalculator(params)
    return calc.calculate_piL(q, w, m, T, dk, Ek, Ekq)


def PiL_T0(q: float, w: float, m: float, T: float, dk: float, Ek: FloatArray, Ekq: FloatArray) -> complex:
    """Longitudinal polarization function at T=0 (Fortran-compatible interface)."""
    params = EpsrtlParameters()
    calc = PolarizationCalculator(params)
    return calc.calculate_piL_T0(q, w, m, T, dk, Ek, Ekq)


def QqGq(ky: FloatArray, Nk: int, dk: float, dw: float,
         EpsR: FloatArray, EpsI: FloatArray, eh: str) -> None:
    """Calculate plasmon frequencies and damping (Fortran-compatible interface)."""
    params = EpsrtlParameters()
    solver = EpsrtlSolver(params)
    solver.qqGq(ky, Nk, dk, dw, EpsR, EpsI, eh)


def ZeroT_L(B: str, m: float, qy: FloatArray, kf: float) -> None:
    """Zero temperature longitudinal calculation (Fortran-compatible interface)."""
    params = EpsrtlParameters()
    solver = EpsrtlSolver(params)
    solver.zeroT_L(B, m, qy, kf)


def ZeroT_T(me: float, mh: float, Egap: float, dcv: float, qy: FloatArray, kf: float) -> None:
    """Zero temperature transverse calculation (Fortran-compatible interface)."""
    params = EpsrtlParameters()
    solver = EpsrtlSolver(params)
    solver.zeroT_T(me, mh, Egap, dcv, qy, kf)


def atanhc(x: complex) -> complex:
    """Inverse hyperbolic tangent (Fortran-compatible interface)."""
    return MathematicalFunctions.atanhc(x)


def atanJG(z: complex) -> complex:
    """Complex arctangent function (Fortran-compatible interface)."""
    return MathematicalFunctions.atanJG(z)


def Eng(m: float, k: float) -> float:
    """Energy dispersion (Fortran-compatible interface)."""
    return MathematicalFunctions.eng(m, k)


def fT0(k: float, kf: float) -> float:
    """Zero temperature Fermi function (Fortran-compatible interface)."""
    return MathematicalFunctions.fT0(k, kf)


def ff0(E: float, T: float, m: float) -> float:
    """Finite temperature distribution function (Fortran-compatible interface)."""
    params = EpsrtlParameters()
    return MathematicalFunctions.ff0(E, T, m, params.n1D)


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def example_usage():
    """Example usage of the epsrtl module."""
    # Create parameters
    params = EpsrtlParameters(
        n1D=1e6,
        dcv=1e-28,
        Te=300.0,
        Th=300.0,
        me=0.067 * me0,
        mh=0.45 * me0,
        Eg=1.42
    )

    # Create momentum grid
    ky = np.linspace(-1e9, 1e9, 100)

    # Create solver
    solver = EpsrtlSolver(params)

    # Calculate dielectric functions
    try:
        solver.get_epsrL_epsrT(params.n1D, params.dcv, params.Te, params.Th,
                              params.me, params.mh, params.Eg, ky)
    except SystemExit:
        print("Calculation completed successfully")


# Create a fast example for epsrtlpythonic.py
def fast_example_usage():
    """Fast example usage of the epsrtl module."""
    # Create parameters with smaller grid for faster execution
    params = EpsrtlParameters(
        n1D=1e6,
        dcv=1e-28,
        Te=300.0,
        Th=300.0,
        me=0.067 * me0,
        mh=0.45 * me0,
        Eg=1.42,
        Nw=20,  # Reduced from 500 to 20 for much faster execution
        dw=2.35e15 / 20.0 * 2  # Adjust dw accordingly
    )

    # Create much smaller momentum grid for faster execution
    ky = np.linspace(-1e8, 1e8, 10)  # Reduced from 100 to 10 points

    # Create solver
    solver = EpsrtlSolver(params)

    # Calculate dielectric functions
    try:
        solver.get_epsrL_epsrT(params.n1D, params.dcv, params.Te, params.Th,
                              params.me, params.mh, params.Eg, ky)
    except SystemExit:
        print("Fast calculation completed successfully")


if __name__ == "__main__":
    # fast_example_usage()
