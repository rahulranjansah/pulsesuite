"""rungekutta.py – high‑performance 4th‑order Runge‑Kutta propagator
for the (modified) nonlinear Schrödinger / UPPE equations.

* No Fortran; entirely Python + Numba + pyFFTW.
* Designed to interoperate with `fastfft.py` and the future
  `gulley.py` plasma module.
* All expensive maths are JIT‑compiled; only plan creation and
  dataclass bookkeeping run in the CPython interpreter.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Optional

import numpy as np
from numba import njit, prange

# Import constants from libpulsesuite (module-level constants)
from .fftw import (
    fft_1d,  # noqa: F401 (re‑export convenience)
    )

# -----------------------------------------------------------------------------
# 1.  Basic physics containers (Field, Medium, Flags)
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class Field:
    """Spatial‑temporal grid and associated cached arrays."""

    Exy: np.ndarray  # complex128 (Nx, Ny, Nt) – ***Fortran‑contiguous***
    dx: float
    dy: float
    dt: float
    lambda0: float

    # cached coordinate arrays (auto‑filled)
    t_arr: np.ndarray = _field(init=False)
    w_arr: np.ndarray = _field(init=False)

    def __post_init__(self):
        Nt = self.Exy.shape[2]
        self.t_arr = np.arange(Nt) * self.dt
        self.w_arr = np.fft.fftfreq(Nt, self.dt, d=1.0) * 2 * np.pi

    # helper accessors (mirror original Get*/Calc* names)
    def w0(self) -> float:
        return 2 * np.pi * 3e8 / self.lambda0

    def intensity(self) -> np.ndarray:  # |E|^2, vacuum
        return (self.Exy.real ** 2 + self.Exy.imag ** 2)


@dataclass(slots=True)
class Medium:
    n0: float          # linear refractive index
    n2: float          # Kerr coeff (m^2/W)
    n4i: float = 0.0   # HO Kerr imaginary
    tc: float = 1.2e-15  # collision time (s)
    sigma: float = 5e-20 # Drude absorption coeff (m^2)
    rho_max: float = 6e28  # max plasma density (m^-3)
    # add more as needed


@dataclass(slots=True)
class RKFlags:
    kerr: bool = True
    kerr4: bool = False
    raman: bool = True
    shock: bool = True

    photo: bool = False
    drude_0: bool = False

    gulley_t: bool = False   # switch to Gulley plasma once intensity high


# -----------------------------------------------------------------------------
# 2.  Cached nonlinear constants (Gamma, etc.)
# -----------------------------------------------------------------------------

def calc_gamma(field: Field, medium: Medium) -> complex:
    k0 = 2 * np.pi / field.lambda0 * medium.n0
    return 2.0 * k0 * medium.n2 / medium.n0  # match Fortran comment


def calc_gamma4(field: Field, medium: Medium) -> complex:
    k0 = 2 * np.pi / field.lambda0 * medium.n0
    return 2.0 * k0 * medium.n4i * (medium.n0 * 3e8 * 8.854187e-12) ** 2


# -----------------------------------------------------------------------------
# 3.  Nyquist‑friendly time‑domain helpers (Numba JIT)
# -----------------------------------------------------------------------------
@njit(fastmath=True)
def _magsq(z: np.ndarray):
    return z.real * z.real + z.imag * z.imag


@njit(parallel=True, fastmath=True)
def _shock_filter(freq: np.ndarray, w0: float):
    """Return T(w)/U(w) factor used in self‑steepening."""
    out = np.empty_like(freq)
    for k in prange(freq.size):
        w = freq[k]
        out[k] = 1j + 0j  # dummy placeholder; replaced below
    return out  # TODO: implement


# -----------------------------------------------------------------------------
# 4.  Raman convolution (FFT domain multiply)
# -----------------------------------------------------------------------------
from functools import lru_cache


@lru_cache(maxsize=8)
def _raman_kernel(Nt: int, dt: float, t1: float, t2: float, fr: float):
    t = np.arange(Nt) * dt
    h_t = (t1**2 + t2**2) / t1 / t2**2 * np.exp(-t / t2) * np.sin(t / t1)
    H_w = np.fft.fft(h_t)
    return fr * H_w  # multiply inside spectral conv


def raman_conv(e_time: np.ndarray, field: Field, t1=12.2e-15, t2=32e-15, fr=0.18):
    H_w = _raman_kernel(e_time.size, field.dt, t1, t2, fr)
    E_w = np.fft.fft(e_time)
    return np.fft.ifft(E_w * H_w)


# -----------------------------------------------------------------------------
# 5.  Right‑hand‑side (nonlinear polarisation & plasma) – Numba kernel
# -----------------------------------------------------------------------------
@njit(fastmath=True)
def _rhs_line(e: np.ndarray, gamma: complex, gamma4: complex,
             do_kerr: bool, do_kerr4: bool) -> np.ndarray:
    """RHS for one temporal line (no Raman, plasma, shock)."""
    Nt = e.size
    out = np.empty_like(e)
    for k in range(Nt):
        val = 0.0 + 0.0j
        if do_kerr:
            val += 1j * gamma * (_magsq(e[k])) * e[k]
        if do_kerr4:
            mag2 = _magsq(e[k])
            val += 1j * gamma4 * (mag2 * mag2) * e[k]
        out[k] = val
    return out


# -----------------------------------------------------------------------------
# 6.  Vectorised wrapper over (Nx, Ny, Nt)
# -----------------------------------------------------------------------------

def rhs(Exy: np.ndarray, field: Field, medium: Medium, flags: RKFlags):
    gamma  = calc_gamma(field, medium)
    gamma4 = calc_gamma4(field, medium) if flags.kerr4 else 0.0j

    Nx, Ny, _ = Exy.shape
    dE = np.empty_like(Exy)

    for j in range(Ny):
        for i in range(Nx):
            dE[i, j, :] = _rhs_line(Exy[i, j, :], gamma, gamma4,
                                     flags.kerr, flags.kerr4)
    return dE


# -----------------------------------------------------------------------------
# 7.  4th‑order Runge–Kutta step (adaptive dz optional)
# -----------------------------------------------------------------------------

def rk4_step(field: Field, medium: Medium, dz: float, flags: RKFlags,
             sub_steps: Optional[int] = None):
    """Advance `field.Exy` by `dz` in‑place using RK4."""
    Exy = field.Exy
    if sub_steps is None:
        sub_steps = 1
    dz_sub = dz / sub_steps

    for _ in range(sub_steps):
        k1 = rhs(Exy, field, medium, flags) * dz_sub
        k2 = rhs(Exy + 0.5 * k1, field, medium, flags) * dz_sub
        k3 = rhs(Exy + 0.5 * k2, field, medium, flags) * dz_sub
        k4 = rhs(Exy + k3,        field, medium, flags) * dz_sub
        Exy += (k1 + 2 * (k2 + k3) + k4) / 6.0


# -----------------------------------------------------------------------------
# 8.  Public convenience integrator class
# -----------------------------------------------------------------------------

class RK4Integrator:
    """Ease‑of‑use wrapper that keeps medium/flags constant."""

    def __init__(self, field: Field, medium: Medium, dz: float, flags: RKFlags):
        self.field = field
        self.medium = medium
        self.dz = dz
        self.flags = flags
        # prime Numba / FFTW plans
        rhs(self.field.Exy, self.field, self.medium, self.flags)

    def step(self, n: int = 1):
        for _ in range(n):
            rk4_step(self.field, self.medium, self.dz, self.flags)

    def run(self, z_max: float):
        nsteps = int(np.ceil(z_max / self.dz))
        self.step(nsteps)


__all__ = [
    "Field", "Medium", "RKFlags", "RK4Integrator",
    "rk4_step", "rhs",
]
