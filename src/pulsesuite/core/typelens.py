from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Annotated, TextIO, Union

import numpy as np
from guardrails.guardrails import with_guardrails
from numba import float64, njit
from typemedium import *

# Project-wide precision alias
dp = np.float64

# File tag constant for lens parameter files
_filetag1 = "[params.lens_v1.0]"

# ----------------------------------------------------------------------
# Data container: Python counterpart of Fortran's `type nlls`
# ----------------------------------------------------------------------
@dataclass(slots=True)
class nlls:
    """
    Nonlinear Lens Structure (plano-convex lens) parameters.
    """
    version:            int                                  = -1
    medium:             nlms                                 = field(default_factory=nlms)
    rc0:                Annotated[float, np.float64]        = dp(1e100)
    wavelength_design:  Annotated[float, np.float64]        = dp(0.0)

    def __post_init__(self):
        # enforce correct types
        self.version = int(self.version)
        self.rc0 = dp(self.rc0)
        self.wavelength_design = dp(self.wavelength_design)

    # ---------------- Pythonic properties --------------------
    @property
    def focal_length(self) -> float:
        """Optical focal length: rc0 / (n0 - 1)"""
        return CalcFocalLength(self)

    @property
    def rc(self) -> float:
        """Radius of curvature (m)"""
        return GetRC(self)
    @rc.setter
    def rc(self, value: float) -> None:
        SetRC(self, value)

    @property
    def design_wavelength(self) -> float:
        """Design wavelength (m)"""
        return GetDesignWavelength(self)
    @design_wavelength.setter
    def design_wavelength(self, value: float) -> None:
        SetDesignWavelength(self, value)

    @property
    def nz(self) -> int:
        """Number of steps"""
        return GetNz_nlls(self)
    @nz.setter
    def nz(self, value: int) -> None:
        SetNz_nlls(self, value)

    @property
    def dz(self) -> float:
        """Step size in propagation direction"""
        return GetDz_nlls(self)
    @dz.setter
    def dz(self, value: float) -> None:
        SetDz_nlls(self, value)

    @property
    def length(self) -> float:
        """Total lens length (m)"""
        return GetLength_nlls(self)
    @length.setter
    def length(self, value: float) -> None:
        SetLength_nlls(self, value)

    @property
    def n0(self) -> float:
        """Index of refraction at center"""
        return GetN0_nlls(self)
    @n0.setter
    def n0(self, value: float) -> None:
        SetN0_nlls(self, value)

    @property
    def k2(self) -> float:
        """Dispersion parameter"""
        return GetK2_nlls(self)
    @k2.setter
    def k2(self, value: float) -> None:
        SetK2_nlls(self, value)

    @property
    def n2i(self) -> float:
        """Intensity-dependent nonlinear index"""
        return GetN2I_nlls(self)
    @n2i.setter
    def n2i(self, value: float) -> None:
        SetN2I_nlls(self, value)

    @property
    def beta(self) -> float:
        """Two-photon absorption coefficient"""
        return GetBeta_nlls(self)
    @beta.setter
    def beta(self, value: float) -> None:
        SetBeta_nlls(self, value)

    @property
    def alpha(self) -> float:
        """Linear absorption coefficient"""
        return GetAlpha_nlls(self)
    @alpha.setter
    def alpha(self, value: float) -> None:
        SetAlpha_nlls(self, value)

    @property
    def material(self) -> str:
        """Material name"""
        return GetMaterial_nlls(self)
    @material.setter
    def material(self, value: str) -> None:
        SetMaterial_nlls(self, value)

    @property
    def wavelength(self) -> float:
        """Operating wavelength (m)"""
        return GetWavelength_nlls(self)
    @wavelength.setter
    def wavelength(self, value: float) -> None:
        SetWavelength_nlls(self, value)

    @property
    def medium_data(self) -> nlms:
        """Underlying medium data"""
        return GetMedium_nlls(self)
    @medium_data.setter
    def medium_data(self, value: nlms) -> None:
        SetMedium_nlls(self, value)

# ----------------------------------------------------------------------
# Free-standing getters and setters (legacy names)
# ----------------------------------------------------------------------
@with_guardrails
def GetVersion_nlls(lens: nlls) -> int:
    return lens.version

@with_guardrails
def CalcFocalLength(lens: nlls) -> Annotated[float, np.float64]:
    return lens.rc0 / (GetN0_nlms(lens.medium) - dp(1.0))

@with_guardrails
def GetNz_nlls(lens: nlls) -> int:
    return GetNz_nlms(lens.medium)

@with_guardrails
def SetNz_nlls(lens: nlls, N: int) -> None:
    SetNz_nlms(lens.medium, N)

@with_guardrails
def GetDz_nlls(lens: nlls) -> Annotated[float, np.float64]:
    return GetDz_nlms(lens.medium)

@with_guardrails
def SetDz_nlls(lens: nlls, Dz: Annotated[float, np.float64]) -> None:
    SetDz_nlms(lens.medium, Dz)

@with_guardrails
def GetLength_nlls(lens: nlls) -> Annotated[float, np.float64]:
    return GetLength_nlms(lens.medium)

@with_guardrails
def SetLength_nlls(lens: nlls, L: Annotated[float, np.float64]) -> None:
    SetLength_nlms(lens.medium, L)

@with_guardrails
def GetN0_nlls(lens: nlls) -> Annotated[float, np.float64]:
    return GetN0_nlms(lens.medium)

@with_guardrails
def SetN0_nlls(lens: nlls, n0: Annotated[float, np.float64]) -> None:
    SetN0_nlms(lens.medium, n0)

@with_guardrails
def GetK2_nlls(lens: nlls) -> Annotated[float, np.float64]:
    return GetK2_nlms(lens.medium)

@with_guardrails
def SetK2_nlls(lens: nlls, k2: Annotated[float, np.float64]) -> None:
    SetK2_nlms(lens.medium, k2)

@with_guardrails
def GetN2I_nlls(lens: nlls) -> Annotated[float, np.float64]:
    return GetN2I_nlms(lens.medium)

@with_guardrails
def SetN2I_nlls(lens: nlls, n2: Annotated[float, np.float64]) -> None:
    SetN2I_nlms(lens.medium, n2)

@with_guardrails
def GetBeta_nlls(lens: nlls) -> Annotated[float, np.float64]:
    return GetBeta_nlms(lens.medium)

@with_guardrails
def SetBeta_nlls(lens: nlls, beta: Annotated[float, np.float64]) -> None:
    SetBeta_nlms(lens.medium, beta)

@with_guardrails
def GetAlpha_nlls(lens: nlls) -> Annotated[float, np.float64]:
    return GetAlpha_nlms(lens.medium)

@with_guardrails
def SetAlpha_nlls(lens: nlls, alpha: Annotated[float, np.float64]) -> None:
    SetAlpha_nlms(lens.medium, alpha)

@with_guardrails
def GetRC(lens: nlls) -> Annotated[float, np.float64]:
    return lens.rc0

@with_guardrails
def SetRC(lens: nlls, rc: Annotated[float, np.float64]) -> None:
    lens.rc0 = dp(rc)

@with_guardrails
def GetMaterial_nlls(lens: nlls) -> str:
    return lens.medium.material

@with_guardrails
def SetMaterial_nlls(lens: nlls, mat: str) -> None:
    lens.medium.material = mat

@with_guardrails
def GetWavelength_nlls(lens: nlls) -> Annotated[float, np.float64]:
    return lens.medium.wavelength

@with_guardrails
def SetWavelength_nlls(lens: nlls, lam: Annotated[float, np.float64]) -> None:
    lens.medium.wavelength = lam

@with_guardrails
def GetMedium_nlls(lens: nlls) -> nlms:
    return lens.medium

@with_guardrails
def SetMedium_nlls(lens: nlls, medium: nlms) -> None:
    lens.medium = medium

@with_guardrails
def GetDesignWavelength(lens: nlls) -> Annotated[float, np.float64]:
    return lens.wavelength_design

@with_guardrails
def SetDesignWavelength(lens: nlls, lam: Annotated[float, np.float64]) -> None:
    lens.wavelength_design = dp(lam)

# ----------------------------------------------------------------------
# I/O: read/write lens parameters
# ----------------------------------------------------------------------
def readlensparams(cmd: Union[str, pathlib.Path, TextIO], lens: nlls) -> None:
    close_when_done = False
    if isinstance(cmd, (str, pathlib.Path)):
        fh = open(cmd, 'rt', encoding='utf-8')
        close_when_done = True
    else:
        fh = cmd
    readmediumparams(fh, lens.medium)
    line = fh.readline().strip()
    if line == _filetag1:
        lens.version = 10
        lens.rc0 = dp(float(fh.readline().split()[0]))
    else:
        if line.startswith('['): raise ValueError(f"Unrecognized params file: {line}")
        lens.version = 0
        lens.rc0 = dp(float(line.split()[0]))
    lens.wavelength_design = dp(float(fh.readline().split()[0]))
    if close_when_done: fh.close()
    dumplens(lens)

def writelensparams(cmd: Union[str, pathlib.Path, TextIO], lens: nlls) -> None:
    close_when_done = False
    if isinstance(cmd, (str, pathlib.Path)):
        fh = open(cmd, 'wt', encoding='utf-8')
        close_when_done = True
    else:
        fh = cmd
    writemediumparams(fh, lens.medium)
    fh.write(f"{_filetag1}\n")
    fh.write(f"{lens.rc0:.15e} : Radius of curvature (m)\n")
    fh.write(f"{lens.wavelength_design:.15e} : Design wavelength (m)\n")
    if close_when_done: fh.close()

def dumplens(lens: nlls, level: int | None = None) -> None:
    writelensparams(sys.stdout, lens)

# ----------------------------------------------------------------------
# Math micro-kernel: calculate 1 - sqrt(1 - x)
# ----------------------------------------------------------------------
@njit(float64(float64), fastmath=True, cache=True)
def BillsMagic(x: float) -> float:
    if x > 0.2:
        return 1.0 - math.sqrt(1.0 - x)
    coeffs = np.array([
        1/2, 1/8, 1/16, 5/128, 7/256, 21/1024, 33/2048,
        429/32768, 715/65536, 2431/262144, 4199/524288,
        29393/4194304, 52003/8388608, 185725/33554432,
        334305/67108864, 9694845/2147483648,
        17678835/4294967296, 64822395/17179869184,
        119409675/34359738368, 883631595/274877906944
    ], dtype=float64)
    y = 0.0
    xp = x
    for c in coeffs:
        y += c * xp
        xp *= x
    return y

# ----------------------------------------------------------------------
# Module exports
# ----------------------------------------------------------------------
__all__ = [
    'nlls', 'GetVersion_nlls', 'CalcFocalLength', 'GetNz_nlls', 'SetNz_nlls',
    'GetDz_nlls', 'SetDz_nlls', 'GetLength_nlls', 'SetLength_nlls',
    'GetN0_nlls', 'SetN0_nlls', 'GetK2_nlls', 'SetK2_nlls',
    'GetN2I_nlls', 'SetN2I_nlls', 'GetBeta_nlls', 'SetBeta_nlls',
    'GetAlpha_nlls', 'SetAlpha_nlls', 'GetRC', 'SetRC',
    'GetMaterial_nlls', 'SetMaterial_nlls', 'GetWavelength_nlls',
    'SetWavelength_nlls', 'GetMedium_nlls', 'SetMedium_nlls',
    'GetDesignWavelength', 'SetDesignWavelength',
    'readlensparams', 'writelensparams', 'dumplens', 'BillsMagic'
]
