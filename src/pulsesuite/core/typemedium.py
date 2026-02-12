from __future__ import annotations

import pathlib
from typing import Annotated, Iterable, TextIO, Union

import numpy as np
from constants import c0, pfrmtA
from guardrails.guardrails import with_guardrails
from type_plasma import (
    plasma_coefficients,
    readplasmaparams_sub as readplasmaparams,
    writeplasmaparams_sub as writeplasmaparams,
)

# -----------------------------------------------------------------------------
# Project-wide aliases and version tags (mimic Fortran kinds)
# -----------------------------------------------------------------------------
dp = np.float64  # double-precision real (Matches `real(dp)` in Fortran)

medium_filetag: tuple[str, ...] = (
    "[params.medium_v1.0]",
    "[params.medium_v1.5]",
    "[params.medium_v2.0]",
    "[params.medium_v2.1]",
    "[params.medium_v2.2]",
)


# -----------------------------------------------------------------------------
# Helper – skip blanks and comments
# -----------------------------------------------------------------------------
def _fp(lines: Iterable[str]) -> Iterable[str]:
    for ln in lines:
        s = ln.strip()
        if s and not s.startswith('#'):
            yield s


def _parse_number(line: str, want_float: bool = True):
    token = line.split(':', 1)[0].strip()
    try:
        return (dp if want_float else int)(token)
    except ValueError as e:
        raise ValueError(f"cannot parse numeric value from line: {line!r}") from e


# -----------------------------------------------------------------------------
# Main data container: Python counterpart of Fortran's `type nlms`
# -----------------------------------------------------------------------------
class nlms:
    __slots__ = (
        "version",
        "material",
        "length",
        "dz",
        "wavelength",
        "n0",
        "k",
        "n2",
        "beta",
        "alpha",
        "Tr",
        "n4",
        "plasma_params",
    )

    def __init__(
        self,
        version: int = -1,
        material: str = "",
        length: Annotated[float, np.float64] = 0.0,
        dz: Annotated[float, np.float64] = 0.0,
        wavelength: Annotated[float, np.float64] = 0.0,
        n0: Annotated[float, np.float64] = 0.0,
        k: Annotated[np.ndarray, np.float64] | None = None,
        n2: Annotated[float, np.float64] = 0.0,
        beta: Annotated[float, np.float64] = 0.0,
        alpha: Annotated[float, np.float64] = 0.0,
        Tr: Annotated[float, np.float64] = 0.0,
        n4: Annotated[float, np.float64] = 0.0,
        plasma_params: plasma_coefficients | None = None,
    ) -> None:
        self.version = int(version)
        self.material = material
        self.length = dp(length)
        self.dz = dp(dz)
        self.wavelength = dp(wavelength)
        self.n0 = dp(n0)
        self.k = np.zeros(5, dtype=np.float64) if k is None else np.array(k, dtype=np.float64)
        self.n2 = dp(n2)
        self.beta = dp(beta)
        self.alpha = dp(alpha)
        self.Tr = dp(Tr)
        self.n4 = dp(n4)
        self.plasma_params = plasma_params.copy() if plasma_params else plasma_coefficients()

    def copy(self) -> nlms:
        """Shallow copy of all fields."""
        return nlms(
            self.version,
            self.material,
            self.length,
            self.dz,
            self.wavelength,
            self.n0,
            self.k.copy(),
            self.n2,
            self.beta,
            self.alpha,
            self.Tr,
            self.n4,
            self.plasma_params.copy(),
        )

    def __repr__(self) -> str:
        return (
            f"nlms(version={self.version}, material={self.material!r}, length={self.length:.3e}, dz={self.dz:.3e}, "
            f"wavelength={self.wavelength:.3e}, n0={self.n0:.3e}, k={self.k!r}, n2={self.n2:.3e}, beta={self.beta:.3e}, "
            f"alpha={self.alpha:.3e}, Tr={self.Tr:.3e}, n4={self.n4:.3e}, plasma_params={self.plasma_params!r})"
        )

    def __len__(self) -> int:
        """len(medium) → number of steps Nz"""
        return int(self.length/self.dz) if self.dz else 0

    def __eq__(self, other):
        if not isinstance(other, nlms):
            return False
        return (
            self.version == other.version and
            self.material == other.material and
            np.isclose(self.length, other.length) and
            np.isclose(self.dz, other.dz) and
            np.isclose(self.wavelength, other.wavelength) and
            np.isclose(self.n0, other.n0) and
            np.allclose(self.k, other.k) and
            np.isclose(self.n2, other.n2) and
            np.isclose(self.beta, other.beta) and
            np.isclose(self.alpha, other.alpha) and
            np.isclose(self.Tr, other.Tr) and
            np.isclose(self.n4, other.n4)
            )

    # ————————————————————————————————
    # 2) Per-coefficient properties (k1…k5)
    # ————————————————————————————————
    @property
    def k1(self) -> float:
        return float(self.k[0])
    @k1.setter
    def k1(self, val: float) -> None:
        self.k[0] = dp(val)

    @property
    def k2(self) -> float:
        return float(self.k[1])
    @k2.setter
    def k2(self, val: float) -> None:
        self.k[1] = dp(val)

    @property
    def k3(self) -> float:
        return float(self.k[2])
    @k3.setter
    def k3(self, val: float) -> None:
        self.k[2] = dp(val)

    @property
    def k4(self) -> float:
        return float(self.k[3])
    @k4.setter
    def k4(self, val: float) -> None:
        self.k[3] = dp(val)

    @property
    def k5(self) -> float:
        return float(self.k[4])
    @k5.setter
    def k5(self, val: float) -> None:
        self.k[4] = dp(val)

    # ————————————————————————————————
    # 3) General “to/from dict” for easy (de)serialization
    # ————————————————————————————————
    def to_dict(self) -> dict:
        return {
            "version":   self.version,
            "material":  self.material,
            "length":    float(self.length),
            "dz":        float(self.dz),
            "wavelength":float(self.wavelength),
            "n0":        float(self.n0),
            "k":         list(self.k),
            "n2":        float(self.n2),
            "beta":      float(self.beta),
            "alpha":     float(self.alpha),
            "Tr":        float(self.Tr),
            "n4":        float(self.n4),
            "plasma":    self.plasma_params.to_dict(),  # assuming you add it there too
        }

    @classmethod
    def from_dict(cls, d: dict) -> nlms:
        inst = cls()
        inst.version   = d["version"]
        inst.material  = d["material"]
        inst.length    = dp(d["length"])
        inst.dz        = dp(d["dz"])
        inst.wavelength= dp(d["wavelength"])
        inst.n0        = dp(d["n0"])
        inst.k         = np.array(d["k"], dtype=np.float64)
        inst.n2        = dp(d["n2"])
        inst.beta      = dp(d["beta"])
        inst.alpha     = dp(d["alpha"])
        inst.Tr        = dp(d["Tr"])
        inst.n4        = dp(d["n4"])
        inst.plasma_params = plasma_coefficients.from_dict(d["plasma"])
        return inst

    # ————————————————————————————————
    # 4) Copy‐module integration and equality
    # ————————————————————————————————
    def __copy__(self) -> nlms:
        return self.copy()

    def __deepcopy__(self, memo) -> nlms:
        return self.copy()


# -----------------------------------------------------------------------------
# I/O operations (mirroring Fortran subroutines)
# -----------------------------------------------------------------------------
@with_guardrails
def readmediumparams(cmd: Union[str, pathlib.Path, TextIO], medium: nlms) -> None:
    """Read medium parameters from file or stream into `medium`."""
    if isinstance(cmd, (str, pathlib.Path)):
        with open(cmd, 'rt', encoding='utf-8') as fh:
            _readmediumparams_sub(fh, medium)
    else:
        _readmediumparams_sub(cmd, medium)


def _readmediumparams_sub(handle: TextIO, medium: nlms) -> None:
    first = next(_fp(handle), None)
    if first is None:
        raise EOFError("empty medium parameter stream")

    try:
        idx = medium_filetag.index(first)
    except ValueError:
        # Legacy or unrecognized: treat as v0.0 header
        idx = -1

    # dispatch based on version tag
    if idx == 4:
        _readmediumparams_sub_2_2(handle, medium)
    elif idx == 3:
        _readmediumparams_sub_2_1(handle, medium)
    elif idx == 2:
        _readmediumparams_sub_2_0(handle, medium)
    elif idx == 1:
        _readmediumparams_sub_1_5(handle, medium)
    elif idx == 0:
        _readmediumparams_sub_1_0(handle, medium)
    else:
        _readmediumparams_sub_0_0(first, handle, medium)


# Version-specific readers
def _readmediumparams_sub_2_2(handle: TextIO, medium: nlms) -> None:
    medium.version = 22
    medium.material = next(_fp(handle)).split(':', 1)[0].strip()
    medium.length = _parse_number(next(_fp(handle)))
    medium.dz = _parse_number(next(_fp(handle)))
    medium.wavelength = _parse_number(next(_fp(handle)))
    medium.n0 = _parse_number(next(_fp(handle)))
    medium.k = np.array([_parse_number(next(_fp(handle))) for _ in range(5)], dtype=np.float64)
    medium.n2 = _parse_number(next(_fp(handle)))
    medium.beta = _parse_number(next(_fp(handle)))
    medium.alpha = _parse_number(next(_fp(handle)))
    medium.Tr = _parse_number(next(_fp(handle)))
    medium.n4 = _parse_number(next(_fp(handle)))
    readplasmaparams(handle, medium.plasma_params)


def _readmediumparams_sub_2_1(handle: TextIO, medium: nlms) -> None:
    medium.version = 21
    medium.material = next(_fp(handle)).split(':', 1)[0].strip()
    medium.length = _parse_number(next(_fp(handle)))
    medium.dz = _parse_number(next(_fp(handle)))
    medium.wavelength = _parse_number(next(_fp(handle)))
    medium.n0 = _parse_number(next(_fp(handle)))
    medium.k = np.array([_parse_number(next(_fp(handle))) for _ in range(5)], dtype=np.float64)
    medium.n2 = _parse_number(next(_fp(handle)))
    medium.beta = _parse_number(next(_fp(handle)))
    medium.alpha = _parse_number(next(_fp(handle)))
    medium.Tr = _parse_number(next(_fp(handle)))
    readplasmaparams(handle, medium.plasma_params)


def _readmediumparams_sub_2_0(handle: TextIO, medium: nlms) -> None:
    medium.version = 20
    medium.material = next(_fp(handle)).split(':', 1)[0].strip()
    medium.length = _parse_number(next(_fp(handle)))
    medium.dz = _parse_number(next(_fp(handle)))
    medium.wavelength = _parse_number(next(_fp(handle)))
    medium.n0 = _parse_number(next(_fp(handle)))
    medium.k = np.array([_parse_number(next(_fp(handle))) for _ in range(5)], dtype=np.float64)
    medium.n2 = _parse_number(next(_fp(handle)))
    medium.beta = _parse_number(next(_fp(handle)))
    medium.alpha = _parse_number(next(_fp(handle)))
    medium.Tr = _parse_number(next(_fp(handle)))


def _readmediumparams_sub_1_5(handle: TextIO, medium: nlms) -> None:
    medium.version = 15
    medium.material = next(_fp(handle)).split(':', 1)[0].strip()
    Nz = int(_parse_number(next(_fp(handle)), want_float=False))
    medium.dz = _parse_number(next(_fp(handle)))
    medium.length = dp(Nz) * medium.dz
    medium.wavelength = _parse_number(next(_fp(handle)))
    medium.n0 = _parse_number(next(_fp(handle)))
    medium.k = np.array([_parse_number(next(_fp(handle))) for _ in range(5)], dtype=np.float64)
    medium.n2 = _parse_number(next(_fp(handle)))
    medium.beta = _parse_number(next(_fp(handle)))
    medium.alpha = _parse_number(next(_fp(handle)))
    medium.Tr = _parse_number(next(_fp(handle)))


def _readmediumparams_sub_1_0(handle: TextIO, medium: nlms) -> None:
    medium.version = 10
    medium.material = next(_fp(handle)).split(':', 1)[0].strip()
    Nz = int(_parse_number(next(_fp(handle)), want_float=False))
    medium.dz = _parse_number(next(_fp(handle)))
    medium.length = dp(Nz) * medium.dz
    medium.wavelength = _parse_number(next(_fp(handle)))
    medium.n0 = medium.n0 / c0
    medium.k = np.array([medium.n0 / c0] + [_parse_number(next(_fp(handle))) for _ in range(1,5)], dtype=np.float64)
    medium.n2 = _parse_number(next(_fp(handle)))
    medium.beta = _parse_number(next(_fp(handle)))
    medium.alpha = _parse_number(next(_fp(handle)))


def _readmediumparams_sub_0_0(first: str, handle: TextIO, medium: nlms) -> None:
    medium.version = 0
    medium.material = first.split(':', 1)[0].strip()
    Nz = int(_parse_number(next(_fp(handle)), want_float=False))
    medium.dz = _parse_number(next(_fp(handle)))
    medium.length = dp(Nz) * medium.dz
    medium.wavelength = _parse_number(next(_fp(handle)))
    medium.n0 = _parse_number(next(_fp(handle)))
    medium.k = np.array([medium.n0 / c0] + [_parse_number(next(_fp(handle))) for _ in range(1,5)], dtype=np.float64)
    medium.n2 = _parse_number(next(_fp(handle)))
    medium.beta = _parse_number(next(_fp(handle)))
    medium.alpha = _parse_number(next(_fp(handle)))


@with_guardrails
def writemediumparams(cmd: Union[str, pathlib.Path, TextIO], medium: nlms) -> None:
    """Write medium parameters from `medium` to file or stream."""
    if isinstance(cmd, (str, pathlib.Path)):
        with open(cmd, 'wt', encoding='utf-8') as fh:
            _writemediumparams_sub(fh, medium)
    else:
        _writemediumparams_sub(cmd, medium)


def _writemediumparams_sub(handle: TextIO, medium: nlms) -> None:
    # Header
    handle.write(f"{medium_filetag[-1]}\n")
    handle.write(f"{medium.material:<25} : The material.\n")
    handle.write(f"{medium.length:{pfrmtA}} : Material length (m)\n")
    handle.write(f"{medium.dz:{pfrmtA}} : Step size (m)\n")
    handle.write(f"{medium.wavelength:{pfrmtA}} : Wavelength (m)\n")
    handle.write(f"{medium.n0:{pfrmtA}} : Index of refraction\n")
    for i, desc in enumerate(["k1 - GV parameter", "k2 - GVD parameter", "k3 - 3OD parameter", "k4 - 4OD parameter", "k5 - 5OD parameter"]):
        handle.write(f"{medium.k[i]:{pfrmtA}} : {desc} (s^{i+1}/m)\n")
    handle.write(f"{medium.n2:{pfrmtA}} : Nonlinear index (m^2/W)\n")
    handle.write(f"{medium.beta:{pfrmtA}} : Two-photon absorption (m/W)\n")
    handle.write(f"{medium.alpha:{pfrmtA}} : Linear absorption (1/m)\n")
    handle.write(f"{medium.Tr:{pfrmtA}} : Raman response time (s)\n")
    handle.write(f"{medium.n4:{pfrmtA}} : 2nd-order nonlinear index (m^4/W^2)\n")
    writeplasmaparams(handle, medium.plasma_params)


# -----------------------------------------------------------------------------
# Logging helper (mirroring `dumpmedium`)
# -----------------------------------------------------------------------------
@with_guardrails
def dumpmedium(medium: nlms, level: int | None = None) -> None:
    """Write medium parameters to stdout if log level is sufficient."""
    # For simplicity, always dump
    _writemediumparams_sub(handle=pathlib.Path('/dev/stdout').open('wt'), medium=medium)


# -----------------------------------------------------------------------------
# Accessors: getters and setters (pure functions)
# -----------------------------------------------------------------------------

# version
@with_guardrails
def GetVersion_nlms(medium: nlms) -> int:
    return medium.version

# Nz
@with_guardrails
def GetNz_nlms(medium: nlms) -> int:
    return int(medium.length / medium.dz) if medium.dz != 0 else 0

# Dz
@with_guardrails
def GetDz_nlms(medium: nlms) -> Annotated[float, np.float64]:
    return medium.dz

# Length
@with_guardrails
def GetLength_nlms(medium: nlms) -> Annotated[float, np.float64]:
    return medium.length

# N0
@with_guardrails
def GetN0_nlms(medium: nlms) -> Annotated[float, np.float64]:
    return medium.n0

# Ks
@with_guardrails
def GetK_nlms(medium: nlms) -> np.ndarray:
    return medium.k.copy()

# Individual K
for i in range(1,6):
    exec(
        f"@with_guardrails\ndef GetK{i}_nlms(medium: nlms) -> Annotated[float, np.float64]: return medium.k[{i-1}]"
    )


@with_guardrails
def GetN2I_nlms(medium: nlms) -> Annotated[float, np.float64]: return medium.n2
@with_guardrails
def GetAlpha_nlms(medium: nlms) -> Annotated[float, np.float64]: return medium.alpha
@with_guardrails
def GetBeta_nlms(medium: nlms) -> Annotated[float, np.float64]: return medium.beta
@with_guardrails
def GetMaterial_nlms(medium: nlms) -> str: return medium.material.strip()
@with_guardrails
def GetWavelength_nlms(medium: nlms) -> Annotated[float, np.float64]: return medium.wavelength
@with_guardrails
def GetTr_nlms(medium: nlms) -> Annotated[float, np.float64]: return medium.Tr
@with_guardrails
def GetN4I_nlms(medium: nlms) -> Annotated[float, np.float64]: return medium.n4
@with_guardrails
def GetPlasma(medium: nlms) -> plasma_coefficients: return medium.plasma_params.copy()

@with_guardrails
def GetKm(m: int, medium: nlms) -> Annotated[float, np.float64]:
    if m < 0 or m >= medium.k.size:
        raise ValueError("Invalid expansion coefficient for GetKm")
    return (2.0 * np.pi * medium.n0 / medium.wavelength) if m == 0 else medium.k[m]


# -----------------------------------------------------------------------------
# Mutators: setters (state-modifying)
# -----------------------------------------------------------------------------
@with_guardrails
def SetNz_nlms(medium: nlms, Nz: int) -> None:
    if medium.length == 0.0:
        raise ValueError("Set the material length before setting the number of steps.")
    medium.dz = dp(medium.length / Nz)

@with_guardrails
def SetDz_nlms(medium: nlms, Dz: float) -> None:
    medium.dz = dp(Dz)

@with_guardrails
def SetLength_nlms(medium: nlms, L: float) -> None:
    medium.length = dp(L)

@with_guardrails
def SetN0_nlms(medium: nlms, n0: float) -> None:
    medium.n0 = dp(n0)

for i in range(1, 6):
    exec(
        f"@with_guardrails\ndef SetK{i}_nlms(medium: nlms, val: float) -> None: medium.k[{i-1}] = dp(val)"
    )

@with_guardrails
def SetK_nlms(medium: nlms, k: Iterable[float]) -> None:
    arr = np.array(list(k), dtype=np.float64)
    if arr.size != 5:
        raise ValueError("Input array for SetK_nlms must have length 5")
    medium.k = arr

@with_guardrails
def SetN2I_nlms(medium: nlms, n2: float) -> None:
    medium.n2 = dp(n2)

@with_guardrails
def SetAlpha_nlms(medium: nlms, alpha: float) -> None:
    medium.alpha = dp(alpha)

@with_guardrails
def SetBeta_nlms(medium: nlms, beta: float) -> None:
    medium.beta = dp(beta)

@with_guardrails
def SetMaterial_nlms(medium: nlms, mat: str) -> None:
    medium.material = mat.strip()

@with_guardrails
def SetWavelength_nlms(medium: nlms, lam: float) -> None:
    medium.wavelength = dp(lam)

@with_guardrails
def SetTr_nlms(medium: nlms, Tr: float) -> None:
    medium.Tr = dp(Tr)

@with_guardrails
def SetN4I_nlms(medium: nlms, n4: float) -> None:
    medium.n4 = dp(n4)

@with_guardrails
def SetPlasma(medium: nlms, plasma: plasma_coefficients) -> None:
    medium.plasma_params = plasma.copy()
