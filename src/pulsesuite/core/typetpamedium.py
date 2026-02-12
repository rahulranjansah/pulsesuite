from __future__ import annotations

from pathlib import Path
from typing import Annotated, Union

import numpy as np
from guardrails.guardrails import with_guardrails
from typemedium import (
    GetAlpha_nlms,
    GetBeta_nlms,
    GetDz_nlms,
    GetK1_nlms,
    GetK2_nlms,
    GetLength_nlms,
    GetMaterial_nlms,
    GetN0_nlms,
    GetN2I_nlms,
    GetNz_nlms,
    GetWavelength_nlms,
    SetAlpha_nlms,
    SetBeta_nlms,
    SetDz_nlms,
    SetK1_nlms,
    SetK2_nlms,
    SetLength_nlms,
    SetMaterial_nlms,
    SetN0_nlms,
    SetN2I_nlms,
    SetNz_nlms,
    SetWavelength_nlms,
    nlms,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TPA_MAX_LEVELS: int = 30  # The maximum levels allowed in a two-photon absorber

dp = np.float64  # double-precision real (Matches `real(dp)` in Fortran)
filetag: tuple[str, str] = (
    "[params.tpa_v1.0]",
    "[params.tpa_v1.1]",
)


# -----------------------------------------------------------------------------
# Main data container: Python counterpart of Fortran's `type tpas`
# -----------------------------------------------------------------------------
class tpas:
    """
    Two-photon absorber parameters (tpas).

    Attributes:
        medium      : nlms                   # base medium parameters
        conc        : float (dp)            # molecule concentration (molecules/m^3)
        levels      : int                    # number of energy levels
        r           : np.ndarray[dp]         # relaxation constants (1/s)
        sigspa      : np.ndarray[dp]         # single-photon absorption cross-section (m^2)
        sigIspa     : np.ndarray[dp]         # intensity-dependent SPA cross-section (m^2)
        sigtpa      : np.ndarray[dp]         # two-photon absorption cross-section (m^4/W)
        sigItpa     : np.ndarray[dp]         # intensity-dependent TPA cross-section (m^4/W)
        tpa_material: str                    # name of the TPA material
    """
    __slots__ = (
        'medium', 'conc', 'levels', 'r', 'sigspa', 'sigIspa', 'sigtpa', 'sigItpa', 'tpa_material'
    )

    def __init__(
        self,
        medium: nlms | None = None,
        conc: Annotated[float, dp] = 0.0,
        levels: int = 0,
        r: np.ndarray | None = None,
        sigspa: np.ndarray | None = None,
        sigIspa: np.ndarray | None = None,
        sigtpa: np.ndarray | None = None,
        sigItpa: np.ndarray | None = None,
        tpa_material: str = ""
    ) -> None:
        self.medium = medium.copy() if medium else nlms()
        self.conc = dp(conc)
        self.levels = int(levels)
        shape = (TPA_MAX_LEVELS, TPA_MAX_LEVELS)
        self.r = np.zeros(shape, dtype=dp) if r is None else np.array(r, dtype=dp)
        self.sigspa = np.zeros(shape, dtype=dp) if sigspa is None else np.array(sigspa, dtype=dp)
        self.sigIspa = np.zeros((TPA_MAX_LEVELS,), dtype=dp) if sigIspa is None else np.array(sigIspa, dtype=dp)
        self.sigtpa = np.zeros(shape, dtype=dp) if sigtpa is None else np.array(sigtpa, dtype=dp)
        self.sigItpa = np.zeros((TPA_MAX_LEVELS,), dtype=dp) if sigItpa is None else np.array(sigItpa, dtype=dp)
        self.tpa_material = tpa_material

    def copy(self) -> tpas:
        """Return a shallow copy of all fields."""
        return tpas(
            medium=self.medium.copy(),
            conc=self.conc,
            levels=self.levels,
            r=self.r.copy(),
            sigspa=self.sigspa.copy(),
            sigIspa=self.sigIspa.copy(),
            sigtpa=self.sigtpa.copy(),
            sigItpa=self.sigItpa.copy(),
            tpa_material=self.tpa_material,
        )

    def __repr__(self) -> str:
        return (
            f"tpas(medium={self.medium!r}, conc={self.conc:.3e}, levels={self.levels}, "
            f"tpa_material={self.tpa_material!r})"
        )

    # ---------------------- Pythonic properties ------------------------------
    @property
    def length(self) -> float:
        return GetLength_tpas(self)

    @length.setter
    def length(self, L: float) -> None:
        SetLength_tpas(self, L)

    @property
    def nz(self) -> int:
        return GetNz_tpas(self)

    @nz.setter
    def nz(self, N: int) -> None:
        SetNz_tpas(self, N)

    @property
    def dz(self) -> float:
        return GetDz_tpas(self)

    @dz.setter
    def dz(self, Dz: float) -> None:
        SetDz_tpas(self, Dz)

    @property
    def n0(self) -> float:
        return GetN0_tpas(self)

    @n0.setter
    def n0(self, n0: float) -> None:
        SetN0_tpas(self, n0)

    @property
    def k2(self) -> float:
        return GetK2_tpas(self)

    @k2.setter
    def k2(self, k2: float) -> None:
        SetK2_tpas(self, k2)

    @property
    def k1(self) -> float:
        return GetK1_tpas(self)

    @k1.setter
    def k1(self, k1: float) -> None:
        SetK1_tpas(self, k1)

    @property
    def n2i(self) -> float:
        return GetN2I_tpas(self)

    @n2i.setter
    def n2i(self, n2: float) -> None:
        SetN2I_tpas(self, n2)

    @property
    def beta(self) -> float:
        return GetBeta_tpas(self)

    @beta.setter
    def beta(self, beta: float) -> None:
        SetBeta_tpas(self, beta)

    @property
    def alpha(self) -> float:
        return GetAlpha_tpas(self)

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        SetAlpha_tpas(self, alpha)

    @property
    def concentration(self) -> float:
        return self.conc

    @concentration.setter
    def concentration(self, conc: float) -> None:
        self.conc = dp(conc)

    @property
    def num_levels(self) -> int:
        return self.levels

    @num_levels.setter
    def num_levels(self, L: int) -> None:
        self.levels = L

    @property
    def relax(self) -> np.ndarray:
        return self.r[:self.levels, :self.levels].copy()

    @relax.setter
    def relax(self, arr: np.ndarray) -> None:
        self.r[:self.levels, :self.levels] = arr[:self.levels, :self.levels]

    @property
    def spa(self) -> np.ndarray:
        return self.sigspa[:self.levels, :self.levels].copy()

    @spa.setter
    def spa(self, arr: np.ndarray) -> None:
        self.sigspa[:self.levels, :self.levels] = arr[:self.levels, :self.levels]

    @property
    def spai(self) -> np.ndarray:
        return self.sigIspa[:self.levels].copy()

    @spai.setter
    def spai(self, arr: np.ndarray) -> None:
        self.sigIspa[:self.levels] = arr[:self.levels]

    @property
    def tpa(self) -> np.ndarray:
        return self.sigtpa[:self.levels, :self.levels].copy()

    @tpa.setter
    def tpa(self, arr: np.ndarray) -> None:
        self.sigtpa[:self.levels, :self.levels] = arr[:self.levels, :self.levels]

    @property
    def tpai(self) -> np.ndarray:
        return self.sigItpa[:self.levels].copy()

    @tpai.setter
    def tpai(self, arr: np.ndarray) -> None:
        self.sigItpa[:self.levels] = arr[:self.levels]

    @property
    def material(self) -> str:
        return f"{self.tpa_material.strip()}:{GetMaterial_nlms(self.medium)}"

    @material.setter
    def material(self, pair: str) -> None:
        tpa_mat, _, mat = pair.partition(":")
        self.tpa_material = tpa_mat
        SetMaterial_nlms(self.medium, mat)

    @property
    def wavelength(self) -> float:
        return GetWavelength_tpas(self)

    @wavelength.setter
    def wavelength(self, lam: float) -> None:
        SetWavelength_tpas(self, lam)

    @property
    def medium_obj(self) -> nlms:
        return self.medium.copy()

    @medium_obj.setter
    def medium_obj(self, m: nlms) -> None:
        self.medium = m.copy()


# -----------------------------------------------------------------------------
# Accessor functions (getters/setters) mirroring Fortran subroutines
# -----------------------------------------------------------------------------

@with_guardrails
def GetLength_tpas(tpa: tpas) -> Annotated[float, dp]:
    return GetLength_nlms(tpa.medium)

@with_guardrails
def SetLength_tpas(tpa: tpas, L: Annotated[float, dp]) -> None:
    SetLength_nlms(tpa.medium, L)

@with_guardrails
def GetNz_tpas(tpa: tpas) -> int:
    return GetNz_nlms(tpa.medium)

@with_guardrails
def SetNz_tpas(tpa: tpas, N: int) -> None:
    SetNz_nlms(tpa.medium, N)

@with_guardrails
def GetDz_tpas(tpa: tpas) -> Annotated[float, dp]:
    return GetDz_nlms(tpa.medium)

@with_guardrails
def SetDz_tpas(tpa: tpas, Dz: Annotated[float, dp]) -> None:
    SetDz_nlms(tpa.medium, Dz)

@with_guardrails
def GetN0_tpas(tpa: tpas) -> Annotated[float, dp]:
    return GetN0_nlms(tpa.medium)

@with_guardrails
def SetN0_tpas(tpa: tpas, n0: Annotated[float, dp]) -> None:
    SetN0_nlms(tpa.medium, n0)

@with_guardrails
def GetK2_tpas(tpa: tpas) -> Annotated[float, dp]:
    return GetK2_nlms(tpa.medium)

@with_guardrails
def SetK2_tpas(tpa: tpas, k2: Annotated[float, dp]) -> None:
    SetK2_nlms(tpa.medium, k2)

@with_guardrails
def GetK1_tpas(tpa: tpas) -> Annotated[float, dp]:
    return GetK1_nlms(tpa.medium)

@with_guardrails
def SetK1_tpas(tpa: tpas, k1: Annotated[float, dp]) -> None:
    SetK1_nlms(tpa.medium, k1)

@with_guardrails
def GetN2I_tpas(tpa: tpas) -> Annotated[float, dp]:
    return GetN2I_nlms(tpa.medium)

@with_guardrails
def SetN2I_tpas(tpa: tpas, n2: Annotated[float, dp]) -> None:
    SetN2I_nlms(tpa.medium, n2)

@with_guardrails
def GetBeta_tpas(tpa: tpas) -> Annotated[float, dp]:
    return GetBeta_nlms(tpa.medium)

@with_guardrails
def SetBeta_tpas(tpa: tpas, beta: Annotated[float, dp]) -> None:
    SetBeta_nlms(tpa.medium, beta)

@with_guardrails
def GetAlpha_tpas(tpa: tpas) -> Annotated[float, dp]:
    return GetAlpha_nlms(tpa.medium)

@with_guardrails
def SetAlpha_tpas(tpa: tpas, alpha: Annotated[float, dp]) -> None:
    SetAlpha_nlms(tpa.medium, alpha)

@with_guardrails
def GetConcentration_tpas(tpa: tpas) -> Annotated[float, dp]:
    return tpa.conc

@with_guardrails
def SetConcentration_tpas(tpa: tpas, conc: Annotated[float, dp]) -> None:
    tpa.conc = conc

@with_guardrails
def GetLevels_tpas(tpa: tpas) -> int:
    return tpa.levels

@with_guardrails
def SetLevels_tpas(tpa: tpas, levels: int) -> None:
    tpa.levels = levels

@with_guardrails
def GetRelax_tpas(tpa: tpas) -> np.ndarray:
    return tpa.r[:tpa.levels, :tpa.levels].copy()

@with_guardrails
def SetRelax_tpas(tpa: tpas, r: np.ndarray) -> None:
    tpa.r[:tpa.levels, :tpa.levels] = r[:tpa.levels, :tpa.levels]

@with_guardrails
def GetSigSpa_tpas(tpa: tpas) -> np.ndarray:
    return tpa.sigspa[:tpa.levels, :tpa.levels].copy()

@with_guardrails
def SetSigSpa_tpas(tpa: tpas, sig: np.ndarray) -> None:
    tpa.sigspa[:tpa.levels, :tpa.levels] = sig[:tpa.levels, :tpa.levels]

@with_guardrails
def GetSigSpaI_tpas(tpa: tpas) -> np.ndarray:
    return tpa.sigIspa[:tpa.levels].copy()

@with_guardrails
def SetSigSpaI_tpas(tpa: tpas, sig: np.ndarray) -> None:
    tpa.sigIspa[:tpa.levels] = sig[:tpa.levels]

@with_guardrails
def GetSigTpa_tpas(tpa: tpas) -> np.ndarray:
    return tpa.sigtpa[:tpa.levels, :tpa.levels].copy()

@with_guardrails
def SetSigTpa_tpas(tpa: tpas, sigtpa: np.ndarray) -> None:
    tpa.sigtpa[:tpa.levels, :tpa.levels] = sigtpa[:tpa.levels, :tpa.levels]

@with_guardrails
def GetSigTpaI_tpas(tpa: tpas) -> np.ndarray:
    return tpa.sigItpa[:tpa.levels].copy()

@with_guardrails
def SetSigTpaI_tpas(tpa: tpas, sig: np.ndarray) -> None:
    tpa.sigItpa[:tpa.levels] = sig[:tpa.levels]

@with_guardrails
def GetMaterial_tpas(tpa: tpas) -> str:
    base = GetMaterial_nlms(tpa.medium)
    return f"{tpa.tpa_material.strip()}:{base}"

@with_guardrails
def SetMaterial_tpas(tpa: tpas, mat: str, tpa_mat: str) -> None:
    SetMaterial_nlms(tpa.medium, mat)
    tpa.tpa_material = tpa_mat

@with_guardrails
def GetWavelength_tpas(tpa: tpas) -> Annotated[float, dp]:
    return GetWavelength_nlms(tpa.medium)

@with_guardrails
def SetWavelength_tpas(tpa: tpas, lam: Annotated[float, dp]) -> None:
    SetWavelength_nlms(tpa.medium, lam)

@with_guardrails
def GetMedium_tpas(tpa: tpas) -> nlms:
    return tpa.medium.copy()

@with_guardrails
def SetMedium_tpas(tpa: tpas, medium: nlms) -> None:
    tpa.medium = medium.copy()





# Doesnt talk with readmediumparams because its not being used change the file I/O of readmediumparams to match rb+
@with_guardrails
def writetpaparams(cmd: Union[str, Path], tpa: tpas) -> None:
    """
    Write all TPA parameters in a compact binary form:
      [conc, levels, r, sigspa, sigIspa, sigtpa, sigItpa]
    Arrays are dumped in C-order; only the first `levels`×`levels` (or `levels`) entries are stored.
    """
    n = tpa.levels
    with open(cmd, "wb") as f:
        # header
        np.array([tpa.conc], dtype=dp).tofile(f)
        np.array([n], dtype=np.int32).tofile(f)
        # matrices & vectors
        tpa.r[:n, :n].tofile(f)
        tpa.sigspa[:n, :n].tofile(f)
        tpa.sigIspa[:n].tofile(f)
        tpa.sigtpa[:n, :n].tofile(f)
        tpa.sigItpa[:n].tofile(f)

@with_guardrails
def readtpaparams(cmd: Union[str, Path], tpa: tpas) -> None:
    """
    Read back the binary format written by writetpaparams.
    Overwrites the fields in the given tpas in-place.
    """
    with open(cmd, "rb") as f:
        # header
        tpa.conc = np.fromfile(f, dtype=dp, count=1)[0]
        tpa.levels = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        n = tpa.levels
        # matrices & vectors
        tpa.r[:n, :n]      = np.fromfile(f, dtype=dp, count=n*n).reshape(n, n)
        tpa.sigspa[:n, :n] = np.fromfile(f, dtype=dp, count=n*n).reshape(n, n)
        tpa.sigIspa[:n]    = np.fromfile(f, dtype=dp, count=n)
        tpa.sigtpa[:n, :n] = np.fromfile(f, dtype=dp, count=n*n).reshape(n, n)
        tpa.sigItpa[:n]    = np.fromfile(f, dtype=dp, count=n)

def memmap_matrix(
    cmd: Union[str, Path],
    offset_bytes: int,
    shape: tuple[int, int]
) -> np.memmap:
    """
    Generic helper: memory-map a 2D float64 block out of a binary file.
    Example: after conc+levels header, r starts at byte offset = 8+4.
    """
    return np.memmap(cmd, dtype=dp, mode="r", offset=offset_bytes,
                     shape=shape, order="C")

# # -----------------------------------------------------------------------------
# # Pythonic File I/O routines with original submodule names (for writing txt?)
# # -----------------------------------------------------------------------------
# @with_guardrails
# def readtpaparams(cmd: Union[str, pathlib.Path], tpa: tpas) -> None:
#     """
#     Reads tpa parameters from file.
#     """
#     with open(cmd, 'r') as f:
#         # zero arrays
#         tpa.r.fill(0); tpa.sigspa.fill(0)
#         tpa.sigIspa.fill(0); tpa.sigtpa.fill(0)
#         tpa.sigItpa.fill(0)
#         # read base medium
#         _readmediumparams(f, tpa.medium)
#         header = f.readline().strip()
#         if header == filetag[1]:
#             readtpaparams_sub_v1_1(f, tpa)
#         elif header == filetag[0]:
#             readtpaparams_sub_v1_0(f, tpa)
#         else:
#             readtpaparams_sub_v0_0(f, tpa, header)

# def readtpaparams_sub_v1_1(f: TextIO, tpa: tpas) -> None:
#     tpa.tpa_material = f.readline().strip()[:25]
#     tpa.conc = dp(float(f.readline().split()[0]))
#     tpa.levels = int(float(f.readline().split()[0]))
#     for i in range(tpa.levels):
#         tpa.r[i, :tpa.levels] = np.fromstring(f.readline(), sep=' ')
#     for i in range(tpa.levels):
#         tpa.sigspa[i, :tpa.levels] = np.fromstring(f.readline(), sep=' ')
#     tpa.sigIspa[:tpa.levels] = np.fromstring(f.readline(), sep=' ')
#     for i in range(tpa.levels):
#         tpa.sigtpa[i, :tpa.levels] = np.fromstring(f.readline(), sep=' ')
#     tpa.sigItpa[:tpa.levels] = np.fromstring(f.readline(), sep=' ')


# def readtpaparams_sub_v1_0(f: TextIO, tpa: tpas) -> None:
#     # legacy v1.0: first scalar is K1
#     k1 = float(f.readline().split()[0])
#     SetK1_tpas(tpa, dp(k1))
#     readtpaparams_sub_v1_1(f, tpa)


# def readtpaparams_sub_v0_0(f: TextIO, tpa: tpas, firstline: str) -> None:
#     k1 = float(firstline.split()[0])
#     SetK1_tpas(tpa, dp(k1))
#     readtpaparams_sub_v1_0(f, tpa)

# @with_guardrails
# def writetpaparams(cmd: Union[str, pathlib.Path], tpa: tpas) -> None:
#     """
#     Writes tpa parameters to file.
#     """
#     with open(cmd, 'w') as f:
#         writetpaparams_sub(f, tpa)


# def writetpaparams_sub(f: TextIO, tpa: tpas) -> None:
#     # write base medium
#     _writemediumparams(f, tpa.medium)
#     # header
#     f.write(f"{filetag[1]}\n")
#     f.write(f"{tpa.tpa_material:<25} : TPA Material.\n")
#     f.write(f"{tpa.conc:{pfrmtA}} : concentration (molecules/m^3)\n")
#     f.write(f"{tpa.levels:25d} : number of energy levels.\n")
#     # relaxation rates
#     for i in range(tpa.levels):
#         f.write(' '.join(f"{v:{pfrmtA}}" for v in tpa.r[i, :tpa.levels]) + " : Relaxation rates (1/s)\n")
#     # single-photon absorption
#     for i in range(tpa.levels):
#         f.write(' '.join(f"{v:{pfrmtA}}" for v in tpa.sigspa[i, :tpa.levels]) + " : SPA cross section (m^2)\n")
#     f.write(' '.join(f"{v:{pfrmtA}}" for v in tpa.sigIspa[:tpa.levels]) + " : SPA intensity cross section (m^2)\n")
#     # two-photon absorption
#     for i in range(tpa.levels):
#         f.write(' '.join(f"{v:{pfrmtA}}" for v in tpa.sigtpa[i, :tpa.levels]) + " : TPA cross section (m^4/W)\n")
#     f.write(' '.join(f"{v:{pfrmtA}}" for v in tpa.sigItpa[:tpa.levels]) + " : TPA intensity cross section (m^4/W)\n")

# @with_guardrails
# def dumptpa(tpa: tpas, level: int | None = None) -> None:
#     """Dump parameters to stdout for debugging."""
#     writetpaparams_sub(sys.stdout, tpa)

# End of module



# # -----------------------------------------------------------------------------
# # File I/O routines and submodules ported from Fortran
# # -----------------------------------------------------------------------------
# @with_guardrails
# def readtpaparams(cmd: Union[str, pathlib.Path], tpa: tpas) -> None:
#     U = open_file_unit(cmd, 'r')
#     # zero arrays
#     tpa.r.fill(0); tpa.sigspa.fill(0); tpa.sigIspa.fill(0)
#     tpa.sigtpa.fill(0); tpa.sigItpa.fill(0)
#     # read base medium
#     _readmediumparams(U, tpa.medium)
#     header = get_next_line(U).strip()
#     if header == filetag[1]:
#         readtpaparams_sub_v1_1(U, tpa)
#     elif header == filetag[0]:
#         readtpaparams_sub_v1_0(U, tpa)
#     else:
#         readtpaparams_sub_v0_0(U, tpa, header)
#     U.close()

# @with_guardrails
# def readtpaparams_sub_v1_1(handle: TextIO, tpa: tpas) -> None:
#     tpa.tpa_material = get_next_line(handle).strip()[:25]
#     tpa.conc = dp(GetFileParam(handle))
#     tpa.levels = int(GetFileParam(handle))
#     # relaxation matrix
#     for i in range(tpa.levels):
#         tpa.r[i, :tpa.levels] = np.fromstring(get_next_line(handle), sep=' ')
#     # SPA cross-sections
#     for i in range(tpa.levels):
#         tpa.sigspa[i, :tpa.levels] = np.fromstring(get_next_line(handle), sep=' ')
#     # SPA intensity
#     tpa.sigIspa[:tpa.levels] = np.fromstring(get_next_line(handle), sep=' ')
#     # TPA cross-sections
#     for i in range(tpa.levels):
#         tpa.sigtpa[i, :tpa.levels] = np.fromstring(get_next_line(handle), sep=' ')
#     # TPA intensity
#     tpa.sigItpa[:tpa.levels] = np.fromstring(get_next_line(handle), sep=' ')

# @with_guardrails
# def readtpaparams_sub_v1_0(handle: TextIO, tpa: tpas) -> None:
#     # legacy v1.0: first scalar is K1
#     SetK1_tpas(tpa, GetFileParam(handle))
#     # then same block as v1_1
#     readtpaparams_sub_v1_1(handle, tpa)

# @with_guardrails
# def readtpaparams_sub_v0_0(handle: TextIO, tpa: tpas, firstline: str) -> None:
#     # firstline contains K1
#     k1_val = dp(firstline.split()[0])
#     SetK1_tpas(tpa, k1_val)
#     # then delegate to v1_0 reader
#     readtpaparams_sub_v1_0(handle, tpa)

# @with_guardrails
# def writetpaparams(cmd: Union[str, pathlib.Path], tpa: tpas) -> None:
#     U = open_file_unit(cmd, 'w')
#     writetpaparams_sub(U, tpa)
#     U.close()

# @with_guardrails
# def writetpaparams_sub(handle: TextIO, tpa: tpas) -> None:
#     # write medium
#     _writemediumparams(handle, tpa.medium)
#     # header
#     handle.write(f"{filetag[1]}\n")
#     handle.write(f"{tpa.tpa_material:<25} : TPA Material.\n")
#     handle.write(f"{tpa.conc:{pfrmtA}} : concentration (molecules/m^3)\n")
#     handle.write(f"{tpa.levels:25d} : number of energy levels.\n")
#     # matrices and arrays
#     for name, arr, desc in [
#         ('relax', tpa.r, 'Relaxation rates (1/s)'),
#         ('spa', tpa.sigspa, 'SPA cross section (m^2)'),
#         ('tp a', tpa.sigtpa, 'TPA cross section (m^4/W)')]:
#         for i in range(tpa.levels):
#             vals = arr[i, :tpa.levels]
#             handle.write(' '.join(f"{v:{pfrmtA}}" for v in vals) + f" : {desc}\n")
#     handle.write(' '.join(f"{v:{pfrmtA}}" for v in tpa.sigIspa[:tpa.levels]) + " : SPA intensity\n")
#     handle.write(' '.join(f"{v:{pfrmtA}}" for v in tpa.sigItpa[:tpa.levels]) + " : TPA intensity\n")

# @with_guardrails
# def dumptpa(tpa: tpas, level: int | None = None) -> None:
#     """Dump parameters to stdout for debugging."""
#     import sys
#     writetpaparams_sub(sys.stdout, tpa)

# # End of module

