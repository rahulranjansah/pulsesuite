"""
Port of Fortran module `type_plasma` ➜ Python
===========================================
High‑performance, HPC‑ready rewrite that preserves the original Fortran naming
conventions one‑for‑one while exploiting Python/NumPy ergonomics.

*   Structure `plasma_coefficients` → class ``plasma_coefficients`` (slots for
    cache‑friendliness; dtype preserved as float64/int64).
*   Getter / setter routines exported with the **exact** original names
    (``GetMass``, ``SetMass`` …) so downstream code can transition painlessly.
*   Human‑readable parameter files (same layout as the Fortran writer) handled
    by ``readplasmaparams_sub`` / ``writeplasmaparams_sub``.
*   Rigorous runtime dtype checking via ``@with_guardrails`` decorator.
*   Ready for further acceleration (e.g. Numba) if bulk processing of plasma
    parameter arrays becomes a bottleneck.

Assumptions
-----------
*   All project‑wide constants live in ``constants.py`` (``dp = np.float64``).
*   A lightweight ``logger`` module provides ``Logger.getInstance().error``.
*   No Fortran‑style unit numbers – we just pass an already‑opened `TextIO` or
    a *path* (the helper takes care of opening).
"""
from __future__ import annotations

import pathlib
import re
from typing import Annotated, Iterable, TextIO, Union

import numpy as np
from guardrails.guardrails import with_guardrails

# from logger import Logger  # project‑wide tiny singleton

# -----------------------------------------------------------------------------
# Project‑wide aliases (mimic Fortran kinds)
# -----------------------------------------------------------------------------
dp = np.float64  # double‑precision real (Matches `real(dp)` in Fortran)

# -----------------------------------------------------------------------------
# Version tag(s) (extend the tuple when you add a new writer)
# -----------------------------------------------------------------------------
plasma_filetag: tuple[str, ...] = ("[params.plasma_v1.0]",)


# -----------------------------------------------------------------------------
# Helper – robust line‑by‑line numeric parser (float or int)  ------------------
# -----------------------------------------------------------------------------
_num_re = re.compile(r"^[^#]*?([-+]?\d*(?:\.\d+)?(?:[eE][-+]?\d+)?).*")

def _parse_number(line: str, want_float: bool = True):
    """Extract the first numeric token from *line* – raise on failure."""
    m = _num_re.match(line.strip())
    if not m or m.group(1) == "":
        raise ValueError(f"cannot parse numeric value from line: {line!r}")
    return (dp if want_float else int)(m.group(1))


def _fp(lines: Iterable[str]):
    """Yield successive *non‑blank* and non-comment lines (skip comments and blanks)."""
    for ln in lines:
        s = ln.strip()
        if s and not s.startswith('#'):
            yield s


# -----------------------------------------------------------------------------
# Main data container  ---------------------------------------------------------
# -----------------------------------------------------------------------------
class plasma_coefficients:
    """Direct Python counterpart of Fortran's `type plasma_coefficients`."""

    __slots__ = (
        "version",
        "mass",
        "band_gap",
        "trap_time",
        "collision_time",
        "max_density",
        "order",
        "sigma_multi",
    )

    # --------------------------- construction --------------------------------
    def __init__(
        self,
        version: int = -1,
        mass: Annotated[float, np.float64] = 0.0,
        band_gap: Annotated[float, np.float64] = 0.0,
        trap_time: Annotated[float, np.float64] = 0.0,
        collision_time: Annotated[float, np.float64] = 0.0,
        max_density: Annotated[float, np.float64] = 0.0,
        order: int = 0,
        sigma_multi: Annotated[float, np.float64] = 0.0,
    ) -> None:
        self.version = int(version)
        self.mass = dp(mass)
        self.band_gap = dp(band_gap)
        self.trap_time = dp(trap_time)
        self.collision_time = dp(collision_time)
        self.max_density = dp(max_density)
        self.order = int(order)
        self.sigma_multi = dp(sigma_multi)

    # --------------------------- convenience ---------------------------------
    def copy(self) -> "plasma_coefficients":
        """Return a *shallow* copy (all fields are immutable scalars)."""
        return plasma_coefficients(
            self.version,
            self.mass,
            self.band_gap,
            self.trap_time,
            self.collision_time,
            self.max_density,
            self.order,
            self.sigma_multi,
        )

    # --------------------------- dunder helpers ------------------------------
    def __repr__(self) -> str:  # pragma: no cover  (debug helper)
        return (
            "plasma_coefficients("  # noqa: E501
            f"version={self.version}, mass={self.mass:.3e}, band_gap={self.band_gap:.3e}, "
            f"trap_time={self.trap_time:.3e}, collision_time={self.collision_time:.3e}, "
            f"max_density={self.max_density:.3e}, order={self.order}, sigma_multi={self.sigma_multi:.3e})"
        )

    def __eq__(self, other):
        if not isinstance(other, plasma_coefficients):
            return NotImplemented
        return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)


# -----------------------------------------------------------------------------
# I/O helpers – preserve Fortran file layout for round‑trip fidelity (Could've used python error handling instead of logger)
# -----------------------------------------------------------------------------
@with_guardrails
def readplasmaparams_sub(
    U: Union[str, pathlib.Path, TextIO], plasma: plasma_coefficients
) -> None:
    """Populate *plasma* by parsing an existing parameter file/stream."""

    def _inner(handle: TextIO):
        first = next(_fp(handle), None)
        if first is None:
            raise EOFError("empty plasma parameter stream")

        if first not in plasma_filetag:
            # Logger.getInstance().error(
            #     f"Unrecognized plasma section version: {first}", file=__file__, line=__import__('inspect').currentframe().f_lineno
            # )
            raise ValueError(f"Unrecognized plasma section version: {first}")
        else:
            plasma.version = 10  # v1.0 → BCD 10, keep parity with Fortran
            # ordered extraction, mirroring the original Fortran reader
            plasma.mass = _parse_number(next(_fp(handle)))
            plasma.band_gap = _parse_number(next(_fp(handle)))
            plasma.trap_time = _parse_number(next(_fp(handle)))
            plasma.collision_time = _parse_number(next(_fp(handle)))
            plasma.max_density = _parse_number(next(_fp(handle)))
            plasma.order = int(_parse_number(next(_fp(handle)), want_float=False))
            plasma.sigma_multi = _parse_number(next(_fp(handle)))

    if isinstance(U, (str, pathlib.Path)):
        with open(U, "rt", encoding="utf-8") as fh:
            _inner(fh)
    else:  # assume TextIO
        _inner(U)


@with_guardrails
def writeplasmaparams_sub(
    U: Union[str, pathlib.Path, TextIO], plasma: plasma_coefficients
) -> None:
    """Write *plasma* in the canonical Fortran text format."""

    def _inner(handle: TextIO):
        handle.write(f"{plasma_filetag[-1]}\n")
        handle.write(f"{plasma.mass :.15e} : Effective electron mass (kg)\n")
        handle.write(f"{plasma.band_gap :.15e} : Material band gap (J)\n")
        handle.write(f"{plasma.trap_time :.15e} : Free electron trapping time (s)\n")
        handle.write(f"{plasma.collision_time :.15e} : Free electron collision time (s)\n")
        handle.write(f"{plasma.max_density :.15e} : Maximum plasma density (1/m^3)\n")
        handle.write(f"{plasma.order :d} : Multi-photon ionization order\n")
        handle.write(f"{plasma.sigma_multi :.15e} : Multi-photon ionization cross-section (1/m^2)\n")

    if isinstance(U, (str, pathlib.Path)):
        with open(U, "wt", encoding="utf-8") as fh:
            _inner(fh)
    else:
        _inner(U)


# -----------------------------------------------------------------------------
# Pure getters / setters (mirrored 1‑to‑1 from Fortran)  -----------------------
# -----------------------------------------------------------------------------
@with_guardrails
def GetVersion_plasma(plasma: plasma_coefficients) -> int:
    return plasma.version


@with_guardrails
def GetMass(plasma: plasma_coefficients) -> Annotated[float, np.float64]:
    return plasma.mass


@with_guardrails
def SetMass(
    plasma: plasma_coefficients, mass: Annotated[float, np.float64]
) -> None:
    plasma.mass = dp(mass)


@with_guardrails
def GetBandGap(plasma: plasma_coefficients) -> Annotated[float, np.float64]:
    return plasma.band_gap


@with_guardrails
def SetBandGap(
    plasma: plasma_coefficients, gap: Annotated[float, np.float64]
) -> None:
    plasma.band_gap = dp(gap)


@with_guardrails
def GetTrapTime(plasma: plasma_coefficients) -> Annotated[float, np.float64]:
    return plasma.trap_time


@with_guardrails
def SetTrapTime(
    plasma: plasma_coefficients, time: Annotated[float, np.float64]
) -> None:
    plasma.trap_time = dp(time)


@with_guardrails
def GetCollisionTime(plasma: plasma_coefficients) -> Annotated[float, np.float64]:
    return plasma.collision_time


@with_guardrails
def SetCollisionTime(
    plasma: plasma_coefficients, time: Annotated[float, np.float64]
) -> None:
    plasma.collision_time = dp(time)


@with_guardrails
def GetMaxDensity(plasma: plasma_coefficients) -> Annotated[float, np.float64]:
    return plasma.max_density


@with_guardrails
def SetMaxDensity(
    plasma: plasma_coefficients, density: Annotated[float, np.float64]
) -> None:
    plasma.max_density = dp(density)


@with_guardrails
def GetOrder(plasma: plasma_coefficients) -> int:
    return plasma.order


@with_guardrails
def SetOrder(plasma: plasma_coefficients, order: int) -> None:
    plasma.order = int(order)


@with_guardrails
def GetCrossSection(plasma: plasma_coefficients) -> Annotated[float, np.float64]:
    return plasma.sigma_multi


@with_guardrails
def SetCrossSection(
    plasma: plasma_coefficients, sigma_multi: Annotated[float, np.float64]
) -> None:
    plasma.sigma_multi = dp(sigma_multi)


# -----------------------------------------------------------------------------
# Re‑export list (makes * from module safe & Fortran‑familiar)
# -----------------------------------------------------------------------------
__all__ = [
    "plasma_coefficients",
    "plasma_filetag",
    # I/O helpers
    "readplasmaparams_sub",
    "writeplasmaparams_sub",
    # getters / setters (keep Fortran names!)
    "GetVersion_plasma",
    "GetMass",
    "SetMass",
    "GetBandGap",
    "SetBandGap",
    "GetTrapTime",
    "SetTrapTime",
    "GetCollisionTime",
    "SetCollisionTime",
    "GetMaxDensity",
    "SetMaxDensity",
    "GetOrder",
    "SetOrder",
    "GetCrossSection",
    "SetCrossSection",
]
