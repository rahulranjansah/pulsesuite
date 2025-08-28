# typetime.py — port of FORTRAN module `typetime`

from __future__ import annotations
import sys
import numpy as np
from numpy.typing import NDArray

F64 = np.float64
C128 = np.complex128
TWOPI = F64(2.0 * np.pi)

# -----------------------------------------------------------------------------
# Fortran type ts → Python class ts
# -----------------------------------------------------------------------------
class ts:
    __slots__ = ("t", "tf", "dt", "n")  # current time, final time, step, index
    def __init__(self, t: float, tf: float, dt: float, n: int):
        self.t  = F64(t)
        self.tf = F64(tf)
        self.dt = F64(dt)
        self.n  = int(n)

# -----------------------------------------------------------------------------
# Parameter I/O (text-mode parity with readtimeparams_sub/writetimeparams_sub)
# -----------------------------------------------------------------------------
def readtimeparams_sub(u, time: ts) -> None:
    """Read 4 lines; first token per line is value."""
    vals: list[str] = []
    for _ in range(4):
        line = u.readline()
        if not line:
            raise EOFError("readtimeparams_sub: unexpected EOF")
        vals.append(line.strip().split()[0])
    time.t  = F64(vals[0])
    time.tf = F64(vals[1])
    time.dt = F64(vals[2])
    time.n  = int(vals[3])

def ReadTimeParams(cmd: str, time: ts) -> None:
    with open(cmd, "r") as fh:
        readtimeparams_sub(fh, time)
    dumptime(time)

def WriteTimeParams_sub(u, time: ts) -> None:
    u.write(f"{time.t : .16e} : Current time of simulation. (s)\n")
    u.write(f"{time.tf: .16e} : Final time of simulation. (s)\n")
    u.write(f"{time.dt: .16e} : Time pixel size [dt]. (s)\n")
    u.write(f"{time.n:25d} : Current time index.\n")

def writetimeparams(cmd: str, time: ts) -> None:
    with open(cmd, "w") as fh:
        WriteTimeParams_sub(fh, time)

def dumptime(params: ts, level: int | None = None) -> None:
    # Minimal logger parity
    WriteTimeParams_sub(sys.stdout, params)

# -----------------------------------------------------------------------------
# Getters / setters (name parity)
# -----------------------------------------------------------------------------
def GetT(time: ts)  -> F64: return time.t
def GetTf(time: ts) -> F64: return time.tf
def GetDt(time: ts) -> F64: return time.dt
def GetN(time: ts)  -> int: return time.n

def SetT(time: ts, t: float)   -> None: time.t  = F64(t)
def SetTf(time: ts, tf: float) -> None: time.tf = F64(tf)
def SetDt(time: ts, dt: float) -> None: time.dt = F64(dt)
def SetN(time: ts, n: int)     -> None: time.n  = int(n)

# -----------------------------------------------------------------------------
# Counts / updates
# -----------------------------------------------------------------------------
def CalcNt(time: ts) -> int:
    """Remaining steps: int((tf - t)/dt)."""
    return int((time.tf - time.t) / time.dt)

def UpdateT(time: ts, dt: float) -> None:
    time.t = F64(time.t + F64(dt))

def UpdateN(time: ts, dn: int) -> None:
    time.n = int(time.n + int(dn))

# -----------------------------------------------------------------------------
# Arrays & spectra
# -----------------------------------------------------------------------------
def GetTArray(time: ts) -> NDArray[F64]:
    Nt = CalcNt(time)
    if Nt <= 1:
        return np.asfortranarray(np.array([0.0], dtype=F64))
    t = F64(time.t) + F64(time.dt) * np.arange(Nt, dtype=F64)
    return np.asfortranarray(t)

def GetOmegaArray(time: ts) -> NDArray[F64]:
    Nt = CalcNt(time)
    Tw = F64(Nt) * F64(time.dt)
    if Nt <= 0:
        return np.asfortranarray(np.zeros(0, dtype=F64))
    w = np.empty(Nt, dtype=F64, order="F")
    half = Nt // 2
    # Fortran 1-based logic kept: if n <= Nt/2 then 2π*(n-1)/Tw else 2π*(n-Nt-1)/Tw
    for n in range(1, Nt + 1):
        w[n-1] = TWOPI * (F64(n - 1) / Tw if n <= half else F64(n - Nt - 1) / Tw)
    return w

def GetdOmega(time: ts) -> F64:
    Nt = CalcNt(time)
    return F64(0.0) if Nt == 0 else TWOPI / (F64(Nt) * F64(time.dt))

# -----------------------------------------------------------------------------
# Field init helper (parity stub)
# -----------------------------------------------------------------------------
def initialize_field(e: NDArray[C128]) -> None:
    e[...] = 0.0 + 0.0j
