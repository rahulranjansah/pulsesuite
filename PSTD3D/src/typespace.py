# typespace.py — port of FORTRAN module `typespace`
from __future__ import annotations
import sys
import numpy as np
from numpy.typing import NDArray

F64 = np.float64                      
C128 = np.complex128                  
twopi = F64(2.0 * np.pi)                                  


class ss:
    __slots__ = ("Dims", "Nx", "Ny", "Nz", "dx", "dy", "dz", "epsr")
    def __init__(self, Dims: int, Nx: int, Ny: int, Nz: int,
                 dx: float, dy: float, dz: float, epsr: float):
        self.Dims = int(Dims)
        self.Nx   = int(Nx)
        self.Ny   = int(Ny)
        self.Nz   = int(Nz)
        self.dx   = F64(dx)
        self.dy   = F64(dy)
        self.dz   = F64(dz)
        self.epsr = F64(epsr)

# -----------------------------------------------------------------------------
# I/O for the space parameters (parity with readspaceparams_sub/writespaceparams_sub)
# -----------------------------------------------------------------------------

def readspaceparams_sub(u, space: ss) -> None:
    """Read 8 lines; take first token on each line as the value."""
    vals: list[str] = []
    for _ in range(8):
        line = u.readline()
        if not line:
            raise EOFError("readspaceparams_sub: unexpected EOF")
        vals.append(line.strip().split()[0])
    space.Dims = int(vals[0])
    space.Nx   = int(vals[1])
    space.Ny   = int(vals[2])
    space.Nz   = int(vals[3])
    space.dx   = F64(vals[4])
    space.dy   = F64(vals[5])
    space.dz   = F64(vals[6])
    space.epsr = F64(vals[7])

def ReadSpaceParams(cmd: str, space: ss) -> None:
    with open(cmd, "r") as fh:
        readspaceparams_sub(fh, space)
    dumpspace(space)

def writespaceparams_sub(u, space: ss) -> None:
    u.write(f"{space.Dims:25d} : Number of Dimensions\n")
    u.write(f"{space.Nx:25d} : Number of space X points.\n")
    u.write(f"{space.Ny:25d} : Number of space Y points.\n")
    u.write(f"{space.Nz:25d} : Number of space Z points.\n")
    u.write(f"{space.dx: .16e} : The Width of the X pixel. (m)\n")
    u.write(f"{space.dy: .16e} : The Width of the Y pixel. (m)\n")
    u.write(f"{space.dz: .16e} : The Width of the Z pixel. (m)\n")
    u.write(f"{space.epsr: .16e} : The relative background dielectric constant\n")

def writespaceparams(cmd: str, space: ss) -> None:
    with open(cmd, "w") as fh:
        writespaceparams_sub(fh, space)

def dumpspace(params: ss, level: int | None = None) -> None:
    # Minimal parity: print to stdout when called
    writespaceparams_sub(sys.stdout, params)

# -----------------------------------------------------------------------------
# Getters / setters (name parity)
# -----------------------------------------------------------------------------

def GetNx(space: ss) -> int: return space.Nx
def GetNy(space: ss) -> int: return space.Ny
def GetNz(space: ss) -> int: return space.Nz

def GetDx(space: ss) -> F64: return F64(1.0) if space.Nx == 1 else space.dx
def GetDy(space: ss) -> F64: return F64(1.0) if space.Ny == 1 else space.dy
def GetDz(space: ss) -> F64: return F64(1.0) if space.Nz == 1 else space.dz

def GetEpsr(space: ss) -> F64: return space.epsr

def SetNx(space: ss, N: int) -> None: space.Nx = int(N)
def SetNy(space: ss, N: int) -> None: space.Ny = int(N)
def SetNz(space: ss, N: int) -> None: space.Nz = int(N)
def SetDx(space: ss, dl: float) -> None: space.dx = F64(dl)
def SetDy(space: ss, dl: float) -> None: space.dy = F64(dl)
def SetDz(space: ss, dl: float) -> None: space.dz = F64(dl)

# -----------------------------------------------------------------------------
# Window widths
# -----------------------------------------------------------------------------

def GetXWidth(space: ss) -> F64: return F64(space.dx * (space.Nx - 1))
def GetYWidth(space: ss) -> F64: return F64(space.dy * (space.Ny - 1))
def GetZWidth(space: ss) -> F64: return F64(space.dz * (space.Nz - 1))

# -----------------------------------------------------------------------------
# Coordinate arrays (space & conjugate k-space)
# -----------------------------------------------------------------------------

def _GetSpaceArray(N: int, width: float) -> NDArray[F64]:
    if N <= 1: return np.asfortranarray(np.zeros(1, dtype=F64))
    dx = width / F64(N - 1)
    return np.asfortranarray(np.arange(N, dtype=F64) * dx)

def _GetKArray(N: int, width: float) -> NDArray[F64]:
    if N <= 1: return np.asfortranarray(np.zeros(1, dtype=F64))
    # FFT convention: k = 2π * freq, with dx = width/(N-1)
    dx = width / F64(N - 1)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    return np.asfortranarray(k.astype(F64, copy=False))

def GetXArray(space: ss) -> NDArray[F64]:
    return np.asfortranarray(np.zeros(1, dtype=F64)) if space.Nx == 1 else _GetSpaceArray(space.Nx, GetXWidth(space))

def GetYArray(space: ss) -> NDArray[F64]:
    return np.asfortranarray(np.zeros(1, dtype=F64)) if space.Ny == 1 else _GetSpaceArray(space.Ny, GetYWidth(space))

def GetZArray(space: ss) -> NDArray[F64]:
    return np.asfortranarray(np.zeros(1, dtype=F64)) if space.Nz == 1 else _GetSpaceArray(space.Nz, GetZWidth(space))

def GetKxArray(space: ss) -> NDArray[F64]:
    return np.asfortranarray(np.zeros(1, dtype=F64)) if space.Nx == 1 else _GetKArray(GetNx(space), GetXWidth(space))

def GetKyArray(space: ss) -> NDArray[F64]:
    return np.asfortranarray(np.zeros(1, dtype=F64)) if space.Ny == 1 else _GetKArray(GetNy(space), GetYWidth(space))

def GetKzArray(space: ss) -> NDArray[F64]:
    return np.asfortranarray(np.zeros(1, dtype=F64)) if space.Nz == 1 else _GetKArray(GetNz(space), GetZWidth(space))

# -----------------------------------------------------------------------------
# Spectral increments & volume elements
# -----------------------------------------------------------------------------

def GetDQx(space: ss) -> F64: return twopi / GetXWidth(space)
def GetDQy(space: ss) -> F64: return twopi / GetYWidth(space)
def GetDQz(space: ss) -> F64: return twopi / GetZWidth(space)

def GetDVol(space: ss) -> F64:  return GetDx(space) * GetDy(space) * GetDz(space)
def GetDQVol(space: ss) -> F64: return GetDQx(space) * GetDQy(space) * GetDQz(space)

# -----------------------------------------------------------------------------
# Field read/write (formatted and simple binary). Name parity kept.
# -----------------------------------------------------------------------------

def writefield(fnout: str,
               e: NDArray[C128],
               space: ss,
               binmode: bool,
               single: bool,
               fnspace: str | None = None) -> None:
    """
    If single: write space+field to same file.
      - text: decorated space lines; then 're im' per value (i, j, k nested)
      - binary: int64[4] (Dims,Nx,Ny,Nz) + float64[4] (dx,dy,dz,epsr) + raw complex128
    Else:
      - write space to fnspace (text)
      - write field to fnout (text or raw complex128)
    """
    e = np.asfortranarray(e, dtype=C128)
    if single:
        if binmode:
            with open(fnout, "wb") as f:
                np.array([space.Dims, space.Nx, space.Ny, space.Nz], dtype=np.int64).tofile(f)
                np.array([space.dx, space.dy, space.dz, space.epsr], dtype=F64).tofile(f)
                e.tofile(f)
        else:
            with open(fnout, "w") as f:
                writespaceparams_sub(f, space)
                writefield_to_unit(f, e, False)
    else:
        if fnspace is not None:
            writespaceparams(fnspace, space)
        if binmode:
            e.tofile(fnout)
        else:
            with open(fnout, "w") as f:
                writefield_to_unit(f, e, False)

def readspace_only(fnin: str,
                   space: ss,
                   binmode: bool,
                   single: bool,
                   fnspace: str | None = None) -> None:
    if single:
        if binmode:
            with open(fnin, "rb") as f:
                hdr_i = np.fromfile(f, dtype=np.int64, count=4)
                hdr_f = np.fromfile(f, dtype=F64,     count=4)
            space.Dims, space.Nx, space.Ny, space.Nz = map(int, hdr_i)
            space.dx, space.dy, space.dz, space.epsr = map(F64, hdr_f)
        else:
            with open(fnin, "r") as f:
                readspaceparams_sub(f, space)
    else:
        if fnspace is None:
            raise ValueError("readspace_only: fnspace required when single=False")
        ReadSpaceParams(fnspace, space)

def readfield(fnin: str,
              e: NDArray[C128] | None,
              space: ss,
              binmode: bool,
              single: bool,
              fnspace: str | None = None) -> NDArray[C128]:
    # read/resolve space first
    readspace_only(fnin, space, binmode, single, fnspace)

    shape = (GetNx(space), GetNy(space), GetNz(space))
    if e is None or e.shape != shape or e.dtype != C128 or not np.isfortran(e):
        e = np.zeros(shape, dtype=C128, order="F")
        initialize_field(e)

    if single and binmode:
        with open(fnin, "rb") as f:
            _ = np.fromfile(f, dtype=np.int64, count=4)
            _ = np.fromfile(f, dtype=F64,     count=4)
            data = np.fromfile(f, dtype=C128, count=e.size)
        e[...] = data.reshape(shape, order="F")
    elif binmode and not single:
        data = np.fromfile(fnin, dtype=C128, count=e.size)
        e[...] = data.reshape(shape, order="F")
    else:
        with open(fnin, "r") as f:
            readfield_from_unit(f, e, False)
    return e

# -----------------------------------------------------------------------------
# Unit-based text I/O helpers (formatted)
# -----------------------------------------------------------------------------

def readfield_from_unit(u, e: NDArray[C128], binmode: bool) -> None:
    if binmode:
        raise NotImplementedError("Binary unit read handled in readfield().")
    it = np.nditer(e, flags=["multi_index", "refs_ok"], op_flags=["readwrite"])
    for _ in it:
        line = u.readline()
        if not line:
            raise EOFError("readfield_from_unit: unexpected EOF")
        parts = line.split()
        if len(parts) < 2:
            raise ValueError("Expected 're im' per line")
        re = F64(parts[0]); im = F64(parts[1])
        e[it.multi_index] = re + 1j * im

def writefield_to_unit(u, e: NDArray[C128], binmode: bool) -> None:
    if binmode:
        raise NotImplementedError("Binary unit write handled in writefield().")
    it = np.nditer(e, flags=["multi_index"])
    for _ in it:
        z = e[it.multi_index]
        u.write(f"{z.real: .16e} {z.imag: .16e}\n")

def initialize_field(e: NDArray[C128]) -> None:
    e[...] = 0.0 + 0.0j
