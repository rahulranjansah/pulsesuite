"""
Temporal grid structure for electromagnetic simulations.

This module provides the temporal grid structure (ts type) and associated
functions for reading, writing, and computing time-domain and frequency-domain
arrays used in PSTD simulations.

Physics reference:
    The time grid spans [t, tf] with step dt.  The number of remaining
    time steps is Nt = int((tf - t) / dt).

    The conjugate (frequency) domain uses FFT-ordered angular frequencies:
        w(n) = 2*pi*(n-1) / (Nt*dt)   for n <= Nt/2
        w(n) = 2*pi*(n-Nt-1) / (Nt*dt) for n > Nt/2
    with spacing dw = 2*pi / (Nt*dt).

Ported from: typetime.f90
"""

import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Physical constants (matching Fortran constants module)
# ---------------------------------------------------------------------------
pi = np.pi
twopi = 2.0 * pi


# ---------------------------------------------------------------------------
# File I/O helper (shared with typespace)
# ---------------------------------------------------------------------------
def GetFileParam(file_handle):
    """Read a single numeric parameter from a file handle.

    Reads one line, extracts the first whitespace-delimited token, and
    converts it to float.  Handles trailing Fortran-style ``!`` comments.

    Parameters
    ----------
    file_handle : file-like
        Open text stream.

    Returns
    -------
    float
        The parsed numeric value.

    Raises
    ------
    ValueError
        If the line is empty or cannot be parsed.
    """
    line = file_handle.readline()
    if not line:
        raise ValueError("Unexpected end of file while reading parameter")
    parts = line.split()
    if not parts:
        raise ValueError(f"Empty line in parameter file: {line!r}")
    # First token that isn't a comment
    token = parts[0]
    if token.startswith("!"):
        raise ValueError(f"Comment-only line: {line!r}")
    return float(token)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------
@dataclass
class ts:
    """
    Temporal grid structure.

    Attributes
    ----------
    t : float
        Current simulation time (s).
    tf : float
        Final simulation time (s).
    dt : float
        Time step size (s).
    n : int
        Current time-step index.
    """
    t: float
    tf: float
    dt: float
    n: int


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def readtimeparams_sub(fh, time):
    """Read time parameters from an open file handle.

    Expected format (one value per line, optional trailing comments)::

        0.0         ! t
        100e-15     ! tf
        0.5e-15     ! dt
        0           ! n

    Parameters
    ----------
    fh : file-like
        Readable text stream.
    time : ts
        Time structure to populate (modified in-place).
    """
    time.t = GetFileParam(fh)
    time.tf = GetFileParam(fh)
    time.dt = GetFileParam(fh)
    time.n = int(GetFileParam(fh))


def ReadtimeParams(filename, time):
    """Read time parameters from a named file.

    Parameters
    ----------
    filename : str
        Path to the parameter file.
    time : ts
        Time structure to populate (modified in-place).
    """
    with open(filename, "r") as fh:
        readtimeparams_sub(fh, time)


def writetimeparams_sub(fh, time):
    """Write time parameters to an open file handle.

    Parameters
    ----------
    fh : file-like
        Writable text stream.
    time : ts
        Time structure to write.
    """
    fh.write(f"{time.t:25.14E} : Current time of simulation. (s)\n")
    fh.write(f"{time.tf:25.14E} : Final time of simulation. (s)\n")
    fh.write(f"{time.dt:25.14E} : Time pixel size [dt]. (s)\n")
    fh.write(f"{time.n:25d} : Current time index.\n")


def writetimeparams(filename, time):
    """Write time parameters to a named file.

    Parameters
    ----------
    filename : str
        Path to the output file.
    time : ts
        Time structure to write.
    """
    with open(filename, "w") as fh:
        writetimeparams_sub(fh, time)


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------
def GetT(time):
    """Return current simulation time (s)."""
    return time.t

def GetTf(time):
    """Return final simulation time (s)."""
    return time.tf

def GetDt(time):
    """Return time step dt (s)."""
    return time.dt

def GetN(time):
    """Return current time-step index."""
    return time.n


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------
def SetT(time, t):
    """Set current simulation time (s)."""
    time.t = t

def SetTf(time, tf):
    """Set final simulation time (s)."""
    time.tf = tf

def SetDt(time, dt):
    """Set time step dt (s)."""
    time.dt = dt

def SetN(time, n):
    """Set current time-step index."""
    time.n = n


# ---------------------------------------------------------------------------
# Computed quantities
# ---------------------------------------------------------------------------
def CalcNt(time):
    """Compute number of remaining time steps: int((tf - t) / dt).

    Parameters
    ----------
    time : ts
        Time structure.

    Returns
    -------
    int
        Number of time steps from current time to final time.
    """
    return int(round((time.tf - time.t) / time.dt))


def UpdateT(time, dt):
    """Advance current time by dt: t <- t + dt.

    Parameters
    ----------
    time : ts
        Time structure (modified in-place).
    dt : float
        Time increment (s).  May differ from time.dt (e.g. half-steps).
    """
    time.t = time.t + dt


def UpdateN(time, dn):
    """Advance time-step index by dn: n <- n + dn.

    Parameters
    ----------
    time : ts
        Time structure (modified in-place).
    dn : int
        Index increment.
    """
    time.n = time.n + dn


# ---------------------------------------------------------------------------
# Array generation
# ---------------------------------------------------------------------------
def GetTArray(time):
    """Generate the time-point array from current time to final time.

    Returns an array of Nt values starting at time.t with spacing time.dt.
    Special case: if Nt==1, returns [0.0] (matching Fortran behaviour).

    Parameters
    ----------
    time : ts
        Time structure.

    Returns
    -------
    numpy.ndarray
        1-D array of time values (s), length CalcNt(time).
    """
    Nt = CalcNt(time)
    if Nt <= 0:
        return np.array([], dtype=float)
    t = np.empty(Nt, dtype=float)
    for i in range(Nt):
        if Nt == 1:
            t[i] = 0.0
        else:
            t[i] = time.t + i * time.dt
    return t


def GetOmegaArray(time):
    """Generate FFT-ordered angular frequency array.

    The array follows the standard FFT convention:
        w[n] = 2*pi*n / (Nt*dt)        for n = 0 .. Nt/2 - 1
        w[n] = 2*pi*(n - Nt) / (Nt*dt) for n = Nt/2 .. Nt - 1

    This places DC (w=0) at index 0 and negative frequencies in the
    upper half of the array, matching numpy.fft conventions.

    Parameters
    ----------
    time : ts
        Time structure.

    Returns
    -------
    numpy.ndarray
        1-D array of angular frequencies (rad/s), length CalcNt(time).
    """
    Nt = CalcNt(time)
    Tw = Nt * time.dt
    w = np.empty(Nt, dtype=float)
    for n in range(Nt):
        # Fortran uses 1-based indexing; n_fortran = n+1
        n1 = n + 1
        if n1 <= Nt // 2:
            w[n] = twopi * (n1 - 1) / Tw
        else:
            w[n] = twopi * (n1 - Nt - 1) / Tw
    return w


def GetdOmega(time):
    """Compute angular frequency spacing dw = 2*pi / (Nt * dt).

    Parameters
    ----------
    time : ts
        Time structure.

    Returns
    -------
    float
        Frequency-domain step size (rad/s).
    """
    Nt = CalcNt(time)
    return twopi / (Nt * time.dt)


# ---------------------------------------------------------------------------
# Field initialisation helper
# ---------------------------------------------------------------------------
def initialize_field(e):
    """Zero a complex 3-D field array in-place.

    Parameters
    ----------
    e : numpy.ndarray
        Complex array of any shape (modified in-place to all zeros).
    """
    e[:] = 0.0 + 0.0j