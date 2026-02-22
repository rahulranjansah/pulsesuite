"""
Pulse parameter structure for electromagnetic simulations.

This module provides the pulse parameter structure (ps type) and associated
functions for reading, writing, and computing derived physical quantities
of ultrafast laser pulses in vacuum.

Physics reference:
    A Gaussian pulse in the time domain is defined as:
        E(x,t) = Amp * exp(-(t')^2 / (2*Tw^2)) * exp(i*(omega0*t' + chirp*t'^2))
    where t' = t - Tp - x/c0 is the retarded time.

    Derived quantities:
        k0     = 2pi / lam            (wave number)
        nu0    = c0 / lam             (optical frequency)
        omega0 = 2pi c0 / lam         (angular frequency)
        tau    = Tw / (2*sqrt(ln2))   (1/e half-width of intensity)
        Domega = 0.44 / tau           (Fourier-limited bandwidth)
        TBP    = Domega * tau = 0.44  (time-bandwidth product, transform-limited)

    Spatial beam quantities (using omega0 as beam waist w0, per Fortran convention):
        z_R  = pi * w0**2 / lam       (Rayleigh range)
        R(x) = x * (1 + (z_R/x)**2)   (wavefront curvature)
        phi  = arctan(x / z_R)         (Gouy phase)

Ported from: typepulse.f90
"""

import math
import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Physical constants (must match Fortran constants module exactly)
# ---------------------------------------------------------------------------
c0 = 299792458.0                                    # speed of light  (m/s)
eps0 = 8.8541878176203898505365630317107e-12         # permittivity    (F/m)
mu0 = 1.2566370614359172953850573533118e-6           # permeability    (H/m)
pi = np.pi
twopi = 2.0 * pi


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------
@dataclass
class ps:
    """
    Pulse parameter structure.

    Attributes
    ----------
    lam : float
        Vacuum wavelength of the field (m).  Named 'lambda' in Fortran.
    Amp : float
        Peak pulse amplitude (V/m).
    Tw : float
        Pulse width -- FWHM of the *field* envelope (s).
    Tp : float
        Time the pulse peak passes through the origin (s).
    chirp : float
        Linear chirp coefficient (rad/s^2).
    pol : int
        Polarisation index (user-defined convention).
    """
    lam: float
    Amp: float
    Tw: float
    Tp: float
    chirp: float
    pol: int


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------
def _get_file_param(fh):
    """Read one numeric value from a file handle (one value per line).

    Handles trailing comments separated by whitespace.
    """
    line = fh.readline()
    if not line:
        raise ValueError("Unexpected end of file while reading parameter")
    parts = line.split()
    if not parts:
        raise ValueError(f"Empty line in parameter file: {line!r}")
    # Strip Fortran-style comment marker
    token = parts[0]
    if token.startswith("!"):
        raise ValueError(f"Comment-only line: {line!r}")
    return float(token)


def readpulseparams_sub(fh, pulse):
    """Read pulse parameters from an open file handle into *pulse*.

    Expected file format (one value per line, optional trailing comments)::

        800e-9      ! lambda
        1.25e8      ! Amp
        5.0e-15     ! Tw
        60e-15      ! Tp
        0.0         ! chirp

    Parameters
    ----------
    fh : file-like
        Readable text stream positioned at the first parameter line.
    pulse : ps
        Pulse structure to populate (modified in-place).
    """
    pulse.lam = _get_file_param(fh)
    pulse.Amp = _get_file_param(fh)
    pulse.Tw = _get_file_param(fh)
    pulse.Tp = _get_file_param(fh)
    pulse.chirp = _get_file_param(fh)


def ReadPulseParams(filename, pulse):
    """Read pulse parameters from a named file.

    Parameters
    ----------
    filename : str
        Path to the parameter file.
    pulse : ps
        Pulse structure to populate (modified in-place).
    """
    with open(filename, "r") as fh:
        readpulseparams_sub(fh, pulse)


def writepulseparams_sub(fh, pulse):
    """Write pulse parameters to an open file handle.

    Parameters
    ----------
    fh : file-like
        Writable text stream.
    pulse : ps
        Pulse structure to write.
    """
    fh.write(f"{pulse.lam:25.14E} : The pulse wavelength. (m)\n")
    fh.write(f"{pulse.Amp:25.14E} : The pulse amplitude. (V/m)\n")
    fh.write(f"{pulse.Tw:25.14E} : The pulsewidth. (s)\n")
    fh.write(f"{pulse.Tp:25.14E} : The time the pulse crosses the origin. (s)\n")
    fh.write(f"{pulse.chirp:25.14E} : The pulse chirp constant. (rad/s^2)\n")


def WritePulseParams(filename, pulse):
    """Write pulse parameters to a named file.

    Parameters
    ----------
    filename : str
        Path to the output file.
    pulse : ps
        Pulse structure to write.
    """
    with open(filename, "w") as fh:
        writepulseparams_sub(fh, pulse)


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------
def GetLambda(pulse):
    """Return vacuum wavelength lambda (m)."""
    return pulse.lam

def GetAmp(pulse):
    """Return peak field amplitude (V/m)."""
    return pulse.Amp

def GetTw(pulse):
    """Return pulse width Tw (s)."""
    return pulse.Tw

def GetTp(pulse):
    """Return peak-crossing time Tp (s)."""
    return pulse.Tp

def GetChirp(pulse):
    """Return chirp coefficient (rad/s^2)."""
    return pulse.chirp

def GetPol(pulse):
    """Return polarisation index."""
    return pulse.pol


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------
def SetLambda(pulse, lam):
    """Set vacuum wavelength lambda (m)."""
    pulse.lam = lam

def SetAmp(pulse, Amp):
    """Set peak field amplitude (V/m)."""
    pulse.Amp = Amp

def SetTw(pulse, Tw):
    """Set pulse width Tw (s)."""
    pulse.Tw = Tw

def SetTp(pulse, Tp):
    """Set peak-crossing time Tp (s)."""
    pulse.Tp = Tp

def SetChirp(pulse, chirp):
    """Set chirp coefficient (rad/s^2)."""
    pulse.chirp = chirp

def SetPol(pulse, pol):
    """Set polarisation index."""
    pulse.pol = pol


# ---------------------------------------------------------------------------
# Derived temporal / spectral quantities
# ---------------------------------------------------------------------------
def CalcK0(pulse):
    """Wave number k0 = 2*pi / lambda  (rad/m)."""
    return twopi / pulse.lam


def CalcFreq0(pulse):
    """Optical frequency nu0 = c0 / lambda  (Hz)."""
    return c0 / pulse.lam


def CalcOmega0(pulse):
    """Angular frequency omega0 = 2*pi*c0 / lambda  (rad/s).

    Satisfies the dispersion relation omega0 = k0 * c0.
    """
    return twopi * c0 / pulse.lam


def CalcTau(pulse):
    """Gaussian parameter tau = Tw / (2*sqrt(ln2)).

    tau is the 1/e half-width of the field intensity envelope for a
    transform-limited Gaussian pulse.  Tw = FWHM of the intensity.
    """
    return pulse.Tw / (2.0 * math.sqrt(math.log(2.0)))


def CalcDeltaOmega(pulse):
    """Fourier-limited spectral bandwidth Delta_omega = 0.44 / tau  (rad/s).

    For a transform-limited Gaussian the time-bandwidth product
    Delta_omega * tau equals 0.44.
    """
    return 0.44 / CalcTau(pulse)


def CalcTime_BandWidth(pulse):
    """Time-bandwidth product Delta_omega * tau.

    Equals 0.44 for a transform-limited Gaussian (no chirp broadening
    in spectral domain for this definition).
    """
    return CalcDeltaOmega(pulse) * CalcTau(pulse)


# ---------------------------------------------------------------------------
# Derived spatial / beam quantities
# ---------------------------------------------------------------------------
def CalcRayleigh(pulse):
    """Rayleigh range z_R = pi * w0^2 / lambda.

    In the Fortran code, CalcOmega0(pulse) is used as the beam-waist
    parameter w0.  This is the original code's naming convention.
    """
    w0 = CalcOmega0(pulse)
    return pi * w0**2 / pulse.lam


def CalcCurvature(pulse, x):
    """Wavefront radius of curvature R(x).

    R(x) = x * (1 + (z_R / x)^2)   for x != 0
    R(0) = inf                       (flat phase at the waist)

    Parameters
    ----------
    x : float
        Propagation distance from beam waist (m).

    Returns
    -------
    float
        Radius of curvature (m).  Returns float('inf') at x=0.
    """
    if x == 0.0:
        return float("inf")
    xR = CalcRayleigh(pulse)
    return x * (1.0 + (xR / x) ** 2)


def CalcGouyPhase(pulse, x):
    """Gouy phase phi(x) = arctan(x / z_R).

    Parameters
    ----------
    x : float
        Propagation distance from beam waist (m).

    Returns
    -------
    float
        Gouy phase (rad).
    """
    return math.atan(x / CalcRayleigh(pulse))


# ---------------------------------------------------------------------------
# Pulse field generation
# ---------------------------------------------------------------------------
def PulseFieldXT(x, t, pulse):
    """Compute complex electric field E(x, t) of the Gaussian pulse.

    E(x,t) = Amp * exp(-delay^2 / (2*Tw^2)) * exp(i*(omega0*delay + chirp*delay^2))

    where delay = t - Tp - x/c0 is the retarded time.

    Parameters
    ----------
    x : float
        Spatial position (m).
    t : float
        Time (s).
    pulse : ps
        Pulse parameters.

    Returns
    -------
    complex
        Complex electric field amplitude (V/m).
    """
    Tw = pulse.Tw
    Tp = pulse.Tp
    omega0 = CalcOmega0(pulse)
    chirp = pulse.chirp

    # Retarded time in the pulse rest frame
    delay = t - Tp - x / c0

    # Gaussian envelope (real-valued)
    envelope = pulse.Amp * math.exp(-(delay ** 2) / (2.0 * Tw ** 2))

    # Total phase: carrier + chirp
    phase = omega0 * delay + chirp * delay ** 2

    return complex(envelope * math.cos(phase), envelope * math.sin(phase))