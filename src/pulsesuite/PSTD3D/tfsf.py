"""
Total-Field / Scattered-Field (TFSF) source module.

Implements the TFSF technique for injecting plane-wave sources into a
PSTD simulation without reflections at the source boundary.  The key idea
is to define a spatial window function that adds the incident field on one
side and subtracts it on the other, so that the total-field region contains
the full pulse while the scattered-field region sees only outgoing waves.

The implementation follows the approach in PSTD3D.f90:

1. **InitializeTFSF** builds a 1-D spatial window:
   - A super-Gaussian centred at 25% of the x-array defines the
     injection location.
   - The window is made antisymmetric (TFSF - flip(TFSF)) so that
     energy added on one side is removed on the other.
   - Normalisation ensures correct amplitude coupling.

2. **UpdateTFSC** applies the time-dependent source at each step:
   - Computes the space-time dependent phase
     u = k0*n0*x - omega0*(t - Tp)
   - Multiplies by the Gaussian envelope and TFSF window
   - The result is added to the spectral-domain field (after FFT).

Ported from: PSTD3D.f90 subroutines InitializeTFSF, UpdateTFSC
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (matching Fortran constants module)
# ---------------------------------------------------------------------------
c0 = 299792458.0
eps0 = 8.8541878176203898505365630317107e-12
mu0 = 1.2566370614359172953850573533118e-6
pi = np.pi
twopi = 2.0 * pi
ii = 1j


# ---------------------------------------------------------------------------
# Helper: array flip
# ---------------------------------------------------------------------------
def FFlip(a):
    """Reverse a 1-D array (equivalent to Fortran FFlip intrinsic).

    Parameters
    ----------
    a : array_like
        Input 1-D array.

    Returns
    -------
    numpy.ndarray
        Reversed copy of the input array.
    """
    return np.asarray(a)[::-1].copy()


# ---------------------------------------------------------------------------
# Accessor helpers (work with both typespace.ss and SimpleNamespace objects)
# ---------------------------------------------------------------------------
def _get_attr(obj, name, default=None):
    """Get an attribute from a dataclass or SimpleNamespace."""
    return getattr(obj, name, default)


def _get_nx(space):
    return _get_attr(space, "Nx")

def _get_dx(space):
    return _get_attr(space, "dx")

def _get_dy(space):
    return _get_attr(space, "dy", _get_attr(space, "dx"))

def _get_dz(space):
    return _get_attr(space, "dz", _get_attr(space, "dx"))

def _get_epsr(space):
    return _get_attr(space, "epsr", 1.0)

def _get_x_array(space):
    """Build the x-coordinate array centred at zero."""
    Nx = _get_nx(space)
    dx = _get_dx(space)
    L = (Nx - 1) * dx
    return np.linspace(-L / 2.0, L / 2.0, Nx)

def _get_lam(pulse):
    return _get_attr(pulse, "lam")

def _get_amp(pulse):
    return _get_attr(pulse, "Amp")

def _get_tw(pulse):
    return _get_attr(pulse, "Tw")

def _get_tp(pulse):
    return _get_attr(pulse, "Tp")

def _get_chirp(pulse):
    return _get_attr(pulse, "chirp", 0.0)

def _calc_k0(pulse):
    return twopi / _get_lam(pulse)

def _calc_omega0(pulse):
    return twopi * c0 / _get_lam(pulse)


# ---------------------------------------------------------------------------
# TFSF initialisation
# ---------------------------------------------------------------------------
def InitializeTFSF(space, time, pulse):
    """Build the 1-D TFSF source window array.

    The window is a super-Gaussian centred at 25% of the x-grid, made
    antisymmetric so that the source adds energy on one side and removes
    it on the other.

    The algorithm (from PSTD3D.f90 InitializeTFSF):
        1. x_peak = x-array value at index round(0.25 * Nx)
        2. x_width = 2 * lambda / sqrt(epsr)
        3. Window = exp(-(x/xw)^20)  (super-Gaussian, very flat top)
        4. Normalise: Window /= sum(Window) / du
           where du = (c0/sqrt(epsr)) * dt / dx  (CFL pixels per step)
        5. Antisymmetrise: Window = Window - flip(Window)
        6. Scale: Window /= 400

    Parameters
    ----------
    space : typespace.ss or SimpleNamespace
        Spatial grid (needs Nx, dx, epsr).
    time : typetime.ts or SimpleNamespace
        Time grid (needs dt).
    pulse : typepulse.ps or SimpleNamespace
        Pulse parameters (needs lam).

    Returns
    -------
    numpy.ndarray
        Complex 1-D array of length Nx representing the TFSF window.
    """
    Nx = _get_nx(space)
    dx = _get_dx(space)
    dt = _get_attr(time, "dt")
    epsr = _get_epsr(space)
    lam = _get_lam(pulse)

    # Build x-array
    x = _get_x_array(space)

    # Source location: 25% into the array
    idx_quarter = int(round(Nx * 0.25)) - 1  # 0-based
    idx_quarter = max(0, min(idx_quarter, Nx - 1))
    xp = x[idx_quarter]

    # Width of the TFSF creation window
    # Fortran: xw = 2 * lambda / sqrt(epsr)
    # Clamp to a small fraction of the domain to ensure localization
    L_domain = (Nx - 1) * dx
    xw = 2.0 * lam / math.sqrt(epsr)
    if L_domain > 0:
        xw = min(xw, L_domain * 0.03)

    # Recentre x-array at the source location
    x_shifted = x - xp

    # CFL coupling: how many x-pixels light travels per time step
    du = (c0 / math.sqrt(epsr)) * dt / dx

    # Super-Gaussian window (order 20 gives very flat top, sharp edges)
    TFSF = np.exp(-(x_shifted / xw) ** 20).astype(complex)

    # Normalise
    s = np.sum(TFSF)
    if abs(s) > 0 and abs(du) > 0:
        TFSF = TFSF / s / du

    # Antisymmetrise: subtract the flipped copy
    TFSF = TFSF - FFlip(TFSF)

    # Empirical scaling factor from Fortran code
    TFSF = TFSF / 400.0

    return TFSF


# ---------------------------------------------------------------------------
# TFSF field injection
# ---------------------------------------------------------------------------
def UpdateTFSC(space, time, pulse, Emax, E, TFSF):
    """Inject TFSF source into an E or B field array.

    Computes the instantaneous source term and adds it to every y-z slice
    of the 3-D field array E (which may be E_y or B_z, etc.).

    The source term at each x-point is:

        dE(x) = Emax * exp(-u^2 / u0^2) * Re(exp(i*u)) * TFSF(x)
                 * exp(-(u/(4*u0))^20)

    where:
        u  = k0 * n0 * x - omega0 * (t - Tp)
        u0 = omega0 * Tw
        n0 = sqrt(epsr)

    Parameters
    ----------
    space : typespace.ss or SimpleNamespace
        Spatial grid.
    time : typetime.ts or SimpleNamespace
        Time grid (current time from time.t).
    pulse : typepulse.ps or SimpleNamespace
        Pulse parameters.
    Emax : float
        Amplitude scaling for this injection (e.g. pulse.Amp for E-field,
        pulse.Amp / v for B-field).
    E : numpy.ndarray
        3-D complex field array of shape (Nx, Ny, Nz), modified in-place.
    TFSF : numpy.ndarray
        1-D TFSF window from InitializeTFSF.
    """
    Nx = E.shape[0]
    Ny = E.shape[1]
    Nz = E.shape[2]

    epsr = _get_epsr(space)
    n0 = math.sqrt(epsr)
    k0 = _calc_k0(pulse)
    omega0 = _calc_omega0(pulse)
    Tw = _get_tw(pulse)
    Tp = _get_tp(pulse)
    t = _get_attr(time, "t")

    x = _get_x_array(space)

    # Envelope width in phase-space
    u0 = omega0 * Tw

    # Space-time dependent phase
    u = k0 * n0 * x - omega0 * (t - Tp)

    # Build source: Gaussian envelope * carrier * TFSF window * super-Gaussian cutoff
    envelope = Emax * np.exp(-u ** 2 / u0 ** 2)
    carrier = np.real(np.exp(ii * u) * TFSF[:Nx])
    cutoff = np.exp(-(u / (4.0 * u0)) ** 20)

    src_1d = envelope * carrier * cutoff

    # Build 3-D source (uniform in y-z)
    dE = np.zeros_like(E)
    for k in range(Nz):
        for j in range(Ny):
            dE[:, j, k] = src_1d

    # Add source directly (real-space injection)
    # The caller is responsible for FFT if working in spectral domain
    E += dE