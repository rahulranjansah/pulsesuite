"""
Pulse envelope generator — Python port of ``pulsegenerator.F90``.

Generates temporal pulse envelopes (Gaussian, Sech², Square, file-based)
on arbitrary grids.  All public names preserve the Fortran API so that
existing users can grep for ``pulsegen``, ``GaussPulse``, etc.

Dependencies
------------
- ``helpers.gauss_dp``, ``helpers.sech``, ``helpers.sech_dp``
- ``spliner.rescale_1D``
"""

from typing import Optional

import numpy as np

from .helpers import gauss_dp, sech, sech_dp
from .spliner import rescale_1D

# ── Analytic envelopes ──────────────────────────────────────────────────

def GaussPulse(x: np.ndarray, FWHM: float) -> np.ndarray:
    """Gaussian envelope normalised to unit peak with given FWHM.

    Parameters
    ----------
    x : ndarray
        Coordinate array.
    FWHM : float
        Full width at half maximum.
    """
    return gauss_dp(x / FWHM * np.sqrt(2.0 * np.log(2.0)))


def Sech2Pulse(x: np.ndarray, FWHM: float) -> np.ndarray:
    """Sech² envelope normalised to unit peak with given FWHM.

    Parameters
    ----------
    x : ndarray
        Coordinate array.
    FWHM : float
        Full width at half maximum.
    """
    return sech_dp(x / FWHM * (2.0 * np.log(1.0 + np.sqrt(2.0))))


def SquarePulse(x: np.ndarray, FWHM: float) -> np.ndarray:
    """Rectangular (top-hat) pulse of width *FWHM*.

    Parameters
    ----------
    x : ndarray
        Coordinate array.
    FWHM : float
        Full width.
    """
    return np.where(np.abs(x) < FWHM / 2.0, 1.0, 0.0)


def FilePulse(fn: str, X: np.ndarray) -> np.ndarray:
    """Load a pulse profile from a text file and interpolate onto *X*.

    The file must be a whitespace-delimited matrix where:
    - column 0 = coordinate
    - column 1 = real part (or amplitude)
    - column 2 = imaginary part (optional)

    Uses ``np.loadtxt`` (replaces Fortran ``readmatrix`` from ``fileio``).

    Parameters
    ----------
    fn : str
        Path to the data file.
    X : ndarray
        Target coordinate grid.
    """
    P = np.loadtxt(fn).T  # shape (ncols, npoints)
    X1 = P[0]
    if P.shape[0] == 2:
        Z1 = P[1].astype(np.complex128)
    else:
        Z1 = P[1] + 1j * P[2]

    Y = np.empty(X.size, dtype=np.complex128)
    rescale_1D(X1, Z1, X, Y)
    return Y


# ── Dispatchers ─────────────────────────────────────────────────────────

def pulsegen(shp: str, FWHM: float,
             X: Optional[np.ndarray] = None,
             N: Optional[int] = None,
             dx: Optional[float] = None) -> np.ndarray:
    """Generate a pulse envelope.

    Mirrors the Fortran ``pulsegen`` interface (``pulsegen1`` / ``pulsegen2``).

    * If *X* is given, evaluates the envelope at those coordinates.
    * If *N* and *dx* are given, builds a centred grid first.

    Parameters
    ----------
    shp : str
        Shape identifier: ``"gauss"``, ``"sech2"``, ``"square"``,
        or ``"file:<path>"`` to read from a data file.
    FWHM : float
        Full width at half maximum.
    X : ndarray, optional
        Coordinate array (pulsegen2 path).
    N : int, optional
        Number of grid points (pulsegen1 path).
    dx : float, optional
        Grid spacing (pulsegen1 path).

    Returns
    -------
    Y : ndarray of complex128
        Pulse envelope on the grid.
    """
    if X is None:
        if N is None or dx is None:
            raise ValueError("Provide either X, or both N and dx.")
        X = (np.arange(N, dtype=np.float64) - N / 2.0 - 1.0) * dx

    key = shp.strip().lower()[:5]
    if key == "gauss":
        return GaussPulse(X, FWHM).astype(np.complex128)
    elif key == "sech2":
        return Sech2Pulse(X, FWHM).astype(np.complex128)
    elif key == "squar":
        return SquarePulse(X, FWHM).astype(np.complex128)
    elif key == "file:":
        return FilePulse(shp.strip()[5:], X)
    else:
        raise ValueError(f"Unknown pulse shape: {shp!r}")


# ── Multi-pulse trains ──────────────────────────────────────────────────

def multipulsegen(shp: str, t0: float, t: np.ndarray,
                  sep: float, num: int) -> np.ndarray:
    """Generate a train of 2 or 3 identical pulses.

    Parameters
    ----------
    shp : str
        ``"gauss"`` for Gaussian envelopes, ``"sech"`` for sech envelopes,
        ``"uneven2a"`` / ``"uneven2b"`` for asymmetric two-pulse trains.
    t0 : float
        Pulse duration parameter (1/e half-width for Gaussian, t0 for sech).
    t : ndarray
        Time grid.
    sep : float
        Centre-to-centre separation between adjacent pulses.
    num : int
        Number of pulses (2 or 3).

    Returns
    -------
    Y : ndarray of complex128
        Envelope on the time grid.
    """
    t = np.asarray(t, dtype=np.float64)
    key = shp.strip().lower()

    if num == 2:
        tx = sep / 2.0
        if key == "gauss":
            y = np.exp(-((t - tx) / t0) ** 2) + np.exp(-((t + tx) / t0) ** 2)
        elif key == "uneven2a":
            y = np.exp(-((t - tx) / t0) ** 2) + np.exp(-((t + tx) / t0) ** 2) / np.sqrt(2.0)
        elif key == "uneven2b":
            y = np.exp(-((t + tx) / t0) ** 2) + np.exp(-((t - tx) / t0) ** 2) / np.sqrt(2.0)
        else:
            y = sech((t - tx) / t0) + sech((t + tx) / t0)
    elif num == 3:
        tx = sep
        if key == "gauss":
            y = (np.exp(-((t - tx) / t0) ** 2)
                 + np.exp(-(t / t0) ** 2)
                 + np.exp(-((t + tx) / t0) ** 2))
        else:
            y = sech((t - tx) / t0) + sech(t / t0) + sech((t + tx) / t0)
    else:
        raise ValueError(f"multipulsegen: num must be 2 or 3, got {num}")

    return y.astype(np.complex128)
