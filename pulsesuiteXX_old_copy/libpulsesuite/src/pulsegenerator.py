# pulsegenerator.py

import numpy as np
from typing import Tuple, Optional, Annotated
from guardrails.guardrails import with_guardrails

__all__ = [
    "pulsegen1",
    "pulsegen2",
    "FilePulse",
    "multipulsegen",
]

# Elemental helpers: no guardrails, keep Annotated hints for clarity

def GaussPulse(
    x: Annotated[np.ndarray, np.float64],
    FWHM: Annotated[float, np.float64]
) -> Annotated[np.ndarray, np.float64]:
    """
    Elemental Gaussian pulse.

    Parameters
    ----------
    x : float64 ndarray
        Sample points.
    FWHM : float64
        Full-width at half-maximum.

    Returns
    -------
    y : float64 ndarray
        Gaussian pulse values at x.
    """
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * (x / sigma) ** 2)


def Sech2Pulse(
    x: Annotated[np.ndarray, np.float64],
    FWHM: Annotated[float, np.float64]
) -> Annotated[np.ndarray, np.float64]:
    """
    Elemental hyperbolic secant pulse.

    Parameters
    ----------
    x : float64 ndarray
        Sample points.
    FWHM : float64
        Full-width at half-maximum.

    Returns
    -------
    y : float64 ndarray
        Sech-shaped pulse values at x.
    """
    factor = 2 * np.log(1 + np.sqrt(2))
    u = x / FWHM * factor
    return 1.0 / np.cosh(u)


def SquarePulse(
    x: Annotated[np.ndarray, np.float64],
    FWHM: Annotated[float, np.float64]
) -> Annotated[np.ndarray, np.float64]:
    """
    Elemental square pulse.

    Parameters
    ----------
    x : float64 ndarray
        Sample points.
    FWHM : float64
        Full-width at half-maximum.

    Returns
    -------
    y : float64 ndarray
        1 inside |x|<FWHM/2, else 0.
    """
    return np.where(np.abs(x) < 0.5 * FWHM, 1.0, 0.0)

# Public-facing routines: enforce guards

@with_guardrails
def FilePulse(
    fn: str,
    X: Annotated[np.ndarray, np.float64]
) -> Annotated[np.ndarray, np.complex128]:
    """
    Read a user-defined pulse from file and resample onto X.

    Parameters
    ----------
    fn : str
        Path to file containing matrix P.
    X : float64 ndarray
        Target grid for interpolation.

    Returns
    -------
    Y : complex128 ndarray
        Pulse values at X.
    """
    P = np.loadtxt(fn, dtype=np.float64)
    X1 = P[0, :]
    if P.shape[0] == 2:
        Z1 = P[1, :].astype(np.complex128)
    else:
        Z1 = P[1, :] + 1j * P[2, :]

    real_interp = np.interp(X, X1, Z1.real)
    imag_interp = np.interp(X, X1, Z1.imag)
    return real_interp + 1j * imag_interp

@with_guardrails
def pulsegen2(
    shp: str,
    FWHM: Annotated[float, np.float64],
    X: Annotated[np.ndarray, np.float64]
) -> Annotated[np.ndarray, np.complex128]:
    """
    Core pulse generator: dispatch by shape.

    Parameters
    ----------
    shp : str
        Shape key ("gauss", "sech2", "square", "file:<fn>").
    FWHM : float64
        Width parameter for built-in shapes.
    X : float64 ndarray
        Sample grid.

    Returns
    -------
    Y : complex128 ndarray
        Pulse values.
    """
    key = shp[:5].lower()
    if key == "gauss":
        real_y = GaussPulse(X, FWHM)
        return real_y.astype(np.complex128)
    elif key == "sech2":
        real_y = Sech2Pulse(X, FWHM)
        return real_y.astype(np.complex128)
    elif key == "squar":
        real_y = SquarePulse(X, FWHM)
        return real_y.astype(np.complex128)
    elif key == "file:":
        return FilePulse(shp[5:], X)
    else:
        raise ValueError(f"Unknown shape '{shp}'")

@with_guardrails
def pulsegen1(
    shp: str,
    FWHM: Annotated[float, np.float64],
    N: Annotated[int, np.int32],
    dx: Annotated[float, np.float64],
    X_out: Optional[Annotated[np.ndarray, np.float64]] = None
) -> Tuple[
    Annotated[np.ndarray, np.complex128],
    Annotated[np.ndarray, np.float64]
]:
    """
    Top-level pulse generator: build grid + call pulsegen2.

    Parameters
    ----------
    shp : str
        Shape key.
    FWHM : float64
        Width parameter.
    N : int32
        Number of samples.
    dx : float64
        Grid spacing.
    X_out : float64 ndarray, optional
        If provided, filled with grid.

    Returns
    -------
    Y : complex128 ndarray
    X : float64 ndarray
    """
    X = (np.arange(N, dtype=np.float64) - N/2) * dx
    Y = pulsegen2(shp, FWHM, X)
    if X_out is not None:
        X_out[:] = X
    return Y, X

@with_guardrails
def multipulsegen(
    shp: str,
    t0: Annotated[float, np.float64],
    t: Annotated[np.ndarray, np.float64],
    sep: Annotated[float, np.float64],
    num: Annotated[int, np.int32]
) -> Annotated[np.ndarray, np.complex128]:
    """
    Generate 2 or 3 pulses on grid t.

    Parameters
    ----------
    shp : str
        "gauss", "uneven2a", "uneven2b", or else sech.
    t0 : float64
        Pulse width/scale.
    t : float64 ndarray
        Time grid.
    sep : float64
        Separation.
    num : int32
        Number of pulses (2 or 3).

    Returns
    -------
    y : complex128 ndarray
    """
    if num not in (2, 3):
        raise ValueError("'num' must be 2 or 3")

    y = np.zeros_like(t, dtype=np.complex128)
    if num == 2:
        tx = (num - 1) * sep / 2.0
        if shp == "gauss":
            y = np.exp(-((t - tx) / t0) ** 2) + np.exp(-((t + tx) / t0) ** 2)
        elif shp == "uneven2a":
            y = (np.exp(-((t - tx) / t0) ** 2)
                 + np.exp(-((t + tx) / t0) ** 2) / np.sqrt(2))
        elif shp == "uneven2b":
            y = (np.exp(-((t + tx) / t0) ** 2)
                 + np.exp(-((t - tx) / t0) ** 2) / np.sqrt(2))
        else:
            y = 1.0/np.cosh((t - tx)/t0) + 1.0/np.cosh((t + tx)/t0)
    else:
        tx = sep
        if shp == "gauss":
            y = (np.exp(-((t - tx) / t0) ** 2)
                 + np.exp(-(t / t0) ** 2)
                 + np.exp(-((t + tx) / t0) ** 2))
        else:
            y = (1.0/np.cosh((t - tx)/t0)
                 + 1.0/np.cosh(t/t0)
                 + 1.0/np.cosh((t + tx)/t0))
    return y
