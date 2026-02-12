"""
Material properties module.

Accesses the materials database and retrieves various material parameters
such as refractive index, dispersion coefficients, nonlinear index,
absorption, and plasma parameters.

Converted from materialproperties.F90.

Author: Rahul R. Sah
"""

import logging
import os
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from pulsesuite.core.constants import c0, eps0, pi, twopi
from pulsesuite.libpulsesuite.spliner import seval, spline

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error codes matching Fortran integer parameters
# ---------------------------------------------------------------------------

class MatError(IntEnum):
    NOERROR = 0
    NOFILE = 1
    NOTFOUND = 2
    FILE_FORMAT = 3
    OUTOFRANGE = 4
    BADVALUE = 5
    DEFAULTUSED = 6


# ---------------------------------------------------------------------------
# Database file paths
# ---------------------------------------------------------------------------

_PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"
_MASTER_DATABASE_FILE = str(_PACKAGE_DATA_DIR / "materials.ini")

# User-overridable local database file
_database_file: str = "materials.ini"


def set_database_file(path: str) -> None:
    """Set the local materials database file path."""
    global _database_file
    _database_file = path
    if not os.path.isfile(path):
        log.warning("Cannot find material databasefile, %s", path)


# ---------------------------------------------------------------------------
# INI file reader (matches Fortran ReadIniTagStr behaviour)
# ---------------------------------------------------------------------------

def _read_ini_tag_str(
    filepath: str, section: str, tag: str
) -> Tuple[Optional[str], int]:
    """
    Read a tag value from an INI-formatted materials database.

    The INI format uses ``[SECTION]`` headers (case-sensitive) and
    ``tag=value`` entries.  Lines starting with ``::`` are comments.

    Returns (value_string, err) where *err* is 0 on success or non-zero
    on failure.
    """
    try:
        with open(filepath, "r") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return None, 1

    # Find section
    section_header = f"[{section}]"
    in_section = False
    for line in lines:
        stripped = line.strip()
        if not in_section:
            if stripped == section_header:
                in_section = True
            continue

        # We are inside the target section
        if stripped.startswith("["):
            # Hit the next section without finding the tag
            return None, -1

        if stripped.startswith("::") or stripped == "":
            continue

        # Check for tag=
        if stripped.startswith(tag + "="):
            value = stripped[len(tag) + 1:].strip()
            return value, 0

    return None, -1


# ---------------------------------------------------------------------------
# Database reading helpers
# ---------------------------------------------------------------------------

def _read_dbs_tag_str(mat: str, param: str) -> Tuple[Optional[str], int]:
    """Read a string value from the materials database for section *mat* and *param*."""
    global _database_file

    err = MatError.NOERROR
    exists_local = os.path.isfile(_database_file)
    exists_master = os.path.isfile(_MASTER_DATABASE_FILE)

    result = None
    err0 = -1

    if exists_local:
        log.debug("Looking in materials database: %s", _database_file)
        result, err0 = _read_ini_tag_str(_database_file, mat.upper(), param)

    if err0 != 0 or not exists_local:
        if exists_master:
            log.debug("Looking in materials database: %s", _MASTER_DATABASE_FILE)
            result, err0 = _read_ini_tag_str(_MASTER_DATABASE_FILE, mat.upper(), param)

    if not exists_local and not exists_master:
        _mat_error_handler(MatError.NOFILE, param, mat, 0.0)
        return None, MatError.NOFILE

    if err0 != 0:
        return None, MatError.NOTFOUND

    return result, MatError.NOERROR


def _read_dbs_tag_val(mat: str, param: str) -> Tuple[float, int]:
    """Read a single float value from the materials database."""
    s, err = _read_dbs_tag_str(mat, param)
    if err != MatError.NOERROR:
        return 0.0, err
    try:
        val = float(s.split(",")[0].strip())
    except (ValueError, IndexError):
        return 0.0, MatError.FILE_FORMAT
    return val, MatError.NOERROR


def _read_dbs_tag_array(mat: str, param: str) -> Tuple[Optional[np.ndarray], int]:
    """Read a comma-separated array of floats from the materials database."""
    s, err = _read_dbs_tag_str(mat, param)
    if err != MatError.NOERROR:
        return None, err
    try:
        vals = np.array([float(x.strip()) for x in s.split(",") if x.strip()])
    except ValueError:
        return None, MatError.FILE_FORMAT
    return vals, MatError.NOERROR


# ---------------------------------------------------------------------------
# Discrete database value lookup
# ---------------------------------------------------------------------------

def _discrete_dbs_val(
    param: str, mat: str, lam: float
) -> Tuple[float, int]:
    """
    Look up a discrete parameter from the database, matching by wavelength.

    Tries to find the closest wavelength match:
    - Within 1 %: exact match (NOERROR)
    - Within 5 %: approximate match (OUTOFRANGE)
    - Negative wavelength entry: default value (DEFAULTUSED)
    - Otherwise: NOTFOUND
    """
    params, err0 = _read_dbs_tag_array(mat, param)
    if err0 != MatError.NOERROR:
        return 0.0, MatError.NOTFOUND

    lams, err0 = _read_dbs_tag_array(mat, f"{param}-wavelength")
    if err0 != MatError.NOERROR:
        return 0.0, MatError.NOTFOUND

    if len(params) != len(lams):
        _mat_error_handler(MatError.FILE_FORMAT, param, mat, lam)
        return 0.0, MatError.FILE_FORMAT

    rel_diff = np.abs(lams - lam) / abs(lam) if lam != 0.0 else np.abs(lams - lam)

    # Within 1 %?
    if np.min(rel_diff) < 0.01:
        idx = int(np.argmin(rel_diff))
        return float(params[idx]), MatError.NOERROR

    # Within 5 %?
    if np.min(rel_diff) < 0.05:
        idx = int(np.argmin(rel_diff))
        return float(params[idx]), MatError.OUTOFRANGE

    # Default value (negative wavelength)?
    if np.min(lams) < 0.0:
        idx = int(np.argmin(lams))
        return float(params[idx]), MatError.DEFAULTUSED

    return 0.0, MatError.NOTFOUND


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

def _mat_error_handler(err: int, param: str, mat: str, lam: float) -> None:
    """Default error handler mirroring the Fortran ``mat_error_handler``."""
    if err == MatError.NOERROR:
        return
    if err == MatError.NOFILE:
        raise FileNotFoundError(
            f"No materials database exists.  Tried: {_database_file} and {_MASTER_DATABASE_FILE}"
        )
    if err == MatError.NOTFOUND:
        raise LookupError(
            f"Unknown material '{mat}', unknown parameter '{param}', "
            f"or wavelength {lam:.2e} m out of range."
        )
    if err == MatError.FILE_FORMAT:
        raise ValueError(
            f"File format error reading material '{mat}', parameter '{param}', "
            f"at wavelength {lam:.2e} m."
        )
    if err == MatError.OUTOFRANGE:
        log.warning(
            "Wavelength %.2e m is >1%% different from nearest wavelength for %s in %s.",
            lam, param, mat,
        )
    if err == MatError.BADVALUE:
        log.warning(
            "Got a 'bad value' while getting %s for %s at %.2e m.", param, mat, lam
        )
    if err == MatError.DEFAULTUSED:
        log.warning(
            "Default value used for %s of %s at %.2e m.", param, mat, lam
        )


def _handle_err(err0: int, param: str, mat: str, lam: float, err: Optional[list] = None):
    """If *err* list is provided, store error; otherwise call handler."""
    if err is not None:
        if err0 != MatError.NOERROR:
            err[0] = err0
    else:
        _mat_error_handler(err0, param, mat, lam)


# ---------------------------------------------------------------------------
# Sellmeier coefficients
# ---------------------------------------------------------------------------

def sellmeiercoeff(
    mat: str, lam: float
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """
    Retrieve the Sellmeier coefficients from the materials database.

    Returns (A, B, C, err).
    """
    A, err0 = _read_dbs_tag_val(mat, "Sellmeier-A")
    if err0 != MatError.NOERROR:
        A = 1.0
    err0_reset = MatError.NOERROR

    B, err0_reset = _read_dbs_tag_array(mat, "Sellmeier-B")
    if err0_reset != MatError.NOERROR:
        return A, np.array([]), np.array([]), MatError.NOTFOUND

    C, err0_reset = _read_dbs_tag_array(mat, "Sellmeier-C")
    if err0_reset != MatError.NOERROR:
        return A, B, np.array([]), MatError.NOTFOUND

    if len(B) != len(C):
        _mat_error_handler(MatError.FILE_FORMAT, "Sellmeier", mat, lam)
        return A, B, C, MatError.FILE_FORMAT

    err = MatError.NOERROR

    # Check wavelength limits
    s, limit_err = _read_dbs_tag_str(mat, "Sellmeier-limits")
    if limit_err == MatError.NOERROR and s is not None:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) >= 2:
            lam1, lam2 = parts[0], parts[1]
            if lam < lam1 or lam > lam2:
                err = MatError.OUTOFRANGE

    return A, B, C, err


# ---------------------------------------------------------------------------
# Refractive index from Sellmeier (wavelength domain)
# ---------------------------------------------------------------------------

def n0_sellmeier(A: float, B: np.ndarray, C: np.ndarray, lam: float) -> float:
    """
    Calculate the index of refraction from Sellmeier coefficients.

    n0 = sqrt(A + sum(B * lam^2 / (lam^2 - C)))
    """
    val = np.sqrt(A + np.sum(B * lam ** 2 / (lam ** 2 - C)))
    if np.isnan(val) or val < 1.0 or val > 1.0e100:
        return 1.0
    return float(val)


def _dn_dl(A: float, B: np.ndarray, C: np.ndarray, lam: float) -> float:
    """dn/dlambda for dispersion calculations."""
    n = n0_sellmeier(A, B, C, lam)
    return float(-lam / n * np.sum(B * C / (lam ** 2 - C) ** 2))


def _ddn_dll(A: float, B: np.ndarray, C: np.ndarray, lam: float) -> float:
    """d^2n/dlambda^2 for dispersion calculations."""
    n = n0_sellmeier(A, B, C, lam)
    dndl = _dn_dl(A, B, C, lam)
    return float(
        dndl / lam - dndl ** 2 / n
        + 4.0 * lam ** 2 / n * np.sum(B * C / (lam ** 2 - C) ** 3)
    )


# ---------------------------------------------------------------------------
# Internal dispersion functions G (frequency domain)
# ---------------------------------------------------------------------------

def _G(B: np.ndarray, D: np.ndarray, w: float) -> np.ndarray:
    """G(w) = B / (1 - D * w^2)"""
    return B / (1.0 - D * w ** 2)


def _dG_dw(B: np.ndarray, D: np.ndarray, w: float) -> np.ndarray:
    g = _G(B, D, w)
    result = np.where(B == 0.0, 0.0, 2.0 * D / B * w * g ** 2)
    return result


def _ddG_dww(B: np.ndarray, D: np.ndarray, w: float) -> np.ndarray:
    g = _G(B, D, w)
    dg = _dG_dw(B, D, w)
    return np.where(B == 0.0, 0.0, 2.0 * D / B * (g ** 2 + 2.0 * w * g * dg))


def _dddG_dwww(B: np.ndarray, D: np.ndarray, w: float) -> np.ndarray:
    g = _G(B, D, w)
    dg = _dG_dw(B, D, w)
    ddg = _ddG_dww(B, D, w)
    return np.where(
        B == 0.0, 0.0,
        4.0 * D / B * (2.0 * g * dg + w * dg ** 2 + w * g * ddg),
    )


def _ddddG_dwwww(B: np.ndarray, D: np.ndarray, w: float) -> np.ndarray:
    g = _G(B, D, w)
    dg = _dG_dw(B, D, w)
    ddg = _ddG_dww(B, D, w)
    dddg = _dddG_dwww(B, D, w)
    return np.where(
        B == 0.0, 0.0,
        4.0 * D / B * (
            3.0 * dg ** 2 + 3.0 * g * ddg
            + 3.0 * w * dg * ddg + w * g * dddg
        ),
    )


def _dddddG_dwwwww(B: np.ndarray, D: np.ndarray, w: float) -> np.ndarray:
    g = _G(B, D, w)
    dg = _dG_dw(B, D, w)
    ddg = _ddG_dww(B, D, w)
    dddg = _dddG_dwww(B, D, w)
    ddddg = _ddddG_dwwww(B, D, w)
    return np.where(
        B == 0.0, 0.0,
        4.0 * D / B * (
            12.0 * dg * ddg + 4.0 * g * dddg
            + 3.0 * w * ddg ** 2 + 4.0 * w * dg * dddg
            + w * g * ddddg
        ),
    )


# ---------------------------------------------------------------------------
# Refractive index and derivatives in frequency domain
# ---------------------------------------------------------------------------

def _n0_w(A: float, B: np.ndarray, D: np.ndarray, w: float) -> float:
    return float(np.sqrt(A + np.sum(_G(B, D, w))))


def _dn_dw(A: float, B: np.ndarray, D: np.ndarray, w: float) -> float:
    return float(np.sum(_dG_dw(B, D, w)) / (2.0 * _n0_w(A, B, D, w)))


def _ddn_dww(A: float, B: np.ndarray, D: np.ndarray, w: float) -> float:
    n = _n0_w(A, B, D, w)
    dn = _dn_dw(A, B, D, w)
    return float((np.sum(_ddG_dww(B, D, w)) / 2.0 - dn ** 2) / n)


def _dddn_dwww(A: float, B: np.ndarray, D: np.ndarray, w: float) -> float:
    n = _n0_w(A, B, D, w)
    dn = _dn_dw(A, B, D, w)
    ddn = _ddn_dww(A, B, D, w)
    return float((np.sum(_dddG_dwww(B, D, w)) / 2.0 - 3.0 * dn * ddn) / n)


def _ddddn_dwwww(A: float, B: np.ndarray, D: np.ndarray, w: float) -> float:
    n = _n0_w(A, B, D, w)
    dn = _dn_dw(A, B, D, w)
    ddn = _ddn_dww(A, B, D, w)
    dddn = _dddn_dwww(A, B, D, w)
    return float(
        (np.sum(_ddddG_dwwww(B, D, w)) / 2.0 - 3.0 * ddn ** 2 - 4.0 * dn * dddn) / n
    )


def _dddddn_dwwwww(A: float, B: np.ndarray, D: np.ndarray, w: float) -> float:
    n = _n0_w(A, B, D, w)
    dn = _dn_dw(A, B, D, w)
    ddn = _ddn_dww(A, B, D, w)
    dddn = _dddn_dwww(A, B, D, w)
    ddddn = _ddddn_dwwww(A, B, D, w)
    return float(
        (np.sum(_dddddG_dwwwww(B, D, w)) / 2.0
         - 10.0 * ddn * dddn - 5.0 * dn * ddddn) / n
    )


# ---------------------------------------------------------------------------
# Wavelength / frequency helpers
# ---------------------------------------------------------------------------

def _l2w(lam: float) -> float:
    return float(twopi * c0 / lam)


def _w2l(w) -> float:
    return twopi * c0 / w


# ---------------------------------------------------------------------------
# Public API — material property functions
# ---------------------------------------------------------------------------

def n0(mat: str, lam: float, err: Optional[list] = None) -> float:
    """
    Calculate or retrieve the index of refraction.

    Uses Sellmeier coefficients when available, otherwise looks for
    discrete values in the database.

    Parameters
    ----------
    mat : str
        Material name (case-insensitive).
    lam : float
        Wavelength in metres.
    err : list, optional
        Single-element list ``[0]``; on return holds the error code.

    Returns
    -------
    float
        Linear refractive index.
    """
    A, B, C, err0 = sellmeiercoeff(mat, lam)

    if err0 != MatError.NOTFOUND:
        if err is not None:
            err[0] = err0
        return n0_sellmeier(A, B, C, lam)

    val, err0 = _discrete_dbs_val("n0", mat, lam)
    if val == 0.0 and err0 == MatError.NOERROR:
        val = 1.0

    _handle_err(err0, "n0", mat, lam, err)
    return val if val != 0.0 else 1.0


def n2I(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Retrieve the intensity-based nonlinear refractive index n2I [m^2/W]."""
    n2I_val, err0 = _discrete_dbs_val("n2I", mat, lam)

    if err0 == MatError.NOTFOUND:
        n2F_val, err0 = _discrete_dbs_val("n2F", mat, lam)
        if err0 != MatError.NOTFOUND:
            err1 = [0]
            n0_val = n0(mat, lam, err1)
            n2I_val = n2F_val / (eps0 * c0 * n0_val)

    _handle_err(err0, "n2I", mat, lam, err)
    return n2I_val


def n2F(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Retrieve the field-based nonlinear refractive index n2F [m^2/V^2]."""
    n2F_val, err0 = _discrete_dbs_val("n2F", mat, lam)

    if err0 == MatError.NOTFOUND:
        n2I_val, err0 = _discrete_dbs_val("n2I", mat, lam)
        if err0 != MatError.NOTFOUND:
            err1 = [0]
            n0_val = n0(mat, lam, err1)
            n2F_val = n2I_val * (eps0 * c0 * n0_val)

    _handle_err(err0, "n2F", mat, lam, err)
    return n2F_val


def alpha(mat: str, lam: float, err: Optional[list] = None) -> float:
    """
    Return the linear absorption coefficient [1/m].

    Uses spline interpolation of the Absorption array when available,
    falls back to discrete lookup.
    """
    err0 = MatError.NOERROR
    alpha_val = 0.0

    A_arr, err0 = _read_dbs_tag_array(mat, "Absorption")

    if err0 == MatError.NOERROR:
        lams, err0 = _read_dbs_tag_array(mat, "Absorption-lams")

    if err0 == MatError.NOERROR and A_arr is not None and lams is not None:
        if len(A_arr) != len(lams):
            _mat_error_handler(MatError.FILE_FORMAT, "alpha", mat, lam)
            return 0.0

        if lam < np.min(lams) or lam > np.max(lams):
            alpha_val = 1e100
            err0 = MatError.OUTOFRANGE
        else:
            b = np.zeros(len(A_arr))
            spline(lams, A_arr, b)
            alpha_val = float(seval(lam, lams, A_arr, b))

        if alpha_val < 0.0:
            alpha_val = 0.0
            err0 = MatError.BADVALUE
    else:
        alpha_val, err0 = _discrete_dbs_val("alpha", mat, lam)

    _handle_err(err0, "alpha", mat, lam, err)
    return alpha_val


def beta(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Retrieve the two-photon absorption coefficient."""
    val, err0 = _discrete_dbs_val("beta", mat, lam)
    _handle_err(err0, "beta", mat, lam, err)
    return val


def Vp(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Calculate the phase velocity [m/s]."""
    err0_list = [MatError.NOERROR]
    val = c0 / n0(mat, lam, err0_list)
    _handle_err(err0_list[0], "Vp", mat, lam, err)
    return float(val)


def k0_val(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Calculate the magnitude of the wavevector k0 = 2*pi*n0/lam."""
    err0_list = [MatError.NOERROR]
    val = 2.0 * pi * n0(mat, lam, err0_list) / lam
    _handle_err(err0_list[0], "k0", mat, lam, err)
    return float(val)


def Vg(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Calculate the group velocity as 1/k1 [m/s]."""
    err0_list = [MatError.NOERROR]
    val = 1.0 / k1(mat, lam, err0_list)
    _handle_err(err0_list[0], "Vg", mat, lam, err)
    return float(val)


def k1(mat: str, lam: float, err: Optional[list] = None) -> float:
    """
    Calculate dk/dw (first-order dispersion, inverse group velocity) [s/m].

    Uses frequency-domain Sellmeier when available, then falls back to
    discrete k1 or Vg values.
    """
    w = _l2w(lam)
    err0 = MatError.NOERROR
    val = 0.0

    A, B, D, err0 = sellmeiercoeff(mat, lam)
    if err0 != MatError.NOTFOUND:
        D = D / (twopi * c0) ** 2
        val = (_n0_w(A, B, D, w) + w * _dn_dw(A, B, D, w)) / c0
    else:
        val, err0 = _discrete_dbs_val("k1", mat, lam)

    if err0 == MatError.NOTFOUND:
        vg_val, err0 = _discrete_dbs_val("Vg", mat, lam)
        if err0 != MatError.NOTFOUND:
            val = 1.0 / vg_val

    _handle_err(err0, "k1", mat, lam, err)
    return float(val)


def k2(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Calculate d^2k/dw^2 (group velocity dispersion) [s^2/m]."""
    w = _l2w(lam)
    err0 = MatError.NOERROR
    val = 0.0

    A, B, D, err0 = sellmeiercoeff(mat, lam)
    if err0 != MatError.NOTFOUND:
        D = D / (twopi * c0) ** 2
        val = (2.0 * _dn_dw(A, B, D, w) + w * _ddn_dww(A, B, D, w)) / c0
    else:
        val, err0 = _discrete_dbs_val("k2", mat, lam)

    _handle_err(err0, "k2", mat, lam, err)
    return float(val)


def k3(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Calculate d^3k/dw^3 (third-order dispersion) [s^3/m]."""
    w = _l2w(lam)
    err0 = MatError.NOERROR
    val = 0.0

    A, B, D, err0 = sellmeiercoeff(mat, lam)
    if err0 != MatError.NOTFOUND:
        D = D / (twopi * c0) ** 2
        val = (3.0 * _ddn_dww(A, B, D, w) + w * _dddn_dwww(A, B, D, w)) / c0
    else:
        val, err0 = _discrete_dbs_val("k3", mat, lam)

    _handle_err(err0, "k3", mat, lam, err)
    return float(val)


def k4(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Calculate d^4k/dw^4 (fourth-order dispersion) [s^4/m]."""
    w = _l2w(lam)
    err0 = MatError.NOERROR
    val = 0.0

    A, B, D, err0 = sellmeiercoeff(mat, lam)
    if err0 != MatError.NOTFOUND:
        D = D / (twopi * c0) ** 2
        val = (4.0 * _dddn_dwww(A, B, D, w) + w * _ddddn_dwwww(A, B, D, w)) / c0
    else:
        val, err0 = _discrete_dbs_val("k4", mat, lam)

    _handle_err(err0, "k4", mat, lam, err)
    return float(val)


def k5(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Calculate d^5k/dw^5 (fifth-order dispersion) [s^5/m]."""
    w = _l2w(lam)
    err0 = MatError.NOERROR
    val = 0.0

    A, B, D, err0 = sellmeiercoeff(mat, lam)
    if err0 != MatError.NOTFOUND:
        D = D / (twopi * c0) ** 2
        val = (5.0 * _ddddn_dwwww(A, B, D, w) + w * _dddddn_dwwwww(A, B, D, w)) / c0
    else:
        val, err0 = _discrete_dbs_val("k5", mat, lam)

    _handle_err(err0, "k5", mat, lam, err)
    return float(val)


def GetKW(mat: str, W: np.ndarray, err: Optional[list] = None) -> np.ndarray:
    """
    Create an array of k values for the given angular frequencies.

    Parameters
    ----------
    mat : str
        Material name.
    W : ndarray
        Angular frequency array [rad/s].
    err : list, optional
        Single-element list for error code.

    Returns
    -------
    ndarray
        k(w) array.
    """
    err0 = MatError.NOERROR
    Kw = np.zeros(len(W))
    lams = _w2l(W)
    mid = len(lams) // 2

    A, B, C, err0 = sellmeiercoeff(mat, float(lams[mid]))

    if err0 != MatError.NOTFOUND:
        for i in range(len(lams)):
            Kw[i] = twopi * n0_sellmeier(A, B, C, abs(float(lams[i]))) / float(lams[i])

    _handle_err(err0, "K(w)", mat, float(lams[mid]), err)
    return Kw


def Tr(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Retrieve the Raman response parameter [s]."""
    val, err0 = _discrete_dbs_val("Raman-tr", mat, lam)

    if err0 == MatError.NOTFOUND:
        err0 = MatError.DEFAULTUSED
        val = 0.0

    _handle_err(err0, "Raman-tr", mat, lam, err)
    return val


# ---------------------------------------------------------------------------
# Plasma parameter getters
# ---------------------------------------------------------------------------

def _plasma_getter(param: str, mat: str, lam: float, err: Optional[list] = None) -> float:
    """Generic getter for plasma parameters."""
    val, err0 = _discrete_dbs_val(param, mat, lam)
    _handle_err(err0, param, mat, lam, err)
    return val


def GetPlasmaElectronMass(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Retrieve the plasma electron mass."""
    return _plasma_getter("Plasma-mass", mat, lam, err)


def GetPlasmaBandGap(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Retrieve the plasma band gap."""
    return _plasma_getter("Plasma-band_gap", mat, lam, err)


def GetPlasmaTrappingTime(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Retrieve the plasma trapping time."""
    return _plasma_getter("Plasma-trap_time", mat, lam, err)


def GetPlasmaCollisionTime(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Retrieve the plasma collision time."""
    return _plasma_getter("Plasma-collision_time", mat, lam, err)


def GetPlasmaMaximumDensity(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Retrieve the plasma maximum density."""
    return _plasma_getter("Plasma-max_density", mat, lam, err)


def GetPlasmaOrder(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Retrieve the plasma multi-photon ionisation order."""
    return _plasma_getter("Plasma-multi_order", mat, lam, err)


def GetPlasmaCrossSection(mat: str, lam: float, err: Optional[list] = None) -> float:
    """Retrieve the plasma multi-photon cross section."""
    return _plasma_getter("Plasma-multi_cross_section", mat, lam, err)
