"""
Coulomb interaction calculations for quantum wire simulations.

This module calculates the electron-hole, electron-electron, and hole-hole
collision integrals and other carrier-optical related calculations required
for the Semiconductor Bloch Equations in support of simulations of pulse
propagation through a quantum wire.

Author: Rahul R. Sah

"""

import numpy as np
from scipy.special import kv
from scipy.constants import e as e0, epsilon_0 as eps0, hbar as hbar_SI
from .usefulsubs import K03, theta
from numba import jit


# Physical constants
# e0: Elementary charge (C) - using scipy.constants.e
# eps0: Vacuum permittivity (F/m) - using scipy.constants.epsilon_0
# hbar: Reduced Planck constant (J·s) - using scipy.constants.hbar
# eV: Electron volt (J)
pi = np.pi
twopi = 2.0 * np.pi
hbar = hbar_SI
eV = 1.602176634e-19  # Electron volt in Joules
ii = 1j  # Imaginary unit

# Module-level state variables (matching Fortran module variables)
_LorentzDelta = False
_Chi1De = None
_Chi1Dh = None
_qe = None
_qh = None
_Veh0 = None
_Vee0 = None
_Vhh0 = None
_UnDel = None
_k3 = None
_Ceh = None
_Cee = None
_Chh = None

def InitializeCoulomb(y, ky, L, Delta0, me, mh, Ee, Eh, ge, gh, alphae, alphah, er, Qy, kkp, screened):
    """
    Initialize the coulomb module and all of its needed quantities.

    This is the main initialization function that sets up all module-level arrays
    required for Coulomb interaction calculations. It calls the various setup functions
    in the correct order, checking if arrays are already initialized to avoid
    redundant calculations.

    Parameters
    ----------
    y : ndarray
        Length coordinates of quantum wire (m), 1D array
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    L : float
        Length of the quantum wire (m)
    Delta0 : float
        Thickness of the quantum wire (m)
    me : float
        Effective electron mass (kg)
    mh : float
        Effective hole mass (kg)
    Ee : ndarray
        Electron energies (J), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    ge : float
        Inverse electron lifetime (Hz)
    gh : float
        Inverse hole lifetime (Hz)
    alphae : float
        Level separation between ground and 1st excited state for electrons (1/m)
    alphah : float
        Level separation between ground and 1st excited state for holes (1/m)
    er : float
        Background dielectric constant (dimensionless)
    Qy : ndarray
        Momentum difference array (1/m), 1D array
    kkp : ndarray
        Index mapping array, 2D integer array. Maps (k,q) indices to Qy indices.
    screened : bool
        Whether to use screened interactions (not used in initialization, but passed through)

    Returns
    -------
    None
        All arrays are stored as module-level variables (matching Fortran behavior).

    Notes
    -----
    The function initializes arrays in the following order:
    1. UnDel - inverse delta function array
    2. k3 - momentum conservation indexing array
    3. qe, qh - momentum difference arrays
    4. Ceh, Cee, Chh - many-body interaction arrays (requires k3 and UnDel)
    5. Veh0, Vee0, Vhh0 - unscreened Coulomb arrays
    6. Chi1De, Chi1Dh - susceptibility arrays (requires qe and qh)

    Each array is only computed if it hasn't been initialized yet, allowing
    for incremental initialization or re-initialization with different parameters.
    """
    global _UnDel, _k3, _qe, _qh, _Ceh, _Cee, _Chh, _Veh0, _Vee0, _Vhh0, _Chi1De, _Chi1Dh

    # Make the inverse delta function
    if _UnDel is None:
        _UnDel = MakeUnDel(ky)

    # Make the 3D index array for k3 = k1 + k2 - k4
    if _k3 is None:
        _k3 = MakeK3(ky)

    # Make the q, qe, and qh arrays
    if _qe is None or _qh is None:
        _qe, _qh = MakeQs(ky, alphae, alphah)

    # Calculate the 3D many-body interaction arrays
    # Note: CalcMBArrays needs k3 and UnDel, which we just ensured are initialized
    if _Ceh is None or _Cee is None or _Chh is None:
        _Ceh, _Cee, _Chh = CalcMBArrays(ky, Ee, Eh, ge, gh, _k3, _UnDel)

    # Calculate the unscreened Coulomb collision arrays
    if _Veh0 is None or _Vee0 is None or _Vhh0 is None:
        _Veh0, _Vee0, _Vhh0 = CalcCoulombArrays(y, ky, er, alphae, alphah, L, Delta0, Qy, kkp)

    # Calculate the Chi1D array for Coulomb screening
    # Note: CalcChi1D needs qe and qh, which we just ensured are initialized
    if _Chi1De is None or _Chi1Dh is None:
        _Chi1De, _Chi1Dh = CalcChi1D(ky, alphae, alphah, Delta0, er, me, mh, _qe, _qh)

# JIT-compatible K0 approximation for modified Bessel function
# nopython friendly version of K03 which is in usefulsubs.py
@jit(nopython=True, cache=True)
def _K03_jit(x):
    """
    JIT-compatible approximation of modified Bessel function K0(x).

    Uses asymptotic expansion for large x and series expansion for small x.
    This is a simplified version that works with numba nopython mode.
    """
    if x > 100.0:
        return 0.0

    if x < 1e-10:
        # K0(x) ~ -ln(x/2) - gamma for small x
        return 23.0  # Approximate large value for very small x

    if x < 2.0:
        # Series expansion for small x
        x2 = x * x
        result = -np.log(0.5 * x) - 0.5772156649015329
        term = 1.0
        for k in range(1, 20):
            term = term * 0.25 * x2 / (k * k)
            result = result + term * (1.0 / k - 0.5772156649015329)
            if abs(term) < 1e-15:
                break
        return result
    else:
        # Asymptotic expansion for large x: K0(x) ~ sqrt(pi/(2x)) * exp(-x)
        return np.sqrt(np.pi / (2.0 * x)) * np.exp(-x)

@jit(nopython=True, cache=True)
def _Vint_jit(Qyk, y, alphae, alphah, Delta0, N1, N2, Ny):
    """
    JIT-compiled version of Vint calculation.

    This is the core computation loop that can be JIT-compiled.
    """
    Vint_val = 0.0

    # Pre-compute arrays (need to pass as parameters for JIT)
    aey2 = np.zeros(Ny)
    ahy2 = np.zeros(Ny)
    for i in range(Ny):
        aey2[i] = (alphae * y[i]) ** 2
        ahy2[i] = (alphah * y[i]) ** 2

    # Minimum momentum and actual momentum difference
    kmin = (alphae + alphah) / 4.0
    dk = max(abs(Qyk), kmin)

    # Multiplication constant
    multconst = alphae * alphah / np.pi * (y[1] - y[0]) ** 2

    # Double loop over integration region
    for i in range(N1, N2 + 1):
        for j in range(N1, N2 + 1):
            # Distance in y-direction with thickness contribution
            r_sq = (y[i] - y[j]) ** 2 + Delta0 ** 2
            r = np.sqrt(r_sq)

            # Exponential factor from wavefunction overlap
            exp_factor = np.exp(-aey2[i] - ahy2[j])

            # Modified Bessel function K0 for Coulomb interaction
            k0_arg = dk * r
            k0_val = _K03_jit(k0_arg)

            Vint_val += exp_factor * multconst * k0_val

    return Vint_val


def Vint(Qyk, y, alphae, alphah, Delta0):
    """
    Calculate the interaction integral for Coulomb potential.

    Computes the integral over spatial coordinates for the Coulomb interaction
    between particles with level separations alphae and alphah. This is used to
    calculate the unscreened Coulomb collision arrays.

    Parameters
    ----------
    Qyk : float
        Momentum difference (1/m)
    y : ndarray
        Length coordinates of quantum wire (m), 1D array
    alphae : float
        Level separation between ground and 1st excited state for electrons (1/m)
    alphah : float
        Level separation between ground and 1st excited state for holes (1/m)
    Delta0 : float
        Thickness of the quantum wire (m)

    Returns
    -------
    float
        Interaction integral value (dimensionless)

    Notes
    -----
    The integration is performed over a central region of the y array
    (from Ny/4 to 3*Ny/4) to focus on the relevant quantum wire region.
    Uses the modified Bessel function K0 to represent the Coulomb interaction.

    Uses JIT compilation for performance with automatic fallback to pure Python.
    """
    Ny = len(y)
    N1 = Ny // 4
    N2 = 3 * Ny // 4

    # Try JIT-compiled version first
    try:
        result = _Vint_jit(Qyk, y, alphae, alphah, Delta0, N1, N2, Ny)
        if not hasattr(Vint, '_jit_used_printed'):
            print("Vint: Using JIT-compiled version")
            Vint._jit_used_printed = True
        return result
    except Exception:
        # Fallback to original implementation
        if not hasattr(Vint, '_fallback_used_printed'):
            print("Vint: JIT compilation failed, using fallback (slower)")
            Vint._fallback_used_printed = True

        Vint_val = 0.0
        # Pre-compute arrays
        aey2 = (alphae * y) ** 2
        ahy2 = (alphah * y) ** 2

        # Minimum momentum and actual momentum difference
        kmin = (alphae + alphah) / 4.0
        dk = max(abs(Qyk), kmin)

        # Multiplication constant
        multconst = alphae * alphah / pi * (y[1] - y[0]) ** 2

        # Double loop over integration region
        for i in range(N1, N2 + 1):
            for j in range(N1, N2 + 1):
                # Distance in y-direction with thickness contribution
                r_sq = (y[i] - y[j]) ** 2 + Delta0 ** 2
                r = np.sqrt(r_sq)

                # Exponential factor from wavefunction overlap
                exp_factor = np.exp(-aey2[i] - ahy2[j])

                # Modified Bessel function K0 for Coulomb interaction
                k0_arg = dk * r
                k0_val = K03(k0_arg)

                Vint_val += exp_factor * multconst * k0_val

        return Vint_val

def Vehint(k, q, y, ky, alphae, alphah, Delta0):
    """
    Calculate electron-hole interaction integral.

    Computes the interaction integral between an electron at momentum index k
    and a hole at momentum index q. This is similar to Vint but uses the
    momentum difference from the ky array directly.

    Parameters
    ----------
    k : int
        Electron momentum index. Note: Fortran uses 1-based indexing, so if
        calling from Python with 0-based indices, pass k+1 and q+1, or adjust
        the indices before calling.
    q : int
        Hole momentum index. Note: Fortran uses 1-based indexing, so if
        calling from Python with 0-based indices, pass k+1 and q+1, or adjust
        the indices before calling.
    y : ndarray
        Length coordinates of quantum wire (m), 1D array
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    alphae : float
        Level separation between ground and 1st excited state for electrons (1/m)
    alphah : float
        Level separation between ground and 1st excited state for holes (1/m)
    Delta0 : float
        Thickness of the quantum wire (m)

    Returns
    -------
    float
        Electron-hole interaction integral value (dimensionless)

    Notes
    -----
    The function uses the momentum difference |ky(k) - ky(q)| to compute
    the interaction. The integration is performed over a central region
    of the y array (from Ny/4 to 3*Ny/4).

    Note on indexing: The Fortran version uses 1-based indexing. If k and q
    are passed as 1-based indices (matching Fortran), they will be converted
    to 0-based for array access. If passed as 0-based, subtract 1 is not needed.
    """
    Vehint_val = 0.0
    Ny = len(y)

    # Pre-compute arrays
    aey2 = (alphae * y) ** 2
    ahy2 = (alphah * y) ** 2

    # Minimum momentum and actual momentum difference
    # Convert from 1-based Fortran indexing to 0-based Python
    # If k and q are passed as 1-based (matching Fortran), subtract 1
    # If they're already 0-based, don't subtract
    # We'll assume they're 1-based to match Fortran behavior
    k_idx = k - 1 if k > 0 else 0
    q_idx = q - 1 if q > 0 else 0
    kmin = (alphae + alphah) / 4.0
    dk = max(abs(ky[k_idx] - ky[q_idx]), kmin)

    # Multiplication constant
    multconst = alphae * alphah / pi * (y[1] - y[0]) ** 2

    # Integration region: central portion of the array
    N1 = Ny // 4
    N2 = 3 * Ny // 4

    # Double loop over integration region
    for i in range(N1, N2 + 1):
        for j in range(N1, N2 + 1):
            # Distance in y-direction with thickness contribution
            r_sq = (y[i] - y[j]) ** 2 + Delta0 ** 2
            r = np.sqrt(r_sq)

            # Exponential factor from wavefunction overlap
            exp_factor = np.exp(-aey2[i] - ahy2[j])

            # Modified Bessel function K0 for Coulomb interaction
            k0_arg = dk * r
            k0_val = K03(k0_arg)

            Vehint_val += exp_factor * multconst * k0_val

    return Vehint_val


def CalcCoulombArrays(y, ky, er, alphae, alphah, L, Delta0, Qy, kkp,
                      ReadArrays=False, ScrewThis=False):
    """
    Construct the unscreened Coulomb collision arrays.

    Calculates the electron-hole (Veh0), electron-electron (Vee0), and
    hole-hole (Vhh0) unscreened Coulomb interaction matrices for a quantum wire.
    These arrays are used in the Semiconductor Bloch Equations to compute
    many-body interactions.

    Parameters
    ----------
    y : ndarray
        Length coordinates of quantum wire (m), 1D array
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    er : float
        Background dielectric constant (dimensionless)
    alphae : float
        Level separation between ground and 1st excited state for electrons (1/m)
    alphah : float
        Level separation between ground and 1st excited state for holes (1/m)
    L : float
        Length of the quantum wire (m)
    Delta0 : float
        Thickness of the quantum wire (m)
    Qy : ndarray
        Momentum difference array (1/m), 1D array
    kkp : ndarray
        Index mapping array, 2D integer array. Maps (k,q) indices to Qy indices.
        Values >= 0 indicate valid mappings, < 0 indicate invalid.
    ReadArrays : bool, optional
        If True, read pre-calculated arrays from files (not implemented).
        Default is False.
    ScrewThis : bool, optional
        If True, skip calculation and return zero arrays. Default is False.

    Returns
    -------
    tuple of ndarray
        (Veh0, Vee0, Vhh0) where:
        - Veh0: Electron-hole interaction matrix (J), shape (N, N)
        - Vee0: Electron-electron interaction matrix (J), shape (N, N)
        - Vhh0: Hole-hole interaction matrix (J), shape (N, N)
        where N = len(ky)

    Notes
    -----
    The function computes the interaction integrals for each momentum difference
    in Qy, then maps these to the (k,q) grid using the kkp index array.
    The calculation uses the Vint function to compute the spatial integrals.

    The interaction matrices are computed as:
    - Veh0(k,q) = (e0^2 / (2π * ε0 * er * L)) * Vint(Qy[kkp(k,q)], y, alphae, alphah, Delta0)
    - Vee0(k,q) = (e0^2 / (2π * ε0 * er * L)) * Vint(Qy[kkp(k,q)], y, alphae, alphae, Delta0)
    - Vhh0(k,q) = (e0^2 / (2π * ε0 * er * L)) * Vint(Qy[kkp(k,q)], y, alphah, alphah, Delta0)
    """
    N = len(ky)
    NQ = len(Qy)

    # Initialize arrays
    Veh0 = np.zeros((N, N))
    Vee0 = np.zeros((N, N))
    Vhh0 = np.zeros((N, N))

    if ReadArrays:
        # File I/O not implemented - would need to implement ReadIt equivalent
        # For now, raise NotImplementedError or return zeros
        print("Warning: ReadArrays=True not implemented, returning zero arrays")
        return Veh0, Vee0, Vhh0

    if ScrewThis:
        # Do nothing, return zero arrays
        return Veh0, Vee0, Vhh0

    print("Calculating Coulomb Arrays")

    # Pre-compute interaction integrals for each Qy value
    eh = np.zeros(NQ)
    ee = np.zeros(NQ)
    hh = np.zeros(NQ)

    # Pre-factor for Coulomb interaction
    prefactor = e0 ** 2 / (twopi * eps0 * er * L)

    # Calculate integrals for each momentum difference
    for k in range(NQ):
        if k % 10 == 0:  # Print every 10th value
            print(f"  Progress: {k}/{NQ} ({100*k/NQ:.1f}%)")
        eh[k] = prefactor * Vint(Qy[k], y, alphae, alphah, Delta0)
        ee[k] = prefactor * Vint(Qy[k], y, alphae, alphae, Delta0)
        hh[k] = prefactor * Vint(Qy[k], y, alphah, alphah, Delta0)

    # Map integrals to (k,q) grid using kkp index array
    # Note: kkp is expected to use 1-based Fortran indexing
    # In Python, arrays are 0-based, so we need to handle the conversion
    for k in range(N):
        for q in range(N):
            # kkp is 1-based in Fortran, so kkp(k+1, q+1) in Fortran
            # becomes kkp[k, q] in Python if kkp is already 0-based
            # But if kkp values are indices into Qy, they might be 0-based or 1-based
            # We'll assume kkp values are 0-based indices into Qy array
            kkp_idx = kkp[k, q]
            if kkp_idx >= 0 and kkp_idx < NQ:
                Veh0[k, q] = eh[kkp_idx]
                Vee0[k, q] = ee[kkp_idx]
                Vhh0[k, q] = hh[kkp_idx]

    print("Finished Calculating Unscreened Coulomb Arrays")

    # Store as module-level variables (matching Fortran)
    global _Veh0, _Vee0, _Vhh0
    _Veh0 = Veh0
    _Vee0 = Vee0
    _Vhh0 = Vhh0

    return Veh0, Vee0, Vhh0


def GaussDelta(a, b):
    """
    (Was commented out in the Fortran code and had implications on the CalcMBArrays function)
    Gaussian delta function approximation.

    Computes a Gaussian approximation to the delta function, used in
    many-body interaction calculations when LorentzDelta is False.

    Parameters
    ----------
    a : float
        Energy difference argument (J)
    b : float
        Broadening parameter (J)

    Returns
    -------
    float
        Gaussian delta function value (1/J)

    Notes
    -----
    The function implements: 1 / (sqrt(pi) * b) * exp(-(a/b)^2)
    This is a smooth approximation to the delta function with width b.
    """
    if abs(b) < 1e-300:
        return 0.0
    return 1.0 / (np.sqrt(pi) * b) * np.exp(-(a / b) ** 2)


def MakeK3(ky):
    """
    Construct the k3 indexing array for momentum conservation.

    Creates a 3D array k3 where k3(k1, k2, k4) represents the index k3
    that satisfies the momentum conservation relation: k1 + k2 = k3 + k4.
    This is used in many-body interaction calculations.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array

    Returns
    -------
    ndarray
        3D integer array of shape (N, N, N) where N = len(ky).
        Values are 1-based indices (matching Fortran) if valid,
        or 0 if the momentum combination is invalid (out of bounds).

    Notes
    -----
    The array is computed as: k3i = k1 + k2 - k4
    If k3i is outside the valid range [1, N], it is set to 0.
    The returned array uses 1-based indexing to match Fortran behavior,
    so valid indices are in the range [1, N], with 0 indicating invalid.
    """
    N = len(ky)
    k3 = np.zeros((N, N, N), dtype=np.int32)

    # Fortran uses 1-based indexing, so k1, k2, k4 go from 1 to N
    # Python uses 0-based, so we iterate from 0 to N-1 but store 1-based values
    for k4 in range(N):
        for k2 in range(N):
            for k1 in range(N):
                # Compute k3 index: k3i = k1 + k2 - k4
                # Convert to 1-based for calculation
                k3i = (k1 + 1) + (k2 + 1) - (k4 + 1)

                # Check if valid (1-based range: 1 to N)
                if k3i < 1 or k3i > N:
                    k3i = 0

                # Store 1-based index (0 indicates invalid)
                k3[k1, k2, k4] = k3i

    # Store as module-level variable (matching Fortran)
    global _k3
    _k3 = k3

    return k3


def MakeQs(ky, ae, ah):
    """
    Construct the qe and qh momentum difference arrays.

    Creates arrays representing the momentum differences between states,
    with minimum values set by the level separation parameters ae and ah.
    These arrays are used in screening and interaction calculations.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    ae : float
        Level separation parameter for electrons (1/m)
    ah : float
        Level separation parameter for holes (1/m)

    Returns
    -------
    tuple of ndarray
        (qe, qh) where:
        - qe: Electron momentum difference array (1/m), shape (N, N)
        - qh: Hole momentum difference array (1/m), shape (N, N)
        where N = len(ky)

    Notes
    -----
    The arrays are computed as:
    - qe(k1, k2) = max(|ky(k2) - ky(k1)|, ae/2)
    - qh(k1, k2) = max(|ky(k2) - ky(k1)|, ah/2)
    This ensures a minimum momentum difference to avoid singularities.
    """
    Nk = len(ky)

    # Initialize arrays
    qe = np.zeros((Nk, Nk))
    qh = np.zeros((Nk, Nk))

    # Compute qe array
    for k2 in range(Nk):
        for k1 in range(Nk):
            qe[k1, k2] = max(abs(ky[k2] - ky[k1]), ae / 2.0)

    # Compute qh array
    for k2 in range(Nk):
        for k1 in range(Nk):
            qh[k1, k2] = max(abs(ky[k2] - ky[k1]), ah / 2.0)

    # Store as module-level variables (matching Fortran)
    global _qe, _qh
    _qe = qe
    _qh = qh

    return qe, qh


def MakeUnDel(ky):
    """
    Construct the UnDel (1 - delta) array.

    Creates an array representing (1 - delta_ij) where delta_ij is the
    Kronecker delta. This is used to exclude self-interaction terms
    in many-body calculations.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array

    Returns
    -------
    ndarray
        2D array of shape (N+1, N+1) where N = len(ky).
        Values are 1.0 except:
        - Row 0 and column 0 are all 0.0
        - Diagonal elements (i, i) for i=1..N are 0.0

    Notes
    -----
    The array has an extra row and column (index 0) compared to the
    momentum array size, matching the Fortran implementation which uses
    0-based indexing for the first row/column.
    """
    N = len(ky)
    # Create array with extra row/column for index 0
    UnDel = np.ones((N + 1, N + 1))

    # Set row 0 and column 0 to zero
    UnDel[0, :] = 0.0
    UnDel[:, 0] = 0.0

    # Set diagonal elements (1-based indices 1..N) to zero
    # In Python 0-based, these are indices 1..N
    for i in range(1, N + 1):
        UnDel[i, i] = 0.0

    # Store as module-level variable (matching Fortran)
    global _UnDel
    _UnDel = UnDel

    return UnDel


def SetLorentzDelta(boolean):
    """
    Set the LorentzDelta flag for many-body array calculations.

    This function sets a module-level flag that determines whether to use
    Lorentzian broadening (True) or Gaussian delta function (False) in
    the CalcMBArrays function.

    Parameters
    ----------
    boolean : bool
        If True, use Lorentzian broadening. If False, use Gaussian delta function.

    Notes
    -----
    This matches the Fortran interface where SetLorentzDelta sets a module-level
    variable. The flag affects the behavior of CalcMBArrays when called without
    explicitly specifying the LorentzDelta parameter.
    """
    global _LorentzDelta
    _LorentzDelta = bool(boolean)


def CalcMBArrays(ky, Ee, Eh, ge, gh, k3, UnDel, LorentzDelta=None):
    """
    Calculate the many-body interaction arrays.

    Computes the electron-hole (Ceh), electron-electron (Cee), and
    hole-hole (Chh) many-body interaction arrays used in the Semiconductor
    Bloch Equations. These arrays represent the collision integrals for
    carrier-carrier interactions.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    Ee : ndarray
        Electron energies (J), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    ge : float
        Electron inverse lifetime (Hz)
    gh : float
        Hole inverse lifetime (Hz)
    k3 : ndarray
        3D indexing array from MakeK3, shape (N, N, N)
    UnDel : ndarray
        2D array from MakeUnDel, shape (N+1, N+1)
    LorentzDelta : bool, optional
        If True, use Lorentzian broadening. If False, use Gaussian delta function.
        If None (default), uses the module-level value set by SetLorentzDelta.

    Returns
    -------
    tuple of ndarray
        (Ceh, Cee, Chh) where:
        - Ceh: Electron-hole interaction array (1/J), shape (N+1, N+1, N+1)
        - Cee: Electron-electron interaction array (1/J), shape (N+1, N+1, N+1)
        - Chh: Hole-hole interaction array (1/J), shape (N+1, N+1, N+1)
        where N = len(ky). The arrays use 0-based indexing for the first dimension
        to match Fortran's 0:N range.

    Notes
    -----
    The function computes interaction rates using either:
    - Lorentzian broadening: 2*gamma * UnDel / ((E_diff)^2 + (hbar*gamma)^2)
    - Gaussian delta function: (2π/hbar) * UnDel * GaussDelta(E_diff, hbar*gamma)

    The arrays are indexed with 0-based first index (0..N) to match Fortran's
    allocation of arrays with bounds (0:N, 0:N, 0:N).
    """
    N = len(ky)

    # Compute average inverse lifetime and broadening parameters
    geh = (ge + gh) / 2.0
    hge2 = (hbar * ge) ** 2
    hgh2 = (hbar * gh) ** 2
    hgeh2 = (hbar * geh) ** 2

    # Allocate arrays with 0-based first index (0..N)
    Ceh = np.zeros((N + 1, N + 1, N + 1))
    Cee = np.zeros((N + 1, N + 1, N + 1))
    Chh = np.zeros((N + 1, N + 1, N + 1))

    # Use module-level value if not explicitly provided
    if LorentzDelta is None:
        LorentzDelta = _LorentzDelta

    if LorentzDelta:
        # Use Lorentzian broadening
        for k1 in range(N):
            for k2 in range(N):
                for k4 in range(N):
                    # Convert to 1-based for array access (Fortran uses 1-based)
                    k1_1b = k1 + 1
                    k2_1b = k2 + 1
                    k4_1b = k4 + 1

                    # Veh(k1,k2,k30,k4)
                    # k3 array: k3[k4, k2, k1] accesses k3(k4+1, k2+1, k1+1) in Fortran
                    # k3 stores 1-based indices
                    k30_1b = k3[k4, k2, k1]
                    if k30_1b > 0:
                        k30 = k30_1b - 1  # Convert to 0-based for Ee/Eh array access
                        E_diff = Ee[k1] + Eh[k2] - Eh[k30] - Ee[k4]
                        # UnDel and Ceh use 1-based indexing in Fortran
                        Ceh[k1_1b, k2_1b, k4_1b] = (2.0 * geh * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] /
                                                   (E_diff ** 2 + hgeh2))

                    # Vee(k1,k2,k30,k4)
                    k30_1b = k3[k1, k2, k4]
                    if k30_1b > 0:
                        k30 = k30_1b - 1
                        E_diff = Ee[k1] + Ee[k2] - Ee[k30] - Ee[k4]
                        Cee[k1_1b, k2_1b, k4_1b] = (2.0 * ge * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] /
                                                   (E_diff ** 2 + hge2))

                    # Vhh(k1,k2,k30,k4)
                    k30_1b = k3[k1, k2, k4]
                    if k30_1b > 0:
                        k30 = k30_1b - 1
                        E_diff = Eh[k1] + Eh[k2] - Eh[k30] - Eh[k4]
                        Chh[k1_1b, k2_1b, k4_1b] = (2.0 * gh * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] /
                                                   (E_diff ** 2 + hgh2))
    else:
        # Use Gaussian delta function
        for k1 in range(N):
            for k2 in range(N):
                for k4 in range(N):
                    # Convert to 1-based for array access (Fortran uses 1-based)
                    k1_1b = k1 + 1
                    k2_1b = k2 + 1
                    k4_1b = k4 + 1

                    # Veh(k1,k2,k30,k4)
                    k30_1b = k3[k4, k2, k1]
                    if k30_1b > 0:
                        k30 = k30_1b - 1  # Convert to 0-based for Ee/Eh array access
                        E_diff = Ee[k1] + Eh[k2] - Eh[k30] - Ee[k4]
                        # UnDel and Ceh use 1-based indexing in Fortran
                        Ceh[k1_1b, k2_1b, k4_1b] = (twopi / hbar * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] *
                                                   GaussDelta(E_diff, hbar * geh))

                    # Vee(k1,k2,k30,k4)
                    k30_1b = k3[k1, k2, k4]
                    if k30_1b > 0:
                        k30 = k30_1b - 1
                        E_diff = Ee[k1] + Ee[k2] - Ee[k30] - Ee[k4]
                        Cee[k1_1b, k2_1b, k4_1b] = (twopi / hbar * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] *
                                                   GaussDelta(E_diff, hbar * ge))

                    # Vhh(k1,k2,k30,k4)
                    k30_1b = k3[k1, k2, k4]
                    if k30_1b > 0:
                        k30 = k30_1b - 1
                        E_diff = Eh[k1] + Eh[k2] - Eh[k30] - Eh[k4]
                        Chh[k1_1b, k2_1b, k4_1b] = (twopi / hbar * UnDel[k1_1b, k4_1b] *
                                                   UnDel[k2_1b, k30_1b] *
                                                   GaussDelta(E_diff, hbar * gh))

    # Store as module-level variables (matching Fortran)
    global _Ceh, _Cee, _Chh
    _Ceh = Ceh
    _Cee = Cee
    _Chh = Chh

    return Ceh, Cee, Chh

####################################################################################################
########################## Coulomb Screeing Calculations ############################################
####################################################################################################
def GetChi1Dqw(alphae, alphah, Delta0, epsr, game, gamh, ky, Ee, Eh, ne, nh, qq, w):
    """
    Calculate the 1D quantum wire susceptibility (chi) for a given momentum and frequency.

    Computes the real and imaginary parts of the susceptibility chi(q, w) for a quantum wire,
    which describes the response of the system to external perturbations. This is used in
    screening calculations.

    Parameters
    ----------
    alphae : float
        Level separation parameter for electrons (1/m)
    alphah : float
        Level separation parameter for holes (1/m)
    Delta0 : float
        Thickness of the quantum wire (m)
    L : float
        Length of the quantum wire (m)
    epsr : float
        Background dielectric constant (dimensionless)
    game : ndarray
        Electron inverse lifetime array (Hz), 1D array
    gamh : ndarray
        Hole inverse lifetime array (Hz), 1D array
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    Ee : ndarray
        Electron energies (J), 1D array
    Eh : ndarray
        Hole energies (J), 1D array
    ne : ndarray
        Electron population array, 1D array
    nh : ndarray
        Hole population array, 1D array
    qq : float
        Momentum value (1/m)
    w : float
        Frequency (rad/s)

    Returns
    -------
    tuple of float
        (chir, chii) where:
        - chir: Real part of susceptibility (dimensionless)
        - chii: Imaginary part of susceptibility (dimensionless)

    Notes
    -----
    The function computes the susceptibility by summing over momentum states,
    taking into account the energy differences and broadening due to lifetimes.
    The calculation uses the K03 Bessel function for the interaction kernel.
    """
    Nk = len(ky)
    dk = ky[1] - ky[0]

    # Compute broadening parameters
    hge = hbar * game
    hgh = hbar * gamh

    # Ensure minimum broadening
    hge = np.maximum(hge, 1e-4 * eV)
    hgh = np.maximum(hgh, 1e-4 * eV)

    # Minimum momentum values
    qmine = alphae / 2.0
    qminh = alphah / 2.0

    # Effective radii
    Re = np.sqrt((2.0 / alphae) ** 2 + Delta0 ** 2)
    Rh = np.sqrt((2.0 / alphah) ** 2 + Delta0 ** 2)

    # Interaction strength
    beta = e0 ** 2 / (4 * pi * eps0 * epsr)

    # ql appears to be qq (momentum value)
    ql = abs(qq)
    Ke = K03(max(ql, qminh) * Re) * beta * 2
    Kh = K03(max(ql, qmine) * Rh) * beta * 2

    # Initialize
    chi = 0.0 + 0.0j
    # Convert momentum to index (Fortran uses nint, which rounds to nearest)
    q = int(round(qq / dk))

    # Sum over momentum states
    # Fortran: do k=max(1,1-q), min(Nk,Nk-q)
    # Convert to 0-based: k goes from max(0, -q) to min(Nk-1, Nk-1-q)
    # But need to ensure k+q is valid
    k_start = max(0, -q)
    k_end = min(Nk, Nk - q)

    for k in range(k_start, k_end):
        k_plus_q = k + q
        if 0 <= k_plus_q < Nk:
            # Electron contribution
            # Note: Fortran uses 1-based indexing, so ne(k) means ne[k-1] in Python
            # Here k is already 0-based
            denom_e = hbar * w - Ee[k_plus_q] + Ee[k] + ii * (hge[k_plus_q] + hge[k])
            if abs(denom_e) > 1e-300:
                chi -= Ke / pi * (ne[k] - ne[k_plus_q]) / denom_e * dk

            # Hole contribution
            denom_h = hbar * w - Eh[k_plus_q] + Eh[k] + ii * (hgh[k_plus_q] + hgh[k])
            if abs(denom_h) > 1e-300:
                chi -= Kh / pi * (nh[k] - nh[k_plus_q]) / denom_h * dk

    chir = np.real(chi)
    chii = np.imag(chi)

    return chir, chii


def GetEps1Dqw(alphae, alphah, Delta0, epsr, me, mh, n1D, q, w):
    """
    Calculate the 1D quantum wire dielectric function epsilon(q, w).

    Computes the real and imaginary parts of the dielectric function for a quantum wire,
    which describes how the system responds to electromagnetic fields. This is used in
    screening calculations.

    Parameters
    ----------
    alphae : float
        Level separation parameter for electrons (1/m)
    alphah : float
        Level separation parameter for holes (1/m)
    Delta0 : float
        Thickness of the quantum wire (m)
    L : float
        Length of the quantum wire (m)
    epsr : float
        Background dielectric constant (dimensionless)
    me : float
        Electron effective mass (kg)
    mh : float
        Hole effective mass (kg)
    n1D : float
        1D carrier density (1/m)
    q : float
        Momentum (1/m)
    w : float
        Frequency (rad/s)

    Returns
    -------
    tuple of float
        (epr, epi) where:
        - epr: Real part of dielectric function (dimensionless)
        - epi: Imaginary part of dielectric function (dimensionless)

    Notes
    -----
    The function computes the dielectric function using the Lindhard formula
    for a 1D system, with contributions from both electrons and holes.
    """
    # Minimum momentum values
    qmine = alphae / 2.0
    qminh = alphah / 2.0

    # Effective radii
    Re = np.sqrt((2.0 / alphae) ** 2 + Delta0 ** 2)
    Rh = np.sqrt((2.0 / alphah) ** 2 + Delta0 ** 2)

    # Interaction strength
    beta = e0 ** 2 / (4 * pi * eps0 * epsr)

    # Ensure minimum momentum
    if abs(q) < 1.0:
        q = 1.0
    ql = abs(q)

    # Coefficients
    Ce = 2 * beta * me / pi / hbar ** 2 / ql
    Ch = 2 * beta * mh / pi / hbar ** 2 / ql

    # Interaction kernels
    Ke = K03(max(ql, qminh) * Re)
    Kh = K03(max(ql, qmine) * Rh)

    # Plasma frequencies
    OhmEp = hbar * ql / 2.0 / me * abs(ql + pi * n1D)
    OhmEm = hbar * ql / 2.0 / me * abs(ql - pi * n1D)
    OhmHp = hbar * ql / 2.0 / mh * abs(ql + pi * n1D)
    OhmHm = hbar * ql / 2.0 / mh * abs(ql - pi * n1D)

    # Real part of dielectric function
    epr = 1.0 - Ce * Ke * np.log(abs((w ** 2 - OhmEm ** 2) / (w ** 2 - OhmEp ** 2))) - \
          Ch * Kh * np.log(abs((w ** 2 - OhmHm ** 2) / (w ** 2 - OhmHp ** 2)))

    # Imaginary part (non-zero only in certain frequency ranges)
    epi = 0.0
    if min(OhmEm, OhmEp) < w < max(OhmEm, OhmEp):
        if min(OhmHm, OhmHp) < w < max(OhmHm, OhmHp):
            epi = -Ce * Ke - Ch * Kh

    # Check for NaN
    if np.isnan(epr) or np.isnan(epi):
        print(f"NAN in EpsL(q,w) at (q,w) = {q}, {w}")
        print(f"Re[EpsL(q,w)] = {epr}")
        print(f"Im[EpsL(q,w)] = {epi}")

    return epr, epi


def CalcChi1D(ky, alphae, alphah, Delta0, epsr, me, mh, qe, qh):
    """
    Calculate the 1D susceptibility arrays Chi1De and Chi1Dh.

    Computes the electron and hole susceptibility matrices used in screening calculations.
    These arrays are stored as module-level variables for use in other functions.

    Parameters
    ----------
    ky : ndarray
        Momentum coordinates of quantum wire (1/m), 1D array
    alphae : float
        Level separation parameter for electrons (1/m)
    alphah : float
        Level separation parameter for holes (1/m)
    Delta0 : float
        Thickness of the quantum wire (m)
    epsr : float
        Background dielectric constant (dimensionless)
    me : float
        Electron effective mass (kg)
    mh : float
        Hole effective mass (kg)
    qe : ndarray
        Electron momentum difference array from MakeQs, shape (N, N)
    qh : ndarray
        Hole momentum difference array from MakeQs, shape (N, N)

    Returns
    -------
    tuple of ndarray
        (Chi1De, Chi1Dh) where:
        - Chi1De: Electron susceptibility array, shape (N, N)
        - Chi1Dh: Hole susceptibility array, shape (N, N)
        where N = len(ky)

    Notes
    -----
    The arrays are also stored as module-level variables _Chi1De and _Chi1Dh
    for compatibility with Fortran module behavior.
    """
    global _Chi1De, _Chi1Dh

    N = len(ky)

    # Initialize arrays
    Chi1De = np.zeros((N, N))
    Chi1Dh = np.zeros((N, N))

    # Effective radii
    Re = np.sqrt((2.0 / alphae) ** 2 + Delta0 ** 2)
    Rh = np.sqrt((2.0 / alphah) ** 2 + Delta0 ** 2)

    # Compute Chi1De (electron susceptibility)
    for k2 in range(N):
        for k1 in range(N):
            Chi1De[k1, k2] = me * K03(qe[k1, k2] * Re) / qe[k1, k2]

    # Compute Chi1Dh (hole susceptibility)
    for k2 in range(N):
        for k1 in range(N):
            Chi1Dh[k1, k2] = mh * K03(qh[k1, k2] * Rh) / qh[k1, k2]

    # Apply scaling factor
    scale = e0 ** 2 / (twopi * eps0 * epsr * hbar ** 2)
    Chi1De = Chi1De * scale
    Chi1Dh = Chi1Dh * scale

    # Store as module-level variables
    global _Chi1De, _Chi1Dh
    _Chi1De = Chi1De
    _Chi1Dh = Chi1Dh

    return Chi1De, Chi1Dh


def Eps1D(n1D, Nk):
    """
    Calculate the 1D dielectric function matrix.

    Computes the dielectric function matrix Eps1D for a given 1D carrier density,
    using the pre-computed module-level susceptibility arrays Chi1De and Chi1Dh.

    Parameters
    ----------
    n1D : float
        1D carrier density (1/m)
    Nk : int
        Size of momentum grid (should match size of module-level arrays)

    Returns
    -------
    ndarray
        2D array of shape (Nk, Nk) representing the dielectric function matrix.

    Notes
    -----
    The function computes: Eps1D = 1 - Chi1De * 2*log(...) - Chi1Dh * 2*log(...)
    where the logarithms involve momentum-dependent terms.
    Uses module-level variables _Chi1De, _Chi1Dh, _qe, _qh (matching Fortran behavior).
    """
    global _Chi1De, _Chi1Dh, _qe, _qh

    if _Chi1De is None or _Chi1Dh is None or _qe is None or _qh is None:
        raise ValueError("Chi1De, Chi1Dh, qe, and qh must be computed first")

    eps1d = np.ones((Nk, Nk)) - \
            _Chi1De * 2 * np.log(np.abs((_qe - pi * n1D) / (_qe + n1D))) - \
            _Chi1Dh * 2 * np.log(np.abs((_qh - pi * n1D) / (_qh + n1D)))

    return eps1d


def CalcScreenedArrays(screened, L, ne, nh, VC, E1D):
    """
    Calculate screened Coulomb interaction arrays.

    Computes the screened version of the Coulomb interaction matrices by dividing
    by the dielectric function. If screening is disabled, returns the unscreened arrays.

    Parameters
    ----------
    screened : bool
        If True, apply screening. If False, return unscreened arrays.
    L : float
        Length of the quantum wire (m)
    ne : ndarray
        Electron population array, 1D array
    nh : ndarray
        Hole population array, 1D array
    VC : ndarray
        Input/output array for interaction matrices, shape (N, N, 3).
        On input: should contain unscreened arrays (or will be filled from module-level).
        On output: contains screened arrays if screened=True, unscreened otherwise.
    E1D : ndarray
        Input/output array for dielectric function, shape (N, N).
        On input: can be any array (will be overwritten).
        On output: contains dielectric function matrix if screened=True, ones otherwise.

    Returns
    -------
    None
        Arrays are modified in-place (matching Fortran intent(inout) behavior).

    Notes
    -----
    The function uses module-level variables _Veh0, _Vee0, _Vhh0, _qe, _qh
    (matching Fortran module behavior). If screening is enabled, computes the
    dielectric function and divides the interaction matrices by it.
    """
    global _Veh0, _Vee0, _Vhh0, _qe, _qh

    N = len(ne)

    # Check that module-level arrays are initialized
    if _Veh0 is None or _Vee0 is None or _Vhh0 is None:
        raise ValueError("Veh0, Vee0, Vhh0 must be computed first using CalcCoulombArrays")

    # Copy unscreened arrays from module-level (matching Fortran)
    VC[:, :, 0] = _Veh0
    VC[:, :, 1] = _Vee0
    VC[:, :, 2] = _Vhh0
    E1D[:, :] = 1.0

    if screened:
        # Check that qe and qh are initialized
        if _qe is None or _qh is None:
            raise ValueError("qe and qh must be computed first using MakeQs")

        # Calculate 1D density
        density_1D = np.sum(np.real(ne) + np.real(nh)) / 2.0 / L

        # Maximum density (avoid singularities)
        # Fortran: min(qe(2,2), qh(2,2)) - note 2,2 is 1-based, so 1,1 in 0-based
        density_max = min(_qe[1, 1], _qh[1, 1]) / pi * 0.99
        density_1D = min(density_1D, density_max)

        # Compute dielectric function (uses module-level variables)
        E1D[:, :] = Eps1D(density_1D, N)

        # Apply screening
        VC[:, :, 0] = VC[:, :, 0] / E1D
        VC[:, :, 1] = VC[:, :, 1] / E1D
        VC[:, :, 2] = VC[:, :, 2] / E1D


####################################################################################################
########################## Coulomb Calculations for the SBEs #######################################
####################################################################################################

@jit(nopython=True, parallel=True)
def _CalcMVeh_core(p, Veh, MVeh, k3, UnDel):
    """
    Core JIT-compiled computation for CalcMVeh.

    This is the inner loop that does the actual computation.
    """
    N = p.shape[0]
    Nf = p.shape[2]

    # Initialize output
    for f in range(Nf):
        for kp in range(N):
            for k in range(N):
                for q in range(N):
                    # Get qp from momentum conservation: k3(kp, q, k)
                    # k3 stores 1-based indices, so convert
                    qp_1b = k3[kp, q, k]
                    if qp_1b > 0:
                        qp = qp_1b - 1  # Convert to 0-based
                        if 0 <= qp < N:
                            # Convert to 1-based for UnDel access (UnDel uses 1-based indexing)
                            k_1b = k + 1
                            kp_1b = kp + 1
                            q_1b = q + 1
                            qp_1b_val = qp + 1

                            MVeh[k, kp, f] += (p[q, qp, f] * Veh[k, q] *
                                             UnDel[k_1b, q_1b] *
                                             UnDel[kp_1b, qp_1b_val])


def CalcMVeh(p, VC, MVeh, k3=None, UnDel=None):
    """
    Calculate the MVeh array for Semiconductor Bloch Equations.

    Computes the many-body electron-hole interaction term MVeh used in the
    Semiconductor Bloch Equations. This represents the Coulomb interaction
    contribution to the polarization dynamics.

    Parameters
    ----------
    p : ndarray
        Polarization array, shape (N, N, Nf) where N is momentum grid size
        and Nf is number of frequency/time points
    VC : ndarray
        Screened Coulomb interaction matrices, shape (N, N, 3)
        VC[:, :, 0] = Veh (electron-hole)
        VC[:, :, 1] = Vee (electron-electron)
        VC[:, :, 2] = Vhh (hole-hole)
    MVeh : ndarray
        Output array for MVeh calculation, shape (N, N, Nf).
        Will be modified in-place (matching Fortran intent(inout)).
    k3 : ndarray, optional
        3D indexing array from MakeK3, shape (N, N, N).
        If None, uses module-level _k3.
    UnDel : ndarray, optional
        2D array from MakeUnDel, shape (N+1, N+1).
        If None, uses module-level _UnDel.

    Returns
    -------
    None
        MVeh is modified in-place (matching Fortran intent(inout) behavior).

    Notes
    -----
    The calculation sums over momentum states q, using the momentum conservation
    relation k3(kp, q, k) to find qp. The UnDel factors exclude self-interaction terms.
    Uses JIT compilation with parallel execution (matching Fortran OpenMP behavior).
    """
    global _k3, _UnDel

    # Use provided arrays or fall back to module-level
    if k3 is None:
        if _k3 is None:
            raise ValueError("k3 must be initialized first using InitializeCoulomb")
        k3 = _k3
    if UnDel is None:
        if _UnDel is None:
            raise ValueError("UnDel must be initialized first using InitializeCoulomb")
        UnDel = _UnDel

    # Extract electron-hole interaction matrix
    Veh = VC[:, :, 0]

    # Initialize output
    MVeh[:, :, :] = 0.0

    # Call JIT-compiled core function
    _CalcMVeh_core(p, Veh, MVeh, k3, UnDel)


def undell(k, q):
    """
    Wrapper function for UnDel array access.

    Provides a simple interface to access the UnDel (1 - delta) array.
    This is a convenience function matching the Fortran interface.

    Parameters
    ----------
    k : int
        First index (1-based, matching Fortran)
    q : int
        Second index (1-based, matching Fortran)

    Returns
    -------
    float
        Value of UnDel(k, q) from module-level array.

    Notes
    -----
    This function accesses the module-level _UnDel array. The indices
    are expected to be 1-based to match Fortran behavior.
    """
    global _UnDel
    if _UnDel is None:
        raise ValueError("UnDel must be initialized first using MakeUnDel or InitializeCoulomb")
    return _UnDel[k, q]


@jit(nopython=True, parallel=True)
def _BGRenorm_core(ne, nh, Vee, Vhh, BGR, UnDel):
    """
    Core JIT-compiled computation for BGRenorm.
    """
    N = len(ne)

    # Initialize output
    for kp in range(N):
        for k in range(N):
            # Convert to 1-based for UnDel access
            k_1b = k + 1
            kp_1b = kp + 1

            # Sum over all momentum states
            sum_hh = 0.0
            sum_ee = 0.0

            for i in range(N):
                i_1b = i + 1
                # Vhh(:,kp) * UnDel(:,kp) - hole-hole contribution
                sum_hh += nh[i] * Vhh[i, kp] * UnDel[i_1b, kp_1b]
                # Vee(:,k) * UnDel(:,k) - electron-electron contribution
                sum_ee += ne[i] * Vee[i, k] * UnDel[i_1b, k_1b]

            BGR[k, kp] = -sum_hh - sum_ee


def BGRenorm(C, D, VC, BGR, UnDel=None):
    """
    Calculate band gap renormalization.

    Computes the band gap renormalization BGR due to many-body Coulomb interactions.
    This accounts for the shift in band gap energy due to carrier-carrier interactions.

    Parameters
    ----------
    C : ndarray
        Electron density matrix, shape (N, N), complex
    D : ndarray
        Hole density matrix, shape (N, N), complex
    VC : ndarray
        Screened Coulomb interaction matrices, shape (N, N, 3)
        VC[:, :, 0] = Veh (electron-hole)
        VC[:, :, 1] = Vee (electron-electron)
        VC[:, :, 2] = Vhh (hole-hole)
    BGR : ndarray
        Output array for band gap renormalization, shape (N, N), complex.
        Will be modified in-place (matching Fortran intent(inout)).
    UnDel : ndarray, optional
        2D array from MakeUnDel, shape (N+1, N+1).
        If None, uses module-level _UnDel.

    Returns
    -------
    None
        BGR is modified in-place (matching Fortran intent(inout) behavior).

    Notes
    -----
    The function extracts diagonal elements from C and D to get carrier populations,
    then computes the renormalization using hole-hole and electron-electron interactions.
    The UnDel factors exclude self-interaction terms.
    Uses JIT compilation with parallel execution (matching Fortran OpenMP behavior).
    """
    global _UnDel

    if UnDel is None:
        if _UnDel is None:
            raise ValueError("UnDel must be initialized first using InitializeCoulomb")
        UnDel = _UnDel

    # Extract interaction matrices
    Vee = VC[:, :, 1]
    Vhh = VC[:, :, 2]

    # Extract diagonal elements (carrier populations)
    ne = np.diag(C)  # Electron population
    nh = np.diag(D)  # Hole population

    # Initialize output
    BGR[:, :] = 0.0

    # Call JIT-compiled core function
    _BGRenorm_core(ne, nh, Vee, Vhh, BGR, UnDel)


@jit(nopython=True, parallel=True)
def _EeRenorm_core(ne, Vee, BGR, UnDel):
    """
    Core JIT-compiled computation for EeRenorm.
    """
    N = len(ne)

    # Initialize output
    for kp in range(N):
        for k in range(N):
            # Convert to 1-based for UnDel access
            k_1b = k + 1
            kp_1b = kp + 1

            # Compute sums matching Fortran formula exactly
            # Note: Some terms cancel but we compute as written in Fortran
            sum_vkp_kp = 0.0  # sum(ne(:) * Vee(kp,kp))
            sum_vkp_ud = 0.0  # sum(ne(:) * Vee(:,kp) * UnDel(:,kp))
            sum_vk_k = 0.0    # sum(ne(:) * Vee(k,k))
            sum_vk_ud = 0.0   # sum(ne(:) * Vee(:,k) * UnDel(:,k))

            for i in range(N):
                i_1b = i + 1
                sum_vkp_kp += ne[i] * Vee[kp, kp]
                sum_vkp_ud += ne[i] * Vee[i, kp] * UnDel[i_1b, kp_1b]
                sum_vk_k += ne[i] * Vee[k, k]
                sum_vk_ud += ne[i] * Vee[i, k] * UnDel[i_1b, k_1b]

            # Fortran formula: + 2*sum(ne*Vee(kp,kp)) - sum(ne*Vee(:,kp)*UnDel(:,kp))
            #                  + 2*sum(ne*Vee(k,k)) - sum(ne*Vee(:,k)*UnDel(:,k))
            #                  - 2*sum(ne*Vee(kp,kp)) - sum(ne*Vee(k,k))
            # This simplifies but we keep it as written
            BGR[k, kp] = (2.0 * sum_vkp_kp - sum_vkp_ud +
                          2.0 * sum_vk_k - sum_vk_ud -
                          2.0 * sum_vkp_kp - sum_vk_k)


def EeRenorm(ne, VC, BGR, UnDel=None):
    """
    Calculate electron energy renormalization.

    Computes the electron energy renormalization due to many-body Coulomb interactions.
    This accounts for the shift in electron energy levels due to electron-electron interactions.

    Parameters
    ----------
    ne : ndarray
        Electron population array, shape (N,), complex
    VC : ndarray
        Screened Coulomb interaction matrices, shape (N, N, 3)
        VC[:, :, 1] = Vee (electron-electron) is used
    BGR : ndarray
        Output array for electron energy renormalization, shape (N, N), complex.
        Will be modified in-place (matching Fortran intent(inout)).
    UnDel : ndarray, optional
        2D array from MakeUnDel, shape (N+1, N+1).
        If None, uses module-level _UnDel.

    Returns
    -------
    None
        BGR is modified in-place (matching Fortran intent(inout) behavior).

    Notes
    -----
    The function computes the electron energy shift using electron-electron
    interactions, with terms that include both direct and exchange contributions.
    The UnDel factors exclude self-interaction terms.
    Uses JIT compilation with parallel execution (matching Fortran OpenMP behavior).
    """
    global _UnDel

    if UnDel is None:
        if _UnDel is None:
            raise ValueError("UnDel must be initialized first using InitializeCoulomb")
        UnDel = _UnDel

    # Extract electron-electron interaction matrix
    Vee = VC[:, :, 1]

    # Initialize output
    BGR[:, :] = 0.0

    # Call JIT-compiled core function
    _EeRenorm_core(ne, Vee, BGR, UnDel)


@jit(nopython=True, parallel=True)
def _EhRenorm_core(nh, Vhh, BGR, UnDel):
    """
    Core JIT-compiled computation for EhRenorm.
    """
    N = len(nh)

    # Initialize output
    for kp in range(N):
        for k in range(N):
            # Convert to 1-based for UnDel access
            k_1b = k + 1
            kp_1b = kp + 1

            # Compute sums matching Fortran formula exactly
            # Note: Some terms cancel but we compute as written in Fortran
            sum_vkp_kp = 0.0  # sum(nh(:) * Vhh(kp,kp))
            sum_vkp_ud = 0.0  # sum(nh(:) * Vhh(:,kp) * UnDel(:,kp))
            sum_vk_k = 0.0    # sum(nh(:) * Vhh(k,k))
            sum_vk_ud = 0.0   # sum(nh(:) * Vhh(:,k) * UnDel(:,k))

            for i in range(N):
                i_1b = i + 1
                sum_vkp_kp += nh[i] * Vhh[kp, kp]
                sum_vkp_ud += nh[i] * Vhh[i, kp] * UnDel[i_1b, kp_1b]
                sum_vk_k += nh[i] * Vhh[k, k]
                sum_vk_ud += nh[i] * Vhh[i, k] * UnDel[i_1b, k_1b]

            # Fortran formula: + 2*sum(nh*Vhh(kp,kp)) - sum(nh*Vhh(:,kp)*UnDel(:,kp))
            #                  + 2*sum(nh*Vhh(k,k)) - sum(nh*Vhh(:,k)*UnDel(:,k))
            #                  - 2*sum(nh*Vhh(kp,kp)) - sum(nh*Vhh(k,k))
            # This simplifies but we keep it as written
            BGR[k, kp] = (2.0 * sum_vkp_kp - sum_vkp_ud +
                          2.0 * sum_vk_k - sum_vk_ud -
                          2.0 * sum_vkp_kp - sum_vk_k)


def EhRenorm(nh, VC, BGR, UnDel=None):
    """
    Calculate hole energy renormalization.

    Computes the hole energy renormalization due to many-body Coulomb interactions.
    This accounts for the shift in hole energy levels due to hole-hole interactions.

    Parameters
    ----------
    nh : ndarray
        Hole population array, shape (N,), complex
    VC : ndarray
        Screened Coulomb interaction matrices, shape (N, N, 3)
        VC[:, :, 2] = Vhh (hole-hole) is used
    BGR : ndarray
        Output array for hole energy renormalization, shape (N, N), complex.
        Will be modified in-place (matching Fortran intent(inout)).
    UnDel : ndarray, optional
        2D array from MakeUnDel, shape (N+1, N+1).
        If None, uses module-level _UnDel.

    Returns
    -------
    None
        BGR is modified in-place (matching Fortran intent(inout) behavior).

    Notes
    -----
    The function computes the hole energy shift using hole-hole interactions,
    with terms that include both direct and exchange contributions.
    The UnDel factors exclude self-interaction terms.
    Uses JIT compilation with parallel execution (matching Fortran OpenMP behavior).
    """
    global _UnDel

    if UnDel is None:
        if _UnDel is None:
            raise ValueError("UnDel must be initialized first using InitializeCoulomb")
        UnDel = _UnDel

    # Extract hole-hole interaction matrix
    Vhh = VC[:, :, 2]

    # Initialize output
    BGR[:, :] = 0.0

    # Call JIT-compiled core function
    _EhRenorm_core(nh, Vhh, BGR, UnDel)


####################################################################################################
########################## MANY BODY RELAXATION EFFECTS ###########################################
########################## i.e., Non-Hartree-Fock Terms ##########################################
####################################################################################################

@jit(nopython=True, parallel=True)
def _MBCE2_core(ne, nh, Veh2, Vee2, Win, Wout, k3, Ceh, Cee):
    """
    Core JIT-compiled computation for MBCE2.

    Note: ne and nh are arrays with 0-index unused, indices 1..Nk are used (1-based in Fortran).
    k3, Ceh, Cee use 1-based indexing.
    """
    Nk = len(ne) - 1  # ne has size Nk+1 (0..Nk), but we use 1..Nk

    for k in range(1, Nk + 1):  # k = 1..Nk (1-based)
        for q1 in range(1, Nk + 1):
            for q2 in range(1, Nk + 1):
                # Electron-Hole In/Out rates
                kp = q1
                k1 = q2

                # k1p = k3(kp, k1, k) - k3 uses 1-based, convert to 0-based for access
                k1p_1b = k3[kp - 1, k1 - 1, k - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b  # k1p is 1-based
                    # Ceh(k, kp, k1) - 1-based indices
                    Win[k - 1] += (Veh2[k - 1, k1 - 1] * (1.0 - nh[kp]) * nh[k1p] *
                                   ne[k1] * Ceh[k - 1, kp - 1, k1 - 1])

                # k1p = k3(kp, k, k1)
                k1p_1b = k3[kp - 1, k - 1, k1 - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b
                    # Ceh(k1, kp, k) - 1-based indices
                    Wout[k - 1] += (Veh2[k1 - 1, k - 1] * (1.0 - ne[k1]) * (1.0 - nh[kp]) *
                                    nh[k1p] * Ceh[k1 - 1, kp - 1, k - 1])

                # Electron-Electron In/Out rates
                k2 = q1
                k4 = q2

                # k30 = k3(k, k2, k4)
                k30_1b = k3[k - 1, k2 - 1, k4 - 1]
                if k30_1b > 0:
                    k30 = k30_1b
                    # Cee(k, k2, k4) - 1-based indices
                    Win[k - 1] += (Vee2[k - 1, k4 - 1] * (1.0 - ne[k2]) * ne[k30] *
                                   ne[k4] * Cee[k - 1, k2 - 1, k4 - 1])

                # k30 = k3(k4, k2, k)
                k30_1b = k3[k4 - 1, k2 - 1, k - 1]
                if k30_1b > 0:
                    k30 = k30_1b
                    # Cee(k4, k2, k) - 1-based indices
                    Wout[k - 1] += (Vee2[k4 - 1, k - 1] * (1.0 - ne[k4]) * (1.0 - ne[k2]) *
                                    ne[k30] * Cee[k4 - 1, k2 - 1, k - 1])


def MBCE2(ne0, nh0, ky, Ee, Eh, VC, geh, ge, Win, Wout, k3=None, Ceh=None, Cee=None):
    """
    Calculate the Many-body Coulomb In/Out rates for electrons (version 2).

    Computes the many-body relaxation rates Win and Wout for electrons due to
    electron-hole and electron-electron Coulomb interactions. These represent
    non-Hartree-Fock terms in the many-body dynamics.

    Parameters
    ----------
    ne0 : ndarray
        Electron population array, shape (Nk,), real
    nh0 : ndarray
        Hole population array, shape (Nk,), real
    ky : ndarray
        Momentum coordinates (not used in calculation but kept for interface compatibility)
    Ee : ndarray
        Electron energies (not used in calculation but kept for interface compatibility)
    Eh : ndarray
        Hole energies (not used in calculation but kept for interface compatibility)
    VC : ndarray
        Screened Coulomb interaction matrices, shape (Nk, Nk, 3)
        VC[:, :, 0] = Veh (electron-hole)
        VC[:, :, 1] = Vee (electron-electron)
    geh : float
        Electron-hole inverse lifetime (not used in calculation but kept for interface)
    ge : float
        Electron inverse lifetime (not used in calculation but kept for interface)
    Win : ndarray
        Input/output array for in-scattering rates, shape (Nk,), real.
        Will be modified in-place (matching Fortran intent(inout)).
    Wout : ndarray
        Input/output array for out-scattering rates, shape (Nk,), real.
        Will be modified in-place (matching Fortran intent(inout)).
    k3 : ndarray, optional
        3D indexing array from MakeK3, shape (Nk, Nk, Nk).
        If None, uses module-level _k3.
    Ceh : ndarray, optional
        Electron-hole many-body interaction array, shape (Nk, Nk, Nk).
        If None, uses module-level _Ceh.
    Cee : ndarray, optional
        Electron-electron many-body interaction array, shape (Nk, Nk, Nk).
        If None, uses module-level _Cee.

    Returns
    -------
    None
        Win and Wout are modified in-place (matching Fortran intent(inout) behavior).

    Notes
    -----
    Uses JIT compilation with parallel execution (matching Fortran OpenMP behavior).
    The function computes scattering rates due to electron-hole and electron-electron
    interactions, accounting for Pauli blocking factors (1 - n) and occupation factors n.
    """
    global _k3, _Ceh, _Cee

    if k3 is None:
        if _k3 is None:
            raise ValueError("k3 must be initialized first using InitializeCoulomb")
        k3 = _k3
    if Ceh is None:
        if _Ceh is None:
            raise ValueError("Ceh must be initialized first using InitializeCoulomb")
        Ceh = _Ceh
    if Cee is None:
        if _Cee is None:
            raise ValueError("Cee must be initialized first using InitializeCoulomb")
        Cee = _Cee

    Nk = len(ne0)

    # Compute squared interaction matrices
    Veh2 = VC[:, :, 0] ** 2
    Vee2 = VC[:, :, 1] ** 2

    # Create arrays with 0-index (unused) and indices 1..Nk (matching Fortran ne(0:Nk))
    # In Python, this is indices 0..Nk, where 0 is unused and 1..Nk are used
    ne = np.zeros(Nk + 1)
    nh = np.zeros(Nk + 1)
    ne[1:Nk + 1] = np.abs(ne0)  # Fill indices 1..Nk (0-based: 1..Nk)
    nh[1:Nk + 1] = np.abs(nh0)

    # Call JIT-compiled core function
    _MBCE2_core(ne, nh, Veh2, Vee2, Win, Wout, k3, Ceh, Cee)


@jit(nopython=True, parallel=True)
def _MBCE_core(ne, nh, Veh2, Vee2, Win, Wout, k3, Ceh, Cee):
    """
    Core JIT-compiled computation for MBCE.

    Note: This is identical to _MBCE2_core but kept separate to match Fortran structure.
    """
    Nk = len(ne) - 1

    for k in range(1, Nk + 1):
        for q1 in range(1, Nk + 1):
            for q2 in range(1, Nk + 1):
                # Electron-Hole In/Out rates
                kp = q1
                k1 = q2

                k1p_1b = k3[kp - 1, k1 - 1, k - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b
                    Win[k - 1] += (Veh2[k - 1, k1 - 1] * (1.0 - nh[kp]) * nh[k1p] *
                                   ne[k1] * Ceh[k - 1, kp - 1, k1 - 1])

                k1p_1b = k3[kp - 1, k - 1, k1 - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b
                    Wout[k - 1] += (Veh2[k1 - 1, k - 1] * (1.0 - ne[k1]) * (1.0 - nh[kp]) *
                                    nh[k1p] * Ceh[k1 - 1, kp - 1, k - 1])

                # Electron-Electron In/Out rates
                k2 = q1
                k4 = q2

                k30_1b = k3[k - 1, k2 - 1, k4 - 1]
                if k30_1b > 0:
                    k30 = k30_1b
                    Win[k - 1] += (Vee2[k - 1, k4 - 1] * (1.0 - ne[k2]) * ne[k30] *
                                   ne[k4] * Cee[k - 1, k2 - 1, k4 - 1])

                k30_1b = k3[k4 - 1, k2 - 1, k - 1]
                if k30_1b > 0:
                    k30 = k30_1b
                    Wout[k - 1] += (Vee2[k4 - 1, k - 1] * (1.0 - ne[k4]) * (1.0 - ne[k2]) *
                                    ne[k30] * Cee[k4 - 1, k2 - 1, k - 1])


def MBCE(ne0, nh0, ky, Ee, Eh, VC, geh, ge, Win, Wout, k3=None, Ceh=None, Cee=None):
    """
    Calculate the Many-body Coulomb In/Out rates for electrons.

    Computes the many-body relaxation rates Win and Wout for electrons due to
    electron-hole and electron-electron Coulomb interactions. This is identical
    to MBCE2 but kept as a separate function to match Fortran structure.

    Parameters
    ----------
    ne0 : ndarray
        Electron population array, shape (Nk,), real
    nh0 : ndarray
        Hole population array, shape (Nk,), real
    ky : ndarray
        Momentum coordinates (not used in calculation but kept for interface compatibility)
    Ee : ndarray
        Electron energies (not used in calculation but kept for interface compatibility)
    Eh : ndarray
        Hole energies (not used in calculation but kept for interface compatibility)
    VC : ndarray
        Screened Coulomb interaction matrices, shape (Nk, Nk, 3)
        VC[:, :, 0] = Veh (electron-hole)
        VC[:, :, 1] = Vee (electron-electron)
    geh : float
        Electron-hole inverse lifetime (not used in calculation but kept for interface)
    ge : float
        Electron inverse lifetime (not used in calculation but kept for interface)
    Win : ndarray
        Input/output array for in-scattering rates, shape (Nk,), real.
        Will be modified in-place (matching Fortran intent(inout)).
    Wout : ndarray
        Input/output array for out-scattering rates, shape (Nk,), real.
        Will be modified in-place (matching Fortran intent(inout)).
    k3 : ndarray, optional
        3D indexing array from MakeK3, shape (Nk, Nk, Nk).
        If None, uses module-level _k3.
    Ceh : ndarray, optional
        Electron-hole many-body interaction array, shape (Nk, Nk, Nk).
        If None, uses module-level _Ceh.
    Cee : ndarray, optional
        Electron-electron many-body interaction array, shape (Nk, Nk, Nk).
        If None, uses module-level _Cee.

    Returns
    -------
    None
        Win and Wout are modified in-place (matching Fortran intent(inout) behavior).

    Notes
    -----
    Uses JIT compilation with parallel execution (matching Fortran OpenMP behavior).
    This function is identical to MBCE2 but kept separate to match Fortran code structure.
    """
    global _k3, _Ceh, _Cee

    if k3 is None:
        if _k3 is None:
            raise ValueError("k3 must be initialized first using InitializeCoulomb")
        k3 = _k3
    if Ceh is None:
        if _Ceh is None:
            raise ValueError("Ceh must be initialized first using InitializeCoulomb")
        Ceh = _Ceh
    if Cee is None:
        if _Cee is None:
            raise ValueError("Cee must be initialized first using InitializeCoulomb")
        Cee = _Cee

    Nk = len(ne0)

    # Compute squared interaction matrices
    Veh2 = VC[:, :, 0] ** 2
    Vee2 = VC[:, :, 1] ** 2

    # Create arrays with 0-index (unused) and indices 1..Nk
    ne = np.zeros(Nk + 1)
    nh = np.zeros(Nk + 1)
    ne[1:Nk + 1] = np.abs(ne0)
    nh[1:Nk + 1] = np.abs(nh0)

    # Call JIT-compiled core function
    _MBCE_core(ne, nh, Veh2, Vee2, Win, Wout, k3, Ceh, Cee)


@jit(nopython=True, parallel=True)
def _MBCH_core(ne, nh, Veh2, Vhh2, Win, Wout, k3, Ceh, Chh):
    """
    Core JIT-compiled computation for MBCH.

    Note: ne and nh are arrays with 0-index unused, indices 1..Nk are used (1-based in Fortran).
    k3, Ceh, Chh use 1-based indexing.
    """
    Nk = len(ne) - 1

    for kp in range(1, Nk + 1):  # kp = 1..Nk (1-based)
        for q1 in range(1, Nk + 1):
            for q2 in range(1, Nk + 1):
                # Electron-Hole In/Out rates
                k = q1
                k1 = q2

                # k1p = k3(kp, k1, k)
                k1p_1b = k3[kp - 1, k1 - 1, k - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b
                    # Ceh(k, kp, k1) - 1-based indices
                    Win[kp - 1] += (Veh2[k - 1, k1 - 1] * (1.0 - ne[k]) * nh[k1p] *
                                    ne[k1] * Ceh[k - 1, kp - 1, k1 - 1])

                # k1p = k3(kp, k, k1)
                k1p_1b = k3[kp - 1, k - 1, k1 - 1]
                if k1p_1b > 0:
                    k1p = k1p_1b
                    # Ceh(k, k1p, k1) - 1-based indices
                    Wout[kp - 1] += (Veh2[k1 - 1, k - 1] * (1.0 - ne[k]) * (1.0 - nh[k1p]) *
                                     ne[k1] * Ceh[k - 1, k1p - 1, k1 - 1])

                # Hole-Hole In/Out rates
                k2p = q1
                k4p = q2

                # k3p = k3(kp, k2p, k4p)
                k3p_1b = k3[kp - 1, k2p - 1, k4p - 1]
                if k3p_1b > 0:
                    k3p = k3p_1b
                    # Chh(kp, k2p, k4p) - 1-based indices
                    Win[kp - 1] += (Vhh2[kp - 1, k4p - 1] * (1.0 - nh[k2p]) * nh[k3p] *
                                    nh[k4p] * Chh[kp - 1, k2p - 1, k4p - 1])

                # k3p = k3(k4p, k2p, kp)
                k3p_1b = k3[k4p - 1, k2p - 1, kp - 1]
                if k3p_1b > 0:
                    k3p = k3p_1b
                    # Chh(k4p, k2p, kp) - 1-based indices
                    Wout[kp - 1] += (Vhh2[k4p - 1, kp - 1] * (1.0 - nh[k4p]) * (1.0 - nh[k2p]) *
                                     nh[k3p] * Chh[k4p - 1, k2p - 1, kp - 1])


def MBCH(ne0, nh0, ky, Ee, Eh, VC, geh, gh, Win, Wout, k3=None, Ceh=None, Chh=None):
    """
    Calculate the Many-body Coulomb In/Out rates for holes.

    Computes the many-body relaxation rates Win and Wout for holes due to
    electron-hole and hole-hole Coulomb interactions. These represent
    non-Hartree-Fock terms in the many-body dynamics.

    Parameters
    ----------
    ne0 : ndarray
        Electron population array, shape (Nk,), real
    nh0 : ndarray
        Hole population array, shape (Nk,), real
    ky : ndarray
        Momentum coordinates (not used in calculation but kept for interface compatibility)
    Ee : ndarray
        Electron energies (not used in calculation but kept for interface compatibility)
    Eh : ndarray
        Hole energies (not used in calculation but kept for interface compatibility)
    VC : ndarray
        Screened Coulomb interaction matrices, shape (Nk, Nk, 3)
        VC[:, :, 0] = Veh (electron-hole)
        VC[:, :, 2] = Vhh (hole-hole)
    geh : float
        Electron-hole inverse lifetime (not used in calculation but kept for interface)
    gh : float
        Hole inverse lifetime (not used in calculation but kept for interface)
    Win : ndarray
        Input/output array for in-scattering rates, shape (Nk,), real.
        Will be modified in-place (matching Fortran intent(inout)).
    Wout : ndarray
        Input/output array for out-scattering rates, shape (Nk,), real.
        Will be modified in-place (matching Fortran intent(inout)).
    k3 : ndarray, optional
        3D indexing array from MakeK3, shape (Nk, Nk, Nk).
        If None, uses module-level _k3.
    Ceh : ndarray, optional
        Electron-hole many-body interaction array, shape (Nk, Nk, Nk).
        If None, uses module-level _Ceh.
    Chh : ndarray, optional
        Hole-hole many-body interaction array, shape (Nk, Nk, Nk).
        If None, uses module-level _Chh.

    Returns
    -------
    None
        Win and Wout are modified in-place (matching Fortran intent(inout) behavior).

    Notes
    -----
    Uses JIT compilation with parallel execution (matching Fortran OpenMP behavior).
    The function computes scattering rates due to electron-hole and hole-hole
    interactions, accounting for Pauli blocking factors (1 - n) and occupation factors n.
    """
    global _k3, _Ceh, _Chh

    if k3 is None:
        if _k3 is None:
            raise ValueError("k3 must be initialized first using InitializeCoulomb")
        k3 = _k3
    if Ceh is None:
        if _Ceh is None:
            raise ValueError("Ceh must be initialized first using InitializeCoulomb")
        Ceh = _Ceh
    if Chh is None:
        if _Chh is None:
            raise ValueError("Chh must be initialized first using InitializeCoulomb")
        Chh = _Chh

    Nk = len(ne0)

    # Compute squared interaction matrices
    Veh2 = VC[:, :, 0] ** 2
    Vhh2 = VC[:, :, 2] ** 2

    # Create arrays with 0-index (unused) and indices 1..Nk
    ne = np.zeros(Nk + 1)
    nh = np.zeros(Nk + 1)
    ne[1:Nk + 1] = np.abs(ne0)
    nh[1:Nk + 1] = np.abs(nh0)

    # Call JIT-compiled core function
    _MBCH_core(ne, nh, Veh2, Vhh2, Win, Wout, k3, Ceh, Chh)
