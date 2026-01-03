"""
Semiconductor Bloch Equations (SBEs) solver for quantum wire simulations.

This module solves the 1D Semiconductor Bloch equations in support of
propagation simulations for a quantum wire.

"""

import numpy as np
from scipy.constants import m_e as me0_SI, hbar as hbar_SI, e as e0_SI, epsilon_0 as eps0_SI

from numba import jit, prange
try:
    from numba import cuda
    _HAS_CUDA = cuda.is_available()
except (ImportError, RuntimeError):
    _HAS_CUDA = False
    cuda = None
import os

from ..libpulsesuite.spliner import locate
from .usefulsubs import FFTG, iFFTG, printITR, TotalEnergy, Temperature, WriteIt, locator, GetArray0Index
from .dcfield import CalcI0, CalcPD, CalcVD, InitializeDC, GetEDrift, Transport, CalcI0n
from .dephasing import WriteDephasing, CalcGammaE, CalcGammaH, OffDiagDephasing2, InitializeDephasing
from .coulomb import CalcScreenedArrays, SetLorentzDelta, GetEps1Dqw, GetChi1Dqw, InitializeCoulomb, MBCE, MBCH
from .phonons import MBPH, MBPE, InitializePhonons, FermiDistr
from .emission import SpontEmission, InitializeEmission, Calchw
from .typespace import GetSpaceArray, GetKArray
from .qwoptics import Prop2QW, QWChi1, QW2Prop, WritePropFields, QWPolarization3, QWRho5, WriteSBESolns, InitializeQWOptics, Xcv, Ycv, Zcv, yw


# Physical constants
eV = 1.602176634e-19  # Electron volt in Joules
me0 = me0_SI  # Electron rest mass (kg)
hbar = hbar_SI  # Reduced Planck constant (J·s)
e0 = e0_SI  # Elementary charge (C)
eps0 = eps0_SI  # Vacuum permittivity (F/m)
pi = np.pi
twopi = 2.0 * np.pi  # 2π constant
_twopi = twopi  # Module-level alias
ii = 1j  # Imaginary unit


#######################################################
################ Initial QW Parameters #################
#######################################################
_L = 100e-9                 # Length of quantum wire (m)
_Delta0 = 5e-9              # z-thickness of quantum wire (m)
_gap = 1.5 * eV             # Band gap (J, 1.53 eV)
_me = 0.07 * me0            # Electron effective mass (kg)
_mh = 0.45 * me0            # Hole effective mass (kg)
_HO = 100e-3 * eV           # Energy level separation (J, default 100 meV)
_gam_e = 1.0 / 1e-12        # Electron lifetime frequency (Hz)
_gam_h = 1.0 / 1e-12        # Hole lifetime frequency (Hz)
_gam_eh = 1.0 / 1e-12       # Energy level broadening rate (Hz)
_wL = 0.0                   # Laser frequency (rad/s), only for calculating chi1
_epsr = 9.1                 # Background dielectric constant
_Oph = 36e-3 * eV / hbar    # Phonon frequency (Hz)
_Gph = 3e-3 * eV / hbar     # Inverse phonon lifetime (Hz)
_Edc = 0.0e6                # Background DC field in +y-direction (V/m)

#######################################################
################ Booleans SBE terms #################
#######################################################
_Optics = True              # Include basic optical coupling
_Excitons = True            # Include excitons & band gap renormalization (HTF-EH)
_EHs = True                 # Include electron-hole (non-HTF) effects
_Screened = True            # Include carrier coulomb screening
_Phonon = True              # Include electron-phonon collisions
_DCTrans = False            # Include transport from DC field
_LF = True                  # Include influence of the longitudinal field
_FreePot = False            # Include a free charge potential under dipole approximation
_DiagDph = True             # Include Diagonal Dephasing
_OffDiagDph = True          # Include Off-Diagonal Dephasing
_OBE = False                # Ignore all Many-body effects and just solve k-resolved OBEs
_Recomb = False             # Include electron-hole recombination
_ReadDC = False             # Read in DC field value from file
_PLSpec = False             # Record the Photoluminescence spectrum
_ignorewire = False
_debug1 = False
_Xqwparams = False

# ============================================================================
################ Semiconductor Bloch Eq. arrays (time-evolution scheme) #################
#######################################################
_YY1 = None                 # EH Coherence at time t(n-1), shape: (Nk, Nk, Nw)
_YY2 = None                 # EH Coherence at time t(n), shape: (Nk, Nk, Nw)
_YY3 = None                 # EH Coherence at time t(n+1), shape: (Nk, Nk, Nw)
_CC1 = None                 # EE Coherence at time t(n-1), shape: (Nk, Nk, Nw)
_CC2 = None                 # EE Coherence at time t(n), shape: (Nk, Nk, Nw)
_CC3 = None                 # EE Coherence at time t(n+1), shape: (Nk, Nk, Nw)
_DD1 = None                 # HH Coherence at time t(n-1), shape: (Nk, Nk, Nw)
_DD2 = None                 # HH Coherence at time t(n), shape: (Nk, Nk, Nw)
_DD3 = None                 # HH Coherence at time t(n+1), shape: (Nk, Nk, Nw)

_qwPx = None                # QW polarization x-component, shape: (Nr, Nw)
_qwPy = None                # QW polarization y-component, shape: (Nr, Nw)
_qwPz = None                # QW polarization z-component, shape: (Nr, Nw)

_Id = None                  # Identity matrix, shape: (Nk, Nk)
_Ia = None                  # Anti-Identity matrix Ia = 1-Id, shape: (Nk, Nk)

#######################################################
################ QW Arrays Required for Solving SBEs #################
#######################################################
_Ee = None                  # Electron energies (J), shape: (Nk,)
_Eh = None                  # Hole energies (J), shape: (Nk,)
_r = None                   # k_y arrays (1/m) for FFT, shape: (Nr,)
_Qr = None                  # k_y arrays (1/m) for FFT, shape: (Nr,)
_QE = None                  # k_y arrays (1/m) for practical use, shape: (Nr-1,)
_kr = None                  # k_y arrays (1/m) for practical use, shape: (Nk,)

_Px0 = None                 # Old polarizations to calc Jp = dP/dt, shape: (Nr,)
_Px1 = None                 # Old polarizations to calc Jp = dP/dt, shape: (Nr,)
_Px0W = None                # Old polarizations to calc Jp = dP/dt, shape: (Nr,)
_Px1W = None                # Old polarizations to calc Jp = dP/dt, shape: (Nr,)
_Py0 = None                 # Old polarizations to calc Jp = dP/dt, shape: (Nr,)
_Py1 = None                 # Old polarizations to calc Jp = dP/dt, shape: (Nr,)
_Py0W = None                # Old polarizations to calc Jp = dP/dt, shape: (Nr,)
_Py1W = None                # Old polarizations to calc Jp = dP/dt, shape: (Nr,)
_Pz0 = None                 # Old polarizations to calc Jp = dP/dt, shape: (Nr,)
_Pz1 = None                 # Old polarizations to calc Jp = dP/dt, shape: (Nr,)
_Pz0W = None                # Old polarizations to calc Jp = dP/dt, shape: (Nr,)
_Pz1W = None                # Old polarizations to calc Jp = dP/dt, shape: (Nr,)

_EPEnergy = 0.0             # Book keeping for total energy to wires
_EPEnergyW = 0.0            # Book keeping for total energy to wires
_I0 = None                  # Drift Current in each wire, shape: (Nw,)
_ErI0 = None                # E_Current in each wire, shape: (Nw,)

_hw = np.zeros(500)         # Photoluminescence spectrum frequency array
_PLS = np.zeros(500)        # Photoluminescence spectrum array

_dkr = 0.0                  # dk (1/m)
_dr = 0.0                   # dy (m)
_Nr1 = 0                    # Beginning yy-array point
_Nr2 = 0                    # Ending yy-array point
_start = False              # Has this module been initiated?

#######################################################
################ TBD QW Parameters #################
#######################################################
_dcv = None                 # Carrier dipole moment (C m)
_ehint = 1.0                # Electron-Hole space integral, TBD, default is 1.0
_Emax0 = 0.0                # Initial Peak Field Value
_alphae = 0.0               # Gaussian inverse radius of wire (1/m)
_alphah = 0.0               # Gaussian inverse radius of wire (1/m)
_qc = 0.0                   # Gaussian inverse radius of wire (1/m)
_area = 1e-16               # Area of wire (m^2), 1e-16 is default, real value TBD
_t = 0.0                    # Time variable
_wph = 0.0                  # Frequency variable
_chiw = 0.0 + 0.0j          # Susceptibility
_wL = 0.0                   # Laser frequency (rad/s)
_c0 = 299792458.0           # Speed of light (m/s)
_uw = 100                   # Unit number offset for wire output files
_vhh0 = 0.0                 # Velocity
_ETHz = 0.0                 # THz field

#######################################################
################ Numerical Parameters #################
#######################################################
_Nr = 0                     # # of points in quantum wire y-space
_Nk = 0                     # # of points in reduced quantum wire k-space (Nr/2-1)
_small = 1e-200             # Smallest # worthy of consideration
_NK0 = 0                    # Frequency equal zero index
_NQ0 = 0                    # Frequency equal zero index
_nqq = 0
_nqq10 = 0
_kkp = None                 # kr - krp index matrix, shape: (Nk, Nk)

#######################################################
################ SBE Booleans #################
#######################################################
_wireoff = True             # Wait to turn QW on when field is strong

#######################################################
################ Book Keeping Variables for Printing of Data #################
#######################################################
_xxx = 1
_jjj = 1
_jmax = 1000
_ntmax = 100000
_uw = 820
_file_972 = None  # File handle for PdotE.p.dat
_file_973 = None  # File handle for PdotE.w.dat


#######################################################
# Functions
#######################################################

def QWCalculator(Exx, Eyy, Ezz, Vrr, rr, q, dt, w, Pxx, Pyy, Pzz, Rho, DoQWP, DoQWDl):
    """
    Time-evolve the source terms of the quantum wire for Maxwell's equations.

    Solves the 1D Semiconductor Bloch equations and calculates polarization
    and charge density source terms for propagation simulations.

    Parameters
    ----------
    Exx : ndarray (complex)
        X-component total electric field in propagation space
    Eyy : ndarray (complex)
        Y-component total electric field in propagation space
    Ezz : ndarray (complex)
        Z-component electric field in propagation space
    Vrr : ndarray (complex)
        Potential electric field in propagation space
    rr : ndarray (float)
        QW spatial array (m)
    q : ndarray (float)
        QW momentum array (1/m)
    dt : float
        Time step (s)
    w : int
        Wire index (which wire to calculate for)
    Pxx : ndarray (complex)
        X polarization output (modified in-place)
    Pyy : ndarray (complex)
        Y polarization output (modified in-place)
    Pzz : ndarray (complex)
        Z polarization output (modified in-place)
    Rho : ndarray (complex)
        Free charge density output (modified in-place)
    DoQWP : list
        Boolean flag [should propagator use QW polarization] (modified in-place)
    DoQWDl : list
        Boolean flag [should propagator use longitudinal field] (modified in-place)

    Returns
    -------
    None
        All outputs are modified in-place

    Notes
    -----
    This function:
    - Updates module-level time counters (xxx, jjj, t)
    - Allocates arrays on first call
    - Calls Prop2QW to convert propagation fields to QW fields
    - Calls SBECalculator to solve the SBEs
    - Calls QW2Prop to convert back to propagation space
    - Calculates energy transfer and writes to files
    - Only activates when field is strong enough (controlled by _wireoff)
    """
    global _xxx, _jjj, _t, _wireoff, _qwPx, _qwPy, _qwPz
    global _Px0, _Px1, _Px0W, _Px1W, _Py0, _Py1, _Py0W, _Py1W
    global _Pz0, _Pz1, _Pz0W, _Pz1W, _EPEnergy, _EPEnergyW
    global _PLS, _file_972, _file_973, _r, _Qr, _Nr, _dr, _area, _CC1

    # Initialize source terms for propagator
    Pxx[:] = 0.0 + 0.0j
    Pyy[:] = 0.0 + 0.0j
    Pzz[:] = 0.0 + 0.0j
    Rho[:] = 0.0 + 0.0j

    WriteFields = False

    # Book keeping - do this only once per time step (when w==1)
    if w == 1:
        if _jjj == _jmax:
            _jjj = 0

        _xxx += 1
        _jjj += 1
        _t += dt

        if _jjj == _jmax:
            _PLS[:] = 0.0

    # Record results only 1 out of every jmax times
    if _jjj == _jmax:
        WriteFields = True

    # Only do the QW calculation if the field was once large compared
    # to its initial maximum value. Otherwise just do book keeping.
    field_magnitude = np.max(np.sqrt(np.abs(Exx)**2 + np.abs(Eyy)**2 + np.abs(Ezz)**2))

    if _wireoff and field_magnitude < 1e-3 * _Emax0:
        # E isn't big enough yet. Do nothing.
        return
    else:
        # E is (or was) big enough. Commence Calculations!
        _wireoff = False
        DoQWP[0] = _Optics
        DoQWDl[0] = _LF

    # Allocate qwPx, qwPy, qwPz arrays if not already allocated
    if _qwPx is None:
        Nw = _CC1.shape[2] if _CC1 is not None else 1
        _qwPx = np.zeros((len(rr), Nw), dtype=complex)
        _qwPy = np.zeros((len(rr), Nw), dtype=complex)
        _qwPz = np.zeros((len(rr), Nw), dtype=complex)

    Ex = np.zeros(_Nr, dtype=complex)
    Ey = np.zeros(_Nr, dtype=complex)
    Ez = np.zeros(_Nr, dtype=complex)
    Vr = np.zeros(_Nr, dtype=complex)
    Px = np.zeros(_Nr, dtype=complex)
    Py = np.zeros(_Nr, dtype=complex)
    Pz = np.zeros(_Nr, dtype=complex)
    re = np.zeros(_Nr, dtype=complex)
    rh = np.zeros(_Nr, dtype=complex)

    Edc0 = 0.0  # Local variable for DC field

    Prop2QW(rr, Exx, Eyy, Ezz, Vrr, Edc0, _r, Ex, Ey, Ez, Vr, _t, _xxx)

    if _debug1:
        # Debug mode: simple linear response
        Px = eps0 * 1.0 * Ex
        Py = eps0 * 1.0 * Ey
        Pz = eps0 * 1.0 * Ez
        re[:] = 0.0
        rh[:] = 0.0
    else:
        # Solve SBEs for the w'th quantum wire and return
        # the values for the QW sources Px, Py, Pz, Re, Rh
        SBECalculator(Ex, Ey, Ez, Vr, dt, Px, Py, Pz, re, rh, WriteFields, w)

    # FFT QW Qr-fields and interpolate them to the Propagation YY-space
    RhoE = np.zeros(len(rr), dtype=complex)
    RhoH = np.zeros(len(rr), dtype=complex)

    QW2Prop(_r, _Qr, Ex, Ey, Ez, Vr, Px, Py, Pz, re, rh, rr,
            Pxx, Pyy, Pzz, RhoE, RhoH, w, _xxx, WriteFields, _LF)

    Rho[:] = RhoH - RhoE

    if WriteFields:
        WritePropFields(rr, Exx, Eyy, Ezz, Vrr, Pxx, Pyy, Pzz, RhoE, RhoH, 'r', w, _xxx)

    # Allocate polarization history arrays on first call
    if _Px0 is None:
        _Px0 = np.zeros(len(Pxx))
        _Px1 = np.zeros(len(Pxx))
        _Px0W = np.zeros(len(Px))
        _Px1W = np.zeros(len(Px))
        _Py0 = np.zeros(len(Pyy))
        _Py1 = np.zeros(len(Pyy))
        _Py0W = np.zeros(len(Py))
        _Py1W = np.zeros(len(Py))
        _Pz0 = np.zeros(len(Pzz))
        _Pz1 = np.zeros(len(Pzz))
        _Pz0W = np.zeros(len(Pz))
        _Pz1W = np.zeros(len(Pz))
        _EPEnergy = 0.0
        _EPEnergyW = 0.0

        # Open output files
        os.makedirs('output', exist_ok=True)
        _file_972 = open('output/PdotE.p.dat', 'w', encoding='utf-8')
        _file_973 = open('output/PdotE.w.dat', 'w', encoding='utf-8')

    # Calculate energy transfer: dE/dt = P · dE/dt
    drr = rr[1] - rr[0] if len(rr) > 1 else 1.0

    _EPEnergy += (np.sum((np.real(Pxx) - _Px0) * np.real(Exx) * 0.5) * drr * _area +
                  np.sum((np.real(Pyy) - _Py0) * np.real(Eyy) * 0.5) * drr * _area +
                  np.sum((np.real(Pzz) - _Pz0) * np.real(Ezz) * 0.5) * drr * _area)

    _EPEnergyW += (np.sum((np.real(Px) - _Px0W) * np.real(Ex) * 0.5) * _dr * _area +
                   np.sum((np.real(Py) - _Py0W) * np.real(Ey) * 0.5) * _dr * _area +
                   np.sum((np.real(Pz) - _Pz0W) * np.real(Ez) * 0.5) * _dr * _area)

    # Update polarization history
    _Px0[:] = _Px1
    _Px1[:] = np.real(Pxx)
    _Px0W[:] = _Px1W
    _Px1W[:] = np.real(Px)
    _Py0[:] = _Py1
    _Py1[:] = np.real(Pyy)
    _Py0W[:] = _Py1W
    _Py1W[:] = np.real(Py)
    _Pz0[:] = _Pz1
    _Pz1[:] = np.real(Pzz)
    _Pz0W[:] = _Pz1W
    _Pz1W[:] = np.real(Pz)

    # Write energy to files
    if _file_972 is not None:
        _file_972.write(f'{_t} {_EPEnergy / e0}\n')
        _file_972.flush()
    if _file_973 is not None:
        _file_973.write(f'{_t} {_EPEnergyW / e0}\n')
        _file_973.flush()

    # Optionally ignore wire contribution
    if _ignorewire:
        Pxx[:] = 0.0
        Pyy[:] = 0.0
        Pzz[:] = 0.0
        Rho[:] = 0.0


#######################################################
################ SBE Helper functions #################
#######################################################


def Checkout(p1, p2, C1, C2, D1, D2, w):
    """
    Check out coherence matrices from module storage for wire w.

    Retrieves previous time steps from module-level arrays for processing.
    """
    global _YY2, _YY3, _CC2, _CC3, _DD2, _DD3

    p1[:, :] = _YY2[:, :, w - 1]
    p2[:, :] = _YY3[:, :, w - 1]
    C1[:, :] = _CC2[:, :, w - 1]
    C2[:, :] = _CC3[:, :, w - 1]
    D1[:, :] = _DD2[:, :, w - 1]
    D2[:, :] = _DD3[:, :, w - 1]


def Checkin(p1, p2, p3, C1, C2, C3, D1, D2, D3, w):
    """
    Check in updated coherence matrices to module storage for wire w.

    Stores current and future time steps to module-level arrays.
    """
    global _YY1, _YY2, _YY3, _CC1, _CC2, _CC3, _DD1, _DD2, _DD3

    _YY1[:, :, w - 1] = p1
    _YY2[:, :, w - 1] = p2
    _YY3[:, :, w - 1] = p3
    _CC1[:, :, w - 1] = C1
    _CC2[:, :, w - 1] = C2
    _CC3[:, :, w - 1] = C3
    _DD1[:, :, w - 1] = D1
    _DD2[:, :, w - 1] = D2
    _DD3[:, :, w - 1] = D3


def Preparation(p2, C2, D2, Ex, Ey, Ez, Vr, w, Heh, Hee, Hhh, VC, E1D, GamE, GamH, OffG, Rsp):
    """
    Prepare Hamiltonians, screening, and dephasing arrays for SBE time step.

    This function calculates all the arrays needed for one time step of the
    semiconductor Bloch equations:
    - Dipole coupling matrix M^{eh} (light-matter interaction)
    - Monopole coupling matrices W^{ee}, W^{hh} (free-carrier potentials)
    - Hamiltonian matrices H^{eh}, H^{ee}, H^{hh} (including many-body effects)
    - Screened Coulomb interaction arrays VC, E1D
    - Diagonal dephasing rates γ_e, γ_h
    - Off-diagonal dephasing rates Γ^{off}
    - Spontaneous emission rates R_sp

    Parameters
    ----------
    p2 : ndarray (complex), shape (Nk, Nk)
        Electron-hole coherence matrix at previous time step
    C2 : ndarray (complex), shape (Nk, Nk)
        Electron-electron coherence matrix at previous time step
    D2 : ndarray (complex), shape (Nk, Nk)
        Hole-hole coherence matrix at previous time step
    Ex, Ey, Ez : ndarray (complex), shape (Nr,)
        Electric field components in momentum space
    Vr : ndarray (complex), shape (Nr,)
        Electric potential from free charge in momentum space
    w : int
        Wire index
    Heh : ndarray (complex), shape (Nk, Nk)
        Electron-hole Hamiltonian (output, modified in-place)
    Hee : ndarray (complex), shape (Nk, Nk)
        Electron-electron Hamiltonian (output, modified in-place)
    Hhh : ndarray (complex), shape (Nk, Nk)
        Hole-hole Hamiltonian (output, modified in-place)
    VC : ndarray (float), shape (Nr, Nr, 3)
        Screened Coulomb arrays (output, modified in-place)
    E1D : ndarray (float), shape (Nk, Nk)
        1D dielectric screening array (output, modified in-place)
    GamE : ndarray (float), shape (Nk,)
        Electron diagonal dephasing rates (output, modified in-place)
    GamH : ndarray (float), shape (Nk,)
        Hole diagonal dephasing rates (output, modified in-place)
    OffG : ndarray (complex), shape (Nk, Nk, 3)
        Off-diagonal dephasing arrays (output, modified in-place)
    Rsp : ndarray (float), shape (Nk,)
        Spontaneous emission rates (output, modified in-place)

    Returns
    -------
    None
        All output arrays are modified in-place.

    Notes
    -----
    The preparation sequence:
    1. Initialize all arrays to zero/defaults
    2. Extract carrier populations ne, nh from diagonal of C, D
    3. Calculate dipole coupling M^{eh} if optics enabled
    4. Calculate monopole coupling W^{ee}, W^{hh} if longitudinal field enabled
    5. Calculate screened Coulomb arrays VC, E1D
    6. Calculate Hamiltonian matrices H^{eh}, H^{ee}, H^{hh}
    7. Calculate diagonal dephasing γ_e, γ_h if enabled
    8. Calculate off-diagonal dephasing Γ^{off} if enabled
    9. Calculate spontaneous emission rates R_sp if enabled

    The Hamiltonian includes:
    - Single-particle energies (diagonal)
    - Light-matter coupling (via M^{eh})
    - Coulomb many-body effects (via VC)
    - Excitonic correlations (if _Excitons=True)

    Module flags control which effects are included:
    - _Optics: optical field coupling
    - _LF: longitudinal field effects
    - _FreePot: free potential effects
    - _Excitons: excitonic correlations
    - _DiagDph: diagonal dephasing
    - _OffDiagDph: off-diagonal dephasing
    - _Recomb: spontaneous recombination

    See Also
    --------
    CalcMeh : Dipole coupling matrix
    CalcWnn : Monopole coupling matrix
    CalcH : Hamiltonian calculation
    CalcScreenedArrays : Screened Coulomb arrays
    CalcGammaE, CalcGammaH : Diagonal dephasing
    OffDiagDephasing2 : Off-diagonal dephasing
    SpontEmission : Spontaneous emission rates
    """
    global _Nk, _gam_e, _gam_h, _gam_eh, _Optics, _LF, _FreePot, _Screened, _L
    global _DiagDph, _OffDiagDph, _Recomb, _kr, _Ee, _Eh, _gap, _t

    # Initialize arrays
    Meh = np.zeros((_Nk, _Nk), dtype=complex)
    Wee = np.zeros((_Nk, _Nk), dtype=complex)
    Whh = np.zeros((_Nk, _Nk), dtype=complex)
    ne = np.zeros(_Nk, dtype=complex)
    nh = np.zeros(_Nk, dtype=complex)
    VC[:, :, :] = 0.0
    GamE[:] = _gam_e
    GamH[:] = _gam_h
    OffG[:, :, :] = 0.0 + 0.0j
    Rsp[:] = 0.0

    g = np.array([_gam_eh, _gam_e, _gam_h])

    # Extract carrier populations from diagonal elements
    for k in range(_Nk):
        ne[k] = C2[k, k]
        nh[k] = D2[k, k]

    # Calculate dipole coupling matrix (light-matter interaction)
    if _Optics:
        CalcMeh(Ex, Ey, Ez, w, Meh)

    # Calculate monopole matrix elements (free-carrier potential)
    # if _Optics and _LF:
    #     CalcWnn(-e0, Vr, Wee)
    #     CalcWnn(+e0, np.conj(Vr), Whh)

    # Calculate screened Coulomb arrays
    CalcScreenedArrays(_Screened, _L, ne, nh, VC, E1D)

    # Calculate Hamiltonian matrix elements
    CalcH(Meh, Wee, Whh, C2, D2, p2, VC, Heh, Hee, Hhh)

    # Calculate diagonal electron dephasing rate
    if _DiagDph:
        CalcGammaE(_kr, ne, nh, VC, GamE)

    # Calculate diagonal hole dephasing rate
    if _DiagDph:
        CalcGammaH(_kr, ne, nh, VC, GamH)

    # Calculate off-diagonal dephasing
    if _OffDiagDph:
        OffDiagDephasing2(ne, nh, p2, _kr, _Ee, _Eh, g, VC, _t, OffG[:, :, 0])

    # Calculate spontaneous emission rate
    if _Recomb:
        SpontEmission(ne, nh, _Ee, _Eh, _gap, _gam_eh, VC, Rsp)


def CalcH(Meh, Wee, Whh, C, D, p, VC, Heh, Hee, Hhh):
    """
    Calculate the Hamiltonian matrices for the semiconductor Bloch equations.

    Constructs the effective Hamiltonian matrices that appear in the SBEs,
    including single-particle energies, light-matter coupling, Coulomb many-body
    effects, and excitonic correlations.

    The Hamiltonians govern the coherent dynamics:
    - H^{eh}: electron-hole Hamiltonian (drives optical polarization)
    - H^{ee}: electron-electron Hamiltonian (drives electron coherence)
    - H^{hh}: hole-hole Hamiltonian (drives hole coherence)

    Parameters
    ----------
    Meh : ndarray (complex), shape (Nk, Nk)
        Dipole-field coupling matrix M^{eh}_{k_e,k_h}
    Wee : ndarray (complex), shape (Nk, Nk)
        Electron monopole-potential coupling W^{ee}_{k1,k2}
    Whh : ndarray (complex), shape (Nk, Nk)
        Hole monopole-potential coupling W^{hh}_{k1,k2}
    C : ndarray (complex), shape (Nk, Nk)
        Electron-electron coherence matrix
    D : ndarray (complex), shape (Nk, Nk)
        Hole-hole coherence matrix
    p : ndarray (complex), shape (Nk, Nk)
        Electron-hole coherence matrix (interband polarization)
    VC : ndarray (float), shape (Nr, Nr, 3)
        Screened Coulomb interaction arrays
    Heh : ndarray (complex), shape (Nk, Nk)
        Output: electron-hole Hamiltonian (modified in-place)
    Hee : ndarray (complex), shape (Nk, Nk)
        Output: electron-electron Hamiltonian (modified in-place)
    Hhh : ndarray (complex), shape (Nk, Nk)
        Output: hole-hole Hamiltonian (modified in-place)

    Returns
    -------
    None
        Hamiltonians Heh, Hee, Hhh are modified in-place.

    Notes
    -----
    The Hamiltonians are constructed as:

    H^{eh}_{k_e,k_h} = M^{eh}_{k_e,k_h} + Σ_q V(q) p^†_{k_e+q,k_h+q}

    H^{ee}_{k1,k2} = E_e(k1)δ_{k1,k2} + W^{ee}_{k1,k2}
                    - Σ_q V_ee(q) C^†_{k1+q,k2+q}
                    + Σ_k [V_ee(q)C_{k,k+q} - V_eh(q)D_{k+q,k}] (q = k1-k2)

    H^{hh}_{k1,k2} = E_h(k1)δ_{k1,k2} + W^{hh}_{k1,k2}
                    - Σ_q V_hh(q) D^†_{k1+q,k2+q}
                    + Σ_k [V_hh(q)D_{k,k+q} - V_eh(q)C_{k+q,k}] (q = k1-k2)

    where:
    - V(q) is the screened Coulomb interaction
    - q is momentum transfer
    - † denotes transpose (not conjugate transpose here)

    The excitonic terms (involving V and coherence matrices) are included
    only if _Excitons=True.

    If _LF=True (longitudinal field), off-diagonal Coulomb exchange terms
    are included. If _LF=False, only diagonal terms (k1=k2) are included,
    which is the Hartree-Fock approximation.

    The noq0 array excludes q=0 terms to avoid divergences in the Coulomb
    interaction.

    Loop ranges max(1-k1, 1-k2) to min(Nk-k1, Nk-k2) ensure that k+q
    stays within array bounds [1, Nk] in Fortran (or [0, Nk-1] in Python).

    See Also
    --------
    Preparation : Calls this function to set up Hamiltonians
    CalcMeh : Calculates dipole coupling M^{eh}
    CalcWnn : Calculates monopole coupling W^{nn}
    dpdt, dCdt, dDdt : Use these Hamiltonians in SBE time evolution
    """
    global _Nk, _Ee, _Eh, _gap, _Excitons, _LF, _FreePot

    # Transpose matrices for efficient access
    Ct = C.T
    Dt = D.T
    pt = p.T

    # Create V array indexed by momentum transfer q
    # V[q, n] where n=0,1,2 for eh, ee, hh interactions
    V = np.zeros((2*_Nk + 1, 3))  # Range: -Nk to +Nk

    # noq0: exclude q=0 terms (would be divergent)
    noq0 = np.ones(4*_Nk + 1)  # Range: -2*Nk to +2*Nk
    noq0[2*_Nk] = 0  # Set q=0 element to 0

    # Extract 1D Coulomb interaction V(q) from VC array
    # VC is indexed as VC[1+q, 1, :] in Fortran for q > 0
    # Python: VC[q, 0, :] for q > 0
    for q in range(1, _Nk):
        V[_Nk + q, :] = VC[q, 0, :]  # Positive q
        V[_Nk - q, :] = VC[q, 0, :]  # Negative q (symmetric)

    # Initialize Hamiltonians
    Heh[:, :] = 0.0 + 0.0j
    Hee[:, :] = 0.0 + 0.0j
    Hhh[:, :] = 0.0 + 0.0j

    # Single-particle energies (diagonal)
    for k in range(_Nk):
        Hee[k, k] = _Ee[k] + _gap
        Hhh[k, k] = _Eh[k]

    # Dipole and monopole coupling
    Heh[:, :] = Meh
    if _FreePot:
        Hee[:, :] += Wee
        Hhh[:, :] += Whh

    # Excitonic many-body terms
    if _Excitons and _LF:
        # Full off-diagonal Coulomb exchange (expensive)
        # H^{eh}: excitonic attraction
        for k2 in range(_Nk):
            for k1 in range(_Nk):
                qmin = max(-k1, -k2)
                qmax = min(_Nk - 1 - k1, _Nk - 1 - k2)
                for q in range(qmin, qmax + 1):
                    Heh[k1, k2] += V[_Nk + q, 0] * pt[k1 + q, k2 + q] * noq0[2*_Nk + q]

        # H^{ee}: electron-electron Coulomb and exchange
        for k2 in range(_Nk):
            for k1 in range(_Nk):
                qmin = max(-k1, -k2)
                qmax = min(_Nk - 1 - k1, _Nk - 1 - k2)
                # Exchange term
                for q in range(qmin, qmax + 1):
                    Hee[k1, k2] -= V[_Nk + q, 1] * Ct[k1 + q, k2 + q] * noq0[2*_Nk + q]

                # Direct term
                q = k1 - k2
                kmin = max(0, -q)
                kmax = min(_Nk - 1, _Nk - 1 - q)
                for k in range(kmin, kmax + 1):
                    Hee[k1, k2] += (V[_Nk + q, 1] * C[k, k + q] -
                                    V[_Nk + q, 0] * D[k + q, k]) * noq0[2*_Nk + q]

        # H^{hh}: hole-hole Coulomb and exchange
        for k2 in range(_Nk):
            for k1 in range(_Nk):
                qmin = max(-k1, -k2)
                qmax = min(_Nk - 1 - k1, _Nk - 1 - k2)
                # Exchange term
                for q in range(qmin, qmax + 1):
                    Hhh[k1, k2] -= V[_Nk + q, 2] * Dt[k1 + q, k2 + q] * noq0[2*_Nk + q]

                # Direct term
                q = k1 - k2
                kmin = max(0, -q)
                kmax = min(_Nk - 1, _Nk - 1 - q)
                for k in range(kmin, kmax + 1):
                    Hhh[k1, k2] += (V[_Nk + q, 2] * D[k, k + q] -
                                    V[_Nk + q, 0] * C[k + q, k]) * noq0[2*_Nk + q]

    elif _Excitons and not _LF:
        # Hartree-Fock approximation: only diagonal k1=k2
        # H^{eh}: excitonic attraction
        for k2 in range(_Nk):
            for k1 in range(_Nk):
                qmin = max(-k1, -k2)
                qmax = min(_Nk - 1 - k1, _Nk - 1 - k2)
                for q in range(qmin, qmax + 1):
                    Heh[k1, k2] += V[_Nk + q, 0] * pt[k1 + q, k2 + q] * noq0[2*_Nk + q]

        # H^{ee}: only diagonal elements
        for k2 in range(_Nk):
            k1 = k2
            qmin = max(-k1, -k2)
            qmax = min(_Nk - 1 - k1, _Nk - 1 - k2)
            for q in range(qmin, qmax + 1):
                Hee[k1, k2] -= V[_Nk + q, 1] * Ct[k1 + q, k2 + q] * noq0[2*_Nk + q]

        # H^{hh}: only diagonal elements
        for k2 in range(_Nk):
            k1 = k2
            qmin = max(-k1, -k2)
            qmax = min(_Nk - 1 - k1, _Nk - 1 - k2)
            for q in range(qmin, qmax + 1):
                Hhh[k1, k2] -= V[_Nk + q, 2] * Dt[k1 + q, k2 + q] * noq0[2*_Nk + q]


def chiqw():
    """
    Get the current linear optical susceptibility χ(ω).

    Returns the module-level variable _chiw which stores the linear
    susceptibility at the current frequency.

    Returns
    -------
    chi : complex
        Linear optical susceptibility χ(ω) (dimensionless).

    Notes
    -----
    This is a simple accessor function that returns the stored value.
    The susceptibility is calculated elsewhere (e.g., in QWChi1 or CalcXqw)
    and stored in _chiw.

    See Also
    --------
    getqc : Get critical momentum
    QWChi1 : Calculate linear susceptibility (qwoptics module)
    CalcXqw : Calculate susceptibility at given (q,ω)
    """
    global _chiw
    return _chiw


def getqc():
    """
    Get the critical momentum q_c.

    Returns the module-level variable _qc which stores the critical
    momentum for some physical process (e.g., plasmon cutoff, screening length).

    Returns
    -------
    qc : float
        Critical momentum q_c (rad/m).

    Notes
    -----
    This is a simple accessor function that returns the stored value.
    The critical momentum is calculated elsewhere and stored in _qc.

    Physical interpretation depends on context:
    - Plasmon physics: momentum where plasmon dispersion changes character
    - Screening: inverse screening length (q_c ≈ 1/λ_screen)
    - Phase transitions: critical wave vector

    See Also
    --------
    chiqw : Get linear susceptibility
    """
    global _qc
    return _qc


def QWArea():
    """
    Get the quantum wire cross-sectional area.

    Returns the module-level variable _area which stores the quantum wire
    cross-sectional area.

    Returns
    -------
    area : float
        Quantum wire cross-sectional area (m²).

    Notes
    -----
    This is a simple accessor function. The area is set during initialization
    and represents the effective cross-sectional area of the quantum wire.

    The area affects:
    - Carrier density normalization
    - Optical transition strengths
    - Coulomb interaction strengths

    See Also
    --------
    ReadQWParams : Reads quantum wire parameters including area
    """
    global _area
    return _area


def ShutOffOptics():
    """
    Disable optical coupling in the SBE calculations.

    Sets the module-level flag _Optics to False, which disables the
    light-matter interaction terms in the Hamiltonian. This is useful
    for studying purely electronic dynamics without optical fields.

    Returns
    -------
    None

    Notes
    -----
    When _Optics is False:
    - Dipole coupling matrix M^{eh} is not calculated
    - Only electronic Coulomb interactions remain
    - Useful for equilibration or dark dynamics

    To re-enable optics, set _Optics = True directly.

    See Also
    --------
    Preparation : Uses _Optics flag to conditionally calculate coupling
    CalcMeh : Dipole coupling (only called if _Optics=True)
    """
    global _Optics
    _Optics = False


def ReadQWParams():
    """
    Read quantum wire parameters from 'params/qw.params' file.

    Loads fundamental physical parameters of the quantum wire system from
    a parameter file and converts units where necessary. Sets module-level
    variables used throughout the SBE calculations.

    File Format
    -----------
    params/qw.params should contain one parameter per line:
        L       - Wire length (m)
        Delta0  - Wire thickness (m)
        gap     - Band gap (eV, converted to J)
        me      - Electron effective mass (units of me0)
        mh      - Hole effective mass (units of me0)
        HO      - Energy level separation (eV, converted to J)
        gam_e   - Electron dephasing rate (Hz)
        gam_h   - Hole dephasing rate (Hz)
        gam_eh  - Interband dephasing rate (Hz)
        epsr    - Relative permittivity (dimensionless)
        Oph     - Phonon energy (eV, converted to Hz)
        Gph     - Phonon damping rate (eV, converted to Hz)
        Edc     - DC electric field (V/m)
        jmax    - Output interval (time steps)
        ntmax   - Maximum time steps

    Returns
    -------
    None
        Module-level variables are set.

    Notes
    -----
    Unit conversions performed:
    - gap: eV → J (multiply by e0)
    - me, mh: units of me0 → kg (multiply by me0)
    - HO: eV → J (multiply by e0)
    - Oph: eV → Hz (multiply by e0/hbar)
    - Gph: eV → Hz (multiply by e0/hbar)

    Module variables set:
    _L, _Delta0, _gap, _me, _mh, _HO, _gam_e, _gam_h, _gam_eh,
    _epsr, _Oph, _Gph, _Edc, _jmax, _ntmax

    See Also
    --------
    ReadMBParams : Read many-body physics flags
    """
    global _L, _Delta0, _gap, _me, _mh, _HO, _gam_e, _gam_h, _gam_eh
    global _epsr, _Oph, _Gph, _Edc, _jmax, _ntmax

    with open('params/qw.params', 'r', encoding='utf-8') as f:
        _L = float(f.readline().split()[0])
        _Delta0 = float(f.readline().split()[0])
        gap_eV = float(f.readline().split()[0])
        me_rel = float(f.readline().split()[0])
        mh_rel = float(f.readline().split()[0])
        HO_eV = float(f.readline().split()[0])
        _gam_e = float(f.readline().split()[0])
        _gam_h = float(f.readline().split()[0])
        _gam_eh = float(f.readline().split()[0])
        _epsr = float(f.readline().split()[0])
        Oph_eV = float(f.readline().split()[0])
        Gph_eV = float(f.readline().split()[0])
        _Edc = float(f.readline().split()[0])
        _jmax = int(f.readline().split()[0])
        _ntmax = int(f.readline().split()[0])

    # Unit conversions
    _gap = gap_eV * e0
    _me = me_rel * me0
    _mh = mh_rel * me0
    _HO = HO_eV * e0
    _Oph = Oph_eV * e0 / hbar
    _Gph = Gph_eV * e0 / hbar


def ReadMBParams():
    """
    Read many-body physics control flags from 'params/mb.params' file.

    Loads boolean flags that control which physical effects are included
    in the many-body semiconductor Bloch equation calculations. Also sets
    the Lorentzian delta function flag for numerical integration.

    File Format
    -----------
    params/mb.params should contain one flag per line (T/F or 1/0):
        Optics       - Include optical field coupling
        Excitons     - Include excitonic correlations
        EHs          - Include carrier-carrier scattering
        Screened     - Use screened Coulomb interaction
        Phonon       - Include phonon scattering
        DCTrans      - Include DC transport
        LF           - Include longitudinal field
        FreePot      - Include free potential
        DiagDph      - Include diagonal dephasing
        OffDiagDph   - Include off-diagonal dephasing
        Recomb       - Include spontaneous recombination
        PLSpec       - Calculate photoluminescence spectrum
        ignorewire   - Ignore wire effects
        Xqwparams    - Write susceptibility parameters
        LorentzDelta - Use Lorentzian delta function

    Returns
    -------
    None
        Module-level flags are set.

    Notes
    -----
    These flags provide fine-grained control over the physics:

    - _Optics: Enable/disable light-matter interaction
    - _Excitons: Include excitonic Coulomb correlations in Hamiltonian
    - _EHs: Include carrier-carrier scattering (MBCE, MBCH)
    - _screened: Use screened vs bare Coulomb interaction
    - _Phonon: Include carrier-phonon scattering (MBPE, MBPH)
    - _DCTrans: Include DC transport and drift
    - _LF: Include longitudinal field (plasmons, screening)
    - _FreePot: Include free-carrier potential
    - _DiagDph: Momentum-dependent dephasing rates
    - _OffDiagDph: Off-diagonal dephasing (correlations)
    - _Recomb: Spontaneous electron-hole recombination
    - _PLSpec: Track photoluminescence
    - _ignorewire: Single-wire approximation
    - _Xqwparams: Write χ(q,ω) parameter file

    The LorentzDelta flag is passed to SetLorentzDelta in the coulomb module
    to control numerical delta function representation.

    See Also
    --------
    ReadQWParams : Read physical parameters
    SetLorentzDelta : Set delta function representation (coulomb module)
    """
    global _Optics, _Excitons, _EHs, _Screened, _Phonon, _DCTrans
    global _LF, _FreePot, _DiagDph, _OffDiagDph, _Recomb, _PLSpec
    global _ignorewire, _Xqwparams

    with open('params/mb.params', 'r', encoding='utf-8') as f:
        _Optics = bool(int(f.readline().split()[0]))
        _Excitons = bool(int(f.readline().split()[0]))
        _EHs = bool(int(f.readline().split()[0]))
        _Screened = bool(int(f.readline().split()[0]))
        _Phonon = bool(int(f.readline().split()[0]))
        _DCTrans = bool(int(f.readline().split()[0]))
        _LF = bool(int(f.readline().split()[0]))
        _FreePot = bool(int(f.readline().split()[0]))
        _DiagDph = bool(int(f.readline().split()[0]))
        _OffDiagDph = bool(int(f.readline().split()[0]))
        _Recomb = bool(int(f.readline().split()[0]))
        _PLSpec = bool(int(f.readline().split()[0]))
        _ignorewire = bool(int(f.readline().split()[0]))
        _Xqwparams = bool(int(f.readline().split()[0]))
        LorentzDelta = bool(int(f.readline().split()[0]))

    # Set Lorentzian delta function flag in coulomb module
    SetLorentzDelta(LorentzDelta)


def WriteSBEsData(n):
    """
    Write SBE coherence matrices to backup files.

    Saves the current state of all coherence matrices (electron-hole,
    electron-electron, hole-hole) at three time steps to disk for backup
    or restart purposes.

    Parameters
    ----------
    n : int
        File index for output filenames (e.g., n=100 creates CC1.100.dat)

    Returns
    -------
    None
        Data is written to files in 'dataQW/backup/' directory.

    Files Created
    -------------
    For each time step (1, 2, 3) and each matrix type (CC, DD, YY):
        dataQW/backup/CC1.{n}.dat - Electron coherence, step 1
        dataQW/backup/CC2.{n}.dat - Electron coherence, step 2
        dataQW/backup/CC3.{n}.dat - Electron coherence, step 3
        dataQW/backup/DD1.{n}.dat - Hole coherence, step 1
        dataQW/backup/DD2.{n}.dat - Hole coherence, step 2
        dataQW/backup/DD3.{n}.dat - Hole coherence, step 3
        dataQW/backup/YY1.{n}.dat - e-h coherence, step 1
        dataQW/backup/YY2.{n}.dat - e-h coherence, step 2
        dataQW/backup/YY3.{n}.dat - e-h coherence, step 3

    Data Format
    -----------
    Each file contains complex values written as:
        real_part imaginary_part
    One matrix element per line, nested loops over wire, k2, k1.

    Notes
    -----
    The three time steps correspond to the leapfrog integration scheme:
    - Step 1: Previous time (t - dt)
    - Step 2: Current time (t)
    - Step 3: Next time (t + dt)

    Data is written for all wires and all momentum indices.

    File size scales as: Nwires × Nk × Nk × 2 (complex) × 3 (time steps) × 9 (matrices)

    See Also
    --------
    ReadSBEsData : Read coherence matrices from backup files
    Checkin, Checkout : Store/retrieve matrices in memory
    """
    global _CC1, _CC2, _CC3, _DD1, _DD2, _DD3, _YY1, _YY2, _YY3, _Nk

    backup_dir = 'dataQW/backup/'
    os.makedirs(backup_dir, exist_ok=True)

    # Generate filenames
    filenames = {
        'CC1': f'{backup_dir}CC1.{n}.dat',
        'CC2': f'{backup_dir}CC2.{n}.dat',
        'CC3': f'{backup_dir}CC3.{n}.dat',
        'DD1': f'{backup_dir}DD1.{n}.dat',
        'DD2': f'{backup_dir}DD2.{n}.dat',
        'DD3': f'{backup_dir}DD3.{n}.dat',
        'YY1': f'{backup_dir}YY1.{n}.dat',
        'YY2': f'{backup_dir}YY2.{n}.dat',
        'YY3': f'{backup_dir}YY3.{n}.dat',
    }

    # Write data
    arrays = [_CC1, _CC2, _CC3, _DD1, _DD2, _DD3, _YY1, _YY2, _YY3]
    keys = ['CC1', 'CC2', 'CC3', 'DD1', 'DD2', 'DD3', 'YY1', 'YY2', 'YY3']

    for key, arr in zip(keys, arrays):
        with open(filenames[key], 'w', encoding='utf-8') as f:
            for w in range(arr.shape[2]):
                for k2 in range(_Nk):
                    for k1 in range(_Nk):
                        val = arr[k1, k2, w]
                        f.write(f'{np.real(val)} {np.imag(val)}\n')


def ReadSBEsData(Nt):
    """
    Read SBE coherence matrices from backup files.

    Loads the saved state of all coherence matrices (electron-hole,
    electron-electron, hole-hole) at three time steps from disk for
    restart or analysis purposes.

    Parameters
    ----------
    Nt : int
        File index for input filenames (e.g., Nt=100 reads CC1.100.dat)

    Returns
    -------
    None
        Module-level arrays _CC1, _CC2, _CC3, _DD1, _DD2, _DD3,
        _YY1, _YY2, _YY3 are updated.

    Files Read
    ----------
    For each time step (1, 2, 3) and each matrix type (CC, DD, YY):
        dataQW/backup/CC1.{Nt}.dat - Electron coherence, step 1
        dataQW/backup/CC2.{Nt}.dat - Electron coherence, step 2
        dataQW/backup/CC3.{Nt}.dat - Electron coherence, step 3
        dataQW/backup/DD1.{Nt}.dat - Hole coherence, step 1
        dataQW/backup/DD2.{Nt}.dat - Hole coherence, step 2
        dataQW/backup/DD3.{Nt}.dat - Hole coherence, step 3
        dataQW/backup/YY1.{Nt}.dat - e-h coherence, step 1
        dataQW/backup/YY2.{Nt}.dat - e-h coherence, step 2
        dataQW/backup/YY3.{Nt}.dat - e-h coherence, step 3

    Data Format
    -----------
    Each file contains complex values as:
        real_part imaginary_part
    One matrix element per line, nested loops over wire, k2, k1.

    Notes
    -----
    This function is used to restart simulations from a saved state.
    The data must match the current grid dimensions (Nk, Nwires).

    The three time steps enable continuation of the leapfrog integration
    without loss of temporal accuracy.

    After reading, the simulation can continue from time Nt·dt using
    the loaded coherence matrices.

    See Also
    --------
    WriteSBEsData : Write coherence matrices to backup files
    Checkin, Checkout : Store/retrieve matrices in memory
    """
    global _CC1, _CC2, _CC3, _DD1, _DD2, _DD3, _YY1, _YY2, _YY3, _Nk

    backup_dir = 'dataQW/backup/'

    # Generate filenames
    filenames = {
        'CC1': f'{backup_dir}CC1.{Nt}.dat',
        'CC2': f'{backup_dir}CC2.{Nt}.dat',
        'CC3': f'{backup_dir}CC3.{Nt}.dat',
        'DD1': f'{backup_dir}DD1.{Nt}.dat',
        'DD2': f'{backup_dir}DD2.{Nt}.dat',
        'DD3': f'{backup_dir}DD3.{Nt}.dat',
        'YY1': f'{backup_dir}YY1.{Nt}.dat',
        'YY2': f'{backup_dir}YY2.{Nt}.dat',
        'YY3': f'{backup_dir}YY3.{Nt}.dat',
    }

    # Read data
    arrays = [_CC1, _CC2, _CC3, _DD1, _DD2, _DD3, _YY1, _YY2, _YY3]
    keys = ['CC1', 'CC2', 'CC3', 'DD1', 'DD2', 'DD3', 'YY1', 'YY2', 'YY3']

    for key, arr in zip(keys, arrays):
        with open(filenames[key], 'r', encoding='utf-8') as f:
            for w in range(arr.shape[2]):
                for k2 in range(_Nk):
                    for k1 in range(_Nk):
                        line = f.readline().split()
                        arr[k1, k2, w] = float(line[0]) + 1j * float(line[1])


def Relaxation(ne, nh, VC, E1D, Rsp, dt, w, WriteFields):
    """
    Apply phonon scattering and carrier-carrier relaxation to electron and hole populations.

    This function calculates the time evolution of carrier populations due to
    many-body scattering processes including:
    - Electron-electron scattering
    - Hole-hole scattering
    - Electron-hole scattering
    - Electron-phonon scattering
    - Hole-phonon scattering
    - Spontaneous electron-hole recombination

    The scattering rates are calculated using many-body perturbation theory
    and applied using an exact exponential solution to the rate equations.

    Parameters
    ----------
    ne : ndarray (complex), shape (Nk,)
        Electron occupation numbers for each momentum state k.
        Modified in-place to apply relaxation effects.
    nh : ndarray (complex), shape (Nk,)
        Hole occupation numbers for each momentum state k.
        Modified in-place to apply relaxation effects.
    VC : ndarray (float), shape (Nk, Nk, 3)
        Real-time screened Coulomb interaction arrays.
        VC[:,:,0] - electron-hole interaction
        VC[:,:,1] - electron-electron interaction
        VC[:,:,2] - hole-hole interaction
    E1D : ndarray (float), shape (Nk, Nk)
        Real-time 1D dielectric screening array used for carrier-photon
        coupling calculations.
    Rsp : ndarray (float), shape (Nk,)
        Spontaneous emission rate for each momentum state (Hz).
        Used if _Recomb is True.
    dt : float
        Time step for integration (seconds).
    w : int
        Wire index identifying which quantum wire is being calculated.
    WriteFields : bool
        If True, write scattering rates to output files via printITR.

    Returns
    -------
    None
        ne and nh are modified in-place.

    Notes
    -----
    The carrier population evolution is governed by:

    dn_e/dt = W_in^e * (1 - n_e) - W_out^e * n_e
    dn_h/dt = W_in^h * (1 - n_h) - W_out^h * n_h

    where W_in and W_out are the in-scattering and out-scattering rates.

    This is solved exactly as:
    n(t+dt) = exp(-W*dt) * [(-1 + exp(W*dt) + n)*W_in + n*W_out] / W
    where W = W_in + W_out

    This exponential solution is more accurate than a simple Euler step and
    remains stable even for large time steps or scattering rates.

    If spontaneous recombination is enabled (_Recomb=True), an additional
    decay is applied:
    n_e(t+dt) = n_e * exp(-R_sp * n_h * dt)
    n_h(t+dt) = n_h * exp(-R_sp * n_e * dt)

    The scattering rates are calculated by RelaxationE and RelaxationH which
    call the many-body perturbation theory functions MBPE and MBPH from the
    phonons module.

    .. note::
        **Possible Modifications:** WinE, WoutE, WinH, WoutH could be inplace arguments.

    Examples
    --------
    >>> ne = np.array([0.1, 0.2, 0.3], dtype=complex)
    >>> nh = np.array([0.1, 0.2, 0.3], dtype=complex)
    >>> VC = np.zeros((3, 3, 3))
    >>> E1D = np.zeros((3, 3))
    >>> Rsp = np.zeros(3)
    >>> Relaxation(ne, nh, VC, E1D, Rsp, 1e-15, 0, False)
    """
    global _EHs, _Phonon, _Recomb, _kr, _xxx, _small

    if _EHs or _Phonon:
        WinE, WoutE = RelaxationE(ne, nh, VC, E1D)
        WinH, WoutH = RelaxationH(ne, nh, VC, E1D)

        if WriteFields:
            wire_str = f'{w:02d}'
            printITR(WinE * (1 - np.real(ne)), _kr, _xxx, f'Wire/Win/Win.e.{wire_str}.k.')
            printITR(WinH * (1 - np.real(nh)), _kr, _xxx, f'Wire/Win/Win.h.{wire_str}.k.')
            printITR(WoutE * np.real(ne), _kr, _xxx, f'Wire/Wout/Wout.e.{wire_str}.k.')
            printITR(WoutH * np.real(nh), _kr, _xxx, f'Wire/Wout/Wout.h.{wire_str}.k.')

        We = WinE + WoutE
        Wh = WinH + WoutH

        ne[:] = (np.exp(-We * dt) *
                ((-1 + np.exp(We * dt) + ne) * WinE + ne * WoutE) /
                (We + _small))

        nh[:] = (np.exp(-Wh * dt) *
                ((-1 + np.exp(Wh * dt) + nh) * WinH + nh * WoutH) /
                (Wh + _small))

    if _Recomb:
        ne[:] = ne * np.exp(-Rsp * nh * dt)
        nh[:] = nh * np.exp(-Rsp * ne * dt)

def RelaxationE(ne, nh, VC, E1D):
    """
    Calculate electron relaxation rates using many-body perturbation theory.

    Computes in-scattering (W_in) and out-scattering (W_out) rates for electrons
    due to:
    - Carrier-carrier scattering (electron-electron, electron-hole)
    - Carrier-phonon scattering (electron-LO phonon, electron-acoustic phonon)

    The rates are calculated using second-order Born approximation within the
    screened Hartree-Fock approximation, accounting for Pauli blocking factors.

    Parameters
    ----------
    ne : ndarray (complex), shape (Nk,)
        Electron occupation numbers for each momentum state
    nh : ndarray (complex), shape (Nk,)
        Hole occupation numbers for each momentum state
    VC : ndarray (float), shape (Nk, Nk, 3)
        Screened Coulomb interaction matrices
    E1D : ndarray (float), shape (Nk, Nk)
        One-dimensional dielectric screening array

    Returns
    -------
    WinE : ndarray (float), shape (Nk,)
        In-scattering rate into each electron state (Hz).
        W_in^e(k) represents the rate of scattering INTO state k.
    WoutE : ndarray (float), shape (Nk,)
        Out-scattering rate from each electron state (Hz).
        W_out^e(k) represents the rate of scattering OUT OF state k.

    Notes
    -----
    The scattering rates satisfy detailed balance and ensure proper
    thermalization. Pauli exclusion is enforced through (1-n_e) factors.

    Calls MBPE (Many-Body Perturbation theory for Electrons) from the phonons
    module, which implements the full quantum kinetic theory calculation.

    .. note::
        **Possible Modifications:** WinE and WoutE could be inplace arguments.

    See Also
    --------
    RelaxationH : Equivalent calculation for holes
    MBPE : The underlying many-body calculation (in phonons module)
    """
    global _Ee, _kr, _EHs, _Phonon, _gam_eh, _gam_e, _Eh

    Nk = len(ne)
    WinE = np.zeros(Nk, dtype=float)
    WoutE = np.zeros(Nk, dtype=float)

    # Carrier-carrier scattering (if enabled)
    if _EHs:
        MBCE(np.real(ne), np.real(nh), _kr, _Ee, _Eh, VC, _gam_eh, _gam_e, WinE, WoutE)

    # Phonon scattering (if enabled)
    if _Phonon:
        MBPE(np.real(ne), VC, E1D, WinE, WoutE)

    return WinE, WoutE


def RelaxationH(ne, nh, VC, E1D):
    """
    Calculate hole relaxation rates using many-body perturbation theory.

    Computes in-scattering (W_in) and out-scattering (W_out) rates for holes
    due to:
    - Carrier-carrier scattering (hole-hole, hole-electron)
    - Carrier-phonon scattering (hole-LO phonon, hole-acoustic phonon)

    The rates are calculated using second-order Born approximation within the
    screened Hartree-Fock approximation, accounting for Pauli blocking factors.

    Parameters
    ----------
    ne : ndarray (complex), shape (Nk,)
        Electron occupation numbers for each momentum state
    nh : ndarray (complex), shape (Nk,)
        Hole occupation numbers for each momentum state
    VC : ndarray (float), shape (Nk, Nk, 3)
        Screened Coulomb interaction matrices
    E1D : ndarray (float), shape (Nk, Nk)
        One-dimensional dielectric screening array

    Returns
    -------
    WinH : ndarray (float), shape (Nk,)
        In-scattering rate into each hole state (Hz).
        W_in^h(k) represents the rate of scattering INTO state k.
    WoutH : ndarray (float), shape (Nk,)
        Out-scattering rate from each hole state (Hz).
        W_out^h(k) represents the rate of scattering OUT OF state k.

    Notes
    -----
    The scattering rates satisfy detailed balance and ensure proper
    thermalization. Pauli exclusion is enforced through (1-n_h) factors.

    Calls MBPH (Many-Body Perturbation theory for Holes) from the phonons
    module, which implements the full quantum kinetic theory calculation.

    .. note::
        **Possible Modifications:** WinH, WoutH could be inplace arguments.

    See Also
    --------
    RelaxationE : Equivalent calculation for electrons
    MBPH : The underlying many-body calculation (in phonons module)
    """
    global _Eh, _kr, _EHs, _Phonon, _gam_eh, _gam_h, _Ee

    Nk = len(nh)
    WinH = np.zeros(Nk, dtype=float)
    WoutH = np.zeros(Nk, dtype=float)

    # Carrier-carrier scattering (if enabled)
    if _EHs:
        MBCH(np.real(ne), np.real(nh), _kr, _Ee, _Eh, VC, _gam_eh, _gam_h, WinH, WoutH)


    # Phonon scattering (if enabled)
    if _Phonon:
        MBPH(np.real(nh), VC, E1D, WinH, WoutH)

    return WinH, WoutH

def dpdt(C, D, p, Heh, Hee, Hhh, GamE, GamH, OffP):
    """
    Calculate the time derivative of the electron-hole coherence (interband polarization).

    Implements the Semiconductor Bloch Equation for the electron-hole coherence p,
    which represents the microscopic interband polarization that gives rise to the
    macroscopic optical polarization.

    The equation solved is:
    iℏ dp_{k_e,k_h}/dt =
        + Σ_k' H_{k_h,k'}^{hh} p_{k',k_e}           [hole Hamiltonian term]
        + Σ_k' H_{k_e,k'}^{ee} p_{k_h,k'}           [electron Hamiltonian term]
        - Σ_k' H_{k',k_h}^{eh†} C_{k',k_e}          [electron-electron correlation]
        - Σ_k' H_{k_e,k'}^{eh} D_{k',k_h}           [hole-hole correlation]
        + H_{k_e,k_h}^{eh}                          [light-matter coupling]
        - iℏ(γ_e(k_e) + γ_h(k_h)) p_{k_e,k_h}      [diagonal dephasing]
        + Γ_p^{off}(k_e,k_h)                        [off-diagonal dephasing]

    Parameters
    ----------
    C : ndarray (complex), shape (Nk, Nk)
        Electron-electron coherence matrix C_{k1,k2}.
        Diagonal elements C_{k,k} = n_e(k) are electron occupation numbers.
        Off-diagonal elements represent electron-electron quantum correlations.
    D : ndarray (complex), shape (Nk, Nk)
        Hole-hole coherence matrix D_{k1,k2}.
        Diagonal elements D_{k,k} = n_h(k) are hole occupation numbers.
        Off-diagonal elements represent hole-hole quantum correlations.
    p : ndarray (complex), shape (Nk, Nk)
        Electron-hole coherence matrix p_{k_e,k_h}.
        This is the microscopic polarization between electron state k_e
        and hole state k_h.
    Heh : ndarray (complex), shape (Nk, Nk)
        Electron-hole Hamiltonian matrix elements H_{k_e,k_h}^{eh}.
        Contains the light-matter coupling and screened Coulomb attraction.
    Hee : ndarray (complex), shape (Nk, Nk)
        Electron-electron Hamiltonian matrix elements H_{k1,k2}^{ee}.
        Contains kinetic energy (diagonal) and Coulomb repulsion (off-diagonal).
    Hhh : ndarray (complex), shape (Nk, Nk)
        Hole-hole Hamiltonian matrix elements H_{k1,k2}^{hh}.
        Contains kinetic energy (diagonal) and Coulomb repulsion (off-diagonal).
    GamE : ndarray (float), shape (Nk,)
        Diagonal electron dephasing rate γ_e(k) for each momentum state (Hz).
        Due to carrier-carrier and carrier-phonon scattering.
    GamH : ndarray (float), shape (Nk,)
        Diagonal hole dephasing rate γ_h(k) for each momentum state (Hz).
        Due to carrier-carrier and carrier-phonon scattering.
    OffP : ndarray (complex), shape (Nk, Nk)
        Off-diagonal dephasing term Γ_p^{off}(k_e, k_h).
        Accounts for correlations in scattering processes.

    Returns
    -------
    dpdt_result : ndarray (complex), shape (Nk, Nk)
        Time derivative dp/dt of the electron-hole coherence (1/s).

    Notes
    -----
    This is one of the three coupled Semiconductor Bloch Equations (SBEs).
    The equations for C (electrons) and D (holes) are given by dCdt and dDdt.

    The electron-hole coherence p is related to the macroscopic polarization P by:
    P(r,t) = Σ_{k_e,k_h} d_{cv} p_{k_e,k_h}(t) exp(i(k_e-k_h)·r)
    where d_{cv} is the interband dipole matrix element.

    Physical interpretation:
    - The Hamiltonian terms (Hee, Hhh) cause phase evolution
    - The correlation terms (via C, D) include Coulomb many-body effects
    - The Heh term couples to the external field (light)
    - Dephasing terms model irreversible decay processes

    Uses JIT compilation for performance with automatic fallback to pure Python.

    References
    ----------
    V. M. Axt and T. Kuhn, "Femtosecond spectroscopy in semiconductors:
    a key to coherences, correlations and quantum kinetics",
    Rep. Prog. Phys. 67, 433 (2004), Eqs. 2.15-2.19.

    See Also
    --------
    dCdt : Time derivative of electron-electron coherence
    dDdt : Time derivative of hole-hole coherence
    """
    # Try CUDA first, then JIT, then fallback
    if _HAS_CUDA:
        try:
            return _dpdt_cuda(C, D, p, Heh, Hee, Hhh, GamE, GamH, OffP, ii, hbar)
        except Exception:
            # Fallback to JIT
            try:
                return _dpdt_jit(C, D, p, Heh, Hee, Hhh, GamE, GamH, OffP, ii, hbar)
            except Exception:
                return _dpdt_fallback(C, D, p, Heh, Hee, Hhh, GamE, GamH, OffP)
    else:
        # No CUDA, use JIT
        try:
            return _dpdt_jit(C, D, p, Heh, Hee, Hhh, GamE, GamH, OffP, ii, hbar)
        except Exception:
            return _dpdt_fallback(C, D, p, Heh, Hee, Hhh, GamE, GamH, OffP)


@jit(nopython=True, cache=True, parallel=True)
def _dpdt_jit(C, D, p, Heh, Hee, Hhh, GamE, GamH, OffP, ii_val, hbar_val):
    """JIT-compiled dpdt calculation."""
    Nk = p.shape[0]
    dpdt_result = np.zeros((Nk, Nk), dtype=np.complex128)

    for ke in prange(Nk):
        for kh in range(Nk):
            term1 = np.sum(Hhh[kh, :] * p[:, ke])
            term2 = np.sum(Hee[ke, :] * p[kh, :])
            term3 = np.sum(Heh[:, kh] * C[:, ke])
            term4 = np.sum(Heh[ke, :] * D[:, kh])

            dpdt_result[kh, ke] = (term1 + term2 - term3 - term4 + Heh[ke, kh]
                                   - ii_val * hbar_val * (GamE[ke] + GamH[kh]) * p[kh, ke]
                                   + OffP[kh, ke])

    dpdt_result = dpdt_result / (ii_val * hbar_val)
    return dpdt_result


# CUDA implementation for dpdt
if _HAS_CUDA:
    @cuda.jit
    def _dpdt_cuda_kernel_main(C_real, C_imag, D_real, D_imag, p_real, p_imag,
                                Heh_real, Heh_imag, Hee_real, Hee_imag, Hhh_real, Hhh_imag,
                                GamE, GamH, OffP_real, OffP_imag,
                                dpdt_real, dpdt_imag, Nk, ii_real, ii_imag, hbar_val):
        """Main CUDA kernel for dpdt calculation."""
        ke = cuda.blockIdx.x
        kh = cuda.threadIdx.x

        if ke < Nk and kh < Nk:
            # Compute reductions using shared memory
            # For simplicity, we'll compute sums directly in each thread
            # This is less efficient but simpler than multi-stage reduction

            # term1 = sum(Hhh[kh, :] * p[:, ke])
            term1_real = 0.0
            term1_imag = 0.0
            for k in range(Nk):
                h_re = Hhh_real[kh, k]
                h_im = Hhh_imag[kh, k]
                p_re = p_real[k, ke]
                p_im = p_imag[k, ke]
                term1_real += h_re * p_re - h_im * p_im
                term1_imag += h_re * p_im + h_im * p_re

            # term2 = sum(Hee[ke, :] * p[kh, :])
            term2_real = 0.0
            term2_imag = 0.0
            for k in range(Nk):
                h_re = Hee_real[ke, k]
                h_im = Hee_imag[ke, k]
                p_re = p_real[kh, k]
                p_im = p_imag[kh, k]
                term2_real += h_re * p_re - h_im * p_im
                term2_imag += h_re * p_im + h_im * p_re

            # term3 = sum(Heh[:, kh] * C[:, ke])
            term3_real = 0.0
            term3_imag = 0.0
            for k in range(Nk):
                h_re = Heh_real[k, kh]
                h_im = Heh_imag[k, kh]
                c_re = C_real[k, ke]
                c_im = C_imag[k, ke]
                term3_real += h_re * c_re - h_im * c_im
                term3_imag += h_re * c_im + h_im * c_re

            # term4 = sum(Heh[ke, :] * D[:, kh])
            term4_real = 0.0
            term4_imag = 0.0
            for k in range(Nk):
                h_re = Heh_real[ke, k]
                h_im = Heh_imag[ke, k]
                d_re = D_real[k, kh]
                d_im = D_imag[k, kh]
                term4_real += h_re * d_re - h_im * d_im
                term4_imag += h_re * d_im + h_im * d_re

            # Additional terms
            heh_re = Heh_real[ke, kh]
            heh_im = Heh_imag[ke, kh]

            # -ii * hbar * (GamE[ke] + GamH[kh]) * p[kh, ke]
            gam_sum = GamE[ke] + GamH[kh]
            p_re = p_real[kh, ke]
            p_im = p_imag[kh, ke]
            deph_re = -ii_real * hbar_val * gam_sum * p_re - (-ii_imag) * hbar_val * gam_sum * p_im
            deph_im = -ii_real * hbar_val * gam_sum * p_im + (-ii_imag) * hbar_val * gam_sum * p_re

            offp_re = OffP_real[kh, ke]
            offp_im = OffP_imag[kh, ke]

            # Final result
            result_real = term1_real + term2_real - term3_real - term4_real + heh_re + deph_re + offp_re
            result_imag = term1_imag + term2_imag - term3_imag - term4_imag + heh_im + deph_im + offp_im

            # Divide by (ii * hbar)
            denom = ii_real * hbar_val
            denom_im = ii_imag * hbar_val
            denom_sq = denom * denom + denom_im * denom_im
            dpdt_real[kh, ke] = (result_real * denom + result_imag * denom_im) / denom_sq
            dpdt_imag[kh, ke] = (result_imag * denom - result_real * denom_im) / denom_sq

    def _dpdt_cuda(C, D, p, Heh, Hee, Hhh, GamE, GamH, OffP, ii_val, hbar_val):
        """CUDA wrapper for dpdt."""
        Nk = p.shape[0]

        # Split complex arrays
        C_real = np.ascontiguousarray(C.real, dtype=np.float64)
        C_imag = np.ascontiguousarray(C.imag, dtype=np.float64)
        D_real = np.ascontiguousarray(D.real, dtype=np.float64)
        D_imag = np.ascontiguousarray(D.imag, dtype=np.float64)
        p_real = np.ascontiguousarray(p.real, dtype=np.float64)
        p_imag = np.ascontiguousarray(p.imag, dtype=np.float64)
        Heh_real = np.ascontiguousarray(Heh.real, dtype=np.float64)
        Heh_imag = np.ascontiguousarray(Heh.imag, dtype=np.float64)
        Hee_real = np.ascontiguousarray(Hee.real, dtype=np.float64)
        Hee_imag = np.ascontiguousarray(Hee.imag, dtype=np.float64)
        Hhh_real = np.ascontiguousarray(Hhh.real, dtype=np.float64)
        Hhh_imag = np.ascontiguousarray(Hhh.imag, dtype=np.float64)
        GamE_arr = np.ascontiguousarray(GamE, dtype=np.float64)
        GamH_arr = np.ascontiguousarray(GamH, dtype=np.float64)
        OffP_real = np.ascontiguousarray(OffP.real, dtype=np.float64)
        OffP_imag = np.ascontiguousarray(OffP.imag, dtype=np.float64)

        # Allocate device arrays
        d_C_real = cuda.to_device(C_real)
        d_C_imag = cuda.to_device(C_imag)
        d_D_real = cuda.to_device(D_real)
        d_D_imag = cuda.to_device(D_imag)
        d_p_real = cuda.to_device(p_real)
        d_p_imag = cuda.to_device(p_imag)
        d_Heh_real = cuda.to_device(Heh_real)
        d_Heh_imag = cuda.to_device(Heh_imag)
        d_Hee_real = cuda.to_device(Hee_real)
        d_Hee_imag = cuda.to_device(Hee_imag)
        d_Hhh_real = cuda.to_device(Hhh_real)
        d_Hhh_imag = cuda.to_device(Hhh_imag)
        d_GamE = cuda.to_device(GamE_arr)
        d_GamH = cuda.to_device(GamH_arr)
        d_OffP_real = cuda.to_device(OffP_real)
        d_OffP_imag = cuda.to_device(OffP_imag)

        # Allocate output
        d_dpdt_real = cuda.device_array((Nk, Nk), dtype=np.float64)
        d_dpdt_imag = cuda.device_array((Nk, Nk), dtype=np.float64)

        # Launch kernel: one block per ke, Nk threads per block (one per kh)
        blocks_per_grid = Nk
        threads_per_block = Nk

        # Get ii as real/imag
        ii_real = ii_val.real
        ii_imag = ii_val.imag

        _dpdt_cuda_kernel_main[blocks_per_grid, threads_per_block](
            d_C_real, d_C_imag, d_D_real, d_D_imag, d_p_real, d_p_imag,
            d_Heh_real, d_Heh_imag, d_Hee_real, d_Hee_imag, d_Hhh_real, d_Hhh_imag,
            d_GamE, d_GamH, d_OffP_real, d_OffP_imag,
            d_dpdt_real, d_dpdt_imag, Nk, ii_real, ii_imag, hbar_val
        )

        # Copy back and combine
        dpdt_real = d_dpdt_real.copy_to_host()
        dpdt_imag = d_dpdt_imag.copy_to_host()
        return dpdt_real + 1j * dpdt_imag
else:
    def _dpdt_cuda(*args, **kwargs):
        """Dummy CUDA function when CUDA is not available."""
        raise RuntimeError("CUDA not available")


def _dpdt_fallback(C, D, p, Heh, Hee, Hhh, GamE, GamH, OffP):
    """Fallback dpdt calculation without JIT."""
    Nk = p.shape[0]
    dpdt_result = np.zeros((Nk, Nk), dtype=complex)

    for ke in range(Nk):
        for kh in range(Nk):
            dpdt_result[kh, ke] = (np.sum(Hhh[kh, :] * p[:, ke]) +
                                   np.sum(Hee[ke, :] * p[kh, :]) -
                                   np.sum(Heh[:, kh] * C[:, ke]) -
                                   np.sum(Heh[ke, :] * D[:, kh]) +
                                   Heh[ke, kh] -
                                   ii * hbar * (GamE[ke] + GamH[kh]) * p[kh, ke] +
                                   OffP[kh, ke])

    dpdt_result = dpdt_result / (ii * hbar)
    return dpdt_result


def dCdt(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffE):
    """
    Calculate the time derivative of the electron-electron coherence.

    Implements the Semiconductor Bloch Equation for the electron-electron coherence C,
    which describes the quantum correlations between different electron momentum states.
    The diagonal elements C_{k,k} = n_e(k) are the electron occupation numbers, while
    off-diagonal elements represent quantum coherence between different momentum states.

    The equation solved is:
    iℏ dC_{k1,k2}/dt =
        + Σ_k' H_{k2,k'}^{ee} C_{k1,k'}          [k2 electron Hamiltonian]
        - Σ_k' H_{k',k1}^{ee} C_{k',k2}          [k1 electron Hamiltonian, commutator]
        + Σ_k' H_{k2,k'}^{eh} p_{k1,k'}^†        [e-h correlation via p†]
        - Σ_k' H_{k',k1}^{eh†} p_{k',k2}         [e-h correlation via Heh†]

    Note: Off-diagonal dephasing terms (commented out in original Fortran) are not included.

    Parameters
    ----------
    Cee : ndarray (complex), shape (Nk, Nk)
        Electron-electron coherence matrix C_{k1,k2}.
        Diagonal: C_{k,k} = n_e(k) is electron occupation.
        Off-diagonal: represents electron quantum correlations.
    Dhh : ndarray (complex), shape (Nk, Nk)
        Hole-hole coherence matrix D_{k1,k2} (not used in this equation,
        kept for interface compatibility).
    Phe : ndarray (complex), shape (Nk, Nk)
        Electron-hole coherence matrix p_{k_e,k_h}.
        Note: Fortran uses Phe for p, but it's the same as p from dpdt.
    Heh : ndarray (complex), shape (Nk, Nk)
        Electron-hole Hamiltonian matrix H_{k_e,k_h}^{eh}.
    Hee : ndarray (complex), shape (Nk, Nk)
        Electron-electron Hamiltonian matrix H_{k1,k2}^{ee}.
        Diagonal: kinetic energy E_e(k).
        Off-diagonal: screened Coulomb repulsion between electrons.
    Hhh : ndarray (complex), shape (Nk, Nk)
        Hole-hole Hamiltonian (not used, kept for interface compatibility).
    GamE : ndarray (float), shape (Nk,)
        Electron dephasing rates (not used in this equation).
    GamH : ndarray (float), shape (Nk,)
        Hole dephasing rates (not used, kept for interface compatibility).
    OffE : ndarray (complex), shape (Nk, Nk)
        Off-diagonal dephasing for electrons (not used in current implementation).

    Returns
    -------
    dCdt_result : ndarray (complex), shape (Nk, Nk)
        Time derivative dC/dt of electron-electron coherence (1/s).

    Notes
    -----
    This equation governs the evolution of electron correlations, including:
    - Single-particle evolution via Hee (phase evolution, energy renormalization)
    - Coupling to the interband polarization p via Heh
    - Generation of correlations due to Coulomb interaction

    Physical significance:
    - Diagonal elements: rate of change of electron occupation n_e(k)
    - Off-diagonal: buildup/decay of electron quantum coherence
    - Couples to the p equation via Heh, forming a closed system

    The off-diagonal dephasing terms involving GamE are commented out in the
    original Fortran code and not implemented here. If needed, they would add:
    - iℏ·I_{k1≠k2}·(γ_e(k1) + γ_e(k2))·C_{k1,k2}
    where I_{k1≠k2} is the anti-identity matrix.

    Uses JIT compilation for performance with automatic fallback.

    References
    ----------
    V. M. Axt and T. Kuhn, Rep. Prog. Phys. 67, 433 (2004), Eqs. 2.15-2.19.

    See Also
    --------
    dpdt : Time derivative of electron-hole coherence (interband polarization)
    dDdt : Time derivative of hole-hole coherence
    """
    # Try CUDA first, then JIT, then fallback
    if _HAS_CUDA:
        try:
            return _dCdt_cuda(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffE, ii, hbar)
        except Exception:
            # Fallback to JIT
            try:
                return _dCdt_jit(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffE, ii, hbar)
            except Exception:
                return _dCdt_fallback(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffE)
    else:
        # No CUDA, use JIT
        try:
            return _dCdt_jit(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffE, ii, hbar)
        except Exception:
            return _dCdt_fallback(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffE)


@jit(nopython=True, cache=True, parallel=True)
def _dCdt_jit(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffE, ii_val, hbar_val):
    """JIT-compiled dCdt calculation."""
    Nk = Cee.shape[0]
    dCdt_result = np.zeros((Nk, Nk), dtype=np.complex128)

    Hhe = np.conj(Heh.T)
    Peh = np.conj(Phe.T)

    for k2 in prange(Nk):
        for k1 in range(Nk):
            term1 = np.sum(Hee[k2, :] * Cee[k1, :])
            term2 = np.sum(Hee[:, k1] * Cee[:, k2])
            term3 = np.sum(Heh[k2, :] * Peh[k1, :])
            term4 = np.sum(Hhe[:, k1] * Phe[:, k2])

            dCdt_result[k1, k2] = term1 - term2 + term3 - term4

    dCdt_result = dCdt_result / (ii_val * hbar_val)
    return dCdt_result


# CUDA implementation for dCdt
if _HAS_CUDA:
    @cuda.jit
    def _dCdt_cuda_kernel(Cee_real, Cee_imag, Phe_real, Phe_imag,
                          Heh_real, Heh_imag, Hee_real, Hee_imag,
                          dCdt_real, dCdt_imag, Nk, ii_real, ii_imag, hbar_val):
        """CUDA kernel for dCdt calculation."""
        k2 = cuda.blockIdx.x
        k1 = cuda.threadIdx.x

        if k2 < Nk and k1 < Nk:
            # Compute Hhe = conj(Heh.T) and Peh = conj(Phe.T)
            # term1 = sum(Hee[k2, :] * Cee[k1, :])
            term1_real = 0.0
            term1_imag = 0.0
            for k in range(Nk):
                h_re = Hee_real[k2, k]
                h_im = Hee_imag[k2, k]
                c_re = Cee_real[k1, k]
                c_im = Cee_imag[k1, k]
                term1_real += h_re * c_re - h_im * c_im
                term1_imag += h_re * c_im + h_im * c_re

            # term2 = sum(Hee[:, k1] * Cee[:, k2])
            term2_real = 0.0
            term2_imag = 0.0
            for k in range(Nk):
                h_re = Hee_real[k, k1]
                h_im = Hee_imag[k, k1]
                c_re = Cee_real[k, k2]
                c_im = Cee_imag[k, k2]
                term2_real += h_re * c_re - h_im * c_im
                term2_imag += h_re * c_im + h_im * c_re

            # term3 = sum(Heh[k2, :] * Peh[k1, :]) where Peh = conj(Phe.T)
            term3_real = 0.0
            term3_imag = 0.0
            for k in range(Nk):
                h_re = Heh_real[k2, k]
                h_im = Heh_imag[k2, k]
                # Peh[k1, k] = conj(Phe[k, k1])
                p_re = Phe_real[k, k1]
                p_im = -Phe_imag[k, k1]  # Conjugate
                term3_real += h_re * p_re - h_im * p_im
                term3_imag += h_re * p_im + h_im * p_re

            # term4 = sum(Hhe[:, k1] * Phe[:, k2]) where Hhe = conj(Heh.T)
            term4_real = 0.0
            term4_imag = 0.0
            for k in range(Nk):
                # Hhe[k, k1] = conj(Heh[k1, k])
                h_re = Heh_real[k1, k]
                h_im = -Heh_imag[k1, k]  # Conjugate
                p_re = Phe_real[k, k2]
                p_im = Phe_imag[k, k2]
                term4_real += h_re * p_re - h_im * p_im
                term4_imag += h_re * p_im + h_im * p_re

            # Final result
            result_real = term1_real - term2_real + term3_real - term4_real
            result_imag = term1_imag - term2_imag + term3_imag - term4_imag

            # Divide by (ii * hbar)
            denom = ii_real * hbar_val
            denom_im = ii_imag * hbar_val
            denom_sq = denom * denom + denom_im * denom_im
            dCdt_real[k1, k2] = (result_real * denom + result_imag * denom_im) / denom_sq
            dCdt_imag[k1, k2] = (result_imag * denom - result_real * denom_im) / denom_sq

    def _dCdt_cuda(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffE, ii_val, hbar_val):
        """CUDA wrapper for dCdt."""
        Nk = Cee.shape[0]

        # Split complex arrays
        Cee_real = np.ascontiguousarray(Cee.real, dtype=np.float64)
        Cee_imag = np.ascontiguousarray(Cee.imag, dtype=np.float64)
        Phe_real = np.ascontiguousarray(Phe.real, dtype=np.float64)
        Phe_imag = np.ascontiguousarray(Phe.imag, dtype=np.float64)
        Heh_real = np.ascontiguousarray(Heh.real, dtype=np.float64)
        Heh_imag = np.ascontiguousarray(Heh.imag, dtype=np.float64)
        Hee_real = np.ascontiguousarray(Hee.real, dtype=np.float64)
        Hee_imag = np.ascontiguousarray(Hee.imag, dtype=np.float64)

        # Allocate device arrays
        d_Cee_real = cuda.to_device(Cee_real)
        d_Cee_imag = cuda.to_device(Cee_imag)
        d_Phe_real = cuda.to_device(Phe_real)
        d_Phe_imag = cuda.to_device(Phe_imag)
        d_Heh_real = cuda.to_device(Heh_real)
        d_Heh_imag = cuda.to_device(Heh_imag)
        d_Hee_real = cuda.to_device(Hee_real)
        d_Hee_imag = cuda.to_device(Hee_imag)

        # Allocate output
        d_dCdt_real = cuda.device_array((Nk, Nk), dtype=np.float64)
        d_dCdt_imag = cuda.device_array((Nk, Nk), dtype=np.float64)

        # Launch kernel
        blocks_per_grid = Nk
        threads_per_block = Nk

        ii_real = ii_val.real
        ii_imag = ii_val.imag

        _dCdt_cuda_kernel[blocks_per_grid, threads_per_block](
            d_Cee_real, d_Cee_imag, d_Phe_real, d_Phe_imag,
            d_Heh_real, d_Heh_imag, d_Hee_real, d_Hee_imag,
            d_dCdt_real, d_dCdt_imag, Nk, ii_real, ii_imag, hbar_val
        )

        # Copy back
        dCdt_real = d_dCdt_real.copy_to_host()
        dCdt_imag = d_dCdt_imag.copy_to_host()
        return dCdt_real + 1j * dCdt_imag
else:
    def _dCdt_cuda(*args, **kwargs):
        """Dummy CUDA function when CUDA is not available."""
        raise RuntimeError("CUDA not available")


def _dCdt_fallback(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffE):
    """Fallback dCdt calculation without JIT."""
    Nk = Cee.shape[0]
    dCdt_result = np.zeros((Nk, Nk), dtype=complex)

    Hhe = np.conj(Heh.T)
    Peh = np.conj(Phe.T)

    for k2 in range(Nk):
        for k1 in range(Nk):
            dCdt_result[k1, k2] = (np.sum(Hee[k2, :] * Cee[k1, :]) -
                                   np.sum(Hee[:, k1] * Cee[:, k2]) +
                                   np.sum(Heh[k2, :] * Peh[k1, :]) -
                                   np.sum(Hhe[:, k1] * Phe[:, k2]))

    dCdt_result = dCdt_result / (ii * hbar)
    return dCdt_result


def dDdt(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffH):
    """
    Calculate the time derivative of the hole-hole coherence.

    Implements the Semiconductor Bloch Equation for the hole-hole coherence D,
    which describes the quantum correlations between different hole momentum states.
    The diagonal elements D_{k,k} = n_h(k) are the hole occupation numbers, while
    off-diagonal elements represent quantum coherence between different momentum states.

    The equation solved is:
    iℏ dD_{k1,k2}/dt =
        + Σ_k' H_{k2,k'}^{hh} D_{k1,k'}          [k2 hole Hamiltonian]
        - Σ_k' H_{k',k1}^{hh} D_{k',k2}          [k1 hole Hamiltonian, commutator]
        + Σ_k' H_{k',k2}^{eh†} p_{k',k1}^†       [e-h correlation via p†]
        - Σ_k' H_{k1,k'}^{eh†} p_{k2,k'}         [e-h correlation via Heh†]

    Note: Off-diagonal dephasing terms (commented out in original Fortran) are not included.

    Parameters
    ----------
    Cee : ndarray (complex), shape (Nk, Nk)
        Electron-electron coherence (not used, kept for interface compatibility).
    Dhh : ndarray (complex), shape (Nk, Nk)
        Hole-hole coherence matrix D_{k1,k2}.
        Diagonal: D_{k,k} = n_h(k) is hole occupation.
        Off-diagonal: represents hole quantum correlations.
    Phe : ndarray (complex), shape (Nk, Nk)
        Electron-hole coherence matrix p_{k_e,k_h}.
    Heh : ndarray (complex), shape (Nk, Nk)
        Electron-hole Hamiltonian matrix H_{k_e,k_h}^{eh}.
    Hee : ndarray (complex), shape (Nk, Nk)
        Electron-electron Hamiltonian (not used, kept for interface compatibility).
    Hhh : ndarray (complex), shape (Nk, Nk)
        Hole-hole Hamiltonian matrix H_{k1,k2}^{hh}.
        Diagonal: kinetic energy E_h(k).
        Off-diagonal: screened Coulomb repulsion between holes.
    GamE : ndarray (float), shape (Nk,)
        Electron dephasing rates (not used, kept for interface compatibility).
    GamH : ndarray (float), shape (Nk,)
        Hole dephasing rates (not used in this equation).
    OffH : ndarray (complex), shape (Nk, Nk)
        Off-diagonal dephasing for holes (not used in current implementation).

    Returns
    -------
    dDdt_result : ndarray (complex), shape (Nk, Nk)
        Time derivative dD/dt of hole-hole coherence (1/s).

    Notes
    -----
    This equation is analogous to the electron equation (dCdt), but for holes.
    It governs the evolution of hole correlations, including:
    - Single-particle evolution via Hhh (phase evolution, energy renormalization)
    - Coupling to the interband polarization p via Heh
    - Generation of correlations due to Coulomb interaction

    Physical significance:
    - Diagonal elements: rate of change of hole occupation n_h(k)
    - Off-diagonal: buildup/decay of hole quantum coherence
    - Couples to the p equation via Heh, forming a closed system with dpdt and dCdt

    The three equations (dpdt, dCdt, dDdt) form the complete set of Semiconductor
    Bloch Equations that must be solved self-consistently. Together they describe:
    - Optical response (via p)
    - Carrier dynamics (via diagonal C and D)
    - Quantum correlations and many-body effects (via off-diagonal C and D)

    The off-diagonal dephasing terms involving GamH are commented out in the
    original Fortran code and not implemented here. If needed, they would add:
    - iℏ·I_{k1≠k2}·(γ_h(k1) + γ_h(k2))·D_{k1,k2}
    where I_{k1≠k2} is the anti-identity matrix.

    Uses JIT compilation for performance with automatic fallback.

    References
    ----------
    V. M. Axt and T. Kuhn, "Femtosecond spectroscopy in semiconductors",
    Rep. Prog. Phys. 67, 433 (2004), Eqs. 2.15-2.19.

    See Also
    --------
    dpdt : Time derivative of electron-hole coherence
    dCdt : Time derivative of electron-electron coherence
    """
    # Try CUDA first, then JIT, then fallback
    if _HAS_CUDA:
        try:
            return _dDdt_cuda(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffH, ii, hbar)
        except Exception:
            # Fallback to JIT
            try:
                return _dDdt_jit(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffH, ii, hbar)
            except Exception:
                return _dDdt_fallback(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffH)
    else:
        # No CUDA, use JIT
        try:
            return _dDdt_jit(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffH, ii, hbar)
        except Exception:
            return _dDdt_fallback(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffH)


@jit(nopython=True, cache=True, parallel=True)
def _dDdt_jit(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffH, ii_val, hbar_val):
    """JIT-compiled dDdt calculation."""
    Nk = Dhh.shape[0]
    dDdt_result = np.zeros((Nk, Nk), dtype=np.complex128)

    Peh = np.conj(Phe.T)
    Hhe = np.conj(Heh.T)

    for k2 in prange(Nk):
        for k1 in range(Nk):
            term1 = np.sum(Hhh[k2, :] * Dhh[k1, :])
            term2 = np.sum(Hhh[:, k1] * Dhh[:, k2])
            term3 = np.sum(Heh[:, k2] * Peh[:, k1])
            term4 = np.sum(Hhe[k1, :] * Phe[k2, :])

            dDdt_result[k1, k2] = term1 - term2 + term3 - term4

    dDdt_result = dDdt_result / (ii_val * hbar_val)
    return dDdt_result


# CUDA implementation for dDdt
if _HAS_CUDA:
    @cuda.jit
    def _dDdt_cuda_kernel(Dhh_real, Dhh_imag, Phe_real, Phe_imag,
                          Heh_real, Heh_imag, Hhh_real, Hhh_imag,
                          dDdt_real, dDdt_imag, Nk, ii_real, ii_imag, hbar_val):
        """CUDA kernel for dDdt calculation."""
        k2 = cuda.blockIdx.x
        k1 = cuda.threadIdx.x

        if k2 < Nk and k1 < Nk:
            # Compute Peh = conj(Phe.T) and Hhe = conj(Heh.T)
            # term1 = sum(Hhh[k2, :] * Dhh[k1, :])
            term1_real = 0.0
            term1_imag = 0.0
            for k in range(Nk):
                h_re = Hhh_real[k2, k]
                h_im = Hhh_imag[k2, k]
                d_re = Dhh_real[k1, k]
                d_im = Dhh_imag[k1, k]
                term1_real += h_re * d_re - h_im * d_im
                term1_imag += h_re * d_im + h_im * d_re

            # term2 = sum(Hhh[:, k1] * Dhh[:, k2])
            term2_real = 0.0
            term2_imag = 0.0
            for k in range(Nk):
                h_re = Hhh_real[k, k1]
                h_im = Hhh_imag[k, k1]
                d_re = Dhh_real[k, k2]
                d_im = Dhh_imag[k, k2]
                term2_real += h_re * d_re - h_im * d_im
                term2_imag += h_re * d_im + h_im * d_re

            # term3 = sum(Heh[:, k2] * Peh[:, k1]) where Peh = conj(Phe.T)
            term3_real = 0.0
            term3_imag = 0.0
            for k in range(Nk):
                h_re = Heh_real[k, k2]
                h_im = Heh_imag[k, k2]
                # Peh[:, k1] = conj(Phe[k1, :]) = conj(Phe[k1, k])
                p_re = Phe_real[k1, k]
                p_im = -Phe_imag[k1, k]  # Conjugate
                term3_real += h_re * p_re - h_im * p_im
                term3_imag += h_re * p_im + h_im * p_re

            # term4 = sum(Hhe[k1, :] * Phe[k2, :]) where Hhe = conj(Heh.T)
            term4_real = 0.0
            term4_imag = 0.0
            for k in range(Nk):
                # Hhe[k1, k] = conj(Heh[k, k1])
                h_re = Heh_real[k, k1]
                h_im = -Heh_imag[k, k1]  # Conjugate
                p_re = Phe_real[k2, k]
                p_im = Phe_imag[k2, k]
                term4_real += h_re * p_re - h_im * p_im
                term4_imag += h_re * p_im + h_im * p_re

            # Final result
            result_real = term1_real - term2_real + term3_real - term4_real
            result_imag = term1_imag - term2_imag + term3_imag - term4_imag

            # Divide by (ii * hbar)
            denom = ii_real * hbar_val
            denom_im = ii_imag * hbar_val
            denom_sq = denom * denom + denom_im * denom_im
            dDdt_real[k1, k2] = (result_real * denom + result_imag * denom_im) / denom_sq
            dDdt_imag[k1, k2] = (result_imag * denom - result_real * denom_im) / denom_sq

    def _dDdt_cuda(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffH, ii_val, hbar_val):
        """CUDA wrapper for dDdt."""
        Nk = Dhh.shape[0]

        # Split complex arrays
        Dhh_real = np.ascontiguousarray(Dhh.real, dtype=np.float64)
        Dhh_imag = np.ascontiguousarray(Dhh.imag, dtype=np.float64)
        Phe_real = np.ascontiguousarray(Phe.real, dtype=np.float64)
        Phe_imag = np.ascontiguousarray(Phe.imag, dtype=np.float64)
        Heh_real = np.ascontiguousarray(Heh.real, dtype=np.float64)
        Heh_imag = np.ascontiguousarray(Heh.imag, dtype=np.float64)
        Hhh_real = np.ascontiguousarray(Hhh.real, dtype=np.float64)
        Hhh_imag = np.ascontiguousarray(Hhh.imag, dtype=np.float64)

        # Allocate device arrays
        d_Dhh_real = cuda.to_device(Dhh_real)
        d_Dhh_imag = cuda.to_device(Dhh_imag)
        d_Phe_real = cuda.to_device(Phe_real)
        d_Phe_imag = cuda.to_device(Phe_imag)
        d_Heh_real = cuda.to_device(Heh_real)
        d_Heh_imag = cuda.to_device(Heh_imag)
        d_Hhh_real = cuda.to_device(Hhh_real)
        d_Hhh_imag = cuda.to_device(Hhh_imag)

        # Allocate output
        d_dDdt_real = cuda.device_array((Nk, Nk), dtype=np.float64)
        d_dDdt_imag = cuda.device_array((Nk, Nk), dtype=np.float64)

        # Launch kernel
        blocks_per_grid = Nk
        threads_per_block = Nk

        ii_real = ii_val.real
        ii_imag = ii_val.imag

        _dDdt_cuda_kernel[blocks_per_grid, threads_per_block](
            d_Dhh_real, d_Dhh_imag, d_Phe_real, d_Phe_imag,
            d_Heh_real, d_Heh_imag, d_Hhh_real, d_Hhh_imag,
            d_dDdt_real, d_dDdt_imag, Nk, ii_real, ii_imag, hbar_val
        )

        # Copy back
        dDdt_real = d_dDdt_real.copy_to_host()
        dDdt_imag = d_dDdt_imag.copy_to_host()
        return dDdt_real + 1j * dDdt_imag
else:
    def _dDdt_cuda(*args, **kwargs):
        """Dummy CUDA function when CUDA is not available."""
        raise RuntimeError("CUDA not available")


def _dDdt_fallback(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffH):
    """Fallback dDdt calculation without JIT."""
    Nk = Dhh.shape[0]
    dDdt_result = np.zeros((Nk, Nk), dtype=complex)

    Peh = np.conj(Phe.T)
    Hhe = np.conj(Heh.T)

    for k2 in range(Nk):
        for k1 in range(Nk):
            dDdt_result[k1, k2] = (np.sum(Hhh[k2, :] * Dhh[k1, :]) -
                                   np.sum(Hhh[:, k1] * Dhh[:, k2]) +
                                   np.sum(Heh[:, k2] * Peh[:, k1]) -
                                   np.sum(Hhe[k1, :] * Phe[k2, :]))

    dDdt_result = dDdt_result / (ii * hbar)
    return dDdt_result


def write_statistics(w, dt, ne2, nh2, Re, Rh):
    """
    Write statistical data to file.

    Writes detailed carrier statistics including velocities, densities,
    energies, temperatures, and currents.

    Parameters
    ----------
    w : int
        Wire index
    dt : float
        Time step
    ne2 : ndarray
        Electron occupation numbers
    nh2 : ndarray
        Hole occupation numbers
    Re : ndarray
        Electron charge density
    Rh : ndarray
        Hole charge density
    """
    global _uw, _xxx, _kr, _me, _mh, _L, _Ee, _Eh, _gap, _I0, _small

    # Calculate various quantities
    ve = CalcVD(_kr, _me, ne2)
    vh = CalcVD(_kr, _mh, nh2)
    ne_density = np.real(2 * np.sum(ne2) / _L)
    nh_density = np.real(2 * np.sum(nh2) / _L)
    Ee_total = TotalEnergy(ne2, _Ee + _gap) / e0
    Eh_total = TotalEnergy(nh2, _Eh) / e0
    Te = Temperature(ne2, _Ee)
    Th = Temperature(nh2, _Eh)
    re_max = np.max(np.abs(Re))
    rh_max = np.max(np.abs(Rh))
    EDrift = GetEDrift()
    momentum = np.real(hbar * np.sum((ne2 - nh2) * _kr)) * 2.0
    pe = np.real(hbar * np.sum(ne2 * _kr))
    ph = np.real(-hbar * np.sum(nh2 * _kr))
    I0_val = _I0[w] if w < len(_I0) else 0.0

    # Calculate current-related quantities
    Ie = CalcI0n(ne2, _me, _kr)
    Ih = CalcI0n(1 - nh2, -_mh, _kr)

    # Write to file
    os.makedirs('output', exist_ok=True)
    with open(f'output/stats_{_uw+w:03d}.dat', 'a', encoding='utf-8') as f:
        f.write(f"{_xxx*dt:18.6e} {ve:18.6e} {vh:18.6e} "
                f"{ne_density:18.6e} {nh_density:18.6e} "
                f"{Ee_total:18.6e} {Eh_total:18.6e} "
                f"{Te:18.6e} {Th:18.6e} "
                f"{re_max:18.6e} {rh_max:18.6e} "
                f"{EDrift:18.6e} {momentum:18.6e} "
                f"{pe:18.6e} {ph:18.6e} "
                f"{I0_val:18.6e} {Ie:18.6e} {Ih:18.6e}\n")


def CalcMeh(Ex, Ey, Ez, w, Meh):
    """
    Calculate the dipole-field coupling matrix for electron-hole transitions.

    Computes the matrix elements of the light-matter interaction Hamiltonian
    M_{k_e,k_h}^{eh} which couples the electric field to the interband polarization.
    This is the driving term in the optical Bloch equations that causes absorption
    and emission of photons.

    The matrix elements are calculated as:
    M_{k_e,k_h}^{eh} = -η_{int} [X_{cv}(k_e,k_h)·E_x(q) + Y_{cv}(k_e,k_h)·E_y(q)·y_w(w)
                                  + Z_{cv}(k_e,k_h)·E_z(q)·(-1)^w]

    where:
    - η_{int} is the electron-hole spatial overlap integral
    - X_{cv}, Y_{cv}, Z_{cv} are the momentum-dependent dipole matrix elements
    - q = k_e - k_h is the momentum transfer (photon momentum)
    - y_w(w) is a wire-dependent weighting factor
    - (-1)^w alternates sign for different wires

    Parameters
    ----------
    Ex : ndarray (complex), shape (Nr,)
        X-component of electric field in momentum space (Fourier transformed).
        Indexed by momentum q.
    Ey : ndarray (complex), shape (Nr,)
        Y-component of electric field in momentum space.
        Multiplied by wire-dependent factor yw(w).
    Ez : ndarray (complex), shape (Nr,)
        Z-component of electric field in momentum space.
        Multiplied by (-1)^w for wire parity.
    w : int
        Wire index. Used to determine y-polarization weight yw(w) and
        z-polarization sign (-1)^w.
    Meh : ndarray (complex), shape (Nk, Nk)
        Output matrix for dipole-field coupling (modified in-place).

    Returns
    -------
    None
        Meh is modified in-place.

    Notes
    -----
    This function is central to the optical response of the semiconductor.
    The coupling matrix Meh appears in the SBE for the electron-hole coherence p:

    iℏ dp/dt = ... + M^{eh}_{k_e,k_h} + ...

    The momentum index q = k_e - k_h represents the photon momentum absorbed/emitted
    in a transition. This is obtained via the index mapping array kkp(ke, kh).

    The spatial overlap integral η_{int} (ehint) accounts for the finite size
    of the quantum wire and reduces the coupling strength compared to the
    bare dipole moment.

    The factors yw(w) and (-1)^w allow for different polarization responses
    in different wires, enabling modeling of wire arrays with varying orientations.

    Important: The comment in the Fortran code notes that previously there was
    confusion about whether to use FFT or IFFT for the fields, and that q should
    be ke - kh. This has been corrected in the current implementation.

    See Also
    --------
    Xcv, Ycv, Zcv : Dipole matrix elements (in qwoptics module)
    dpdt : The SBE that uses this coupling matrix
    """
    global _kkp, _ehint, _Nk

    Meh[:, :] = 0.0 + 0.0j


    for kh in range(_Nk):
        for ke in range(_Nk):
            # q = ke - kh (momentum transfer, photon momentum)
            # Corrected: We don't use ifft anymore, so q is truly ke - kh
            q = _kkp[ke, kh]

            # Dipole-field coupling: M = -η·(X·Ex + Y·Ey·yw + Z·Ez·(-1)^w)
            Meh[ke, kh] = (-_ehint * Xcv(ke, kh) * Ex[q]
                          - _ehint * Ycv(ke, kh) * Ey[q] * yw(w)
                          - _ehint * Zcv(ke, kh) * Ez[q] * (-1)**w)


def CalcWnn(q0, Vr, Wnn):
    """
    Calculate the monopole-potential coupling matrix for carrier-carrier interactions.

    Computes the matrix elements of the free-charge potential coupling
    W_{k1,k2}^{nn} which represents the effect of long-range Coulomb potential
    from free carriers on the carrier states. This contributes to the longitudinal
    field effects in the SBEs.

    The matrix elements are:
    W_{k1,k2}^{nn} = q_0 · V_r(q)

    where:
    - q_0 is the charge of the carrier (+e for holes, -e for electrons)
    - V_r(q) is the electric potential from free charge in momentum space
    - q = k1 - k2 is the momentum transfer

    Parameters
    ----------
    q0 : float
        Free charge of carrier species:
        +e0 (positive) for holes
        -e0 (negative) for electrons
        where e0 is the elementary charge (C)
    Vr : ndarray (complex), shape (Nr,)
        Electric potential from free charge in momentum space (Fourier transformed).
        This is the long-range Coulomb potential V(r) = ρ(r)/(4πε_0ε_r) in q-space.
    Wnn : ndarray (complex), shape (Nk, Nk)
        Output monopole-potential coupling matrix (modified in-place).

    Returns
    -------
    None
        Wnn is modified in-place.

    Notes
    -----
    This function calculates the coupling between carriers and the self-consistent
    mean-field potential arising from the carrier density distribution. It's used
    when the longitudinal field (_LF) is enabled to include plasmon effects and
    long-range Coulomb interactions.

    The coupling matrix Wnn would appear in the SBEs as additional diagonal terms:
    - For electrons: H_{k1,k2}^{ee} += W_{k1,k2}^{ee} (with q0 = -e0)
    - For holes: H_{k1,k2}^{hh} += W_{k1,k2}^{hh} (with q0 = +e0)

    The momentum transfer q = k1 - k2 is obtained via the index mapping kkp(k1, k2).

    Note: This function is currently not actively used in the CalcH function (it's
    commented out in the Fortran code), but is kept for potential future use or
    for studying longitudinal field effects.

    Important: As noted in the Fortran comments, there was previously confusion
    about FFT vs IFFT conventions, which has been corrected. The q index now
    properly represents q = k1 - k2.

    See Also
    --------
    CalcMeh : Dipole-field coupling matrix for optical transitions
    CalcH : Hamiltonian calculation (where Wnn would be used)
    """
    global _kkp, _Nk

    Wnn[:, :] = 0.0 + 0.0j

    for k2 in range(_Nk):
        for k1 in range(_Nk):
            # q = k1 - k2 (momentum transfer)
            # Corrected: We don't use ifft anymore, so q is truly k1 - k2
            q = _kkp[k1, k2]

            # Monopole-potential coupling
            Wnn[k1, k2] = q0 * Vr[q]


def CalcXqw(iq, w, kr, fe, fh, Ee, Eh, gap, area, game, gamh, dcv):
    """
    Calculate the linear optical susceptibility χ(q,ω) for a quantum wire.

    Computes the frequency and momentum-dependent linear susceptibility using
    the Lindhard formula generalized to include population inversion (fe, fh)
    and dephasing rates (game, gamh). This describes the linear optical response
    of the quantum wire system to an external electromagnetic field.

    The susceptibility includes contributions from both absorption (fc < fv) and
    stimulated emission (fc > fv) processes at each momentum state.

    Physics
    -------
    The linear susceptibility is calculated as:

    χ(q,ω) = Σ_k [N·|d_cv|²/ε₀] × {
        (f_c(k) - f_v(k+q)) / [E_v(k+q) - E_c(k) - ℏω - iℏ(γ_h(k+q) + γ_e(k))]
      + (f_v(k) - f_c(k+q)) / [E_c(k+q) - E_v(k) - ℏω - iℏ(γ_e(k+q) + γ_h(k))]
    }

    where:
    - N = 2·Δk/(2π·A) is the density of states normalization
    - d_cv is the interband dipole matrix element
    - f_c, f_v are electron (conduction) and hole (valence) occupation numbers
    - E_c, E_v are conduction and valence band energies
    - γ_e, γ_h are electron and hole dephasing rates
    - q is the photon momentum (momentum transfer)
    - ω is the photon frequency

    The first term represents transitions where an electron in state k absorbs
    a photon with momentum q, while the second term represents the time-reversed
    process (stimulated emission).

    Parameters
    ----------
    iq : int
        Momentum index for photon momentum q. Can be positive, negative, or zero.
        The actual momentum is q = iq * (kr[1] - kr[0]).
    w : float
        Angular frequency ω (rad/s) of the electromagnetic field.
    kr : ndarray (float), shape (Nk,)
        Momentum grid (rad/m) for carrier states.
    fe : ndarray (float), shape (Nk,)
        Electron occupation numbers f_e(k) at each momentum state.
        Range: [0, 1] for physical occupation.
    fh : ndarray (float), shape (Nk,)
        Hole occupation numbers f_h(k) at each momentum state.
        Range: [0, 1] for physical occupation.
    Ee : ndarray (float), shape (Nk,)
        Electron energies E_e(k) relative to conduction band edge (J).
    Eh : ndarray (float), shape (Nk,)
        Hole energies E_h(k) relative to valence band edge (J).
    gap : float
        Band gap energy E_g (J).
    area : float
        Cross-sectional area A of the quantum wire (m²).
    game : ndarray (float), shape (Nk,)
        Electron dephasing rates γ_e(k) (1/s).
    gamh : ndarray (float), shape (Nk,)
        Hole dephasing rates γ_h(k) (1/s).
    dcv : complex
        Interband dipole matrix element d_cv (C·m).

    Returns
    -------
    Xqw : complex
        Linear optical susceptibility χ(q,ω) (dimensionless).
        Real part: related to refractive index change Δn.
        Imaginary part: related to absorption/gain coefficient α.

    Notes
    -----
    The summation range is carefully chosen to avoid out-of-bounds array access:
    - For iq ≥ 0: kmin = 0, kmax = Nk - 1 - iq (Python indexing)
    - For iq < 0: kmin = -iq, kmax = Nk - 1 (Python indexing)

    This ensures that both k and k+iq remain within [0, Nk-1].

    The minimum dephasing is set to 1e-4 eV to prevent numerical instabilities
    from vanishing denominators.

    The valence band occupations are f_v = 1 - f_h, as holes represent absence
    of electrons in the valence band.

    The imaginary part Im[χ(q,ω)] > 0 indicates absorption, while Im[χ] < 0
    indicates gain (population inversion). The real part Re[χ(q,ω)] is related
    to dispersion via the Kramers-Kronig relations.

    Physical applications:
    - Optical absorption/gain spectra
    - Refractive index changes
    - Electromagnetically induced transparency (EIT)
    - Population inversion and lasing conditions
    - Exciton resonances in quantum wires

    References
    ----------
    H. Haug and S. W. Koch, "Quantum Theory of the Optical and Electronic
    Properties of Semiconductors", 5th ed., World Scientific (2009), Ch. 13.

    See Also
    --------
    QWChi1 : Related function for calculating χ from coulomb module
    RecordXqw : Function that records χ(q,ω) data to files
    GetChi1Dqw : 1D susceptibility calculation from coulomb module

    Examples
    --------
    >>> # Calculate susceptibility at q=0, near band edge
    >>> iq = 0
    >>> w = gap / hbar  # Frequency at band gap
    >>> chi = CalcXqw(iq, w, kr, fe, fh, Ee, Eh, gap, area, game, gamh, dcv)
    >>> alpha = 2 * w / c * np.imag(chi)  # Absorption coefficient
    """
    global _Nk, _twopi

    # Try JIT-compiled version first
    try:
        return _CalcXqw_jit(iq, w, kr, fe, fh, Ee, Eh, gap, area, game, gamh,
                            dcv, _Nk, _twopi, hbar, e0, eps0, ii)
    except Exception:
        return _CalcXqw_fallback(iq, w, kr, fe, fh, Ee, Eh, gap, area, game,
                                 gamh, dcv)


@jit(nopython=True)
def _CalcXqw_jit(iq, w, kr, fe, fh, Ee, Eh, gap, area, game, gamh, dcv,
                 Nk, twopi, hbar_val, e0_val, eps0_val, ii_val):
    """JIT-compiled version of CalcXqw."""
    dkr = kr[1] - kr[0]

    # Allocate arrays
    hge = np.zeros(Nk)
    hgh = np.zeros(Nk)
    Ec = np.zeros(Nk)
    Ev = np.zeros(Nk)
    fc = np.zeros(Nk)
    fv = np.zeros(Nk)

    # Minimum dephasing (1e-4 eV in Joules)
    min_deph = 1e-4 * e0_val

    # Calculate energy-scale dephasing rates with minimum threshold
    for k in range(Nk):
        hge[k] = max(min_deph, hbar_val * game[k])
        hgh[k] = max(min_deph, hbar_val * gamh[k])

    # Energy arrays: conduction and valence band energies
    for k in range(Nk):
        Ec[k] = gap + Ee[k]
        Ev[k] = -Eh[k]
        fc[k] = fe[k]
        fv[k] = 1.0 - fh[k]  # Valence occupation = 1 - hole occupation

    # Normalization factor
    N = 2.0 * dkr / twopi / area
    x = N * dcv * dcv / eps0_val

    Xqw = 0.0 + 0.0j

    # Determine summation range to avoid out-of-bounds access
    # Fortran: kmin = max(1-iq, 1), kmax = min(Nk-iq, Nk)
    # Python:  kmin = max(0-iq, 0), kmax = min(Nk-1-iq, Nk-1) + 1 for range
    if iq >= 0:
        kmin = 0
        kmax = Nk - iq
    else:
        kmin = -iq
        kmax = Nk

    # First sum: absorption from valence to conduction
    for ik in range(kmin, kmax):
        numerator = x * (fc[ik] - fv[ik + iq])
        denominator = (Ev[ik + iq] - Ec[ik] - hbar_val * w
                      - ii_val * (hgh[ik + iq] + hge[ik]))
        Xqw += numerator / denominator

    # Second sum: stimulated emission from conduction to valence
    for ik in range(kmin, kmax):
        numerator = x * (fv[ik] - fc[ik + iq])
        denominator = (Ec[ik + iq] - Ev[ik] - hbar_val * w
                      - ii_val * (hge[ik + iq] + hgh[ik]))
        Xqw += numerator / denominator

    return Xqw


def _CalcXqw_fallback(iq, w, kr, fe, fh, Ee, Eh, gap, area, game, gamh, dcv):
    """Pure Python fallback for CalcXqw."""
    global _Nk, _twopi

    dkr = kr[1] - kr[0]

    # Minimum dephasing (1e-4 eV)
    min_deph = 1e-4 * e0
    hge = np.maximum(min_deph, hbar * game)
    hgh = np.maximum(min_deph, hbar * gamh)

    # Energy arrays
    Ec = gap + Ee
    Ev = -Eh
    fc = fe.copy()
    fv = 1.0 - fh

    # Normalization
    N = 2.0 * dkr / _twopi / area
    x = N * dcv**2 / eps0

    Xqw = 0.0 + 0.0j

    # Determine summation range
    if iq >= 0:
        kmin = 0
        kmax = _Nk - iq
    else:
        kmin = -iq
        kmax = _Nk

    # First sum: absorption
    for ik in range(kmin, kmax):
        Xqw += (x * (fc[ik] - fv[ik + iq]) /
                (Ev[ik + iq] - Ec[ik] - hbar * w - ii * (hgh[ik + iq] + hge[ik])))

    # Second sum: stimulated emission
    for ik in range(kmin, kmax):
        Xqw += (x * (fv[ik] - fc[ik + iq]) /
                (Ec[ik + iq] - Ev[ik] - hbar * w - ii * (hge[ik + iq] + hgh[ik])))

    return Xqw


def RecordEpsLqw(Qr, fe, fh, Ee, Eh, gap, area, gamE, gamH, dcv, ind):
    """
    Record the longitudinal dielectric function ε_L(q,ω) and susceptibility χ_L(q,ω).

    Calculates and saves to file the frequency and momentum-dependent longitudinal
    dielectric function and linear susceptibility for a quantum wire. This provides
    a complete characterization of the collective excitations (plasmons) and
    optical response of the system.

    The longitudinal dielectric function describes the screening of longitudinal
    electric fields (parallel to the wire axis) and determines the dispersion
    relation of plasmon modes via ε_L(q,ω) = 0.

    Parameters
    ----------
    Qr : ndarray (float), shape (Nq,)
        Momentum grid (rad/m) for the output. Determines the q-resolution.
    fe : ndarray (float), shape (Nk,)
        Electron occupation numbers f_e(k).
    fh : ndarray (float), shape (Nk,)
        Hole occupation numbers f_h(k).
    Ee : ndarray (float), shape (Nk,)
        Electron energies E_e(k) (J).
    Eh : ndarray (float), shape (Nk,)
        Hole energies E_h(k) (J).
    gap : float
        Band gap energy E_g (J).
    area : float
        Wire cross-sectional area A (m²).
    gamE : ndarray (float), shape (Nk,)
        Electron dephasing rates γ_e(k) (1/s).
    gamH : ndarray (float), shape (Nk,)
        Hole dephasing rates γ_h(k) (1/s).
    dcv : complex
        Interband dipole matrix element d_cv (C·m).
    ind : int
        File index for output filename generation.

    Returns
    -------
    None
        Data is written to files in 'dataQW/Wire/Xqw/' directory.

    Files Created
    -------------
    dataQW/Wire/Xqw/EpsL.{ind:06d}.dat :
        Contains Re[ε_L(q,ω) - 1] and Im[ε_L(q,ω)] in two columns.
        Data is organized as nested loops: outer loop over ω, inner loop over q.
        Total size: (nwmax+1) × (Nq+1) rows, 2 columns.

    dataQW/Wire/Xqw/ChiL.{ind:06d}.dat :
        Contains Re[χ_L(q,ω)] and Im[χ_L(q,ω)] in two columns.
        Same structure as EpsL file.

    dataQW/Wire/Xqw/EpsL.params :
        Parameter file (written only once when _Xqwparams is True).
        Contains grid information: Nq, Nw, Δω, Δq, frequency range, momentum range.

    Notes
    -----
    The frequency grid is automatically generated:
    - ω_max = 2·E_g/ℏ (twice the band gap frequency)
    - Δω = ω_max / 2000
    - Number of points: 2001 (from ω=0 to ω_max)

    The momentum grid uses the input kr spacing:
    - Δq = kr[1] - kr[0]
    - q ranges from 0 to Nk·Δq

    The 1D carrier density is calculated as:
    n_1D = Σ(f_e + f_h) / (2·L)
    but is currently overridden to a fixed value n_1D = 1.5×10⁸ m⁻¹ for testing.

    Two different methods are used:
    1. GetEps1Dqw: Analytical RPA dielectric function (from coulomb module)
    2. GetChi1Dqw: Numerical susceptibility from actual occupations (from coulomb module)

    The longitudinal dielectric function is related to susceptibility by:
    ε_L(q,ω) = 1 + V(q)·χ_L(q,ω)
    where V(q) is the 1D Coulomb interaction.

    Physical applications:
    - Plasmon dispersion relations (zeros of ε_L)
    - Collective oscillation frequencies
    - Screening length determination
    - Optical absorption with many-body effects
    - Carrier density diagnostics

    The data format allows easy plotting as 2D contour plots with:
    - X-axis: momentum q
    - Y-axis: frequency ω
    - Color: Re[ε_L] or Im[ε_L]

    References
    ----------
    G. D. Mahan, "Many-Particle Physics", 3rd ed., Springer (2000), Ch. 5.

    See Also
    --------
    CalcXqw : Calculate linear susceptibility at single (q,ω) point
    GetEps1Dqw : Analytical 1D dielectric function (coulomb module)
    GetChi1Dqw : Numerical 1D susceptibility (coulomb module)
    RecordXqw : Similar function for recording χ(q,ω) using CalcXqw

    Examples
    --------
    >>> # Record dielectric function at time step 1000
    >>> RecordEpsLqw(kr, fe, fh, Ee, Eh, gap, area, gamE, gamH, dcv, ind=1000)
    >>> # Files created: EpsL.001000.dat, ChiL.001000.dat
    """
    global _alphae, _alphah, _Delta0, _L, _epsr, _me, _mh, _kr, _Xqwparams

    # Frequency grid parameters
    wmax = gap / hbar * 2.0
    dw = wmax / 2000.0
    nwmax = int(wmax / dw)

    # Calculate 1D carrier density
    n1D = np.sum(fe + fh) / 2.0 / _L
    n1D = 1.5e8  # Override with fixed value (as in Fortran: !2.908d8)

    # Create output directory
    os.makedirs('dataQW/Wire/Xqw', exist_ok=True)

    # Generate filenames with zero-padded index
    filename = f'dataQW/Wire/Xqw/EpsL.{ind:06d}.dat'
    filename2 = f'dataQW/Wire/Xqw/ChiL.{ind:06d}.dat'

    # Open output files
    with open(filename, 'w', encoding='utf-8') as f_eps, \
         open(filename2, 'w', encoding='utf-8') as f_chi:

        # Loop over frequencies
        for iw in range(nwmax + 1):
            w = iw * dw

            # Loop over momenta
            for iq in range(len(_kr) + 1):
                q = iq * (_kr[1] - _kr[0])

                # Calculate analytical RPA dielectric function
                epr_eps, epi_eps = GetEps1Dqw(_alphae, _alphah, _Delta0, _L,
                                               _epsr, _me, _mh, n1D, q, w)
                f_eps.write(f'{epr_eps - 1.0:12.3e} {epi_eps:12.3e}\n')

                # Calculate numerical susceptibility from actual populations
                epr_chi, epi_chi = GetChi1Dqw(_alphae, _alphah, _Delta0, _L,
                                               _epsr, gamE, gamH, _kr, Ee, Eh,
                                               fe, fh, q, w)
                f_chi.write(f'{epr_chi:12.3e} {epi_chi:12.3e}\n')

    # Write parameter file (only once)
    if _Xqwparams:
        dkr = _kr[1] - _kr[0]
        with open('dataQW/Wire/Xqw/EpsL.params', 'w', encoding='utf-8') as f_params:
            f_params.write(f"Nq = {len(Qr)}\n")
            f_params.write(f"Nw = 2000\n")
            f_params.write(f" \n")
            f_params.write(f"hbar dw (eV) = {dw * hbar / e0}\n")
            f_params.write(f"dq (rad/m) = {dkr}\n")
            f_params.write(f" \n")
            f_params.write(f"wmin = {-wmax * hbar / e0}\n")
            f_params.write(f"wmax = {wmax * hbar / e0}\n")
            f_params.write(f"qmin = {Qr[0]}\n")
            f_params.write(f"qmax = {Qr[-1]}\n")
        _Xqwparams = False


def RecordXqw(kr, fe, fh, Ee, Eh, gap, area, game, gamh, dcv, ind):
    """
    Record the linear optical susceptibility χ(q,ω) using CalcXqw.

    Calculates and saves to file the frequency and momentum-dependent linear
    susceptibility χ(q,ω) for a quantum wire using the direct CalcXqw function.
    This provides the complete optical response spectrum including absorption,
    gain, and dispersion.

    Unlike RecordEpsLqw which uses analytical RPA functions, this uses the
    CalcXqw function that directly sums over carrier occupations, providing
    a more accurate representation when populations deviate from equilibrium.

    Parameters
    ----------
    kr : ndarray (float), shape (Nk,)
        Momentum grid (rad/m) for carrier states.
    fe : ndarray (float), shape (Nk,)
        Electron occupation numbers f_e(k).
    fh : ndarray (float), shape (Nk,)
        Hole occupation numbers f_h(k).
    Ee : ndarray (float), shape (Nk,)
        Electron energies E_e(k) (J).
    Eh : ndarray (float), shape (Nk,)
        Hole energies E_h(k) (J).
    gap : float
        Band gap energy E_g (J).
    area : float
        Wire cross-sectional area A (m²).
    game : ndarray (float), shape (Nk,)
        Electron dephasing rates γ_e(k) (1/s).
    gamh : ndarray (float), shape (Nk,)
        Hole dephasing rates γ_h(k) (1/s).
    dcv : complex
        Interband dipole matrix element d_cv (C·m).
    ind : int
        File index for output filename generation.

    Returns
    -------
    None
        Data is written to files in 'dataQW/Wire/Xqw/' directory.

    Files Created
    -------------
    dataQW/Wire/Xqw/Xqw.{ind:06d}.dat :
        Contains three columns: ω (rad/s), Re[χ(q,ω)], Im[χ(q,ω)].
        Data is organized as nested loops: outer loop over ω, inner loop over q.
        Total size: (nwmax+1) × (Nk+1) rows, 3 columns.

    dataQW/Wire/Xqw/Xqw.params :
        Parameter file (written only once when _Xqwparams is True).
        Contains grid information: Nq, Nw, Δω, Δq, frequency range, momentum range.

    Notes
    -----
    The frequency grid is automatically generated:
    - ω_max = 2·E_g/ℏ (twice the band gap frequency)
    - Δω = ω_max / 2000
    - Number of points: 2001 (from ω=0 to ω_max)

    The momentum grid uses indices iq from 0 to Nk:
    - q ranges from 0 to Nk·Δkr
    - Δkr = kr[1] - kr[0]

    The output format with three columns (ω, Re[χ], Im[χ]) facilitates plotting
    and analysis of the frequency-dependent response at each momentum.

    Computation time scales as O(Nω × Nq × Nk), which can be substantial for
    large grids. For Nω=2001, Nq=101, Nk=101, this requires ~20M function calls
    to CalcXqw.

    Physical applications:
    - Optical absorption/gain spectra vs. carrier density
    - Refractive index changes (via Kramers-Kronig)
    - Identification of excitonic resonances
    - Population inversion and lasing thresholds
    - Non-equilibrium optical response

    The inclusion of ω in the first column makes it easy to plot susceptibility
    vs. frequency for any given q value.

    Differences from RecordEpsLqw:
    - Uses CalcXqw (direct sum over occupations) vs. GetChi1Dqw (equilibrium)
    - Output format: 3 columns (ω, Re, Im) vs. 2 columns (Re, Im)
    - Single file vs. separate files for ε and χ

    See Also
    --------
    CalcXqw : Function used to calculate χ(q,ω)
    RecordEpsLqw : Similar function using analytical RPA methods
    QWChi1 : Related susceptibility calculation from qwoptics module

    Examples
    --------
    >>> # Record susceptibility after excitation at time step 1000
    >>> RecordXqw(kr, fe, fh, Ee, Eh, gap, area, game, gamh, dcv, ind=1000)
    >>> # File created: Xqw.001000.dat
    """
    global _Nk, _dkr, _Xqwparams

    # Frequency grid parameters
    wmax = gap / hbar * 2.0
    dw = wmax / 2000.0
    nwmax = int(wmax / dw)

    # Pre-calculate susceptibility array for all (q,ω) points
    Xqw = np.zeros((_Nk + 1, nwmax + 1), dtype=complex)

    # Calculate χ(q,ω) for all frequencies and momenta
    for iw in range(nwmax + 1):
        w = iw * dw
        for iq in range(_Nk + 1):
            Xqw[iq, iw] = CalcXqw(iq, w, kr, fe, fh, Ee, Eh, gap, area,
                                  game, gamh, dcv)

    # Create output directory
    os.makedirs('dataQW/Wire/Xqw', exist_ok=True)

    # Generate filename with zero-padded index
    filename = f'dataQW/Wire/Xqw/Xqw.{ind:06d}.dat'

    # Write data to file (3 columns: ω, Re[χ], Im[χ])
    with open(filename, 'w', encoding='utf-8') as f:
        for iw in range(nwmax + 1):
            w = iw * dw
            for iq in range(_Nk + 1):
                f.write(f'{w:16.6e} {np.real(Xqw[iq, iw]):16.6e} '
                       f'{np.imag(Xqw[iq, iw]):16.6e}\n')

    # Write parameter file (only once)
    if _Xqwparams:
        with open('dataQW/Wire/Xqw/Xqw.params', 'w', encoding='utf-8') as f_params:
            f_params.write(f"Nq = {_Nk + 1}\n")
            f_params.write(f"Nw = {nwmax + 1}\n")
            f_params.write(f" \n")
            f_params.write(f"hbar dw (eV) = {dw * hbar / e0}\n")
            f_params.write(f"dq (rad/m) = {_dkr}\n")
            f_params.write(f" \n")
            f_params.write(f"wmin = 0.0\n")  # Start from 0 (not negative)
            f_params.write(f"wmax = {wmax * hbar / e0}\n")
            f_params.write(f"qmin = 0.0\n")  # Start from 0 (not negative)
            f_params.write(f"qmax = {_Nk * _dkr}\n")
        _Xqwparams = False


def GetArrays(x, qx, kx):
    """
    Initialize spatial and momentum arrays for the SBE solver.

    Sets up the fundamental coordinate and momentum space grids used throughout
    the semiconductor Bloch equation calculations. This includes:
    - Real-space positions x for field propagation
    - Momentum space qx for Fourier-transformed fields
    - Carrier momentum grid kx (offset by half grid spacing)

    The carrier momentum grid kx is offset by 0.5 grid spacing to avoid having
    a state exactly at k=0, which simplifies boundary conditions and symmetry.

    Parameters
    ----------
    x : ndarray (float), shape (Nr,)
        Real-space position array (m). Modified in-place.
    qx : ndarray (float), shape (Nr,)
        Momentum array for fields (rad/m). Modified in-place.
    kx : ndarray (float), shape (Nk,)
        Carrier momentum array (rad/m). Modified in-place.

    Returns
    -------
    None
        Arrays x, qx, kx are modified in-place.
        Module-level variables _NK0 and _NQ0 are set.

    Notes
    -----
    The carrier momentum grid kx is centered at k=0 with the formula:
    kx[k] = Δkr × (-dnk + k - 0.5)  for k = 0, 1, ..., Nk-1

    where dnk = (Nk-1)/2, so that:
    - kx is antisymmetric about the origin
    - The grid spacing is Δkr
    - No state sits exactly at k=0 (offset by 0.5)

    The field momentum array qx is obtained from GetKArray and then:
    1. Shifted by cshift(qx, Nr/2) to center the zero-frequency component
    2. The first element is negated: qx[0] = -qx[0]

    Module-level variables set:
    - _NK0: Index where kx ≈ 0 (central carrier momentum index)
    - _NQ0: Index where qx = 0 (zero field momentum index)

    These zero indices are crucial for the kkp mapping array and for properly
    handling momentum conservation in scattering processes.

    The real-space array x spans the domain [-L, L] (total length 2L).
    The momentum array qx corresponds to this spatial grid via FFT convention.

    See Also
    --------
    GetSpaceArray : Generate real-space position array (from usefulsubs)
    GetKArray : Generate momentum array for FFT (from usefulsubs)
    GetArray0Index : Find index where array value is closest to zero (from usefulsubs)
    MakeKKP : Create momentum difference mapping using NQ0

    Examples
    --------
    >>> x = np.zeros(Nr)
    >>> qx = np.zeros(Nr)
    >>> kx = np.zeros(Nk)
    >>> GetArrays(x, qx, kx)
    >>> # Now x, qx, kx are initialized and _NK0, _NQ0 are set
    """
    global _Nk, _Nr, _L, _dkr, _NK0, _NQ0

    # Calculate central index for carrier momentum
    dnk = (_Nk - 1) // 2
    _NK0 = dnk + 1  # Fortran 1-based, Python we store as is for compatibility

    # Generate carrier momentum grid (offset by 0.5 to avoid k=0)
    for k in range(_Nk):
        kx[k] = _dkr * (-dnk + k - 0.5)

    # Generate real-space and momentum arrays for fields
    x[:] = GetSpaceArray(_Nr, 2 * _L)
    qx[:] = GetKArray(_Nr, 2 * _L)

    # Shift momentum array to center zero-frequency component
    qx[:] = np.roll(qx, _Nr // 2)
    qx[0] = -qx[0]

    # Find index where qx is closest to zero
    _NQ0 = GetArray0Index(qx)

@jit(nopython=True, parallel=True)
def _MakeKKP_jit(Nk, kr, dkr, NQ0, kkp):
    """JIT-compiled parallel version of MakeKKP computation."""
    from numba import prange

    # Fill mapping array - kkp is passed in and modified in-place
    for k in prange(Nk):  # Parallel over k
        for kp in range(Nk):
            # Calculate momentum difference
            q = kr[k] - kr[kp]
            # Map to nearest grid index
            kkp[k, kp] = int(np.round(q / dkr)) + NQ0

def MakeKKP():
    """
    Create the momentum difference mapping array kkp.

    Constructs the index mapping array kkp[k, kp] that gives the field momentum
    index for the momentum difference q = k - kp. This is essential for evaluating
    matrix elements of Fourier-transformed fields in the semiconductor Bloch equations.

    The array satisfies:
    qx[kkp[k, kp]] ≈ kr[k] - kr[kp]

    where qx is the field momentum array and kr is the carrier momentum array.

    Returns
    -------
    None
        The module-level array _kkp is allocated and filled.

    Notes
    -----
    The mapping is calculated as:
    kkp[k, kp] = round[(kr[k] - kr[kp]) / Δkr] + NQ0

    where:
    - kr[k] - kr[kp] is the momentum difference
    - Division by Δkr converts to grid index units
    - round() maps to nearest grid point
    - NQ0 is added to shift from centered indexing to array indexing

    This mapping is used extensively in the SBE Hamiltonian terms:
    - Dipole-field coupling: M_{k_e,k_h} ∝ E(q) where q = k_e - k_h
    - Coulomb interaction: V_{k1,k2} ∝ V(q) where q = k1 - k2
    - Screening: ε(q) where q is momentum transfer

    Physical interpretation:
    When an electron scatters from state kp to state k, the momentum transfer
    is q = k - kp. This momentum must be provided by (or absorbed into) a photon,
    phonon, or other field mode. The kkp array provides the field mode index
    corresponding to this momentum transfer.

    Memory: For Nk=101, kkp requires ~10,000 integers (~40 KB).

    The Fortran code includes commented-out bounds checking:
    - if(kr[0] <= q <= kr[Nr]): only map if q is in range
    - if(kkp < 1 or kkp > Nr): set kkp = 0 for out-of-range

    These checks are omitted in the current implementation, assuming that the
    momentum grids are compatible and all differences are in range.

    Global Variables Modified
    -------------------------
    _kkp : ndarray (int), shape (Nk, Nk)
        Momentum difference mapping array. Allocated and filled.

    See Also
    --------
    GetArrays : Must be called first to set _NQ0
    CalcMeh : Uses kkp for dipole-field coupling
    CalcWnn : Uses kkp for potential coupling
    kindex : Convert continuous momentum to grid index

    Examples
    --------
    >>> GetArrays(x, qx, kx)  # Initialize arrays first
    >>> MakeKKP()  # Now create mapping
    >>> # Access field at momentum difference k - kp:
    >>> q_idx = _kkp[k, kp]
    >>> field_at_q = Ex[q_idx]
    """
    global _Nk, _kr, _dkr, _NQ0, _kkp

    # Allocate kkp array (Python side - not JIT)
    _kkp = np.zeros((_Nk, _Nk), dtype=int)

    # Call JIT-compiled function to fill it (pass array, modify in-place)
    try:
        _MakeKKP_jit(_Nk, _kr, _dkr, _NQ0, _kkp)
    except Exception:
        # Fallback to pure Python
        for k in range(_Nk):
            for kp in range(_Nk):
                q = _kr[k] - _kr[kp]
                _kkp[k, kp] = int(np.round(q / _dkr)) + _NQ0

def kindex(k):
    """
    Convert a continuous momentum value to the nearest grid index.

    Maps a continuous momentum k (in rad/m) to the corresponding index in the
    carrier momentum array kr. Useful for finding which discrete momentum state
    corresponds to a given continuous momentum value.

    Parameters
    ----------
    k : float
        Momentum value (rad/m).

    Returns
    -------
    idx : int
        Index in the kr array closest to the given momentum k.

    Notes
    -----
    The mapping formula is:
    idx = round(k / Δkr) + NK0

    where:
    - k / Δkr converts momentum to grid index units
    - round() finds nearest grid point
    - NK0 is added to shift from centered indexing to array indexing

    This function is the inverse of:
    k ≈ (idx - NK0) × Δkr

    Caution: No bounds checking is performed. If k is outside the range of the
    kr array, the returned index may be out of bounds [0, Nk-1].

    Physical applications:
    - Finding initial state indices for given carrier momenta
    - Mapping continuous momentum distributions to discrete grid
    - Identifying states near a specific momentum (e.g., Fermi surface)

    See Also
    --------
    GetArrays : Sets up kr array and NK0
    MakeKKP : Creates momentum difference mapping

    Examples
    --------
    >>> # Find index for momentum near Fermi momentum
    >>> k_fermi = 1e8  # rad/m
    >>> idx = kindex(k_fermi)
    >>> k_actual = _kr[idx]  # Actual momentum on grid
    """
    global _dkr, _NK0
    return int(np.round(k / _dkr)) + _NK0


def InitializeSBE(q, rr, r0, Emaxxx, lam, Nw, QW):
    """
    Initialize the SBE module for calculations.

    This function allocates all module-level arrays, initializes them to zero,
    calculates material constants, and sets up all subsystems required for
    solving the Semiconductor Bloch Equations. It must be called before any
    SBE calculations are performed.

    The initialization process includes:
    1. Reading physical parameters from parameter files
    2. Calculating momentum and spatial grids
    3. Allocating coherence matrices (YY, CC, DD) for all wires
    4. Computing material constants (dipole moment, screening lengths, etc.)
    5. Initializing subsystems (Coulomb, Phonons, DC field, Dephasing, Emission)
    6. Setting up output files
    7. Calculating initial carrier distributions

    Parameters
    ----------
    q : ndarray (float), shape (Nq,)
        Momentum array for fields (rad/m). Used for compatibility but not
        directly modified in this function.
    rr : ndarray (float), shape (Nr_prop,)
        Spatial position array for propagation space (m). Used to determine
        the quantum wire window within the propagation domain.
    r0 : float
        Reference position offset (m). Used to locate the quantum wire
        within the propagation spatial grid.
    Emaxxx : float
        Initial peak electric field magnitude (V/m). Used to determine
        when to activate quantum wire calculations (via _wireoff flag).
    lam : float
        Laser wavelength (m). Used for calculating linear susceptibility
        and initializing QW optics.
    Nw : int
        Number of quantum wires in the simulation. Each wire has its own
        set of coherence matrices (YY, CC, DD).
    QW : bool
        Flag to enable quantum wire calculations. If False, only minimal
        initialization is performed.

    Returns
    -------
    None
        All module-level arrays and variables are set.

    Notes
    -----
    The function performs extensive setup:

    **Grid Calculation:**
    - Maximum momentum: kmax = sqrt(1.2 * gap * 2 * me / hbar²)
    - Momentum spacing: dkr = 2π / (2*L)
    - Number of k-points: Nk = floor(kmax/dkr) * 2 + 1, then Nk = Nk - 1
    - Number of r-points: Nr = Nk * 2

    **Material Constants:**
    - Dipole moment: dcv = sqrt((e0*hbar)² / (6*me0*gap) * (me0/me - 1))
    - Electron confinement: alphae = sqrt(me * HO) / hbar
    - Hole confinement: alphah = sqrt(mh * HO) / hbar
    - Overlap integral: ehint = sqrt(2 * alphae * alphah / (alphae² + alphah²))
    - Critical momentum: qc = 2 * alphae * alphah / (alphae + alphah)
    - Wire area: area = sqrt(2π) / sqrt(alphae² + alphah²) * Delta0

    **Initial Carrier Distribution:**
    - Electrons and holes initialized to Fermi-Dirac distribution at
      energy E = gap/2 (half the band gap)
    - All off-diagonal coherence elements set to zero

    **Subsystem Initialization:**
    - InitializeQWOptics: Sets up optical coupling matrices
    - InitializeCoulomb: Calculates screened Coulomb interactions
    - InitializePhonons: Sets up phonon scattering rates
    - InitializeDC: Prepares DC field transport calculations
    - InitializeDephasing: Sets up dephasing rate calculations
    - InitializeEmission: (if _Recomb=True) Sets up spontaneous emission

    **Output Files:**
    For each wire w:
    - dataQW/info.{w:02d}.t.dat - General information
    - dataQW/EP.{w:02d}.t.dat - Energy and polarization data
    - dataQW/XQ.{w:02d}.t.dat - Susceptibility data
    - dataQW/Etr.dat - Transition energies
    - dataQW/Wire/info/ETHz.t.dat - THz field data

    **Special Modes:**
    If _OBE=True (Optical Bloch Equations only):
    - Disables all many-body effects
    - Only optical coupling remains active
    - Useful for testing basic optical response

    **File I/O:**
    - Reads parameters from 'params/qw.params' and 'params/mb.params'
    - Optionally reads DC field from 'DC.txt' if _ReadDC=True
    - Writes initial arrays using WriteIt function

    The function sets the module-level flag _start to indicate that
    initialization is complete, allowing other functions to check if
    the module has been properly initialized.

    See Also
    --------
    ReadQWParams : Read quantum wire physical parameters
    ReadMBParams : Read many-body physics flags
    GetArrays : Set up spatial and momentum grids
    MakeKKP : Create momentum difference mapping
    InitializeQWOptics : Initialize optical coupling
    InitializeCoulomb : Initialize Coulomb interactions
    InitializePhonons : Initialize phonon scattering
    InitializeDC : Initialize DC field transport
    InitializeDephasing : Initialize dephasing rates
    InitializeEmission : Initialize spontaneous emission
    QWChi1 : Calculate linear susceptibility
    RecordXqw : Record susceptibility data

    Examples
    --------
    >>> import numpy as np
    >>> q = np.linspace(-1e8, 1e8, 201)
    >>> rr = np.linspace(-500e-9, 500e-9, 201)
    >>> InitializeSBE(q, rr, r0=0.0, Emaxxx=1e6, lam=800e-9, Nw=1, QW=True)
    >>> # Module is now initialized and ready for SBE calculations
    """
    global _L, _Delta0, _gap, _me, _mh, _HO, _gam_e, _gam_h, _gam_eh
    global _epsr, _Oph, _Gph, _Edc, _jmax, _ntmax
    global _Optics, _Excitons, _EHs, _Screened, _Phonon, _DCTrans
    global _LF, _FreePot, _DiagDph, _OffDiagDph, _Recomb, _PLSpec
    global _ignorewire, _Xqwparams, _OBE, _ReadDC
    global _YY1, _YY2, _YY3, _CC1, _CC2, _CC3, _DD1, _DD2, _DD3
    global _Id, _Ia, _Ee, _Eh, _r, _Qr, _QE, _kr
    global _I0, _ErI0, _dcv, _ehint, _Emax0, _alphae, _alphah, _qc, _area
    global _t, _wph, _chiw, _wL, _c0, _uw, _ETHz
    global _dkr, _dr, _Nr1, _Nr2, _start
    global _Nr, _Nk, _NK0, _NQ0, _nqq, _nqq10, _kkp
    global _hw, _PLS, _twopi

    if not QW:
        # Minimal initialization if QW is disabled
        os.makedirs('dataQW/Wire/info', exist_ok=True)
        with open('dataQW/Wire/info/ETHz.t.dat', 'w', encoding='utf-8') as f:
            pass  # Create empty file
        print("InitializeSBE? (QW disabled)")
        return

    # Read parameter files
    ReadQWParams()
    ReadMBParams()

    # Set initial values
    _Emax0 = Emaxxx
    _t = 0.0

    # Calculate momentum grid parameters
    # kmax = sqrt((1.2 * gap) * 2 * me / hbar²)
    kmax = np.sqrt(1.2 * _gap * 2.0 * _me / (hbar**2))
    _dkr = twopi / (2.0 * _L)
    _Nk = int(np.floor(kmax / _dkr) * 2 + 1)
    _Nr = _Nk * 2

    # Adjust Nk (Fortran comment: if(Nf==1))
    _Nk = _Nk - 1

    # Allocate field y-space and q-space (FFT transformed space) arrays
    _r = np.zeros(_Nr, dtype=float)
    _Qr = np.zeros(_Nr, dtype=float)

    # Allocate the practical field Q-space array (without extra FFT point)
    _QE = np.zeros(_Nr - 1, dtype=float)

    # Momentum space arrays for the SBEs
    _kr = np.zeros(_Nk, dtype=float)
    _Ee = np.zeros(_Nk, dtype=float)
    _Eh = np.zeros(_Nk, dtype=float)

    # Allocate current arrays
    _I0 = np.zeros(Nw, dtype=float)
    _ErI0 = np.zeros(Nw, dtype=float)

    # Allocate coherence matrices for all wires
    _YY1 = np.zeros((_Nk, _Nk, Nw), dtype=complex)
    _YY2 = np.zeros((_Nk, _Nk, Nw), dtype=complex)
    _YY3 = np.zeros((_Nk, _Nk, Nw), dtype=complex)
    _CC1 = np.zeros((_Nk, _Nk, Nw), dtype=complex)
    _CC2 = np.zeros((_Nk, _Nk, Nw), dtype=complex)
    _CC3 = np.zeros((_Nk, _Nk, Nw), dtype=complex)
    _DD1 = np.zeros((_Nk, _Nk, Nw), dtype=complex)
    _DD2 = np.zeros((_Nk, _Nk, Nw), dtype=complex)
    _DD3 = np.zeros((_Nk, _Nk, Nw), dtype=complex)

    # Allocate identity and anti-identity matrices
    _Id = np.zeros((_Nk, _Nk), dtype=float)
    _Ia = np.zeros((_Nk, _Nk), dtype=float)

    # Initialize arrays
    _ErI0[:] = 0.0
    _I0[:] = 0.0

    # Set up identity matrix
    _Id[:, :] = 0.0
    for k in range(_Nk):
        _Id[k, k] = 1.0
    _Ia[:, :] = 1.0 - _Id

    # Calculate needed spatial & momentum arrays for SBE calculations
    GetArrays(_r, _Qr, _kr)
    _dr = (_r[2] - _r[1]) * (_Nr - 1) / float(_Nr)

    # Calculate Material constants
    # Dipole moment: dcv = sqrt((e0*hbar)² / (6*me0*gap) * (me0/me - 1))
    _dcv = np.sqrt((e0 * hbar)**2 / (6.0 * me0 * _gap) * (me0 / _me - 1.0))

    # Confinement parameters
    _alphae = np.sqrt(_me * _HO) / hbar
    _alphah = np.sqrt(_mh * _HO) / hbar

    # Electron-hole overlap integral
    _ehint = np.sqrt(2.0 * _alphae * _alphah / (_alphae**2 + _alphah**2))

    # Average dephasing rate
    _gam_eh = (_gam_e + _gam_h) / 2.0

    # Critical momentum (Gaussian inverse radius)
    _qc = 2.0 * _alphae * _alphah / (_alphae + _alphah)

    # Wire cross-sectional area (first calculation, then overwritten)
    _area = np.sqrt(pi) / _qc * _Delta0

    # Print material constants
    print(f"dcv / e0 = {_dcv / e0}")
    print(f"alphae = {_alphae}, alphah = {_alphah}")
    print(f"1/qc = {1.0 / _qc}, sqrt(2) / sqrt(alphae² + alphah²) = "
          f"{np.sqrt(2.0) / np.sqrt(_alphae**2 + _alphah**2)}")

    # Final area calculation
    _area = np.sqrt(2.0 * pi) / np.sqrt(_alphae**2 + _alphah**2) * _Delta0

    print(f"ehint = {_ehint}")
    print(f"Wire Radius = {1.0 / _qc}")
    print(f"Wire sqrt(area) = {np.sqrt(_area)}")
    print(f"Wire Thickness = {_Delta0}")

    # Calculate band structure (parabolic dispersion)
    _Ee[:] = hbar**2 * _kr**2 / (2.0 * _me)
    _Eh[:] = hbar**2 * _kr**2 / (2.0 * _mh)

    # Alternative: tight-binding dispersion (commented out in Fortran)
    # a = 5.6e-10  # Lattice constant
    # _Ee = hbar**2 / a**2 / _me * (1 - np.cos(_kr * a))
    # _Eh = hbar**2 / a**2 / _mh * (1 - np.cos(_kr * a))

    # Initialize coherence matrices to zero
    _CC1[:, :, :] = 0.0 + 0.0j
    _DD1[:, :, :] = 0.0 + 0.0j
    _YY1[:, :, :] = 0.0 + 0.0j

    # Initialize diagonal elements with Fermi-Dirac distribution
    # at energy E = gap/2 (half the band gap)
    fermi_e = FermiDistr(_Ee + _gap / 2.0)
    fermi_h = FermiDistr(_Ee + _gap / 2.0)  # Note: Fortran uses Ee here, which is correct

    for k in range(_Nk):
        for w in range(Nw):
            _CC1[k, k, w] = fermi_e[k]
            _DD1[k, k, w] = fermi_h[k]

    # Copy to other time steps
    _CC2[:, :, :] = _CC1
    _CC3[:, :, :] = _CC1
    _DD2[:, :, :] = _DD1
    _DD3[:, :, :] = _DD1
    _YY2[:, :, :] = 0.0 + 0.0j
    _YY3[:, :, :] = 0.0 + 0.0j

    # Calculate QW window within Y-array
    InitializeQWOptics(_r, _L, _dcv, _kr, _Qr, _Ee, _Eh, _ehint, _area, _gap)

    # Write initial arrays to files
    WriteIt(_kr, "kr")
    WriteIt(_Qr, "Qr")
    WriteIt(_r, "R")
    WriteIt(_Ee / eV, "Ee.k")
    WriteIt(_Eh / eV, "Eh.k")
    WriteIt((_Ee + _Eh + _gap - hbar * _c0 * twopi / lam) / eV, "Echw.k")
    WriteIt((_Ee + _Eh + _gap) / eV, "Etrn.k")

    # Write transition energies to file
    os.makedirs('dataQW', exist_ok=True)
    with open('dataQW/Etr.dat', 'w', encoding='utf-8') as f:
        for i in range(len(_kr)):
            f.write(f'{_kr[i]} {(_Ee[i] + _Eh[i] + _gap) / eV}\n')

    # OBE mode: disable all many-body effects
    if _OBE:
        _Optics = True
        _Excitons = False
        _EHs = False
        _Phonon = False
        _Recomb = False
        _LF = False
        _DCTrans = False

    # Create momentum difference mapping array
    MakeKKP()

    # Calculate constant material arrays
    InitializeCoulomb(_r, _kr, _L, _Delta0, _me, _mh, _Ee, _Eh,
                      _gam_e, _gam_h, _alphae, _alphah, _epsr, _Qr, _kkp, _Screened)

    InitializePhonons(_kr, _Ee, _Eh, _L, _epsr, _Gph, _Oph)

    InitializeDC(_kr, _me, _mh)

    InitializeDephasing(_kr, _me, _mh)

    if _Recomb:
        InitializeEmission(_kr, _Ee, _Eh, np.abs(_dcv), _epsr, _gam_eh, _ehint)

    # Determine the beginning and ending points of the quantum wire
    # in the field array y-space
    rr_offset = rr - r0
    _Nr1 = locator(rr_offset, _r[0])
    _Nr2 = locator(rr_offset, _r[_Nr - 1])

    # Set time and frequency variables
    _t = 0.0
    _wph = (_gap + hbar * _Oph - hbar * _wL) / hbar

    # Calculate linear susceptibility
    _chiw = QWChi1(lam, _dkr, _Ee + _gap, _Eh, _area, _gam_eh, _dcv)

    print(f"Quantum Wire Linear Chi = {_chiw}")

    # Record initial susceptibility (with zero populations for testing)
    game = np.full(_Nk, _gam_e, dtype=float)
    gamh = np.full(_Nk, _gam_h, dtype=float)
    RecordXqw(_kr, np.zeros(_Nk), np.zeros(_Nk), _Ee, _Eh, _gap, _area,
              game, gamh, _dcv, 0)

    # Open output files for each wire
    os.makedirs('dataQW', exist_ok=True)
    for w in range(1, Nw + 1):
        wire_str = f'{w:02d}'
        # Files are opened but not stored (Python handles file closing differently)
        # In Fortran, unit numbers are used; in Python we'll use context managers
        # when writing. For now, just create the files.
        with open(f'dataQW/info.{wire_str}.t.dat', 'w', encoding='utf-8') as f:
            pass
        with open(f'dataQW/EP.{wire_str}.t.dat', 'w', encoding='utf-8') as f:
            pass
        with open(f'dataQW/XQ.{wire_str}.t.dat', 'w', encoding='utf-8') as f:
            pass

    # Calculate photoluminescence spectrum frequency grid
    # Calchw sets up frequency array from Estart to Emax
    Estart = 0.8 * _gap
    Emax = 1.1 * (_gap + _Ee[_Nk - 1] + _Eh[_Nk - 1])
    Calchw(_hw, _PLS, Estart, Emax)

    # Find indices for specific momentum values
    _nqq = locate(_Qr, 2.35e7)
    _nqq10 = locate(_Qr, 2.35e8)

    # Read DC field from file if requested
    if _ReadDC:
        with open('DC.txt', 'r', encoding='utf-8') as f:
            _Edc = float(f.readline().split()[0])

    # Open THz field output file
    os.makedirs('dataQW/Wire/info', exist_ok=True)
    with open('dataQW/Wire/info/ETHz.t.dat', 'w', encoding='utf-8') as f:
        pass

    # Mark initialization as complete
    _start = True

    print("InitializeSBE?")


def SBECalculator(Ex, Ey, Ez, Vr, dt, Px, Py, Pz, Re, Rh, WriteFields, w):
    """
    Solve the 1D Semiconductor Bloch Equations and calculate source terms.

    This is the main function that solves the Semiconductor Bloch Equations
    for the w'th quantum wire and calculates the source terms Px, Py, Pz, Re,
    and Rh that are used in Maxwell's equations for field propagation.

    The function performs a leapfrog time integration of the SBEs, including:
    - Electron-hole coherence (interband polarization) p
    - Electron-electron coherence C
    - Hole-hole coherence D
    - Many-body effects (Coulomb, phonons, dephasing)
    - DC transport effects
    - Charge density calculations

    Parameters
    ----------
    Ex : ndarray (complex), shape (Nr,)
        X-component electric field in QW momentum space (modified in-place).
        The field is FFT'd to real space for calculations, then FFT'd back.
    Ey : ndarray (complex), shape (Nr,)
        Y-component electric field in QW momentum space (modified in-place).
    Ez : ndarray (complex), shape (Nr,)
        Z-component electric field in QW momentum space (modified in-place).
    Vr : ndarray (complex), shape (Nr,)
        Free charge potential (voltage) in QW momentum space (modified in-place).
    dt : float
        Time step (s).
    Px : ndarray (complex), shape (Nr,)
        X-component QW polarization field (output, modified in-place).
    Py : ndarray (complex), shape (Nr,)
        Y-component QW polarization field (output, modified in-place).
    Pz : ndarray (complex), shape (Nr,)
        Z-component QW polarization field (output, modified in-place).
    Re : ndarray (complex), shape (Nr,)
        QW electron charge density (output, modified in-place).
    Rh : ndarray (complex), shape (Nr,)
        QW hole charge density (output, modified in-place).
    WriteFields : bool
        Flag to record SBE solutions and write output files this time step.
    w : int
        Quantum wire index (which wire to calculate for).

    Returns
    -------
    None
        All output arrays (Px, Py, Pz, Re, Rh, Ex, Ey, Ez, Vr) are modified in-place.

    Notes
    -----
    The function implements the following sequence:

    1. **Initialize source terms** to zero
    2. **Checkout coherence matrices** from module storage for wire w
    3. **Prepare arrays** (Hamiltonians, screening, dephasing) via Preparation()
    4. **Calculate time derivatives** dp/dt, dC/dt, dD/dt
    5. **Time evolve** using leapfrog scheme: X3 = X1 + dX/dt * 2*dt
    6. **Apply relaxation** (phonon/carrier-carrier scattering) if enabled
    7. **Apply DC transport** if enabled
    8. **Normalize populations** to ensure charge neutrality
    9. **Reshuffle** for stability (convert leapfrog to implicit Euler)
    10. **Calculate polarization** Px, Py, Pz from coherence p
    11. **Calculate charge densities** Re, Rh if longitudinal field enabled
    12. **Write output files** if WriteFields is True
    13. **Checkin updated matrices** to module storage

    **Time Integration Scheme:**
    The leapfrog scheme is:
    - X3 = X1 + dX/dt * 2*dt  (2nd order accurate but potentially unstable)
    - Then reshuffled to: X2 = (X1 + X3) / 2  (1st order accurate but stable)

    This converts the 2nd-order accurate (but unstable) leapfrog scheme into
    a 1st-order accurate (but stable) implicit Euler scheme.

    **Charge Neutrality:**
    The populations are normalized so that:
    - Total electrons = Total holes = (sum(ne) + sum(nh)) / 2
    This ensures charge neutrality in the system.

    **FFT Operations:**
    The electric fields and polarizations are FFT'd to real space for
    calculations, then FFT'd back to momentum space. This is done because:
    - Some calculations are easier in real space
    - The output is written in real space (at center point Nr/2)

    **Output Files:**
    If WriteFields is True:
    - WriteSBESolns: Writes coherence matrices and populations
    - WriteDephasing: Writes dephasing rates (if DiagDph enabled)
    - File (uw+w): Writes statistics (velocities, densities, energies, etc.)
    - File (2*uw+w): Writes fields and polarizations at center point

    **Module Variables Used:**
    - _YY1, _YY2, _YY3: Electron-hole coherence matrices
    - _CC1, _CC2, _CC3: Electron-electron coherence matrices
    - _DD1, _DD2, _DD3: Hole-hole coherence matrices
    - _Id, _Ia: Identity and anti-identity matrices
    - _kr, _Ee, _Eh: Momentum and energy arrays
    - _r, _Qr: Spatial and momentum arrays
    - _I0: Drift current array
    - _xxx, _jjj: Time step counters
    - _gap, _me, _mh, _L, _area, _ehint: Material parameters
    - _Optics, _EHs, _Phonon, _DCTrans, _LF, _DiagDph: Physics flags

    See Also
    --------
    Preparation : Prepare Hamiltonians and arrays for SBE time step
    dpdt : Calculate time derivative of electron-hole coherence
    dCdt : Calculate time derivative of electron-electron coherence
    dDdt : Calculate time derivative of hole-hole coherence
    Relaxation : Apply phonon and carrier-carrier scattering
    Transport : Apply DC field transport effects
    QWPolarization3 : Calculate polarization from coherence
    QWRho5 : Calculate charge densities from coherence
    Checkout : Retrieve coherence matrices from module storage
    Checkin : Store coherence matrices to module storage
    """
    global _YY1, _YY2, _YY3, _CC1, _CC2, _CC3, _DD1, _DD2, _DD3
    global _Id, _Ia, _kr, _Ee, _Eh, _r, _Qr, _Nk, _Nr, _NQ0
    global _I0, _xxx, _jjj, _gap, _me, _mh, _L, _area, _ehint, _dkr
    global _Optics, _EHs, _Phonon, _DCTrans, _LF, _DiagDph, _Edc
    global _small, _uw

    # Arrays for the reduced SBE k-space
    ne1 = np.zeros(_Nk, dtype=complex)  # Electron occupation at t(n-1)
    nh1 = np.zeros(_Nk, dtype=complex)  # Hole occupation at t(n-1)
    ne2 = np.zeros(_Nk, dtype=complex)  # Electron occupation at t(n)
    nh2 = np.zeros(_Nk, dtype=complex)  # Hole occupation at t(n)
    ne3 = np.zeros(_Nk, dtype=complex)  # Electron occupation at t(n+1)
    nh3 = np.zeros(_Nk, dtype=complex)  # Hole occupation at t(n+1)

    p1 = np.zeros((_Nk, _Nk), dtype=complex)  # e-h coherence at t(n-1)
    p2 = np.zeros((_Nk, _Nk), dtype=complex)  # e-h coherence at t(n)
    p3 = np.zeros((_Nk, _Nk), dtype=complex)  # e-h coherence at t(n+1)

    C1 = np.zeros((_Nk, _Nk), dtype=complex)  # e-e coherence at t(n-1)
    C2 = np.zeros((_Nk, _Nk), dtype=complex)  # e-e coherence at t(n)
    C3 = np.zeros((_Nk, _Nk), dtype=complex)  # e-e coherence at t(n+1)

    D1 = np.zeros((_Nk, _Nk), dtype=complex)  # h-h coherence at t(n-1)
    D2 = np.zeros((_Nk, _Nk), dtype=complex)  # h-h coherence at t(n)
    D3 = np.zeros((_Nk, _Nk), dtype=complex)  # h-h coherence at t(n+1)

    dpdt2 = np.zeros((_Nk, _Nk), dtype=complex)  # dp/dt at t(n)
    dCdt2 = np.zeros((_Nk, _Nk), dtype=complex)  # dC/dt at t(n)
    dDdt2 = np.zeros((_Nk, _Nk), dtype=complex)  # dD/dt at t(n)

    Hee = np.zeros((_Nk, _Nk), dtype=complex)  # e-e Hamiltonian
    Hhh = np.zeros((_Nk, _Nk), dtype=complex)  # h-h Hamiltonian
    Heh = np.zeros((_Nk, _Nk), dtype=complex)  # e-h Hamiltonian

    OffG = np.zeros((_Nk, _Nk, 3), dtype=complex)  # Off-diagonal dephasing
    VC = np.zeros((_Nk, _Nk, 3), dtype=float)  # Screened Coulomb arrays
    E1D = np.zeros((_Nk, _Nk), dtype=float)  # Screening array
    GamE = np.zeros(_Nk, dtype=float)  # Electron dephasing rates
    GamH = np.zeros(_Nk, dtype=float)  # Hole dephasing rates
    Rsp = np.zeros(_Nk, dtype=float)  # Spontaneous emission rates

    Ene = np.zeros(_Nk, dtype=complex)  # Electron corrected energies
    Enh = np.zeros(_Nk, dtype=complex)  # Hole corrected energies

    # Initialize source terms
    Px[:] = 0.0 + 0.0j
    Py[:] = 0.0 + 0.0j
    Pz[:] = 0.0 + 0.0j
    Re[:] = 0.0 + 0.0j
    Rh[:] = 0.0 + 0.0j

    # Set up identity and anti-identity matrices
    _Ia[:, :] = 1.0
    _Id[:, :] = 0.0
    for k in range(_Nk):
        _Ia[k, k] = 0.0
        _Id[k, k] = 1.0

    # Checkout coherence matrices from module storage
    Checkout(p1, p2, C1, C2, D1, D2, w)

    # Prepare the needed arrays for the SBEs
    Preparation(p2, C2, D2, Ex, Ey, Ez, Vr, w, Heh, Hee, Hhh, VC, E1D, GamE, GamH, OffG, Rsp)

    # Extract corrected energies (in eV)
    for k in range(_Nk):
        Ene[k] = Hee[k, k] / eV
        Enh[k] = Hhh[k, k] / eV

    # Calculate time derivatives of SBEs
    dpdt2[:, :] = dpdt(C2, D2, p2, Heh, Hee, Hhh, GamE, GamH, OffG[:, :, 0])
    dCdt2[:, :] = dCdt(C2, D2, p2, Heh, Hee, Hhh, GamE, GamH, OffG[:, :, 1])
    dDdt2[:, :] = dDdt(C2, D2, p2, Heh, Hee, Hhh, GamE, GamH, OffG[:, :, 2])

    # Time evolve by leapfrog: X3 = X1 + dX/dt * 2*dt
    C3[:, :] = C1 + dCdt2 * dt * 2.0
    D3[:, :] = D1 + dDdt2 * dt * 2.0
    p3[:, :] = p1 + dpdt2 * dt * 2.0

    # Apply relaxation (phonon and carrier-carrier scattering)
    if _EHs or _Phonon:
        # Extract populations from diagonal
        for k in range(_Nk):
            ne3[k] = C3[k, k]
            nh3[k] = D3[k, k]

        # Apply relaxation
        Relaxation(ne3, nh3, VC, E1D, Rsp, dt, w, WriteFields)

        # Put populations back into diagonal
        for k in range(_Nk):
            C3[k, k] = ne3[k]
            D3[k, k] = nh3[k]

    # Apply DC transport effects
    if _DCTrans:
        Ex_real = np.real(Ex[_NQ0])
        Transport(C3, _Edc, Ex_real, dt, _DCTrans, _LF)
        Transport(D3, _Edc, Ex_real, dt, _DCTrans, _LF)
        Transport(p3, _Edc, Ex_real, dt, _DCTrans, _LF)

    # Extract populations and normalize for charge neutrality
    for k in range(_Nk):
        ne3[k] = np.abs(C3[k, k])
        nh3[k] = np.abs(D3[k, k])

    # Make sure total electrons equals total holes
    total = (np.sum(np.abs(ne3)) + np.sum(np.abs(nh3))) / 2.0
    ne3[:] = ne3 * total / (np.sum(np.abs(ne3)) + _small)
    nh3[:] = nh3 * total / (np.sum(np.abs(nh3)) + _small)

    # Put normalized populations back into diagonal
    for k in range(_Nk):
        C3[k, k] = ne3[k]
        D3[k, k] = nh3[k]

    # Reshuffle for stability: convert leapfrog to implicit Euler
    # This turns the 2nd order accurate (but unstable) leapfrog scheme
    # into a 1st order accurate (but stable) implicit Euler scheme.
    p2[:, :] = (p1 + p3) / 2.0
    C2[:, :] = (C1 + C3) / 2.0
    D2[:, :] = (D1 + D3) / 2.0

    # Extract populations at time t(n)
    for k in range(_Nk):
        ne2[k] = np.abs(C2[k, k])
        nh2[k] = np.abs(D2[k, k])

    # Write SBE solutions if requested
    if WriteFields:
        WriteSBESolns(_kr, ne3, nh3, C3, D3, p3, Ene, Enh, w, _xxx)

    # Write dephasing rates if requested
    if WriteFields and _DiagDph:
        WriteDephasing(_kr, GamE, GamH, w, _xxx)

    # Calculate the X, Y, Z components of the QW Polarization
    if _Optics:
        QWPolarization3(_r, _kr, p3, _ehint, _area, _L, Px, Py, Pz, _xxx, w)

    # Calculate new electron and hole charge densities
    if _LF:
        QWRho5(_Qr, _kr, _r, _L, _kkp, p3, C3, D3, ne3, nh3, Re, Rh, _xxx, _jjj)
        Re[:] = 2.0 * e0 * Re / _area * _ehint
        Rh[:] = 2.0 * e0 * Rh / _area * _ehint
    else:
        Re[:] = 0.0 + 0.0j
        Rh[:] = 0.0 + 0.0j

    # Print statistics for middle wire if WriteFields is True
    if w == int(np.ceil(len(_I0) / 2.0)) and WriteFields:
        print(f"{_xxx:8d} Elec {np.sum(np.real(ne3)):18.6e} {np.max(np.real(ne3)):18.6e} "
              f"{np.min(np.real(ne3)):18.6e} {CalcVD(_kr, _me, ne3):18.6e} "
              f"{np.real(CalcPD(_kr, _me, ne3)):18.6e}")
        print(f"{_xxx:8d} Hole {np.sum(np.real(nh3)):18.6e} {np.max(np.real(nh3)):18.6e} "
              f"{np.min(np.real(nh3)):18.6e} {CalcVD(_kr, _mh, nh3):18.6e} "
              f"{np.real(CalcPD(_kr, _mh, nh3)):18.6e}")

    # Calculate drift current
    _I0[w-1] = CalcI0(ne2, nh2, _Ee, _Eh, VC, _dkr, _kr, _I0[w-1])

    # Write statistics to file
    os.makedirs('dataQW', exist_ok=True)
    with open(f'dataQW/info.{w:02d}.t.dat', 'a', encoding='utf-8') as f:
        f.write(f"{_xxx*dt:18.6e} {CalcVD(_kr, _me, ne2):18.6e} {CalcVD(_kr, _mh, nh2):18.6e} "
                f"{np.real(2*np.sum(ne2)/_L):18.6e} {np.real(2*np.sum(nh2)/_L):18.6e} "
                f"{TotalEnergy(ne2, _Ee+_gap)/e0:18.6e} {TotalEnergy(nh2, _Eh)/e0:18.6e} "
                f"{Temperature(ne2, _Ee):18.6e} {Temperature(nh2, _Eh):18.6e} "
                f"{np.max(np.abs(Re)):18.6e} {np.max(np.abs(Rh)):18.6e} "
                f"{GetEDrift():18.6e} {np.real(hbar*np.sum((ne2-nh2)*_kr))*2.0:18.6e} "
                f"{np.real(hbar*np.sum(ne2*_kr)):18.6e} {np.real(-hbar*np.sum(nh2*_kr)):18.6e} "
                f"{_I0[w-1]:18.6e} {CalcI0n(ne2, _me, _kr):18.6e} "
                f"{CalcI0n(1-nh2, -_mh, _kr):18.6e}\n")

    # FFT to real space for output
    iFFTG(Ex)
    iFFTG(Ey)
    iFFTG(Ez)
    iFFTG(Px)
    iFFTG(Py)
    iFFTG(Pz)

    # Write fields and polarizations at center point
    with open(f'dataQW/EP.{w:02d}.t.dat', 'a', encoding='utf-8') as f:
        f.write(f"{_xxx*dt:18.6e} {np.real(Ex[_Nr//2]):18.6e} {np.real(Ey[_Nr//2]):18.6e} "
                f"{np.real(Ez[_Nr//2]):18.6e} {np.real(Px[_Nr//2]):18.6e} "
                f"{np.real(Py[_Nr//2]):18.6e} {np.real(Pz[_Nr//2]):18.6e}\n")

    # FFT back to momentum space
    FFTG(Ex)
    FFTG(Ey)
    FFTG(Ez)
    FFTG(Px)
    FFTG(Py)
    FFTG(Pz)

    # Checkin updated coherence matrices to module storage
    Checkin(p1, p2, p3, C1, C2, C3, D1, D2, D3, w)


