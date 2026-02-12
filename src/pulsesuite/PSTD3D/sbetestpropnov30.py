"""
SBE Test Program - Full Version with Longitudinal Field Decomposition.

This program tests the Semiconductor Bloch Equations (SBE) solver by simulating
quantum wire response to plane wave electromagnetic fields with complete
longitudinal field analysis. It tracks E-fields from polarization (P), charge
density (Rho), and their combination (P+Rho).

Author: Rahul R. Sah
"""

import os

import numpy as np

try:
    from numba import jit
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False
    # Fallback: create a no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]):
            # Called as @jit without parentheses
            return args[0]
        return decorator

# Import required modules
from .rhoPJ import QuantumWire
from .SBEs import InitializeSBE
from .typespace import (
    GetKyArray,
    GetNx,
    GetNy,
    GetNz,
    GetXArray,
    GetYArray,
    ReadSpaceParams,
    ss,
)
from .usefulsubs import WriteIT2D

# Physical constants
c0 = 299792458.0  # Speed of light (m/s)
pi = np.pi
twopi = 2.0 * pi

# Global parameters (matching Fortran defaults)
DEFAULT_DT = 10e-18  # Time step (s) - note: 10e-18 in this version
DEFAULT_NT = 10000   # Number of time steps
DEFAULT_TP = 50e-15  # Pulse peak time (s)
DEFAULT_LAM = 800e-9  # Wavelength (m)
DEFAULT_TW = 10e-15   # Pulse width (s)
DEFAULT_N0 = 3.1     # Background refractive index
DEFAULT_E0X = 2e8    # Peak Ex field (V/m)
DEFAULT_E0Y = 0.0    # Peak Ey field (V/m)
DEFAULT_E0Z = 0.0    # Peak Ez field (V/m)


def initializefields(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                     F11, F12, F13, F14, F15, F16, F17, F18, F19, F20):
    """
    Initialize all field arrays to zero.

    Sets all 20 complex field arrays to zero for the start of simulation.
    This extended version includes longitudinal field decompositions.

    Parameters
    ----------
    F1, F2, F3 : ndarray
        Electric field components (Ex, Ey, Ez), 3D complex arrays
    F4, F5, F6 : ndarray
        Current density components (Jx, Jy, Jz), 3D complex arrays
    F7 : ndarray
        Charge density (Rho), 3D complex array
    F8, F9, F10 : ndarray
        Longitudinal field components from Helmholtz decomposition (Exl, Eyl, Ezl), 3D complex arrays
    F11, F12, F13 : ndarray
        Longitudinal fields from P+Rho (ExlfromPRho, EylfromPRho, EzlfromPRho), 3D complex arrays
    F14, F15, F16 : ndarray
        Longitudinal fields from P only (ExlfromP, EylfromP, EzlfromP), 3D complex arrays
    F17, F18, F19 : ndarray
        Longitudinal fields from Rho only (ExlfromRho, EylfromRho, EzlfromRho), 3D complex arrays
    F20 : ndarray
        Bound charge density (RhoBound), 3D complex array

    Returns
    -------
    None
        All arrays are modified in-place.
    """
    F1[:] = 0.0 + 0.0j
    F2[:] = 0.0 + 0.0j
    F3[:] = 0.0 + 0.0j
    F4[:] = 0.0 + 0.0j
    F5[:] = 0.0 + 0.0j
    F6[:] = 0.0 + 0.0j
    F7[:] = 0.0 + 0.0j
    F8[:] = 0.0 + 0.0j
    F9[:] = 0.0 + 0.0j
    F10[:] = 0.0 + 0.0j
    F11[:] = 0.0 + 0.0j
    F12[:] = 0.0 + 0.0j
    F13[:] = 0.0 + 0.0j
    F14[:] = 0.0 + 0.0j
    F15[:] = 0.0 + 0.0j
    F16[:] = 0.0 + 0.0j
    F17[:] = 0.0 + 0.0j
    F18[:] = 0.0 + 0.0j
    F19[:] = 0.0 + 0.0j
    F20[:] = 0.0 + 0.0j


@jit(nopython=True, cache=True)
def _compute_plane_wave_jit(u, Emax0, w0, tw):
    """
    JIT-compiled core computation for plane wave envelope.

    Computes the Gaussian envelope with super-Gaussian cutoff:
    E = Emax0 * exp(-u²/(w0*tw)²) * cos(u) * exp(-u²⁰/(2*w0*tw)²⁰)

    Parameters
    ----------
    u : ndarray
        Phase array (k*x - w*t), 1D real array
    Emax0 : float
        Peak field amplitude (V/m)
    w0 : float
        Angular frequency (rad/s)
    tw : float
        Pulse width (s)

    Returns
    -------
    ndarray
        Complex field values, 1D array
    """
    N = len(u)
    result = np.zeros(N, dtype=np.complex128)
    w0tw = w0 * tw
    w0tw20 = (2.0 * w0tw) ** 20

    for i in range(N):
        u_val = u[i]
        gaussian = np.exp(-u_val**2 / w0tw**2)
        supergaussian = np.exp(-u_val**20 / w0tw20)
        carrier = np.cos(u_val)
        result[i] = Emax0 * gaussian * carrier * supergaussian

    return result


def MakePlaneWaveX(Ey, space, t, Emax0, lam, tw, tp):
    """
    Create a plane wave propagating in the x-direction.

    Generates E = (0, Ey(x,t), 0) with Gaussian envelope and super-Gaussian
    cutoff, propagating along the x-axis.

    Parameters
    ----------
    Ey : ndarray
        Y-component electric field to populate (modified in-place), 3D complex array
    space : ss
        Space structure containing grid parameters
    t : float
        Current time (s)
    Emax0 : float
        Peak field amplitude (V/m)
    lam : float
        Wavelength (m)
    tw : float
        Pulse width (s)
    tp : float
        Time of pulse peak (s)

    Returns
    -------
    None
        Ey array is modified in-place.

    Notes
    -----
    Uses a background refractive index n0 = 3.1 (hardcoded).
    Super-Gaussian cutoff (power 20) prevents spurious oscillations.
    """
    n0 = 3.1  # Background refractive index

    # Calculate angular frequencies and wavevector
    w0 = twopi * c0 / lam  # Angular frequency (rad/s)
    k0 = twopi / lam * n0  # Wavevector (rad/m)

    # Get spatial array
    x = GetXArray(space)

    # Calculate phase: k*x - w*(t-tp)
    u = k0 * x - w0 * (t - tp)

    # Compute field envelope
    try:
        field = _compute_plane_wave_jit(u, Emax0, w0, tw)
    except:
        # Fallback to NumPy
        w0tw = w0 * tw
        field = (Emax0 * np.exp(-u**2 / w0tw**2) * np.cos(u) *
                 np.exp(-u**20 / (2.0 * w0tw)**20))

    # Populate 3D array (field is uniform in y and z)
    Nx = GetNx(space)
    Ny = GetNy(space)
    Nz = GetNz(space)

    for k in range(Nz):
        for j in range(Ny):
            Ey[:, j, k] = field


def MakePlaneWaveY(Ex, space, t, Emax0, lam, tw, tp):
    """
    Create a plane wave propagating in the y-direction.

    Generates E = (Ex(y,t), 0, 0) with Gaussian envelope and super-Gaussian
    cutoff, propagating along the y-axis.

    Parameters
    ----------
    Ex : ndarray
        X-component electric field to populate (modified in-place), 3D complex array
    space : ss
        Space structure containing grid parameters
    t : float
        Current time (s)
    Emax0 : float
        Peak field amplitude (V/m)
    lam : float
        Wavelength (m)
    tw : float
        Pulse width (s)
    tp : float
        Time of pulse peak (s)

    Returns
    -------
    None
        Ex array is modified in-place.

    Notes
    -----
    Uses a background refractive index n0 = 3.1 (hardcoded).
    Super-Gaussian cutoff (power 20) prevents spurious oscillations.
    """
    n0 = 3.1  # Background refractive index

    # Calculate angular frequencies and wavevector
    w0 = twopi * c0 / lam  # Angular frequency (rad/s)
    k0 = twopi / lam * n0  # Wavevector (rad/m)

    # Get spatial array
    y = GetYArray(space)

    # Calculate phase: k*y - w*(t-tp)
    u = k0 * y - w0 * (t - tp)

    # Compute field envelope
    try:
        field = _compute_plane_wave_jit(u, Emax0, w0, tw)
    except:
        # Fallback to NumPy
        w0tw = w0 * tw
        field = (Emax0 * np.exp(-u**2 / w0tw**2) * np.cos(u) *
                 np.exp(-u**20 / (2.0 * w0tw)**20))

    # Populate 3D array (field is uniform in x and z)
    Nx = GetNx(space)
    Ny = GetNy(space)
    Nz = GetNz(space)

    for k in range(Nz):
        for i in range(Nx):
            Ex[i, :, k] = field


def MakePlaneWaveZ(Ez, space, t, Emax0, lam, tw, tp):
    """
    Create a plane wave propagating in the x-direction (z-polarized).

    Generates E = (0, 0, Ez(x,t)) with Gaussian envelope and super-Gaussian
    cutoff, propagating along the x-axis.

    Parameters
    ----------
    Ez : ndarray
        Z-component electric field to populate (modified in-place), 3D complex array
    space : ss
        Space structure containing grid parameters
    t : float
        Current time (s)
    Emax0 : float
        Peak field amplitude (V/m)
    lam : float
        Wavelength (m)
    tw : float
        Pulse width (s)
    tp : float
        Time of pulse peak (s)

    Returns
    -------
    None
        Ez array is modified in-place.

    Notes
    -----
    Uses a background refractive index n0 = 3.1 (hardcoded).
    Super-Gaussian cutoff (power 20) prevents spurious oscillations.
    Note: The wave propagates along x-axis, not z-axis.
    """
    n0 = 3.1  # Background refractive index

    # Calculate angular frequencies and wavevector
    w0 = twopi * c0 / lam  # Angular frequency (rad/s)
    k0 = twopi / lam * n0  # Wavevector (rad/m)

    # Get spatial array
    x = GetXArray(space)

    # Calculate phase: k*x - w*(t-tp)
    u = k0 * x - w0 * (t - tp)

    # Compute field envelope
    try:
        field = _compute_plane_wave_jit(u, Emax0, w0, tw)
    except:
        # Fallback to NumPy
        w0tw = w0 * tw
        field = (Emax0 * np.exp(-u**2 / w0tw**2) * np.cos(u) *
                 np.exp(-u**20 / (2.0 * w0tw)**20))

    # Populate 3D array (field is uniform in y and z)
    Nx = GetNx(space)
    Ny = GetNy(space)
    Nz = GetNz(space)

    for k in range(Nz):
        for j in range(Ny):
            Ez[:, j, k] = field


def MakePlaneWaveTemporal(Ex, t, Emax0, lam, tw, tp):
    """
    Create a temporally-varying uniform field.

    Generates E = (Ex(t), 0, 0) that is uniform in space but varies in time
    with a Gaussian envelope.

    Parameters
    ----------
    Ex : ndarray
        X-component electric field to populate (modified in-place), 3D complex array
    t : float
        Current time (s)
    Emax0 : float
        Peak field amplitude (V/m)
    lam : float
        Wavelength (m)
    tw : float
        Pulse width (s)
    tp : float
        Time of pulse peak (s)

    Returns
    -------
    None
        Ex array is modified in-place.

    Notes
    -----
    This creates a spatially uniform field that oscillates in time.
    Useful for testing temporal response without spatial propagation effects.
    """
    # Calculate angular frequency
    w0 = 2.0 * pi * c0 / lam  # Angular frequency (rad/s)

    # Calculate phase
    u = w0 * (t - tp)

    # Compute Gaussian pulse envelope
    env = Emax0 * np.cos(u) * np.exp(-(t - tp)**2 / tw**2)

    # Set uniform field everywhere
    Ex[:, :, :] = env


def ElongSeparate(space, Ex, Ey, Ez, Exl, Eyl, Ezl):
    """
    Separate longitudinal and transverse electric field components.

    Placeholder function for field decomposition using Helmholtz decomposition.
    In the full implementation, this would separate E into transverse
    and longitudinal components.

    Parameters
    ----------
    space : ss
        Space structure containing grid parameters
    Ex, Ey, Ez : ndarray
        Total electric field components, 3D complex arrays
    Exl, Eyl, Ezl : ndarray
        Longitudinal electric field components (output), 3D complex arrays

    Returns
    -------
    None
        Exl, Eyl, Ezl are modified in-place.

    Notes
    -----
    Currently a stub. Full implementation would compute:
    E_long = -∇φ where ∇²φ = ∇·E
    E_trans = E - E_long
    """
    # Stub implementation - just zero out longitudinal fields
    Exl[:] = 0.0 + 0.0j
    Eyl[:] = 0.0 + 0.0j
    Ezl[:] = 0.0 + 0.0j


def int2str(n):
    """
    Convert integer to string for filename generation.

    Simple helper function to convert integers to strings for output filenames.

    Parameters
    ----------
    n : int
        Integer to convert

    Returns
    -------
    str
        String representation of the integer
    """
    return str(n)


def SBETest(space_params_file='params/space.params',
            dt=DEFAULT_DT,
            Nt=DEFAULT_NT,
            tp=DEFAULT_TP,
            lam=DEFAULT_LAM,
            tw=DEFAULT_TW,
            n0=DEFAULT_N0,
            E0x=DEFAULT_E0X,
            E0y=DEFAULT_E0Y,
            E0z=DEFAULT_E0Z,
            output_dir='fields',
            write_2d_slices=True,
            slice_interval=10):
    """
    Main SBE test program with full longitudinal field analysis.

    Simulates quantum wire response to plane wave electromagnetic fields with
    complete tracking of longitudinal field components from polarization (P),
    charge density (Rho), and their combination.

    Parameters
    ----------
    space_params_file : str, optional
        Path to space parameters file (default: 'params/space.params')
    dt : float, optional
        Time step in seconds (default: 10e-18 s)
    Nt : int, optional
        Number of time steps (default: 10000)
    tp : float, optional
        Time of pulse peak in seconds (default: 50e-15 s)
    lam : float, optional
        Field wavelength in meters (default: 800e-9 m)
    tw : float, optional
        Pulse width in seconds (default: 10e-15 s)
    n0 : float, optional
        Background refractive index (default: 3.1)
    E0x : float, optional
        Peak Ex-field value in V/m (default: 2e8 V/m)
    E0y : float, optional
        Peak Ey-field value in V/m (default: 0.0 V/m)
    E0z : float, optional
        Peak Ez-field value in V/m (default: 0.0 V/m)
    output_dir : str, optional
        Directory for output files (default: 'fields')
    write_2d_slices : bool, optional
        Whether to write 2D slice files (default: True)
    slice_interval : int, optional
        Interval for writing 2D slices (default: 10)

    Returns
    -------
    dict
        Dictionary containing final field arrays and statistics

    Notes
    -----
    This function:
    1. Reads spatial grid parameters
    2. Allocates all field arrays including longitudinal decompositions
    3. Initializes SBE solver
    4. Time-loops with plane wave excitation
    5. Calls QuantumWire with full longitudinal field outputs
    6. Tracks max/min values for all fields
    7. Writes time series and 2D slices to files
    """
    print("="*70)
    print("SBE Test Program - Full Longitudinal Field Analysis")
    print("="*70)

    # Read spatial grid structure
    space = ss(Dims=0, Nx=0, Ny=0, Nz=0, dx=0.0, dy=0.0, dz=0.0, epsr=1.0)
    ReadSpaceParams(space_params_file, space)

    # Get spatial grid size integers
    Nx = GetNx(space)
    Ny = GetNy(space)
    Nz = GetNz(space)

    print(f"Grid dimensions: Nx={Nx}, Ny={Ny}, Nz={Nz}")
    print(f"Time steps: {Nt}, dt={dt*1e15:.2f} fs")
    print(f"Wavelength: {lam*1e9:.1f} nm, Pulse width: {tw*1e15:.2f} fs")
    print(f"Peak fields: E0x={E0x:.2e} V/m, E0y={E0y:.2e} V/m, E0z={E0z:.2e} V/m")

    # Allocate Maxwell 3D arrays
    Ex = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Ey = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Ez = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Jx = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Jy = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Jz = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Rho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Exl = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Eyl = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Ezl = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    # Longitudinal field decomposition arrays
    ExlfromPRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EylfromPRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EzlfromPRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    ExlfromP = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EylfromP = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EzlfromP = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    ExlfromRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EylfromRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    EzlfromRho = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    RhoBound = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    # Allocate 1D arrays for quantum wire
    rr = np.zeros(Nx, dtype=np.float64)
    qrr = np.zeros(Nx, dtype=np.float64)

    # Initialize max/min tracking variables
    Ex_max, Ex_min = -np.inf, np.inf
    Ey_max, Ey_min = -np.inf, np.inf
    Ez_max, Ez_min = -np.inf, np.inf
    Rho_max, Rho_min = -np.inf, np.inf
    Jx_max, Jx_min = -np.inf, np.inf
    Jy_max, Jy_min = -np.inf, np.inf
    Jz_max, Jz_min = -np.inf, np.inf
    Exl_max, Exl_min = -np.inf, np.inf
    Eyl_max, Eyl_min = -np.inf, np.inf
    Ezl_max, Ezl_min = -np.inf, np.inf
    ExlPRho_max, ExlPRho_min = -np.inf, np.inf
    EylPRho_max, EylPRho_min = -np.inf, np.inf
    EzlPRho_max, EzlPRho_min = -np.inf, np.inf
    ExlP_max, ExlP_min = -np.inf, np.inf
    EylP_max, EylP_min = -np.inf, np.inf
    EzlP_max, EzlP_min = -np.inf, np.inf
    ExlRho_max, ExlRho_min = -np.inf, np.inf
    EylRho_max, EylRho_min = -np.inf, np.inf
    EzlRho_max, EzlRho_min = -np.inf, np.inf

    # Initialize the Maxwell arrays
    initializefields(Ex, Ey, Ez, Jx, Jy, Jz, Rho, Exl, Eyl, Ezl,
                    ExlfromPRho, EylfromPRho, EzlfromPRho,
                    ExlfromP, EylfromP, EzlfromP,
                    ExlfromRho, EylfromRho, EzlfromRho, RhoBound)

    # Calculate angular frequencies and optical cycle (for Y-direction)
    w0y = twopi * c0 / lam
    k0y = twopi / lam * n0
    Tcy = lam / c0

    # Calculate maximum field possible during simulation
    Emax = np.sqrt(E0x**2 + E0y**2 + E0z**2)
    print(f"Maximum field: {Emax:.2e} V/m")

    # Calculate real-space and q-space arrays
    rr = GetYArray(space)
    qrr = GetKyArray(space)

    # Initialize the SBEs
    print("Initializing SBE solver...")
    InitializeSBE(qrr, rr, 0.0, Emax, lam, 4, True)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open files to record data
    file_handles = {}
    output_files = {
        'Ex': f'{output_dir}/Ex.dat',
        'Ey': f'{output_dir}/Ey.dat',
        'Ez': f'{output_dir}/Ez.dat',
        'Jx': f'{output_dir}/Jx.dat',
        'Jy': f'{output_dir}/Jy.dat',
        'Jz': f'{output_dir}/Jz.dat',
        'Rho': f'{output_dir}/Rho.dat',
        'Eywireloc': f'{output_dir}/Eywireloc.dat',
        'Exl': f'{output_dir}/Exl.dat',
        'Eyl': f'{output_dir}/Eyl.dat',
        'Ezl': f'{output_dir}/Ezl.dat',
        'ExlfromPRho': f'{output_dir}/ExlfromPRho.dat',
        'EylfromPRho': f'{output_dir}/EylfromPRho.dat',
        'EzlfromPRho': f'{output_dir}/EzlfromPRho.dat',
        'ExlfromP': f'{output_dir}/ExlfromP.dat',
        'EylfromP': f'{output_dir}/EylfromP.dat',
        'EzlfromP': f'{output_dir}/EzlfromP.dat',
        'ExlfromRho': f'{output_dir}/ExlfromRho.dat',
        'EylfromRho': f'{output_dir}/EylfromRho.dat',
        'EzlfromRho': f'{output_dir}/EzlfromRho.dat',
        'final_max_min': f'{output_dir}/final_max_min.dat',
        'RhoBound': f'{output_dir}/RhoBound.dat',
    }

    for key, filename in output_files.items():
        file_handles[key] = open(filename, 'w', encoding='utf-8')

    print("Starting time loop...")
    print("="*70)

    # Time loop
    t = 0.0
    for n in range(1, Nt + 1):
        # Update user on command line
        print(f"Step {n}/{Nt}")

        # Create plane wave excitation (Y-direction propagation)
        MakePlaneWaveY(Ex, space, t, E0x, lam, tw, tp)

        # Update quantum wire response with full longitudinal field decomposition
        QuantumWire(space, dt, n, Ex, Ey, Ez, Jx, Jy, Jz, Rho,
                   ExlfromPRho, EylfromPRho, EzlfromPRho,
                   ExlfromP, EylfromP, EzlfromP,
                   ExlfromRho, EylfromRho, EzlfromRho, RhoBound)

        # Check differences (diagnostic)
        if n % 120 == 0:
            diff_x = np.max(np.abs(ExlfromPRho - ExlfromP))
            diff_y = np.max(np.abs(EylfromPRho - EylfromP))
            diff_z = np.max(np.abs(EzlfromPRho - EzlfromP))
            print(f"  After QuantumWire difference check: {diff_x:.4e}, {diff_y:.4e}, {diff_z:.4e}")
            print(f"  ExlfromP:    {np.max(np.abs(ExlfromP)):.4e}")
            print(f"  ExlfromPRho: {np.max(np.abs(ExlfromPRho)):.4e}")
            print(f"  ExlfromRho:  {np.max(np.abs(ExlfromRho)):.4e}")

        # Separate longitudinal field components (Helmholtz decomposition stub)
        ElongSeparate(space, Ex, Ey, Ez, Exl, Eyl, Ezl)

        # Print diagnostics every 10 steps
        if n % 10 == 0:
            print(f"  Ex-max          = {np.max(np.abs(Ex)):.4e}")
            print(f"  Ey-max          = {np.max(np.abs(Ey)):.4e}")
            print(f"  Ez-max          = {np.max(np.abs(Ez)):.4e}")
            print(f"  Rho-max         = {np.max(np.abs(Rho)):.4e}")
            print(f"  Jx-max          = {np.max(np.abs(Jx)):.4e}")
            print(f"  Jy-max          = {np.max(np.abs(Jy)):.4e}")
            print(f"  Jz-max          = {np.max(np.abs(Jz)):.4e}")
            print(f"  ExlfromPRho-max = {np.max(np.real(ExlfromPRho)):.4e}")
            print(f"  EylfromPRho-max = {np.max(np.real(EylfromPRho)):.4e}")
            print(f"  EzlfromPRho-max = {np.max(np.real(EzlfromPRho)):.4e}")
            print(f"  ExlfromP-max    = {np.max(np.real(ExlfromP)):.4e}")
            print(f"  EylfromP-max    = {np.max(np.real(EylfromP)):.4e}")
            print(f"  EzlfromP-max    = {np.max(np.real(EzlfromP)):.4e}")
            print(f"  ExlfromRho-max  = {np.max(np.real(ExlfromRho)):.4e}")
            print(f"  EylfromRho-max  = {np.max(np.real(EylfromRho)):.4e}")
            print(f"  EzlfromRho-max  = {np.max(np.real(EzlfromRho)):.4e}")
            print(f"  RhoBound-max    = {np.max(np.real(RhoBound)):.4e}")
            print(f"  Ex:     max={Ex_max:.4e}  min={Ex_min:.4e}")
            print(f"  Ey:     max={Ey_max:.4e}  min={Ey_min:.4e}")
            print(f"  Rho:    max={Rho_max:.4e}  min={Rho_min:.4e}")

        # Update max/min values
        Ex_max = max(Ex_max, np.max(np.abs(Ex)))
        Ex_min = min(Ex_min, np.min(np.real(Ex)))
        Ey_max = max(Ey_max, np.max(np.abs(Ey)))
        Ey_min = min(Ey_min, np.min(np.real(Ey)))
        Ez_max = max(Ez_max, np.max(np.abs(Ez)))
        Ez_min = min(Ez_min, np.min(np.real(Ez)))

        Rho_max = max(Rho_max, np.max(np.abs(Rho)))
        Rho_min = min(Rho_min, np.min(np.real(Rho)))
        Jx_max = max(Jx_max, np.max(np.abs(Jx)))
        Jx_min = min(Jx_min, np.min(np.real(Jx)))
        Jy_max = max(Jy_max, np.max(np.abs(Jy)))
        Jy_min = min(Jy_min, np.min(np.real(Jy)))
        Jz_max = max(Jz_max, np.max(np.abs(Jz)))
        Jz_min = min(Jz_min, np.min(np.real(Jz)))

        Exl_max = max(Exl_max, np.max(np.abs(Exl)))
        Exl_min = min(Exl_min, np.min(np.real(Exl)))
        Eyl_max = max(Eyl_max, np.max(np.abs(Eyl)))
        Eyl_min = min(Eyl_min, np.min(np.real(Eyl)))
        Ezl_max = max(Ezl_max, np.max(np.abs(Ezl)))
        Ezl_min = min(Ezl_min, np.min(np.real(Ezl)))

        ExlPRho_max = max(ExlPRho_max, np.max(np.real(ExlfromPRho)))
        ExlPRho_min = min(ExlPRho_min, np.min(np.real(ExlfromPRho)))
        EylPRho_max = max(EylPRho_max, np.max(np.real(EylfromPRho)))
        EylPRho_min = min(EylPRho_min, np.min(np.real(EylfromPRho)))
        EzlPRho_max = max(EzlPRho_max, np.max(np.real(EzlfromPRho)))
        EzlPRho_min = min(EzlPRho_min, np.min(np.real(EzlfromPRho)))

        ExlP_max = max(ExlP_max, np.max(np.real(ExlfromP)))
        ExlP_min = min(ExlP_min, np.min(np.real(ExlfromP)))
        EylP_max = max(EylP_max, np.max(np.real(EylfromP)))
        EylP_min = min(EylP_min, np.min(np.real(EylfromP)))
        EzlP_max = max(EzlP_max, np.max(np.real(EzlfromP)))
        EzlP_min = min(EzlP_min, np.min(np.real(EzlfromP)))

        ExlRho_max = max(ExlRho_max, np.max(np.real(ExlfromRho)))
        ExlRho_min = min(ExlRho_min, np.min(np.real(ExlfromRho)))
        EylRho_max = max(EylRho_max, np.max(np.real(EylfromRho)))
        EylRho_min = min(EylRho_min, np.min(np.real(EylfromRho)))
        EzlRho_max = max(EzlRho_max, np.max(np.real(EzlfromRho)))
        EzlRho_min = min(EzlRho_min, np.min(np.real(EzlfromRho)))

        # Write time series data to files
        file_handles['Ex'].write(f"{t:.15e} {np.real(Ex[0, 0, 0]):.15e}\n")
        file_handles['Ey'].write(f"{t:.15e} {np.real(Ey[0, Ny//2, 0]):.15e}\n")
        file_handles['Ez'].write(f"{t:.15e} {np.real(Ez[0, 0, 0]):.15e}\n")
        file_handles['Jx'].write(f"{t:.15e} {np.real(Jx[0, Ny//2, 0]):.15e}\n")
        file_handles['Jy'].write(f"{t:.15e} {np.real(Jy[0, Ny//2, 0]):.15e}\n")
        file_handles['Jz'].write(f"{t:.15e} {np.real(Jz[0, Ny//2, 0]):.15e}\n")
        file_handles['Rho'].write(f"{t:.15e} {np.real(Rho[0, Ny//2, 0]):.15e}\n")
        file_handles['Eywireloc'].write(f"{t:.15e} {np.real(Ey[Nx//4, Ny//2, Nz//2]):.15e}\n")

        file_handles['ExlfromPRho'].write(f"{t:.15e} {np.real(ExlfromPRho[0, 0, 0]):.15e}\n")
        file_handles['EylfromPRho'].write(f"{t:.15e} {np.real(EylfromPRho[0, 0, 0]):.15e}\n")
        file_handles['EzlfromPRho'].write(f"{t:.15e} {np.real(EzlfromPRho[0, 0, 0]):.15e}\n")

        file_handles['ExlfromP'].write(f"{t:.15e} {np.real(ExlfromP[0, 0, 0]):.15e}\n")
        file_handles['EylfromP'].write(f"{t:.15e} {np.real(EylfromP[0, 0, 0]):.15e}\n")
        file_handles['EzlfromP'].write(f"{t:.15e} {np.real(EzlfromP[0, 0, 0]):.15e}\n")

        file_handles['ExlfromRho'].write(f"{t:.15e} {np.real(ExlfromRho[0, 0, 0]):.15e}\n")
        file_handles['EylfromRho'].write(f"{t:.15e} {np.real(EylfromRho[0, 0, 0]):.15e}\n")
        file_handles['EzlfromRho'].write(f"{t:.15e} {np.real(EzlfromRho[0, 0, 0]):.15e}\n")

        file_handles['RhoBound'].write(f"{t:.15e} {np.real(RhoBound[0, Ny//2, 0]):.15e}\n")

        # Write 2D slices at intervals
        if write_2d_slices and (n % slice_interval == 0):
            # ExlfromPRho slices
            WriteIT2D(np.real(ExlfromPRho[:, :, Nz//2]), f'ExPRho.{int2str(n)}.z')
            WriteIT2D(np.real(EylfromPRho[:, :, Nz//2]), f'EyPRho.{int2str(n)}.z')
            WriteIT2D(np.real(EzlfromPRho[:, :, Nz//2]), f'EzPRho.{int2str(n)}.z')

            WriteIT2D(np.real(ExlfromPRho[:, Ny//2, :]), f'ExPRho.{int2str(n)}.y')
            WriteIT2D(np.real(EylfromPRho[:, Ny//2, :]), f'EyPRho.{int2str(n)}.y')
            WriteIT2D(np.real(EzlfromPRho[:, Ny//2, :]), f'EzPRho.{int2str(n)}.y')

            WriteIT2D(np.real(ExlfromPRho[Nx//2, :, :]), f'ExPRho.{int2str(n)}.x')
            WriteIT2D(np.real(EylfromPRho[Nx//2, :, :]), f'EyPRho.{int2str(n)}.x')
            WriteIT2D(np.real(EzlfromPRho[Nx//2, :, :]), f'EzPRho.{int2str(n)}.x')

            # Rho slices
            WriteIT2D(np.real(Rho[:, :, Nz//2]), f'Rho.{int2str(n)}.z')
            WriteIT2D(np.real(Rho[:, Ny//2, :]), f'Rho.{int2str(n)}.y')
            WriteIT2D(np.real(Rho[Nx//2, :, :]), f'Rho.{int2str(n)}.x')

            # ExlfromP slices
            WriteIT2D(np.real(ExlfromP[:, :, Nz//2]), f'ExP.{int2str(n)}.z')
            WriteIT2D(np.real(EylfromP[:, :, Nz//2]), f'EyP.{int2str(n)}.z')
            WriteIT2D(np.real(EzlfromP[:, :, Nz//2]), f'EzP.{int2str(n)}.z')

            WriteIT2D(np.real(ExlfromP[:, Ny//2, :]), f'ExP.{int2str(n)}.y')
            WriteIT2D(np.real(EylfromP[:, Ny//2, :]), f'EyP.{int2str(n)}.y')
            WriteIT2D(np.real(EzlfromP[:, Ny//2, :]), f'EzP.{int2str(n)}.y')

            WriteIT2D(np.real(ExlfromP[Nx//2, :, :]), f'ExP.{int2str(n)}.x')
            WriteIT2D(np.real(EylfromP[Nx//2, :, :]), f'EyP.{int2str(n)}.x')
            WriteIT2D(np.real(EzlfromP[Nx//2, :, :]), f'EzP.{int2str(n)}.x')

            # ExlfromRho slices
            WriteIT2D(np.real(ExlfromRho[:, :, Nz//2]), f'ExRho.{int2str(n)}.z')
            WriteIT2D(np.real(EylfromRho[:, :, Nz//2]), f'EyRho.{int2str(n)}.z')
            WriteIT2D(np.real(EzlfromRho[:, :, Nz//2]), f'EzRho.{int2str(n)}.z')

            WriteIT2D(np.real(ExlfromRho[:, Ny//2, :]), f'ExRho.{int2str(n)}.y')
            WriteIT2D(np.real(EylfromRho[:, Ny//2, :]), f'EyRho.{int2str(n)}.y')
            WriteIT2D(np.real(EzlfromRho[:, Ny//2, :]), f'EzRho.{int2str(n)}.y')

            WriteIT2D(np.real(ExlfromRho[Nx//2, :, :]), f'ExRho.{int2str(n)}.x')
            WriteIT2D(np.real(EylfromRho[Nx//2, :, :]), f'EyRho.{int2str(n)}.x')
            WriteIT2D(np.real(EzlfromRho[Nx//2, :, :]), f'EzRho.{int2str(n)}.x')

            # Exl slices
            WriteIT2D(np.real(Exl[:, :, Nz//2]), f'Exl.{int2str(n)}.z')
            WriteIT2D(np.real(Eyl[:, :, Nz//2]), f'Eyl.{int2str(n)}.z')
            WriteIT2D(np.real(Ezl[:, :, Nz//2]), f'Ezl.{int2str(n)}.z')

            WriteIT2D(np.real(Exl[:, Ny//2, :]), f'Exl.{int2str(n)}.y')
            WriteIT2D(np.real(Eyl[:, Ny//2, :]), f'Eyl.{int2str(n)}.y')
            WriteIT2D(np.real(Ezl[:, Ny//2, :]), f'Ezl.{int2str(n)}.y')

            WriteIT2D(np.real(Exl[Nx//2, :, :]), f'Exl.{int2str(n)}.x')
            WriteIT2D(np.real(Eyl[Nx//2, :, :]), f'Eyl.{int2str(n)}.x')
            WriteIT2D(np.real(Ezl[Nx//2, :, :]), f'Ezl.{int2str(n)}.x')

            # RhoBound slices
            WriteIT2D(np.real(RhoBound[:, :, Nz//2]), f'RhoB.{int2str(n)}.z')
            WriteIT2D(np.real(RhoBound[:, Ny//2, :]), f'RhoB.{int2str(n)}.y')
            WriteIT2D(np.real(RhoBound[Nx//2, :, :]), f'RhoB.{int2str(n)}.x')

        # Write max/min summary every 1000 steps
        if n % 1000 == 0:
            fh = file_handles['final_max_min']
            fh.write(f"Max/Min Field Values Over {n} Time Steps\n")
            fh.write(f"Ex:           Max = {Ex_max:.6e}  Min = {Ex_min:.6e}\n")
            fh.write(f"Ey:           Max = {Ey_max:.6e}  Min = {Ey_min:.6e}\n")
            fh.write(f"Ez:           Max = {Ez_max:.6e}  Min = {Ez_min:.6e}\n")
            fh.write(f"Rho:          Max = {Rho_max:.6e}  Min = {Rho_min:.6e}\n")
            fh.write(f"Jx:           Max = {Jx_max:.6e}  Min = {Jx_min:.6e}\n")
            fh.write(f"Jy:           Max = {Jy_max:.6e}  Min = {Jy_min:.6e}\n")
            fh.write(f"Jz:           Max = {Jz_max:.6e}  Min = {Jz_min:.6e}\n")
            fh.write(f"Exl:          Max = {Exl_max:.6e}  Min = {Exl_min:.6e}\n")
            fh.write(f"Eyl:          Max = {Eyl_max:.6e}  Min = {Eyl_min:.6e}\n")
            fh.write(f"Ezl:          Max = {Ezl_max:.6e}  Min = {Ezl_min:.6e}\n")
            fh.write(f"ExlfromPRho:  Max = {ExlPRho_max:.6e}  Min = {ExlPRho_min:.6e}\n")
            fh.write(f"EylfromPRho:  Max = {EylPRho_max:.6e}  Min = {EylPRho_min:.6e}\n")
            fh.write(f"EzlfromPRho:  Max = {EzlPRho_max:.6e}  Min = {EzlPRho_min:.6e}\n")
            fh.write(f"ExlfromP:     Max = {ExlP_max:.6e}  Min = {ExlP_min:.6e}\n")
            fh.write(f"EylfromP:     Max = {EylP_max:.6e}  Min = {EylP_min:.6e}\n")
            fh.write(f"EzlfromP:     Max = {EzlP_max:.6e}  Min = {EzlP_min:.6e}\n")
            fh.write(f"ExlfromRho:   Max = {ExlRho_max:.6e}  Min = {ExlRho_min:.6e}\n")
            fh.write(f"EylfromRho:   Max = {EylRho_max:.6e}  Min = {EylRho_min:.6e}\n")
            fh.write(f"EzlfromRho:   Max = {EzlRho_max:.6e}  Min = {EzlRho_min:.6e}\n")
            fh.write("\n")

        # Increment time
        t = t + dt

    # Print final statistics
    print("="*70)
    print("FINAL FIELD MAX/MIN VALUES OVER ALL TIME STEPS")
    print("="*70)
    print(f"Ex:           Max = {Ex_max:.6e}  Min = {Ex_min:.6e}")
    print(f"Ey:           Max = {Ey_max:.6e}  Min = {Ey_min:.6e}")
    print(f"Ez:           Max = {Ez_max:.6e}  Min = {Ez_min:.6e}")
    print(f"Rho:          Max = {Rho_max:.6e}  Min = {Rho_min:.6e}")
    print(f"Jx:           Max = {Jx_max:.6e}  Min = {Jx_min:.6e}")
    print(f"Jy:           Max = {Jy_max:.6e}  Min = {Jy_min:.6e}")
    print(f"Jz:           Max = {Jz_max:.6e}  Min = {Jz_min:.6e}")
    print(f"Exl:          Max = {Exl_max:.6e}  Min = {Exl_min:.6e}")
    print(f"Eyl:          Max = {Eyl_max:.6e}  Min = {Eyl_min:.6e}")
    print(f"Ezl:          Max = {Ezl_max:.6e}  Min = {Ezl_min:.6e}")
    print(f"ExlfromPRho:  Max = {ExlPRho_max:.6e}  Min = {ExlPRho_min:.6e}")
    print(f"EylfromPRho:  Max = {EylPRho_max:.6e}  Min = {EylPRho_min:.6e}")
    print(f"EzlfromPRho:  Max = {EzlPRho_max:.6e}  Min = {EzlPRho_min:.6e}")
    print(f"ExlfromP:     Max = {ExlP_max:.6e}  Min = {ExlP_min:.6e}")
    print(f"EylfromP:     Max = {EylP_max:.6e}  Min = {EylP_min:.6e}")
    print(f"EzlfromP:     Max = {EzlP_max:.6e}  Min = {EzlP_min:.6e}")
    print(f"ExlfromRho:   Max = {ExlRho_max:.6e}  Min = {ExlRho_min:.6e}")
    print(f"EylfromRho:   Max = {EylRho_max:.6e}  Min = {EylRho_min:.6e}")
    print(f"EzlfromRho:   Max = {EzlRho_max:.6e}  Min = {EzlRho_min:.6e}")

    # Write final max/min to file
    fh = file_handles['final_max_min']
    fh.write("Max/Min Field Values Over All Time Steps\n")
    fh.write(f"Ex:           Max = {Ex_max:.6e}  Min = {Ex_min:.6e}\n")
    fh.write(f"Ey:           Max = {Ey_max:.6e}  Min = {Ey_min:.6e}\n")
    fh.write(f"Ez:           Max = {Ez_max:.6e}  Min = {Ez_min:.6e}\n")
    fh.write(f"Rho:          Max = {Rho_max:.6e}  Min = {Rho_min:.6e}\n")
    fh.write(f"Jx:           Max = {Jx_max:.6e}  Min = {Jx_min:.6e}\n")
    fh.write(f"Jy:           Max = {Jy_max:.6e}  Min = {Jy_min:.6e}\n")
    fh.write(f"Jz:           Max = {Jz_max:.6e}  Min = {Jz_min:.6e}\n")
    fh.write(f"Exl:          Max = {Exl_max:.6e}  Min = {Exl_min:.6e}\n")
    fh.write(f"Eyl:          Max = {Eyl_max:.6e}  Min = {Eyl_min:.6e}\n")
    fh.write(f"Ezl:          Max = {Ezl_max:.6e}  Min = {Ezl_min:.6e}\n")
    fh.write(f"ExlfromPRho:  Max = {ExlPRho_max:.6e}  Min = {ExlPRho_min:.6e}\n")
    fh.write(f"EylfromPRho:  Max = {EylPRho_max:.6e}  Min = {EylPRho_min:.6e}\n")
    fh.write(f"EzlfromPRho:  Max = {EzlPRho_max:.6e}  Min = {EzlPRho_min:.6e}\n")
    fh.write(f"ExlfromP:     Max = {ExlP_max:.6e}  Min = {ExlP_min:.6e}\n")
    fh.write(f"EylfromP:     Max = {EylP_max:.6e}  Min = {EylP_min:.6e}\n")
    fh.write(f"EzlfromP:     Max = {EzlP_max:.6e}  Min = {EzlP_min:.6e}\n")
    fh.write(f"ExlfromRho:   Max = {ExlRho_max:.6e}  Min = {ExlRho_min:.6e}\n")
    fh.write(f"EylfromRho:   Max = {EylRho_max:.6e}  Min = {EylRho_min:.6e}\n")
    fh.write(f"EzlfromRho:   Max = {EzlRho_max:.6e}  Min = {EzlRho_min:.6e}\n")

    # Close all files
    for fh in file_handles.values():
        fh.close()

    print("="*70)
    print("Simulation complete!")
    print(f"Output files written to: {output_dir}/")
    print("="*70)

    # Return field arrays and statistics
    return {
        'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
        'Jx': Jx, 'Jy': Jy, 'Jz': Jz,
        'Rho': Rho,
        'Exl': Exl, 'Eyl': Eyl, 'Ezl': Ezl,
        'ExlfromPRho': ExlfromPRho, 'EylfromPRho': EylfromPRho, 'EzlfromPRho': EzlfromPRho,
        'ExlfromP': ExlfromP, 'EylfromP': EylfromP, 'EzlfromP': EzlfromP,
        'ExlfromRho': ExlfromRho, 'EylfromRho': EylfromRho, 'EzlfromRho': EzlfromRho,
        'RhoBound': RhoBound,
        'stats': {
            'Ex_max': Ex_max, 'Ex_min': Ex_min,
            'Ey_max': Ey_max, 'Ey_min': Ey_min,
            'Ez_max': Ez_max, 'Ez_min': Ez_min,
            'Rho_max': Rho_max, 'Rho_min': Rho_min,
        }
    }


def main():
    """
    Main entry point for command-line execution.

    Runs the SBE test program with default parameters.
    """
    try:
        results = SBETest()
        print("\nSimulation completed successfully!")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

