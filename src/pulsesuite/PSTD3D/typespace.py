"""
Spatial grid structure for quantum wire simulations.

This module provides the spatial grid structure (ss type) and associated
functions for reading, writing, and accessing spatial grid parameters.
Supports 1D, 2D, and 3D grids with configurable pixel sizes and dielectric constants.

Author: Rahul R. Sah
"""

import sys
from dataclasses import dataclass

import numpy as np

try:
    from numba import jit
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False
    # Fallback: create a no-op decorator
    def jit(*args, **kwargs):  # noqa: ARG001, ARG002
        def decorator(func):
            return func
        if args and callable(args[0]):
            # Called as @jit without parentheses
            return args[0]
        return decorator


# Physical constants (matching Fortran constants module)
pi = np.pi
twopi = 2.0 * np.pi  # 2π constant (matching Fortran twopi)


# Default logging level constants (if not provided by logging module)
LOGVERBOSE = 2


def GetFileParam(file_handle):
    """
    Read a parameter from a file handle.

    Reads a single numeric parameter from the file. Assumes the file format
    has one parameter per line, possibly with comments after the value.

    Parameters
    ----------
    file_handle : file
        Open file handle to read from

    Returns
    -------
    float
        Parameter value read from file

    Notes
    -----
    This is a helper function for reading parameter files. It reads a line,
    extracts the first numeric value, and returns it.

    This function is not defined in typespace.f90 but is used by
    readspaceparams_sub. It is likely defined in the fileio module in Fortran.
    This Python implementation provides equivalent functionality.
    """
    line = file_handle.readline()
    if not line:
        raise ValueError("Unexpected end of file while reading parameter")
    # Extract first numeric value from line (handles comments)
    parts = line.split()
    if not parts:
        raise ValueError(f"Empty line in parameter file: {line}")
    try:
        # Try to convert first part to float
        return float(parts[0])
    except ValueError as exc:
        raise ValueError(f"Could not parse parameter value from line: {line}") from exc


def GetLogLevel():
    """
    Get the current logging level.

    Returns the current logging verbosity level. Higher values mean more verbose.

    Returns
    -------
    int
        Current logging level (default: 0)

    Notes
    -----
    This is a stub implementation. In a full system, this would interface
    with a logging module to get the actual log level.

    This function is not defined in typespace.f90 but is used by dumpspace.
    It is likely defined in the logger module in Fortran. This Python
    implementation provides a default minimal logging level.
    """
    return 0  # Default: minimal logging


@dataclass
class ss:
    """
    Spatial grid structure.

    The spatial grid structure contains all parameters needed to define
    a 1D, 2D, or 3D spatial grid for simulations.

    Attributes
    ----------
    Dims : int
        Dimensionality of grid (1=1D, 2=2D, 3=3D)
    Nx : int
        X-window number of pixels
    Ny : int
        Y-window number of pixels [Ny=1 for 1D]
    Nz : int
        Z-window number of pixels [Nz=1 for 1D and 2D]
    dx : float
        Size of x-pixel (m)
    dy : float
        Size of y-pixel (m), [dy=1.0 for 1D]
    dz : float
        Size of z-pixel (m), [dz=1.0 for 1D and 2D]
    epsr : float
        Background dielectric constant
    """
    Dims: int
    Nx: int
    Ny: int
    Nz: int
    dx: float
    dy: float
    dz: float
    epsr: float


def readspaceparams_sub(file_handle, space):
    """
    Read the space parameters (ss) from a file handle.

    Reads all space parameters from an open file handle and populates
    the space structure.

    Parameters
    ----------
    file_handle : file
        Open file handle to read from
    space : ss
        Space structure to populate (modified in-place)

    Returns
    -------
    None
        space structure is modified in-place

    Notes
    -----
    The file format expects 8 parameters in order:
    1. Dims (integer)
    2. Nx (integer)
    3. Ny (integer)
    4. Nz (integer)
    5. dx (float)
    6. dy (float)
    7. dz (float)
    8. epsr (float)
    """
    space.Dims = int(GetFileParam(file_handle))
    space.Nx = int(GetFileParam(file_handle))
    space.Ny = int(GetFileParam(file_handle))
    space.Nz = int(GetFileParam(file_handle))
    space.dx = GetFileParam(file_handle)
    space.dy = GetFileParam(file_handle)
    space.dz = GetFileParam(file_handle)
    space.epsr = GetFileParam(file_handle)


def ReadSpaceParams(filename, space):
    """
    Read the space parameters (ss) from a file.

    Opens the specified file, reads the space parameters, and populates
    the space structure. Also dumps the parameters to stdout if logging
    level is sufficient.

    Parameters
    ----------
    filename : str
        The filename to read from
    space : ss
        Space structure to populate (modified in-place)

    Returns
    -------
    None
        space structure is modified in-place

    Notes
    -----
    The file is opened, read, and closed automatically.
    After reading, dumpspace is called to log the parameters.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        readspaceparams_sub(f, space)
    dumpspace(space)


def WriteSpaceParams_sub(file_handle, space):
    """
    Write space parameters (ss) to a file handle.

    Writes all space parameters to an open file handle with descriptive
    comments for each parameter.

    Parameters
    ----------
    file_handle : file
        Open file handle to write to
    space : ss
        Space structure to write

    Returns
    -------
    None

    Notes
    -----
    The format matches the Fortran output format:
    - Integers are written with format I25 (right-aligned, 25 characters)
    - Floats are written with format E25.15E3 (scientific notation, 25 chars)
    - Each line has a comment describing the parameter

    Note: The Fortran source has typos in lines 94-96 (says "With" instead of "Width"
    and all three say "X pixel" instead of X/Y/Z). This Python implementation
    corrects those typos.
    """
    # Format: I25 for integers, E25.15E3 for floats (matching Fortran pfrmtA)
    file_handle.write(f"{space.Dims:25d} : Number of dimensions.\n")
    file_handle.write(f"{space.Nx:25d} : Number of space X points.\n")
    file_handle.write(f"{space.Ny:25d} : Number of space Y points.\n")
    file_handle.write(f"{space.Nz:25d} : Number of space Z points.\n")
    # Note: Fixed Fortran typos - original had "With" and all said "X pixel"
    file_handle.write(f"{space.dx:25.15E} : The Width of the X pixel. (m)\n")
    file_handle.write(f"{space.dy:25.15E} : The Width of the Y pixel. (m)\n")
    file_handle.write(f"{space.dz:25.15E} : The Width of the Z pixel. (m)\n")
    file_handle.write(f"{space.epsr:25.15E} : The relative background dielectric constant\n")


def writespaceparams(filename, space):
    """
    Write space parameters (ss) to a file.

    Opens the specified file, writes all space parameters with descriptive
    comments, and closes the file.

    Parameters
    ----------
    filename : str
        The filename to write to
    space : ss
        Space structure to write

    Returns
    -------
    None

    Notes
    -----
    The file is opened in write mode, written to, and closed automatically.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        WriteSpaceParams_sub(f, space)


def dumpspace(params, level=None):
    """
    Dump space parameters to stdout.

    Writes space parameters to stdout (file handle 0) if the logging level
    is sufficient. Used for debugging and verbose output.

    Parameters
    ----------
    params : ss
        Space structure to dump
    level : int, optional
        Minimum logging level required to dump. If not provided, uses
        LOGVERBOSE as the default threshold.

    Returns
    -------
    None

    Notes
    -----
    If level is provided, dumps only if GetLogLevel() >= level.
    Otherwise, dumps only if GetLogLevel() >= LOGVERBOSE.
    """
    if level is not None:
        if GetLogLevel() >= level:
            WriteSpaceParams_sub(sys.stdout, params)
    else:
        if GetLogLevel() >= LOGVERBOSE:
            WriteSpaceParams_sub(sys.stdout, params)


# Getter functions - these are simple property accessors
# JIT compilation not needed for simple attribute access

def GetNx(space):
    """
    Get the number of points in the x direction.

    Returns the number of spatial grid points in the x-direction.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    int
        Number of points in the x direction
    """
    return space.Nx


def GetNy(space):
    """
    Get the number of points in the y direction.

    Returns the number of spatial grid points in the y-direction.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    int
        Number of points in the y direction
    """
    return space.Ny


def GetNz(space):
    """
    Get the number of points in the z direction.

    Returns the number of spatial grid points in the z-direction.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    int
        Number of points in the z direction
    """
    return space.Nz


# Getter functions with special logic - these can benefit from JIT
# but need to be careful about nopython compatibility

@jit
def _GetDx_core(Nx, dx):
    """
    JIT-compiled core for GetDx.

    Returns dx if Nx > 1, otherwise returns 1.0.
    """
    if Nx == 1:
        return 1.0
    else:
        return dx


def GetDx(space):
    """
    Get the step size for the x dimension.

    Returns the x-pixel size. If Nx == 1 (collapsed dimension),
    returns 1.0 instead of the actual dx value.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        Step size for x dimension (m), or 1.0 if Nx == 1
    """
    if JIT_AVAILABLE:
        try:
            return _GetDx_core(space.Nx, space.dx)
        except (TypeError, ValueError, RuntimeError):
            # Fallback to standard Python
            if space.Nx == 1:
                return 1.0
            else:
                return space.dx
    else:
        if space.Nx == 1:
            return 1.0
        else:
            return space.dx


@jit
def _GetDy_core(Ny, dy):
    """
    JIT-compiled core for GetDy.

    Returns dy if Ny > 1, otherwise returns 1.0.
    """
    if Ny == 1:
        return 1.0
    else:
        return dy


def GetDy(space):
    """
    Get the step size for the y dimension.

    Returns the y-pixel size. If Ny == 1 (collapsed dimension),
    returns 1.0 instead of the actual dy value.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        Step size for y dimension (m), or 1.0 if Ny == 1
    """
    if JIT_AVAILABLE:
        try:
            return _GetDy_core(space.Ny, space.dy)
        except (TypeError, ValueError, RuntimeError):
            # Fallback to standard Python
            if space.Ny == 1:
                return 1.0
            else:
                return space.dy
    else:
        if space.Ny == 1:
            return 1.0
        else:
            return space.dy


@jit
def _GetDz_core(Nz, dz):
    """
    JIT-compiled core for GetDz.

    Returns dz if Nz > 1, otherwise returns 1.0.
    """
    if Nz == 1:
        return 1.0
    else:
        return dz


def GetDz(space):
    """
    Get the step size for the z dimension.

    Returns the z-pixel size. If Nz == 1 (collapsed dimension),
    returns 1.0 instead of the actual dz value.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        Step size for z dimension (m), or 1.0 if Nz == 1
    """
    if JIT_AVAILABLE:
        try:
            return _GetDz_core(space.Nz, space.dz)
        except (TypeError, ValueError, RuntimeError):
            # Fallback to standard Python
            if space.Nz == 1:
                return 1.0
            else:
                return space.dz
    else:
        if space.Nz == 1:
            return 1.0
        else:
            return space.dz


@jit
def _GetEpsr_core(Nz, epsr):
    """
    JIT-compiled core for GetEpsr.

    Returns epsr if Nz > 1, otherwise returns 1.0.
    """
    if Nz == 1:
        return 1.0
    else:
        return epsr


def GetEpsr(space):
    """
    Get the relative background dielectric constant.

    Returns the dielectric constant. If Nz == 1 (1D or 2D case),
    returns 1.0 instead of the actual epsr value.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        Relative background dielectric constant, or 1.0 if Nz == 1
    """
    if JIT_AVAILABLE:
        try:
            return _GetEpsr_core(space.Nz, space.epsr)
        except (TypeError, ValueError, RuntimeError):
            # Fallback to standard Python
            if space.Nz == 1:
                return 1.0
            else:
                return space.epsr
    else:
        if space.Nz == 1:
            return 1.0
        else:
            return space.epsr


# Setter functions - simple attribute modifiers
# JIT not needed for simple assignments

def SetNx(space, N):
    """
    Set the number of points in the x dimension.

    Sets the number of spatial grid points in the x-direction.

    Parameters
    ----------
    space : ss
        Space structure (modified in-place)
    N : int
        Number of points in x direction

    Returns
    -------
    None
        space structure is modified in-place
    """
    space.Nx = N


def SetNy(space, N):
    """
    Set the number of points in the y dimension.

    Sets the number of spatial grid points in the y-direction.

    Parameters
    ----------
    space : ss
        Space structure (modified in-place)
    N : int
        Number of points in y direction

    Returns
    -------
    None
        space structure is modified in-place
    """
    space.Ny = N


def SetNz(space, N):
    """
    Set the number of points in the z dimension.

    Sets the number of spatial grid points in the z-direction.

    Parameters
    ----------
    space : ss
        Space structure (modified in-place)
    N : int
        Number of points in z direction

    Returns
    -------
    None
        space structure is modified in-place
    """
    space.Nz = N


def SetDx(space, dl):
    """
    Set the step size for the x dimension.

    Sets the x-pixel size.

    Parameters
    ----------
    space : ss
        Space structure (modified in-place)
    dl : float
        Step size for x dimension (m)

    Returns
    -------
    None
        space structure is modified in-place
    """
    space.dx = dl


def SetDy(space, dl):
    """
    Set the step size for the y dimension.

    Sets the y-pixel size.

    Parameters
    ----------
    space : ss
        Space structure (modified in-place)
    dl : float
        Step size for y dimension (m)

    Returns
    -------
    None
        space structure is modified in-place
    """
    space.dy = dl


def SetDz(space, dl):
    """
    Set the step size for the z dimension.

    Sets the z-pixel size.

    Parameters
    ----------
    space : ss
        Space structure (modified in-place)
    dl : float
        Step size for z dimension (m)

    Returns
    -------
    None
        space structure is modified in-place
    """
    space.dz = dl


# Width calculation functions
# These are simple calculations, JIT may help but not critical

@jit
def _GetXWidth_core(dx, Nx):
    """
    JIT-compiled core for GetXWidth.

    Returns dx * (Nx - 1).
    """
    return dx * (Nx - 1)


def GetXWidth(space):
    """
    Get the width of the x window.

    Returns the total width of the spatial window in the x-direction:
    width = dx * (Nx - 1)

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        Width of x window (m)
    """
    if JIT_AVAILABLE:
        try:
            return _GetXWidth_core(space.dx, space.Nx)
        except (TypeError, ValueError, RuntimeError):
            return space.dx * (space.Nx - 1)
    else:
        return space.dx * (space.Nx - 1)


@jit
def _GetYWidth_core(dy, Ny):
    """
    JIT-compiled core for GetYWidth.

    Returns dy * (Ny - 1).
    """
    return dy * (Ny - 1)


def GetYWidth(space):
    """
    Get the width of the y window.

    Returns the total width of the spatial window in the y-direction:
    width = dy * (Ny - 1)

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        Width of y window (m)
    """
    if JIT_AVAILABLE:
        try:
            return _GetYWidth_core(space.dy, space.Ny)
        except (TypeError, ValueError, RuntimeError):
            return space.dy * (space.Ny - 1)
    else:
        return space.dy * (space.Ny - 1)


@jit
def _GetZWidth_core(dz, Nz):
    """
    JIT-compiled core for GetZWidth.

    Returns dz * (Nz - 1).
    """
    return dz * (Nz - 1)


def GetZWidth(space):
    """
    Get the width of the z window.

    Returns the total width of the spatial window in the z-direction:
    width = dz * (Nz - 1)

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        Width of z window (m)
    """
    if JIT_AVAILABLE:
        try:
            return _GetZWidth_core(space.dz, space.Nz)
        except (TypeError, ValueError, RuntimeError):
            return space.dz * (space.Nz - 1)
    else:
        return space.dz * (space.Nz - 1)


# Array generation functions
# These use numpy operations that may not be JIT-compatible

def GetSpaceArray(N, L):
    """
    Generate a real-space position array.

    Creates an array of N positions evenly spaced across a window of width L,
    centered at zero. The positions range from -L/2 to L/2.

    Parameters
    ----------
    N : int
        Number of points
    L : float
        Total width of the window (m)

    Returns
    -------
    ndarray
        Array of position values (m), 1D array of length N

    Notes
    -----
    The array is centered at zero, so positions range from -L/2 to L/2.
    For N=1, returns array with single value 0.0.

    This function is not defined in typespace.f90 but is used by GetXArray,
    GetYArray, and GetZArray. It is likely defined in the helpers module
    in Fortran. The implementation matches usage patterns in other Fortran files
    (e.g., GetSpaceArray(Ny, (Ny-1) * dyy) in pstd.f90).
    """
    if N == 1:
        return np.array([0.0])
    else:
        # Create array from -L/2 to L/2
        x = np.linspace(-L / 2.0, L / 2.0, N)
        return x


def GetKArray(Nk, L):
    """
    Generate k-space array for Fourier transforms.

    Creates an array of Nk k-values (momentum values) for use with FFT.
    The array is centered at zero and spaced by dk = twopi/L.

    Parameters
    ----------
    Nk : int
        Number of k-space points
    L : float
        Length of the spatial domain (m)

    Returns
    -------
    ndarray
        Array of k values (1/m), 1D array of length Nk

    Notes
    -----
    The k-values are centered at zero, suitable for FFT operations.
    For Nk=1, returns array with single value 0.0.

    This function is not defined in typespace.f90 but is used by GetKxArray,
    GetKyArray, and GetKzArray. It is likely defined in the helpers module
    in Fortran. The implementation matches usage patterns in other Fortran files.
    """
    if Nk == 1:
        return np.array([0.0])
    else:
        dk = twopi / L if L > 0 else 1.0
        k = np.arange(Nk, dtype=float) * dk
        k = k - k[Nk // 2]  # Center at zero
        return k


def GetXArray(space):
    """
    Get an array of x positions.

    Returns an array of x-coordinate positions for the spatial grid.
    If Nx == 1, returns array with single value 0.0.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    ndarray
        Array of x positions (m), 1D array of length space.Nx
    """
    if space.Nx == 1:
        return np.array([0.0])
    else:
        return GetSpaceArray(space.Nx, GetXWidth(space))


def GetYArray(space):
    """
    Get an array of y positions.

    Returns an array of y-coordinate positions for the spatial grid.
    If Ny == 1, returns array with single value 0.0.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    ndarray
        Array of y positions (m), 1D array of length space.Ny
    """
    if space.Ny == 1:
        return np.array([0.0])
    else:
        return GetSpaceArray(space.Ny, GetYWidth(space))


def GetZArray(space):
    """
    Get an array of z positions.

    Returns an array of z-coordinate positions for the spatial grid.
    If Nz == 1, returns array with single value 0.0.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    ndarray
        Array of z positions (m), 1D array of length space.Nz
    """
    if space.Nz == 1:
        return np.array([0.0])
    else:
        return GetSpaceArray(space.Nz, GetZWidth(space))


def GetKxArray(space):
    """
    Get an array of kx values (conjugate coordinate system).

    Returns an array of kx (momentum) values for the conjugate coordinate
    system, suitable for FFT operations. If Nx == 1, returns array with
    single value 0.0.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    ndarray
        Array of kx values (1/m), 1D array of length space.Nx
    """
    if space.Nx == 1:
        return np.array([0.0])
    else:
        return GetKArray(GetNx(space), GetXWidth(space))


def GetKyArray(space):
    """
    Get an array of ky values (conjugate coordinate system).

    Returns an array of ky (momentum) values for the conjugate coordinate
    system, suitable for FFT operations. If Ny == 1, returns array with
    single value 0.0.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    ndarray
        Array of ky values (1/m), 1D array of length space.Ny
    """
    if space.Ny == 1:
        return np.array([0.0])
    else:
        return GetKArray(GetNy(space), GetYWidth(space))


def GetKzArray(space):
    """
    Get an array of kz values (conjugate coordinate system).

    Returns an array of kz (momentum) values for the conjugate coordinate
    system, suitable for FFT operations. If Nz == 1, returns array with
    single value 0.0.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    ndarray
        Array of kz values (1/m), 1D array of length space.Nz
    """
    if space.Nz == 1:
        return np.array([0.0])
    else:
        return GetKArray(GetNz(space), GetZWidth(space))


# Differential functions for conjugate coordinate system

def GetDQx(space):
    """
    Get the differential for the conjugate coordinate system in x.

    Returns the k-space step size in the x-direction:
    dqx = twopi / GetXWidth(space)

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        k-space step size in x direction (1/m)

    Notes
    -----
    Matches Fortran implementation which uses twopi constant.
    Returns 0.0 if width is zero (Fortran would divide by zero).
    """
    dq = 0.0
    width = GetXWidth(space)
    if width > 0:
        dq = twopi / width
    # Note: Fortran doesn't check for zero width, but Python adds safety check
    return dq


def GetDQy(space):
    """
    Get the differential for the conjugate coordinate system in y.

    Returns the k-space step size in the y-direction:
    dqy = twopi / GetYWidth(space)

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        k-space step size in y direction (1/m)

    Notes
    -----
    Matches Fortran implementation which uses twopi constant.
    Returns 0.0 if width is zero (Fortran would divide by zero).
    """
    dq = 0.0
    width = GetYWidth(space)
    if width > 0:
        dq = twopi / width
    # Note: Fortran doesn't check for zero width, but Python adds safety check
    return dq


def GetDQz(space):
    """
    Get the differential for the conjugate coordinate system in z.

    Returns the k-space step size in the z-direction:
    dqz = twopi / GetZWidth(space)

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        k-space step size in z direction (1/m)

    Notes
    -----
    Matches Fortran implementation which uses twopi constant.
    Returns 0.0 if width is zero (Fortran would divide by zero).
    """
    dq = 0.0
    width = GetZWidth(space)
    if width > 0:
        dq = twopi / width
    # Note: Fortran doesn't check for zero width, but Python adds safety check
    return dq


# Volume element functions

def GetDVol(space):
    """
    Get the volume element.

    Returns the volume element (dx * dy * dz) for the spatial grid.

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        Volume element (m^3)
    """
    dVol = 0.0
    dVol = GetDx(space) * GetDy(space) * GetDz(space)
    return dVol


def GetDQVol(space):
    """
    Get the volume element for the conjugate coordinate system.

    Returns the k-space volume element (dqx * dqy * dqz).

    Parameters
    ----------
    space : ss
        Space structure

    Returns
    -------
    float
        k-space volume element ((1/m)^3)
    """
    dQVol = 0.0
    dQVol = GetDQx(space) * GetDQy(space) * GetDQz(space)
    return dQVol


# Field I/O functions
# These handle binary and text file I/O

def writefield(fnout, e, space, binmode, single, fnspace=None):
    """
    Write the space structure and field to a file.

    Writes the space structure (ss) and the complex field array to a single
    file. Can write in binary or text format, and can include space parameters
    in the same file or a separate file.

    Parameters
    ----------
    fnout : str
        Filename to write to, or "stdout" or "-" for stdout
    e : ndarray
        Complex field array, 3D array of shape (Nx, Ny, Nz)
    space : ss
        Space structure
    binmode : bool
        If True, write in binary format; if False, write in text format
    single : bool
        If True, include space parameters in the same file
    fnspace : str, optional
        If single is False, write space parameters to this separate file

    Returns
    -------
    None

    Notes
    -----
    For binary mode, uses unformatted_write_space and unformatted_write_e.
    For text mode, uses WriteSpaceParams_sub and formatted field output.
    """
    if fnout == "stdout" or fnout == "-":
        # Use stdout
        file_handle = sys.stdout
        close_file = False
    else:
        if binmode:
            file_handle = open(fnout, 'wb')
        else:
            file_handle = open(fnout, 'w', encoding='utf-8')
        close_file = True

    try:
        if single:
            if binmode:
                unformatted_write_space(file_handle, space)
            else:
                WriteSpaceParams_sub(file_handle, space)
        else:
            if fnspace is not None:
                writespaceparams(fnspace, space)

        writefield_to_unit(file_handle, e, binmode)
    finally:
        if close_file:
            file_handle.close()


def readspace_only(fnin, space, binmode, single, fnspace=None):
    """
    Read the space structure from a field file.

    Reads the space structure (ss) from a field file. Can read from binary
    or text format, and can read space parameters from the same file or
    a separate file.

    Parameters
    ----------
    fnin : str
        Filename to read from, or "stdin" or "-" for stdin
    space : ss
        Space structure to populate (modified in-place)
    binmode : bool
        If True, read from binary format; if False, read from text format
    single : bool
        If True, read space parameters from the same file
    fnspace : str, optional
        If single is False, read space parameters from this separate file

    Returns
    -------
    None
        space structure is modified in-place

    Notes
    -----
    For binary mode, uses unformatted_read_space.
    For text mode, uses readspaceparams_sub.
    """
    if fnin == "stdin" or fnin == "-":
        # Use stdin
        file_handle = sys.stdin
        close_file = False
    else:
        if binmode:
            file_handle = open(fnin, 'rb')
        else:
            file_handle = open(fnin, 'r', encoding='utf-8')
        close_file = True

    try:
        if single:
            if binmode:
                unformatted_read_space(file_handle, space)
            else:
                readspaceparams_sub(file_handle, space)
        else:
            if fnspace is not None:
                ReadSpaceParams(fnspace, space)
    finally:
        if close_file:
            file_handle.close()


# Helper functions for binary I/O

def writefield_to_unit(file_handle, e, binmode):
    """
    Write field array to a file handle.

    Writes the complex field array to an open file handle in either
    binary or text format.

    Parameters
    ----------
    file_handle : file
        Open file handle to write to
    e : ndarray
        Complex field array, 3D array
    binmode : bool
        If True, write in binary format; if False, write in text format

    Returns
    -------
    None

    Notes
    -----
    For binary mode, uses numpy's save or pickle.
    For text mode, writes real and imaginary parts separately.
    """
    if binmode:
        # Binary mode: use numpy save
        np.save(file_handle, e)
    else:
        # Text mode: write real and imaginary parts
        for k in range(e.shape[2]):
            for j in range(e.shape[1]):
                for i in range(e.shape[0]):
                    file_handle.write(f"{np.real(e[i, j, k]):25.15E} {np.imag(e[i, j, k]):25.15E}\n")


def unformatted_write_space(file_handle, space):
    """
    Write space structure in binary format.

    Writes the space structure to a file handle in binary (unformatted) format.

    Parameters
    ----------
    file_handle : file
        Open binary file handle to write to
    space : ss
        Space structure to write

    Returns
    -------
    None

    Notes
    -----
    Uses numpy's save to write the dataclass fields in binary format.
    """
    # Write space structure fields as a dictionary
    space_dict = {
        'Dims': space.Dims,
        'Nx': space.Nx,
        'Ny': space.Ny,
        'Nz': space.Nz,
        'dx': space.dx,
        'dy': space.dy,
        'dz': space.dz,
        'epsr': space.epsr
    }
    np.save(file_handle, space_dict)


def unformatted_read_space(file_handle, space):
    """
    Read space structure from binary format.

    Reads the space structure from a file handle in binary (unformatted) format.

    Parameters
    ----------
    file_handle : file
        Open binary file handle to read from
    space : ss
        Space structure to populate (modified in-place)

    Returns
    -------
    None
        space structure is modified in-place

    Notes
    -----
    Uses numpy's load to read the dataclass fields from binary format.
    """
    space_dict = np.load(file_handle, allow_pickle=True).item()
    space.Dims = int(space_dict['Dims'])
    space.Nx = int(space_dict['Nx'])
    space.Ny = int(space_dict['Ny'])
    space.Nz = int(space_dict['Nz'])
    space.dx = float(space_dict['dx'])
    space.dy = float(space_dict['dy'])
    space.dz = float(space_dict['dz'])
    space.epsr = float(space_dict['epsr'])


def initialize_field(e):
    """
    Initialize field array to zero.

    Sets all elements of the complex field array to zero.

    Parameters
    ----------
    e : ndarray
        Complex field array, 3D array (modified in-place)

    Returns
    -------
    None
        e array is modified in-place

    Notes
    -----
    In Fortran, this uses OpenMP parallelization when available.
    In Python, we use NumPy's efficient zero assignment.
    """
    e[:] = 0.0 + 0.0j


def unformatted_write_e(file_handle, e):
    """
    Write field array in binary format.

    Writes the complex field array to a file handle in binary (unformatted) format.

    Parameters
    ----------
    file_handle : file
        Open binary file handle to write to
    e : ndarray
        Complex field array, 3D array

    Returns
    -------
    None

    Notes
    -----
    Uses numpy's save to write the array in binary format.
    """
    np.save(file_handle, e)


def unformatted_read_e(file_handle, e):
    """
    Read field array from binary format.

    Reads the complex field array from a file handle in binary (unformatted) format.

    Parameters
    ----------
    file_handle : file
        Open binary file handle to read from
    e : ndarray
        Complex field array to populate (modified in-place), 3D array

    Returns
    -------
    None
        e array is modified in-place

    Notes
    -----
    Uses numpy's load to read the array from binary format.
    The array must already be allocated with the correct shape.
    """
    e[:] = np.load(file_handle)


def readfield_from_unit(file_handle, e, binmode):
    """
    Read field array from a file handle.

    Reads the complex field array from an open file handle in either
    binary or text format.

    Parameters
    ----------
    file_handle : file
        Open file handle to read from
    e : ndarray
        Complex field array to populate (modified in-place), 3D array
    binmode : bool
        If True, read from binary format; if False, read from text format

    Returns
    -------
    None
        e array is modified in-place

    Notes
    -----
    For binary mode, uses unformatted_read_e.
    For text mode, reads real and imaginary parts line by line.
    """
    if binmode:
        unformatted_read_e(file_handle, e)
    else:
        # Text mode: read real and imaginary parts
        for k in range(e.shape[2]):
            for j in range(e.shape[1]):
                for i in range(e.shape[0]):
                    line = file_handle.readline()
                    if not line:
                        raise ValueError("Unexpected end of file while reading field")
                    parts = line.split()
                    if len(parts) < 2:
                        raise ValueError(f"Invalid field data format: {line}")
                    re = float(parts[0])
                    im = float(parts[1])
                    e[i, j, k] = re + 1j * im


def readfield(fnin, e, space, binmode, single, fnspace=None):
    """
    Read the space structure and field from a file.

    Reads the space structure (ss) and the complex field array from a file.
    Can read from binary or text format, and can read space parameters from
    the same file or a separate file. The field array is automatically
    allocated or resized to match the space dimensions.

    Parameters
    ----------
    fnin : str
        Filename to read from, or "stdin" or "-" for stdin
    e : ndarray or None
        Complex field array. If None or wrong shape, will be allocated/resized.
        Modified in-place if provided, or a new array is returned.
    space : ss
        Space structure to populate (modified in-place)
    binmode : bool
        If True, read from binary format; if False, read from text format
    single : bool
        If True, read space parameters from the same file
    fnspace : str, optional
        If single is False, read space parameters from this separate file

    Returns
    -------
    ndarray
        Complex field array. If e was None, returns a new array.
        If e was provided, returns the same array (modified in-place).

    Notes
    -----
    The field array is automatically allocated or resized to match
    GetNx(space), GetNy(space), GetNz(space).
    After reading space parameters, the field is initialized to zero
    before reading the actual field data.
    """
    if fnin == "stdin" or fnin == "-":
        # Use stdin
        file_handle = sys.stdin
        close_file = False
    else:
        if binmode:
            file_handle = open(fnin, 'rb')
        else:
            file_handle = open(fnin, 'r', encoding='utf-8')
        close_file = True

    try:
        # Read space parameters
        if single:
            if binmode:
                unformatted_read_space(file_handle, space)
            else:
                readspaceparams_sub(file_handle, space)
        else:
            if fnspace is not None:
                ReadSpaceParams(fnspace, space)

        # Allocate or resize field array
        nx = GetNx(space)
        ny = GetNy(space)
        nz = GetNz(space)

        # Determine if we need a new array
        need_new_array = (e is None) or (e.shape != (nx, ny, nz))

        if need_new_array:
            # Allocate new array
            e = np.zeros((nx, ny, nz), dtype=complex)
        else:
            # Initialize existing array to zero
            initialize_field(e)

        # Read field data
        readfield_from_unit(file_handle, e, binmode)

    finally:
        if close_file:
            file_handle.close()

    return e

