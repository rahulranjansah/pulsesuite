"""
Quantum wire polarization and current density calculations for 3D Maxwell propagation.

This module handles the interface between 1D quantum wire calculations and 3D
Maxwell field propagation, including polarization placement, current density
calculation, and longitudinal field computation from charge density.

Author: Rahul R. Sah
"""

import numpy as np
import pyfftw

pyfftw.interfaces.cache.enable()

from numba import jit, prange
from scipy.constants import epsilon_0 as eps0_SI

from .SBEs import QWCalculator
from .typespace import (
    GetDx,
    GetEpsr,
    GetKxArray,
    GetKyArray,
    GetKzArray,
    GetNx,
    GetNy,
    GetNz,
    GetXArray,
    GetYArray,
    GetZArray,
)
from .usefulsubs import EAtXYZ

# Physical constants
eps0 = eps0_SI
ii = 1j  # Imaginary unit

# Module-level state variables (matching Fortran module variables)
_Nw = 4  # Number of quantum wires
_y0 = 0.0e-6  # Wire placement along x-axis [m]
_x0 = 0.0e-9  # X-Center of QW array (m)
_z0 = 0.0e-6  # Z-Center of QW array (m)
_dxqw = 0.0e-9  # Wire spacing along x [m]
_ay = 5.0e-9  # SHO oscillator length in y [m]
_az = 5.0e-9  # SHO oscillator length in z [m]

_QW = True  # Turn on the quantum wire(s)
_host = False  # Turn on the host dispersion
_propagate = True  # Is the propagation on?

# Maxwell grid accumulator from Quantum Wire, intermediate vals
_PxxOld = None
_PyyOld = None
_PzzOld = None

# Free charge density storage for continuity eq.
_RhoOld = None


def read_qw_parameters(filename):
    """
    Read quantum wire parameters from file.

    Reads QW configuration parameters from a file and updates module-level
    variables. If the file cannot be opened, prints an error message.

    Parameters
    ----------
    filename : str
        Path to the parameter file

    Returns
    -------
    int
        Error status: 0 for success, non-zero for error
    """
    global _QW, _Nw, _x0, _dxqw, _ay, _az

    def parse_line(line):
        """Parse a line, stripping comments and whitespace."""
        # Remove comments (# style)
        if '#' in line:
            line = line.split('#')[0]
        # Strip whitespace
        return line.strip()

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # Read and parse each line, skipping empty lines
            lines = []
            for line in f:
                parsed = parse_line(line)
                if parsed:  # Skip empty lines
                    lines.append(parsed)

            if len(lines) < 6:
                raise ValueError(f"Expected 6 parameters, found {len(lines)}")

            _QW = bool(int(lines[0]))
            _Nw = int(lines[1])
            _x0 = float(lines[2])
            _dxqw = float(lines[3])
            _ay = float(lines[4])
            _az = float(lines[5])
        return 0
    except IOError as e:
        print(f"Error opening quantum wire parameters file: {e}")
        return 1
    except (ValueError, IndexError) as e:
        print(f"Error reading quantum wire parameters: {e}")
        return 1


# (Porting to 3D)
# def QuantumWire(space, dt, n, Ex, Ey, Ez, Jx, Jy, Jz, Rho)
def QuantumWire(space, dt, n, Ex, Ey, Ez, Jx, Jy, Jz, Rho, ExlfromPRho, EylfromPRho, EzlfromPRho, ExlfromP, EylfromP, EzlfromP, ExlfromRho, EylfromRho, EzlfromRho, RhoBound):
    """
    Main quantum wire calculation routine.

    Calculates polarization and charge density for quantum wires and deposits
    them into 3D Maxwell grid. Handles multiple wires and computes longitudinal
    fields from polarization and charge density.

    Parameters
    ----------
    space : object
        Space configuration object (from typespace)
    dt : float
        Time step (s)
    n : int
        Current time step number
    Ex : ndarray
        X-component electric field (complex), 3D array
    Ey : ndarray
        Y-component electric field (complex), 3D array
    Ez : ndarray
        Z-component electric field (complex), 3D array
    Jx : ndarray
        X-component current density (complex, modified in-place), 3D array
    Jy : ndarray
        Y-component current density (complex, modified in-place), 3D array
    Jz : ndarray
        Z-component current density (complex, modified in-place), 3D array
    Rho : ndarray
        Charge density (complex, modified in-place), 3D array
    ExlfromPRho : ndarray
        X-component longitudinal field from P+Rho (output), 3D array
    EylfromPRho : ndarray
        Y-component longitudinal field from P+Rho (output), 3D array
    EzlfromPRho : ndarray
        Z-component longitudinal field from P+Rho (output), 3D array
    ExlfromP : ndarray
        X-component longitudinal field from P (output), 3D array
    EylfromP : ndarray
        Y-component longitudinal field from P (output), 3D array
    EzlfromP : ndarray
        Z-component longitudinal field from P (output), 3D array
    ExlfromRho : ndarray
        X-component longitudinal field from Rho (output), 3D array
    EylfromRho : ndarray
        Y-component longitudinal field from Rho (output), 3D array
    EzlfromRho : ndarray
        Z-component longitudinal field from Rho (output), 3D array
    RhoBound : ndarray
        Bound charge density (output), 3D array

    Returns
    -------
    None
        All output arrays are modified in-place.
    """
    global _PxxOld, _PyyOld, _PzzOld, _RhoOld, _QW, _Nw, _y0, _z0, _ay, _az, _propagate

    # Local Work arrays (not dummy arguments)
    # 1D arrays for the SBE code
    Nx = GetNx(space)
    Ny = GetNy(space)
    Nz = GetNz(space)

    Exx = np.zeros(Nx, dtype=np.complex128)  # No. of points E field points on each axis
    Eyy = np.zeros(Nx, dtype=np.complex128)
    Ezz = np.zeros(Nx, dtype=np.complex128)
    Jxx = np.zeros(Nx, dtype=np.complex128)  # No. of points Jfield points on each axis
    Jyy = np.zeros(Nx, dtype=np.complex128)
    Jzz = np.zeros(Nx, dtype=np.complex128)
    # Jfx = np.zeros(Nx, dtype=np.complex128)  # Temp. array for free current
    # Vrr(GetNx(space), GetNy(space), GetNz(space))
    Vrr = np.zeros(Nx, dtype=np.complex128)

    # Arrays to be turned 1D after being called through two wires
    Pxx = np.zeros((Nx, _Nw), dtype=np.complex128)
    Pyy = np.zeros((Nx, _Nw), dtype=np.complex128)
    Pzz = np.zeros((Nx, _Nw), dtype=np.complex128)
    RhoEH = np.zeros((Nx, _Nw), dtype=np.complex128)

    # Arrays for QWPlacement aiding E longitudinal computations with Rho
    Px = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Py = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Pz = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    # Real-space and Q-space arrays
    rx = GetXArray(space)  # Maxwell domain
    ry = GetYArray(space)
    rz = GetZArray(space)
    qx = GetKxArray(space)  # SBE domain
    qy = GetKyArray(space)
    qz = GetKzArray(space)
    # x(GetNx(space)) , y(GetNy(space)) , z(GetNz(space))  # X, Y, and Z Space Arrays (m)

    # RhoDot = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    # Rho00 = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    gx = np.zeros(Nx, dtype=np.complex128)
    gy = np.zeros(Ny, dtype=np.complex128)
    gz = np.zeros(Nz, dtype=np.complex128)
    xqw = np.zeros(_Nw, dtype=np.float64)

    # NQ0, k, w
    DoQWP = [True]   # modified as a list in SBEs.py (line 278)
    DoQWDl = [True]  # modified as a list in SBEs.py (line 279)
    DoQWCurr = True

    ps = 1e-12

    xqw[:] = 0.0

    # Read parameters at the start of the subroutine
    iostat = read_qw_parameters('params/qwarray.params')
    if iostat != 0:
        print("Error reading quantum wire parameters, using defaults")
        _QW = False  # Disable QW if there was an error
        return

    # consistent GetNx(space) -> propagation and wire is on x-axis
    if _PxxOld is not None:
        _PxxOld = None
        _PyyOld = None
        _PzzOld = None
        _RhoOld = None

    if _PxxOld is None:
        _PxxOld = np.zeros((Nx, _Nw), dtype=np.complex128)
        _PyyOld = np.zeros((Nx, _Nw), dtype=np.complex128)
        _PzzOld = np.zeros((Nx, _Nw), dtype=np.complex128)
        _RhoOld = np.zeros((Nx, _Nw), dtype=np.complex128)
        _PxxOld[:] = 0.0
        _PyyOld[:] = 0.0
        _PzzOld[:] = 0.0
        _RhoOld[:] = 0.0

    # if (_PxxOld is None or
    #     _PxxOld.shape[0] != Nx or
    #     _PxxOld.shape[1] != _Nw):
    #
    #     if _PxxOld is not None:
    #         _PxxOld = None
    #         _PyyOld = None
    #         _PzzOld = None
    #         _RhoOld = None
    #     _PxxOld = np.zeros((Nx, _Nw), dtype=np.complex128)
    #     _PyyOld = np.zeros((Nx, _Nw), dtype=np.complex128)
    #     _PzzOld = np.zeros((Nx, _Nw), dtype=np.complex128)
    #     _RhoOld = np.zeros((Nx, _Nw), dtype=np.complex128)
    #     _PxxOld[:] = 0.0
    #     _PyyOld[:] = 0.0
    #     _PzzOld[:] = 0.0
    #     _RhoOld[:] = 0.0

    Vrr[:] = 0.0

    Pxx[:] = 0.0
    Pyy[:] = 0.0
    Pzz[:] = 0.0
    Px[:] = 0.0
    Py[:] = 0.0
    Pz[:] = 0.0
    RhoEH[:] = 0.0

    # Exx = EAtXYZ( Ex, rx, ry, rz, 0d0, y0, z0 )
    # Eyy = EAtXYZ( Ey, rx, ry, rz, 0d0, y0, z0 )
    # Ezz = EAtXYZ( Ez, rx, ry, rz, 0d0, y0, z0 )

    # Eyy = sum(sum(Ey(:, Ny/2:Ny/2+1, Nz/2:Nz/2+1),3),2) / 4d0
    # the full 3D field at the wire center not along the whole wire (xqw(w), y0, z0)
    for w in range(_Nw):
        Exx[:] = EAtXYZ(Ex, rx, ry, rz, xqw[w], _y0, _z0)
        Eyy[:] = EAtXYZ(Ey, rx, ry, rz, xqw[w], _y0, _z0)
        Ezz[:] = EAtXYZ(Ez, rx, ry, rz, xqw[w], _y0, _z0)

        # work needs to be done here (for 3D), done down here
        # do k=2, GetNx(space)    # or maybe size(rx)
        #    Vrr(k) = Vrr(k-1) - (Exx(k)+Exx(k-1)) / 2d0 * GetDx(space)
        # end do

        # computing Vrr in discrete uniform grid, trapezoidal rule
        # !$omp parallel do private(k, j, i)
        # do k = 2, nx
        #    do j = 1, ny
        #      do i = 1, nz
        #        Vrr(k,j,i) = Vrr(k-1,j,i) - ((Ex(k,j,i) + Ex(k-1,j,i)) / 2d0) * GetDx(space)
        #      end do
        #    end do
        #  end do
        # !$omp end parallel do

        # spatial solve per wire, and temporal solve for internal quantum states (solving in 1D)
        QWCalculator(Exx, Eyy, Ezz, Vrr, rx, qx, dt, w + 1, Pxx[:, w], Pyy[:, w], Pzz[:, w], RhoEH[:, w], DoQWP, DoQWDl)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # DEBUG PRINT 1: Check values after the QWCalculator loop.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print("Debug QuantumWire: After QWCalculator")
    # print("Max abs(Pxx):   ", np.max(np.abs(Pxx)))
    # print("Max abs(Pyy):   ", np.max(np.abs(Pyy)))
    # print("Max abs(Pzz):   ", np.max(np.abs(Pzz)))
    # print("Max abs(RhoEH): ", np.max(np.abs(RhoEH)))

    # PxxOld handling time evolutions
    for i in range(Nx):
        Jxx[i] = np.sum(Pxx[i, :] - _PxxOld[i, :]) / dt
    for i in range(Nx):
        Jyy[i] = np.sum(Pyy[i, :] - _PyyOld[i, :]) / dt
    for i in range(Nx):
        Jzz[i] = np.sum(Pzz[i, :] - _PzzOld[i, :]) / dt

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # DEBUG PRINT 2: Check J values after calculating the time derivative of P.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print("Debug QuantumWire: After forall statements")
    # print("Max abs(Jxx) before free current: ", np.max(np.abs(Jxx)))
    # print("Max abs(Jyy): ", np.max(np.abs(Jyy)))
    # print("Max abs(Jzz): ", np.max(np.abs(Jzz)))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # To account for two wires, moved down and fixed all errors? I guess!
    Jxx[:] = Jxx[:] / float(_Nw) + CalcJfx(np.sum(RhoEH, axis=1), np.sum(_RhoOld, axis=1), dt, GetDx(space)) / float(_Nw)
    Jyy[:] = Jyy[:] / float(_Nw)
    Jzz[:] = Jzz[:] / float(_Nw)

    # gate profiles, where is real gx?  In QWOptics.f90 -> supergaussian in qwwindow
    gx[:] = 1.0
    # gy(j)   = (pi * ay**2) ** (-1d0 / 4d0) * exp( -((y(j) - y0)) ** 2) / (2d0 * ay ** 2)    # SHO in y
    # gz(k)   = (pi * az**2) ** (-1d0 / 4d0) * exp( -((z(k) - z0)) ** 2) / (2d0 * az ** 2)    # another SHO in z

    for j in range(Ny):
        # gy(j)   = (pi * ay**2) ** (-1d0 / 4d0) * exp( -((ry(j) - y0) ** 2) / (2d0 * ay ** 2) )
        gy[j] = np.exp(-((ry[j] - _y0) ** 2) / (2.0 * _ay ** 2))

    for k in range(Nz):
        # gz(k)   = (pi * az**2) ** (-1d0 / 4d0) * exp( -((rz(k) - z0) ** 2) / (2d0 * az ** 2) )
        gz[k] = np.exp(-((rz[k] - _z0) ** 2) / (2.0 * _az ** 2))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # DEBUG PRINT 3: Check the gate profiles.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print("Debug QuantumWire: Gate Profiles")
    # print("Max abs(gy): ", np.max(np.abs(gy)))
    # print("Max abs(gz): ", np.max(np.abs(gz)))
    # print("ay = ", _ay, "az = ", _az)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if DoQWCurr and _propagate:
        QWPlacement(Jxx, gx, gy, gz, Jx)
        QWPlacement(Jyy, gx, gy, gz, Jy)
        QWPlacement(Jzz, gx, gy, gz, Jz)

    # open(file="fields/gx.x.dat",unit=873)
    #     do i=1, GetNx(space)
    #         write(873,*) rx(i), real(gx(i))
    #     end do
    # close(873)
    # open(file="fields/gy.y.dat",unit=873)
    #     do i=1, GetNy(space)
    #         write(873,*) ry(i), real(gy(i))
    #     end do
    # close(873)
    # open(file="fields/gz.z.dat",unit=873)
    #     do i=1, GetNz(space)
    #         write(873,*) rz(i), real(gz(i))
    #     end do
    # close(873)
    # stop
    if DoQWDl[0] and _propagate:
        QWPlacement((np.sum(RhoEH, axis=1) / float(_Nw)), gx, gy, gz, Rho)

    # projections of 1D QWcalculator in 3D
    if DoQWP[0] and _propagate:
        QWPlacement((np.sum(Pxx, axis=1) / float(_Nw)), gx, gy, gz, Px)
        QWPlacement((np.sum(Pyy, axis=1) / float(_Nw)), gx, gy, gz, Py)
        QWPlacement((np.sum(Pzz, axis=1) / float(_Nw)), gx, gy, gz, Pz)

        # call ElongfromRho(space, Rho, Px, Py, Pz, ExlfromRho, EylfromRho, EzlfromRho)
        ElongfromRho(space, Rho, Px, Py, Pz, ExlfromPRho, EylfromPRho, EzlfromPRho, ExlfromP, EylfromP, EzlfromP, ExlfromRho, EylfromRho, EzlfromRho, RhoBound)

    _PxxOld[:] = Pxx
    _PyyOld[:] = Pyy
    _PzzOld[:] = Pzz
    _RhoOld[:] = RhoEH

    # print(_QW)
    if n % 100 == 0:
        print(n, "RhoEH-max = ", np.max(np.abs(RhoEH[:, 0])), np.max(np.abs(RhoEH[:, 1])))
        print(n, "Jxx-max = ", np.max(np.abs(Jxx[:])))
        print(n, "Jyy-max = ", np.max(np.abs(Jyy[:])))
        print(n, "Jzz-max = ", np.max(np.abs(Jzz[:])))
        print("Difference check E*lfromPRho E*lfromP:", np.max(np.abs(ExlfromPRho - ExlfromP)), np.max(np.abs(EylfromPRho - EylfromP)), np.max(np.abs(EzlfromPRho - EzlfromP)))

        for w in range(_Nw):
            print(n, "w = ", w)
            print(n, "Pxx-max-min-w = ", np.max(np.real(Pxx[:, w])), np.min(np.real(Pxx[:, w])))
            print(n, "Pyy-max-min-w = ", np.max(np.real(Pyy[:, w])), np.min(np.real(Pyy[:, w])))
            print(n, "Pzz-max-min-w = ", np.max(np.real(Pzz[:, w])), np.min(np.real(Pzz[:, w])))


def CalcJfx(RhoNew, RhoPrev, dt, dx):
    """
    Compute free current density from charge density time derivative.

    Calculates J_free = - ∫(∂ρ/∂t) dx using local approximation (Riemann sums).

    Parameters
    ----------
    RhoNew : ndarray
        Current charge density (complex), 1D array
    RhoPrev : ndarray
        Previous charge density (complex), 1D array
    dt : float
        Time step (s)
    dx : float
        Spatial step (m)

    Returns
    -------
    ndarray
        Free current density (complex), 1D array
    """
    # finite differencing
    drhodt = (RhoNew - RhoPrev) / dt

    CalcJfx_result = np.zeros(len(RhoNew), dtype=np.complex128)

    # Using Riemann sums approx. integrals
    for i in range(1, len(RhoNew)):
        CalcJfx_result[i] = CalcJfx_result[i - 1] - drhodt[i] * dx

    return CalcJfx_result


@jit(nopython=True, parallel=True)
def _ElongfromRho_loop_jit(Rho_k, Px, Py, Pz, qx, qy, qz, qx2, qy2, qz2, small, inv_epsr,
                            ExlfromPRho, EylfromPRho, EzlfromPRho, ExlfromP, EylfromP, EzlfromP,
                            ExlfromRho, EylfromRho, EzlfromRho, RhoB_k, Nx, Ny, Nz):
    """
    JIT-compiled parallel version of ElongfromRho k-space loop.

    Parameters
    ----------
    Rho_k : ndarray
        Charge density in k-space (complex), 3D array
    Px, Py, Pz : ndarray
        Polarization components in k-space (complex), 3D arrays
    qx, qy, qz : ndarray
        k-space vectors, 1D arrays
    qx2, qy2, qz2 : ndarray
        Squared k-space vectors, 1D arrays
    small : float
        Small constant for numerical stability
    inv_epsr : float
        Inverse of epsilon_r
    ExlfromPRho, EylfromPRho, EzlfromPRho : ndarray
        Longitudinal fields from P+Rho (output), 3D arrays
    ExlfromP, EylfromP, EzlfromP : ndarray
        Longitudinal fields from P (output), 3D arrays
    ExlfromRho, EylfromRho, EzlfromRho : ndarray
        Longitudinal fields from Rho (output), 3D arrays
    RhoB_k : ndarray
        Bound charge density in k-space (output), 3D array
    Nx, Ny, Nz : int
        Grid dimensions

    Returns
    -------
    None
        All output arrays are modified in-place.
    """
    ii_val = 1j  # Imaginary unit

    # Parallel over k & j, i innermost (collapse outer loops)
    for k in prange(Nz): # pylint: disable=not-an-iterable
        for j in range(Ny):
            for i in range(Nx):
                # combined denominator
                qsquare = qx2[i] + qy2[j] + qz2[k] + small

                # dot-product of k·P
                qdotP = qx[i] * Px[i, j, k] + qy[j] * Py[i, j, k] + qz[k] * Pz[i, j, k]

                RhoB_k[i, j, k] = -ii_val * qdotP  # ρ_bound(k) = -i k·P(k)

                # ρ-term and P-term
                rho_term = -ii_val * Rho_k[i, j, k] * inv_epsr / qsquare

                # deposit into each component
                ExlfromRho[i, j, k] = (rho_term) * qx[i]
                EylfromRho[i, j, k] = (rho_term) * qy[j]
                EzlfromRho[i, j, k] = (rho_term) * qz[k]

                P_term = -qdotP * inv_epsr / qsquare

                # deposit into each component
                ExlfromP[i, j, k] = (P_term) * qx[i]
                EylfromP[i, j, k] = (P_term) * qy[j]
                EzlfromP[i, j, k] = (P_term) * qz[k]

                # deposit into each component
                ExlfromPRho[i, j, k] = (rho_term + P_term) * qx[i]
                EylfromPRho[i, j, k] = (rho_term + P_term) * qy[j]
                EzlfromPRho[i, j, k] = (rho_term + P_term) * qz[k]


@jit(nopython=True, parallel=True)
def _QWPlacement_jit(Fwire, gx, gy, gz, Fgrid, Nx, Ny, Nz):
    """
    JIT-compiled parallel version of QWPlacement loop.

    Parameters
    ----------
    Fwire : ndarray
        Complex field along x-axis, 1D array
    gx : ndarray
        Gate profile in x direction, 1D array
    gy : ndarray
        Gate profile in y direction, 1D array
    gz : ndarray
        Gate profile in z direction, 1D array
    Fgrid : ndarray
        Complex 3D Maxwell grid array (modified in-place), 3D array
    Nx, Ny, Nz : int
        Grid dimensions

    Returns
    -------
    None
        Fgrid is modified in-place.
    """
    # Parallel loop over k, j, i (collapse outer loops for better parallelization)
    for k in prange(Nz): # pylint: disable=not-an-iterable
        for j in range(Ny):
            for i in range(Nx):
                Fgrid[i, j, k] = Fgrid[i, j, k] + gx[i] * gy[j] * gz[k] * Fwire[i]  # with SHO gate in y and z


def QWPlacement(Fwire, gx, gy, gz, Fgrid):
    """
    Place 1D quantum wire field into 3D Maxwell grid.

    Deposits a 1D field Fwire along the x-axis into a 3D grid Fgrid using
    gate profiles gx, gy, gz for spatial weighting.

    Parameters
    ----------
    Fwire : ndarray
        Complex field along x-axis (length Nx), 1D array
    gx : ndarray
        Gate profile in x direction, 1D array
    gy : ndarray
        Gate profile in y direction, 1D array
    gz : ndarray
        Gate profile in z direction, 1D array
    Fgrid : ndarray
        Complex 3D Maxwell grid array to deposit into (modified in-place), 3D array

    Returns
    -------
    None
        Fgrid is modified in-place.
    """
    Nx, Ny, Nz = Fgrid.shape

    Fgrid[:] = 0.0

    # print("QWPlacement Debug:")
    # print("Max Fwire = ", np.max(np.abs(Fwire)))
    # print("Max gx = ", np.max(np.abs(gx)))
    # print("Max gy = ", np.max(np.abs(gy)))
    # print("Max gz = ", np.max(np.abs(gz)))

    # Try JIT-compiled parallel version, fallback to regular loop if it fails
    try:
        _QWPlacement_jit(Fwire, gx, gy, gz, Fgrid, Nx, Ny, Nz)
    except (TypeError, ValueError, RuntimeError):
        # Fallback to regular Python loop
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    Fgrid[i, j, k] = Fgrid[i, j, k] + gx[i] * gy[j] * gz[k] * Fwire[i]  # with SHO gate in y and z


def ElongfromRho(space, Rho, Px, Py, Pz, ExlfromPRho, EylfromPRho, EzlfromPRho, ExlfromP, EylfromP, EzlfromP, ExlfromRho, EylfromRho, EzlfromRho, RhoBound):
    """
    Calculate longitudinal electric fields from polarization and charge density.

    Computes longitudinal (Coulomb) electric fields in k-space from polarization
    and free charge density, then transforms back to real space.

    Parameters
    ----------
    space : object
        Space configuration object (from typespace)
    Rho : ndarray
        Free charge density (complex), 3D array
    Px : ndarray
        X-component polarization (complex, modified in-place), 3D array
    Py : ndarray
        Y-component polarization (complex, modified in-place), 3D array
    Pz : ndarray
        Z-component polarization (complex, modified in-place), 3D array
    ExlfromPRho : ndarray
        X-component longitudinal field from P+Rho (output), 3D array
    EylfromPRho : ndarray
        Y-component longitudinal field from P+Rho (output), 3D array
    EzlfromPRho : ndarray
        Z-component longitudinal field from P+Rho (output), 3D array
    ExlfromP : ndarray
        X-component longitudinal field from P (output), 3D array
    EylfromP : ndarray
        Y-component longitudinal field from P (output), 3D array
    EzlfromP : ndarray
        Z-component longitudinal field from P (output), 3D array
    ExlfromRho : ndarray
        X-component longitudinal field from Rho (output), 3D array
    EylfromRho : ndarray
        Y-component longitudinal field from Rho (output), 3D array
    EzlfromRho : ndarray
        Z-component longitudinal field from Rho (output), 3D array
    RhoBound : ndarray
        Bound charge density (output), 3D array

    Returns
    -------
    None
        All output arrays are modified in-place.
    """
    Nx = GetNx(space)
    Ny = GetNy(space)
    Nz = GetNz(space)

    Rho_k = Rho.copy()

    # Fetch sizes and k-space vectors
    qx = GetKxArray(space)
    qy = GetKyArray(space)
    qz = GetKzArray(space)

    qx2 = qx**2
    qy2 = qy**2
    qz2 = qz**2

    small = (qx[1] - qx[0]) / 1e4
    inv_epsr = 1.0 / (eps0 * GetEpsr(space))

    # Zero outputs
    ExlfromPRho[:] = 0.0
    EylfromPRho[:] = 0.0
    EzlfromPRho[:] = 0.0

    ExlfromP[:] = 0.0
    EylfromP[:] = 0.0
    EzlfromP[:] = 0.0

    ExlfromRho[:] = 0.0
    EylfromRho[:] = 0.0
    EzlfromRho[:] = 0.0

    RhoB_k = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    # Forward FFTs (modify in-place to match Fortran behavior)
    Rho_k = pyfftw.interfaces.numpy_fft.fftn(Rho_k)
    Px[:] = pyfftw.interfaces.numpy_fft.fftn(Px)
    Py[:] = pyfftw.interfaces.numpy_fft.fftn(Py)
    Pz[:] = pyfftw.interfaces.numpy_fft.fftn(Pz)

    # Try JIT-compiled parallel version, fallback to regular loop if it fails
    try:
        _ElongfromRho_loop_jit(Rho_k, Px, Py, Pz, qx, qy, qz, qx2, qy2, qz2, small, inv_epsr,
                                ExlfromPRho, EylfromPRho, EzlfromPRho, ExlfromP, EylfromP, EzlfromP,
                                ExlfromRho, EylfromRho, EzlfromRho, RhoB_k, Nx, Ny, Nz)
    except (TypeError, ValueError, RuntimeError):
        # Fallback to regular Python loop
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    # combined denominator
                    qsquare = qx2[i] + qy2[j] + qz2[k] + small

                    # dot-product of k·P
                    qdotP = qx[i] * Px[i, j, k] + qy[j] * Py[i, j, k] + qz[k] * Pz[i, j, k]

                    RhoB_k[i, j, k] = -ii * qdotP  # ρ_bound(k) = -i k·P(k)

                    # ρ-term and P-term
                    rho_term = -ii * Rho_k[i, j, k] * inv_epsr / qsquare

                    # deposit into each component
                    ExlfromRho[i, j, k] = (rho_term) * qx[i]
                    EylfromRho[i, j, k] = (rho_term) * qy[j]
                    EzlfromRho[i, j, k] = (rho_term) * qz[k]

                    P_term = -qdotP * inv_epsr / qsquare

                    # deposit into each component
                    ExlfromP[i, j, k] = (P_term) * qx[i]
                    EylfromP[i, j, k] = (P_term) * qy[j]
                    EzlfromP[i, j, k] = (P_term) * qz[k]

                    # deposit into each component
                    ExlfromPRho[i, j, k] = (rho_term + P_term) * qx[i]
                    EylfromPRho[i, j, k] = (rho_term + P_term) * qy[j]
                    EzlfromPRho[i, j, k] = (rho_term + P_term) * qz[k]

    RhoBound[:] = RhoB_k
    RhoBound[:] = pyfftw.interfaces.numpy_fft.ifftn(RhoBound)

    ExlfromPRho[:] = pyfftw.interfaces.numpy_fft.ifftn(ExlfromPRho)
    EylfromPRho[:] = pyfftw.interfaces.numpy_fft.ifftn(EylfromPRho)
    EzlfromPRho[:] = pyfftw.interfaces.numpy_fft.ifftn(EzlfromPRho)

    ExlfromP[:] = pyfftw.interfaces.numpy_fft.ifftn(ExlfromP)
    EylfromP[:] = pyfftw.interfaces.numpy_fft.ifftn(EylfromP)
    EzlfromP[:] = pyfftw.interfaces.numpy_fft.ifftn(EzlfromP)

    ExlfromRho[:] = pyfftw.interfaces.numpy_fft.ifftn(ExlfromRho)
    EylfromRho[:] = pyfftw.interfaces.numpy_fft.ifftn(EylfromRho)
    EzlfromRho[:] = pyfftw.interfaces.numpy_fft.ifftn(EzlfromRho)


# ============================================================================
# QWArray class — encapsulates all module-level state
# ============================================================================


class QWArray:
    """Quantum wire array state for 3D Maxwell propagation.

    Encapsulates all mutable module-level state (QW parameters, old
    polarization/charge arrays) enabling multiple independent QW array
    simulations.

    Usage (mirrors Fortran workflow via shim)::

        # Implicit via shim — zero caller changes:
        QuantumWire(space, dt, n, Ex, Ey, Ez, ...)

        # Explicit — for multi-instance:
        qw = QWArray()
        qw.QuantumWire(space, dt, n, Ex, Ey, Ez, ...)
    """

    def __init__(self):
        """Initialize with defaults (matching Fortran module variables)."""
        # QW array parameters (overwritten by read_qw_parameters)
        self.Nw = 4
        self.y0 = 0.0e-6
        self.x0 = 0.0e-9
        self.z0 = 0.0e-6
        self.dxqw = 0.0e-9
        self.ay = 5.0e-9
        self.az = 5.0e-9

        # Flags
        self.QW = True
        self.host = False
        self.propagate = True

        # Old polarization/charge arrays (persistent across timesteps)
        self.PxxOld = None
        self.PyyOld = None
        self.PzzOld = None
        self.RhoOld = None

    def read_qw_parameters(self, filename):
        """Read quantum wire parameters from file.

        Parameters
        ----------
        filename : str
            Path to the parameter file

        Returns
        -------
        int
            Error status: 0 for success, non-zero for error
        """
        def parse_line(line):
            if '#' in line:
                line = line.split('#')[0]
            return line.strip()

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = []
                for line in f:
                    parsed = parse_line(line)
                    if parsed:
                        lines.append(parsed)

                if len(lines) < 6:
                    raise ValueError(f"Expected 6 parameters, found {len(lines)}")

                self.QW = bool(int(lines[0]))
                self.Nw = int(lines[1])
                self.x0 = float(lines[2])
                self.dxqw = float(lines[3])
                self.ay = float(lines[4])
                self.az = float(lines[5])
            return 0
        except IOError as e:
            print(f"Error opening quantum wire parameters file: {e}")
            return 1
        except (ValueError, IndexError) as e:
            print(f"Error reading quantum wire parameters: {e}")
            return 1

    def QuantumWire(self, space, dt, n, Ex, Ey, Ez, Jx, Jy, Jz, Rho,
                    ExlfromPRho, EylfromPRho, EzlfromPRho,
                    ExlfromP, EylfromP, EzlfromP,
                    ExlfromRho, EylfromRho, EzlfromRho, RhoBound):
        """Main quantum wire calculation routine.

        Calculates polarization and charge density for quantum wires and
        deposits them into 3D Maxwell grid.  Handles multiple wires and
        computes longitudinal fields from polarization and charge density.

        All output arrays are modified in-place.
        """
        Nx = GetNx(space)
        Ny = GetNy(space)
        Nz = GetNz(space)

        Exx = np.zeros(Nx, dtype=np.complex128)
        Eyy = np.zeros(Nx, dtype=np.complex128)
        Ezz = np.zeros(Nx, dtype=np.complex128)
        Jxx = np.zeros(Nx, dtype=np.complex128)
        Jyy = np.zeros(Nx, dtype=np.complex128)
        Jzz = np.zeros(Nx, dtype=np.complex128)
        Vrr = np.zeros(Nx, dtype=np.complex128)

        Pxx = np.zeros((Nx, self.Nw), dtype=np.complex128)
        Pyy = np.zeros((Nx, self.Nw), dtype=np.complex128)
        Pzz = np.zeros((Nx, self.Nw), dtype=np.complex128)
        RhoEH = np.zeros((Nx, self.Nw), dtype=np.complex128)

        Px = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        Py = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        Pz = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

        rx = GetXArray(space)
        ry = GetYArray(space)
        rz = GetZArray(space)
        qx = GetKxArray(space)

        gx = np.zeros(Nx, dtype=np.complex128)
        gy = np.zeros(Ny, dtype=np.complex128)
        gz = np.zeros(Nz, dtype=np.complex128)
        xqw = np.zeros(self.Nw, dtype=np.float64)

        DoQWP = [True]
        DoQWDl = [True]
        DoQWCurr = True

        xqw[:] = 0.0

        # Read parameters at the start of the subroutine
        iostat = self.read_qw_parameters('params/qwarray.params')
        if iostat != 0:
            print("Error reading quantum wire parameters, using defaults")
            self.QW = False
            return

        # Reallocate old arrays if needed
        if self.PxxOld is not None:
            self.PxxOld = None
            self.PyyOld = None
            self.PzzOld = None
            self.RhoOld = None

        if self.PxxOld is None:
            self.PxxOld = np.zeros((Nx, self.Nw), dtype=np.complex128)
            self.PyyOld = np.zeros((Nx, self.Nw), dtype=np.complex128)
            self.PzzOld = np.zeros((Nx, self.Nw), dtype=np.complex128)
            self.RhoOld = np.zeros((Nx, self.Nw), dtype=np.complex128)

        Vrr[:] = 0.0
        Pxx[:] = 0.0
        Pyy[:] = 0.0
        Pzz[:] = 0.0
        Px[:] = 0.0
        Py[:] = 0.0
        Pz[:] = 0.0
        RhoEH[:] = 0.0

        for w in range(self.Nw):
            Exx[:] = EAtXYZ(Ex, rx, ry, rz, xqw[w], self.y0, self.z0)
            Eyy[:] = EAtXYZ(Ey, rx, ry, rz, xqw[w], self.y0, self.z0)
            Ezz[:] = EAtXYZ(Ez, rx, ry, rz, xqw[w], self.y0, self.z0)

            QWCalculator(Exx, Eyy, Ezz, Vrr, rx, qx, dt, w + 1,
                         Pxx[:, w], Pyy[:, w], Pzz[:, w], RhoEH[:, w],
                         DoQWP, DoQWDl)

        # Time derivative of polarization → current density
        for i in range(Nx):
            Jxx[i] = np.sum(Pxx[i, :] - self.PxxOld[i, :]) / dt
        for i in range(Nx):
            Jyy[i] = np.sum(Pyy[i, :] - self.PyyOld[i, :]) / dt
        for i in range(Nx):
            Jzz[i] = np.sum(Pzz[i, :] - self.PzzOld[i, :]) / dt

        Jxx[:] = (Jxx / float(self.Nw)
                  + CalcJfx(np.sum(RhoEH, axis=1), np.sum(self.RhoOld, axis=1),
                            dt, GetDx(space)) / float(self.Nw))
        Jyy[:] = Jyy / float(self.Nw)
        Jzz[:] = Jzz / float(self.Nw)

        # Gate profiles
        gx[:] = 1.0
        for j in range(Ny):
            gy[j] = np.exp(-((ry[j] - self.y0) ** 2) / (2.0 * self.ay ** 2))
        for k in range(Nz):
            gz[k] = np.exp(-((rz[k] - self.z0) ** 2) / (2.0 * self.az ** 2))

        if DoQWCurr and self.propagate:
            QWPlacement(Jxx, gx, gy, gz, Jx)
            QWPlacement(Jyy, gx, gy, gz, Jy)
            QWPlacement(Jzz, gx, gy, gz, Jz)

        if DoQWDl[0] and self.propagate:
            QWPlacement((np.sum(RhoEH, axis=1) / float(self.Nw)), gx, gy, gz, Rho)

        if DoQWP[0] and self.propagate:
            QWPlacement((np.sum(Pxx, axis=1) / float(self.Nw)), gx, gy, gz, Px)
            QWPlacement((np.sum(Pyy, axis=1) / float(self.Nw)), gx, gy, gz, Py)
            QWPlacement((np.sum(Pzz, axis=1) / float(self.Nw)), gx, gy, gz, Pz)

            ElongfromRho(space, Rho, Px, Py, Pz, ExlfromPRho, EylfromPRho,
                         EzlfromPRho, ExlfromP, EylfromP, EzlfromP,
                         ExlfromRho, EylfromRho, EzlfromRho, RhoBound)

        self.PxxOld[:] = Pxx
        self.PyyOld[:] = Pyy
        self.PzzOld[:] = Pzz
        self.RhoOld[:] = RhoEH

        if n % 100 == 0:
            print(n, "RhoEH-max = ", np.max(np.abs(RhoEH[:, 0])),
                  np.max(np.abs(RhoEH[:, 1])))
            print(n, "Jxx-max = ", np.max(np.abs(Jxx[:])))
            print(n, "Jyy-max = ", np.max(np.abs(Jyy[:])))
            print(n, "Jzz-max = ", np.max(np.abs(Jzz[:])))
            print("Difference check E*lfromPRho E*lfromP:",
                  np.max(np.abs(ExlfromPRho - ExlfromP)),
                  np.max(np.abs(EylfromPRho - EylfromP)),
                  np.max(np.abs(EzlfromPRho - EzlfromP)))

            for w in range(self.Nw):
                print(n, "w = ", w)
                print(n, "Pxx-max-min-w = ", np.max(np.real(Pxx[:, w])),
                      np.min(np.real(Pxx[:, w])))
                print(n, "Pyy-max-min-w = ", np.max(np.real(Pyy[:, w])),
                      np.min(np.real(Pyy[:, w])))
                print(n, "Pzz-max-min-w = ", np.max(np.real(Pzz[:, w])),
                      np.min(np.real(Pzz[:, w])))


# ============================================================================
# Backward-compatible module-level shims
# ============================================================================

_default_qw = None


def QuantumWire(space, dt, n, Ex, Ey, Ez, Jx, Jy, Jz, Rho,  # noqa: F811
                ExlfromPRho, EylfromPRho, EzlfromPRho,
                ExlfromP, EylfromP, EzlfromP,
                ExlfromRho, EylfromRho, EzlfromRho, RhoBound):
    """Backward-compatible shim — delegates to _default_qw."""
    global _default_qw
    if _default_qw is None:
        _default_qw = QWArray()
    _default_qw.QuantumWire(space, dt, n, Ex, Ey, Ez, Jx, Jy, Jz, Rho,
                            ExlfromPRho, EylfromPRho, EzlfromPRho,
                            ExlfromP, EylfromP, EzlfromP,
                            ExlfromRho, EylfromRho, EzlfromRho, RhoBound)


def read_qw_parameters(filename):  # noqa: F811
    """Backward-compatible shim — delegates to _default_qw."""
    global _default_qw
    if _default_qw is None:
        _default_qw = QWArray()
    return _default_qw.read_qw_parameters(filename)
