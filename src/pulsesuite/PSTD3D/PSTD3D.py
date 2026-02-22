"""
3-D Pseudo-Spectral Time-Domain (PSTD) electromagnetic field propagator.

Implements the spectral-domain Maxwell curl updates for a leapfrog
time-stepping scheme.  All spatial derivatives are computed exactly
in Fourier space (multiplication by i*q), avoiding numerical dispersion
inherent in finite-difference methods.

Physics reference:
    Maxwell's equations in Fourier space (fields are functions of q, t):

    Ampere's law (with current source):
        dE/dt = (v^2) * i*(q x B) - mu0 * v^2 * J

    Faraday's law:
        dB/dt = -i*(q x E)

    where v = c0 / sqrt(epsr) is the phase velocity in the medium.

    Discretised as leapfrog updates:
        E^{n+1} = E^n + i*(q x B^n) * v^2 * dt - mu0 * J^n * v^2 * dt
        B^{n+1} = B^n - i*(q x E^{n+1}) * dt

    The q-vectors are the FFT-ordered wave vectors from the spatial grid.

Ported from: PSTD3D.f90 subroutines UpdateE3D, UpdateB3D, InitializeFields
"""

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
# Accessor helpers (work with dataclasses or SimpleNamespace)
# ---------------------------------------------------------------------------
def _get_attr(obj, name, default=None):
    return getattr(obj, name, default)


def _get_kx_array(space):
    """Build FFT-ordered kx array."""
    Nx = _get_attr(space, "Nx")
    dx = _get_attr(space, "dx")
    L = (Nx - 1) * dx
    if L <= 0 or Nx <= 1:
        return np.zeros(Nx)
    dk = twopi / L
    k = np.arange(Nx, dtype=float) * dk
    k = k - k[Nx // 2]
    return k


def _get_ky_array(space):
    """Build FFT-ordered ky array."""
    Ny = _get_attr(space, "Ny", 1)
    dy = _get_attr(space, "dy", _get_attr(space, "dx"))
    if Ny <= 1:
        return np.zeros(1)
    L = (Ny - 1) * dy
    if L <= 0:
        return np.zeros(Ny)
    dk = twopi / L
    k = np.arange(Ny, dtype=float) * dk
    k = k - k[Ny // 2]
    return k


def _get_kz_array(space):
    """Build FFT-ordered kz array."""
    Nz = _get_attr(space, "Nz", 1)
    dz = _get_attr(space, "dz", _get_attr(space, "dx"))
    if Nz <= 1:
        return np.zeros(1)
    L = (Nz - 1) * dz
    if L <= 0:
        return np.zeros(Nz)
    dk = twopi / L
    k = np.arange(Nz, dtype=float) * dk
    k = k - k[Nz // 2]
    return k


# ---------------------------------------------------------------------------
# Field initialisation
# ---------------------------------------------------------------------------
def InitializeFields(Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz):
    """Zero all electromagnetic field and current arrays.

    Parameters
    ----------
    Ex, Ey, Ez : ndarray, complex
        Electric field components (modified in-place).
    Bx, By, Bz : ndarray, complex
        Magnetic flux density components (modified in-place).
    Jx, Jy, Jz : ndarray, complex
        Current density components (modified in-place).
    """
    for arr in (Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz):
        arr[:] = 0.0


# ---------------------------------------------------------------------------
# E-field update: Ampere's law in Fourier space
# ---------------------------------------------------------------------------
def UpdateE3D(space, time, Bx, By, Bz, Jx, Jy, Jz, Ex, Ey, Ez):
    """Spectral E-field update via Ampere's law.

    E += i*(q x B) * v^2 * dt  -  mu0 * J * v^2 * dt

    where v^2 = c0^2 / epsr.

    The curl in Fourier space is computed component-wise:
        (q x B)_x = qy*Bz - qz*By
        (q x B)_y = qz*Bx - qx*Bz
        (q x B)_z = qx*By - qy*Bx

    Parameters
    ----------
    space : typespace.ss or SimpleNamespace
        Spatial grid (needs Nx, Ny, Nz, dx, epsr).
    time : typetime.ts or SimpleNamespace
        Time grid (needs dt).
    Bx, By, Bz : ndarray (Nx, Ny, Nz), complex
        Magnetic flux density in Fourier space.
    Jx, Jy, Jz : ndarray (Nx, Ny, Nz), complex
        Current density in Fourier space.
    Ex, Ey, Ez : ndarray (Nx, Ny, Nz), complex
        Electric field in Fourier space (modified in-place).
    """
    epsr = _get_attr(space, "epsr", 1.0)
    dt = _get_attr(time, "dt")
    v2 = c0 ** 2 / epsr

    Nx = _get_attr(space, "Nx")
    Ny = _get_attr(space, "Ny", 1)
    Nz = _get_attr(space, "Nz", 1)

    qx = _get_kx_array(space)  # shape (Nx,)
    qy = _get_ky_array(space)  # shape (Ny,)
    qz = _get_kz_array(space)  # shape (Nz,)

    # Loop structure matches Fortran: outer k, inner j, vectorised over i (x)
    for k in range(Nz):
        for j in range(Ny):
            # (q x B)_x = qy*Bz - qz*By
            Ex[:, j, k] += ii * (qy[j] * Bz[:, j, k] - qz[k] * By[:, j, k]) * v2 * dt \
                           - mu0 * Jx[:, j, k] * v2 * dt

            # (q x B)_y = qz*Bx - qx*Bz
            Ey[:, j, k] += ii * (qz[k] * Bx[:, j, k] - qx[:] * Bz[:, j, k]) * v2 * dt \
                           - mu0 * Jy[:, j, k] * v2 * dt

            # (q x B)_z = qx*By - qy*Bx
            Ez[:, j, k] += ii * (qx[:] * By[:, j, k] - qy[j] * Bx[:, j, k]) * v2 * dt \
                           - mu0 * Jz[:, j, k] * v2 * dt


# ---------------------------------------------------------------------------
# B-field update: Faraday's law in Fourier space
# ---------------------------------------------------------------------------
def UpdateB3D(space, time, Ex, Ey, Ez, Bx, By, Bz):
    """Spectral B-field update via Faraday's law.

    B -= i*(q x E) * dt

    Note: Faraday's law has NO dependence on epsr.

    The curl in Fourier space:
        (q x E)_x = qy*Ez - qz*Ey
        (q x E)_y = qz*Ex - qx*Ez
        (q x E)_z = qx*Ey - qy*Ex

    Parameters
    ----------
    space : typespace.ss or SimpleNamespace
        Spatial grid.
    time : typetime.ts or SimpleNamespace
        Time grid (needs dt).
    Ex, Ey, Ez : ndarray (Nx, Ny, Nz), complex
        Electric field in Fourier space.
    Bx, By, Bz : ndarray (Nx, Ny, Nz), complex
        Magnetic flux density in Fourier space (modified in-place).
    """
    dt = _get_attr(time, "dt")

    Nx = _get_attr(space, "Nx")
    Ny = _get_attr(space, "Ny", 1)
    Nz = _get_attr(space, "Nz", 1)

    qx = _get_kx_array(space)
    qy = _get_ky_array(space)
    qz = _get_kz_array(space)

    for k in range(Nz):
        for j in range(Ny):
            # (q x E)_x = qy*Ez - qz*Ey
            Bx[:, j, k] -= ii * (qy[j] * Ez[:, j, k] - qz[k] * Ey[:, j, k]) * dt

            # (q x E)_y = qz*Ex - qx*Ez
            By[:, j, k] -= ii * (qz[k] * Ex[:, j, k] - qx[:] * Ez[:, j, k]) * dt

            # (q x E)_z = qx*Ey - qy*Ex
            Bz[:, j, k] -= ii * (qx[:] * Ey[:, j, k] - qy[j] * Ex[:, j, k]) * dt