"""
Convolutional Perfectly Matched Layer (CPML) for 3-D PSTD simulations.

Implements absorbing boundary conditions using the CPML formulation.
The PML region is a layer of cells at each boundary where auxiliary
convolution fields (psi) damp outgoing waves exponentially.

Physics reference:
    The CPML uses graded conductivity profiles sigma(x) and coordinate
    stretching kappa(x) to create a reflectionless absorbing region.
    The key parameters are:

    - sigma(x): conductivity profile, graded as (x/L)^m from 0 at the
      interior boundary to sigma_max at the outer boundary.
    - kappa(x): coordinate stretching factor, graded similarly from 1
      to kappa_max.
    - alpha(x): CFS (complex frequency shift) parameter that improves
      absorption of evanescent and grazing-incidence waves.

    Update coefficients:
        b_E = exp(-(sigma/kappa + alpha) * dt / eps0)
        c_E = sigma / (sigma + kappa*alpha*eps0) * (b_E - 1)

    The E-field update in the PML region is:
        psi_new = b * psi_old + c * (curl H component)
        E = Ca * E + Cb * (curl_H + psi - J)

    where Ca = 1/kappa, Cb = dt/(eps0*kappa).

Ported from: cpml.f90
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (matching Fortran constants module)
# ---------------------------------------------------------------------------
c0 = 299792458.0
eps0 = 8.8541878176203898505365630317107e-12
mu0 = 1.2566370614359172953850573533118e-6

# ---------------------------------------------------------------------------
# Module-level CPML parameters (matching Fortran module variables)
# ---------------------------------------------------------------------------
M_PROFILE = 4           # polynomial grading order
KAPPA_MAX = 8.0         # maximum coordinate-stretching factor
ALPHA_MAX = 0.05        # CFS parameter for low-frequency absorption
R_TARGET = 1.0e-8       # target reflection coefficient

# Module-level state (set by InitCPML, read by Update routines)
Nx = Ny = Nz = 0
dx = dy = dz = dt = 0.0
npml_x = npml_y = npml_z = 0

# 1-D profiles
sigma_x = sigma_y = sigma_z = None
kappa_x = kappa_y = kappa_z = None
alpha_x = alpha_y = alpha_z = None

# 1-D update coefficients
bEx = cEx = bHx = cHx = None
bEy = cEy = bHy = cHy = None
bEz = cEz = bHz = cHz = None

# 3-D update scaling coefficients
CaX = CbX = DaX = DbX = None
CaY = CbY = DaY = DbY = None
CaZ = CbZ = DaZ = DbZ = None

# Auxiliary (psi) fields
psi_Exy = psi_Exz = None
psi_Eyx = psi_Eyz = None
psi_Ezx = psi_Ezy = None
psi_Hxy = psi_Hxz = None
psi_Hyx = psi_Hyz = None
psi_Hzx = psi_Hzy = None


# ---------------------------------------------------------------------------
# PML thickness heuristic
# ---------------------------------------------------------------------------
def CalcNPML(N_in):
    """Compute safe CPML thickness for a given grid dimension.

    The heuristic targets 5% of the grid with a minimum of 6.
    For larger grids (N > 40), the result is clamped to N/10.
    For smaller grids, the minimum of 6 is kept unless it would
    leave fewer than 2 interior points, in which case npml is
    reduced to (N-2)//2.

    Parameters
    ----------
    N_in : int
        Total number of grid points along one axis.

    Returns
    -------
    int
        Number of PML cells on each side for this axis (>= 1).
    """
    npml = max(6, int(0.05 * float(N_in)))
    # For larger grids, cap at 10% to avoid excessive PML
    if N_in > 40:
        npml = min(npml, N_in // 10)
    # Safety: ensure at least 2 interior points remain
    max_pml = max(1, (N_in - 2) // 2)
    npml = min(npml, max_pml)
    # Absolute minimum of 1
    npml = max(1, npml)
    return npml


# ---------------------------------------------------------------------------
# Stable sigma_max
# ---------------------------------------------------------------------------
def sigma_stable(L):
    """Compute maximum stable sigma for a given PML physical thickness.

    Uses the analytic formula from Taflove & Hagness:
        sigma_max = -(m+1) * eps0 * c0 * ln(R) / (2*L)

    A Courant-safety clamp is applied: sigma * dt / eps0 <= 0.5

    Parameters
    ----------
    L : float
        Physical thickness of the PML region (m).

    Returns
    -------
    float
        Maximum conductivity (S/m).
    """
    pm = float(M_PROFILE)
    sig = -(pm + 1.0) * eps0 * c0 * np.log(R_TARGET) / (2.0 * L)
    # Courant safety clamp (uses module-level dt)
    dt_eff = dt if dt > 0 else 1e-18
    if sig * dt_eff / eps0 > 0.5:
        sig = 0.5 * eps0 / dt_eff
    return sig


# ---------------------------------------------------------------------------
# Coefficient computation for one axis
# ---------------------------------------------------------------------------
def Calc_CoefficientsCPML(N, sigma, kappa, alpha):
    """Compute CPML b/c update coefficients for one axis.

    Parameters
    ----------
    N : int
        Number of grid points along this axis.
    sigma : ndarray (N,)
        Conductivity profile.
    kappa : ndarray (N,)
        Coordinate-stretching profile.
    alpha : ndarray (N,)
        CFS alpha profile.

    Returns
    -------
    bE, cE, bH, cH : ndarray (N,)
        Update coefficients for E-field and H-field auxiliary variables.

    Notes
    -----
    The coefficients implement the recursive convolution:
        psi_new = b * psi_old + c * (spatial derivative)

    For sigma=0, kappa=1: bE = exp(-alpha*dt/eps0), cE = 0
    (no PML damping, just the CFS term).

    For sigma>0: bE < 1 and cE < 0, providing exponential damping.
    """
    bE = np.zeros(N)
    cE = np.zeros(N)
    bH = np.zeros(N)
    cH = np.zeros(N)

    # Use module-level dt; if it's zero (InitCPML not yet called), use default
    dt_eff = dt if dt > 0 else 1e-18

    for i in range(N):
        # E-field coefficients
        bE[i] = np.exp(-(sigma[i] / kappa[i] + alpha[i]) * dt_eff / eps0)
        denom_E = sigma[i] + kappa[i] * alpha[i] * eps0 + 1e-30
        cE[i] = sigma[i] / denom_E * (bE[i] - 1.0)

        # H-field coefficients
        bH[i] = np.exp(-(sigma[i] / kappa[i] + alpha[i]) * dt_eff / mu0)
        denom_H = sigma[i] + kappa[i] * alpha[i] * mu0 + 1e-30
        cH[i] = sigma[i] / denom_H * (bH[i] - 1.0)

    return bE, cE, bH, cH


# ---------------------------------------------------------------------------
# Build graded profile for one axis
# ---------------------------------------------------------------------------
def _build_profiles(N, npml, d, sig_max):
    """Build sigma, kappa, alpha profiles for one axis.

    Parameters
    ----------
    N : int
        Total grid points.
    npml : int
        PML thickness in grid points.
    d : float
        Grid spacing (m).
    sig_max : float
        Maximum conductivity.

    Returns
    -------
    sigma, kappa, alpha : ndarray (N,)
    """
    pm = float(M_PROFILE)
    sigma = np.zeros(N)
    kappa = np.ones(N)
    alpha = np.full(N, ALPHA_MAX)

    for i in range(N):
        i1 = i + 1  # 1-based index matching Fortran
        if i1 <= npml:
            pos = float(npml - i1 + 0.5) / npml
        elif i1 > N - npml:
            pos = float(i1 - (N - npml) - 0.5) / npml
        else:
            pos = 0.0

        sigma[i] = sig_max * pos ** pm
        kappa[i] = 1.0 + (KAPPA_MAX - 1.0) * pos ** pm
        alpha[i] = ALPHA_MAX * (1.0 - pos)

    return sigma, kappa, alpha


# ---------------------------------------------------------------------------
# Full initialisation
# ---------------------------------------------------------------------------
def InitCPML(Nx_in, Ny_in, Nz_in, dx_in, dy_in, dz_in, dt_in, espr=1.0):
    """Initialise all CPML arrays, profiles, and coefficients.

    This must be called before any UpdateCPML_E / UpdateCPML_H calls.

    Parameters
    ----------
    Nx_in, Ny_in, Nz_in : int
        Grid dimensions.
    dx_in, dy_in, dz_in : float
        Grid spacings (m).
    dt_in : float
        Time step (s).
    espr : float
        Relative permittivity (used for Courant safety in sigma_stable).
    """
    # Import into module namespace
    global Nx, Ny, Nz, dx, dy, dz, dt
    global npml_x, npml_y, npml_z
    global sigma_x, kappa_x, alpha_x
    global sigma_y, kappa_y, alpha_y
    global sigma_z, kappa_z, alpha_z
    global bEx, cEx, bHx, cHx
    global bEy, cEy, bHy, cHy
    global bEz, cEz, bHz, cHz
    global CaX, CbX, DaX, DbX
    global CaY, CbY, DaY, DbY
    global CaZ, CbZ, DaZ, DbZ
    global psi_Exy, psi_Exz, psi_Eyx, psi_Eyz, psi_Ezx, psi_Ezy
    global psi_Hxy, psi_Hxz, psi_Hyx, psi_Hyz, psi_Hzx, psi_Hzy

    Nx, Ny, Nz = Nx_in, Ny_in, Nz_in
    dx, dy, dz = dx_in, dy_in, dz_in
    dt = dt_in

    npml_x = CalcNPML(Nx)
    npml_y = CalcNPML(Ny)
    npml_z = CalcNPML(Nz)

    # Physical PML thickness
    Lx = npml_x * dx
    Ly = npml_y * dy
    Lz = npml_z * dz

    # Build 1-D profiles
    sigma_x, kappa_x, alpha_x = _build_profiles(Nx, npml_x, dx, sigma_stable(Lx))
    sigma_y, kappa_y, alpha_y = _build_profiles(Ny, npml_y, dy, sigma_stable(Ly))
    sigma_z, kappa_z, alpha_z = _build_profiles(Nz, npml_z, dz, sigma_stable(Lz))

    # Compute 1-D CPML coefficients
    bEx, cEx, bHx, cHx = Calc_CoefficientsCPML(Nx, sigma_x, kappa_x, alpha_x)
    bEy, cEy, bHy, cHy = Calc_CoefficientsCPML(Ny, sigma_y, kappa_y, alpha_y)
    bEz, cEz, bHz, cHz = Calc_CoefficientsCPML(Nz, sigma_z, kappa_z, alpha_z)

    # Build 3-D coefficient arrays
    CaX = np.zeros((Nx, Ny, Nz))
    CbX = np.zeros((Nx, Ny, Nz))
    DaX = np.zeros((Nx, Ny, Nz))
    DbX = np.zeros((Nx, Ny, Nz))
    CaY = np.zeros((Nx, Ny, Nz))
    CbY = np.zeros((Nx, Ny, Nz))
    DaY = np.zeros((Nx, Ny, Nz))
    DbY = np.zeros((Nx, Ny, Nz))
    CaZ = np.zeros((Nx, Ny, Nz))
    CbZ = np.zeros((Nx, Ny, Nz))
    DaZ = np.zeros((Nx, Ny, Nz))
    DbZ = np.zeros((Nx, Ny, Nz))

    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                CaX[i, j, k] = 1.0 / kappa_x[i]
                CbX[i, j, k] = dt / (eps0 * kappa_x[i])
                DaX[i, j, k] = 1.0 / kappa_x[i]
                DbX[i, j, k] = dt / (mu0 * kappa_x[i])

                CaY[i, j, k] = 1.0 / kappa_y[j]
                CbY[i, j, k] = dt / (eps0 * kappa_y[j])
                DaY[i, j, k] = 1.0 / kappa_y[j]
                DbY[i, j, k] = dt / (mu0 * kappa_y[j])

                CaZ[i, j, k] = 1.0 / kappa_z[k]
                CbZ[i, j, k] = dt / (eps0 * kappa_z[k])
                DaZ[i, j, k] = 1.0 / kappa_z[k]
                DbZ[i, j, k] = dt / (mu0 * kappa_z[k])

    # Allocate and zero auxiliary fields
    shape = (Nx, Ny, Nz)
    psi_Exy = np.zeros(shape)
    psi_Exz = np.zeros(shape)
    psi_Eyx = np.zeros(shape)
    psi_Eyz = np.zeros(shape)
    psi_Ezx = np.zeros(shape)
    psi_Ezy = np.zeros(shape)
    psi_Hxy = np.zeros(shape)
    psi_Hxz = np.zeros(shape)
    psi_Hyx = np.zeros(shape)
    psi_Hyz = np.zeros(shape)
    psi_Hzx = np.zeros(shape)
    psi_Hzy = np.zeros(shape)


# ---------------------------------------------------------------------------
# E-field CPML update (real-space finite differences in PML regions)
# ---------------------------------------------------------------------------
def UpdateCPML_E(Ex, Ey, Ez, Hx, Hy, Hz, Jx, Jy, Jz,
                 Nx_in, Ny_in, Nz_in, dx_in, dy_in, dz_in, dt_in):
    """Apply CPML corrections to E-field components in PML regions.

    This operates in real space using finite differences of the H-field.
    Only grid points inside the PML layers are modified.

    Parameters
    ----------
    Ex, Ey, Ez : ndarray (Nx, Ny, Nz), complex
        Electric field components (modified in-place).
    Hx, Hy, Hz : ndarray (Nx, Ny, Nz), complex
        Magnetic field components (read-only).
    Jx, Jy, Jz : ndarray (Nx, Ny, Nz), float
        Current density components.
    Nx_in, Ny_in, Nz_in : int
        Grid dimensions.
    dx_in, dy_in, dz_in, dt_in : float
        Grid spacings and time step.
    """
    global psi_Exy, psi_Exz, psi_Eyx, psi_Eyz, psi_Ezx, psi_Ezy

    # --- Ex component: PML in x-direction ---
    for k in range(Nz_in - 1):
        for j in range(Ny_in - 1):
            # Lower x-boundary
            for i in range(npml_x):
                curlHz = (Hz[i, j + 1, k] - Hz[i, j, k]) / dy_in
                curlHy = (Hy[i, j, k + 1] - Hy[i, j, k]) / dz_in
                psi_Exy[i, j, k] = bEy[j] * psi_Exy[i, j, k] + cEy[j] * (Hz[i, j + 1, k] - Hz[i, j, k]) / dy_in
                psi_Exz[i, j, k] = bEz[k] * psi_Exz[i, j, k] + cEz[k] * (Hy[i, j, k + 1] - Hy[i, j, k]) / dz_in
                Ex[i, j, k] = CaX[i, j, k] * Ex[i, j, k] + CbX[i, j, k] * (
                    (curlHz - curlHy) + psi_Exy[i, j, k] + psi_Exz[i, j, k] - Jx[i, j, k])

            # Upper x-boundary
            for i in range(Nx_in - npml_x, Nx_in):
                psi_Exy[i, j, k] = bEy[j] * psi_Exy[i, j, k] + cEy[j] * (Hz[i, j + 1, k] - Hz[i, j, k]) / dy_in
                psi_Exz[i, j, k] = bEz[k] * psi_Exz[i, j, k] + cEz[k] * (Hy[i, j, k + 1] - Hy[i, j, k]) / dz_in
                curlHz = (Hz[i, j + 1, k] - Hz[i, j, k]) / dy_in
                curlHy = (Hy[i, j, k + 1] - Hy[i, j, k]) / dz_in
                Ex[i, j, k] = CaX[i, j, k] * Ex[i, j, k] + CbX[i, j, k] * (
                    (curlHz - curlHy) + psi_Exy[i, j, k] + psi_Exz[i, j, k] - Jx[i, j, k])

    # --- Ey component: PML in y-direction ---
    for k in range(Nz_in - 1):
        for i in range(Nx_in - 1):
            # Lower y-boundary
            for j in range(npml_y):
                psi_Eyx[i, j, k] = bEx[i] * psi_Eyx[i, j, k] + cEx[i] * (Hz[i + 1, j, k] - Hz[i, j, k]) / dx_in
                psi_Eyz[i, j, k] = bEz[k] * psi_Eyz[i, j, k] + cEz[k] * (Hx[i, j, k + 1] - Hx[i, j, k]) / dz_in
                curlHx = (Hx[i, j, k + 1] - Hx[i, j, k]) / dz_in
                curlHz = (Hz[i + 1, j, k] - Hz[i, j, k]) / dx_in
                Ey[i, j, k] = CaY[i, j, k] * Ey[i, j, k] + CbY[i, j, k] * (
                    (curlHx - curlHz) + psi_Eyx[i, j, k] + psi_Eyz[i, j, k] - Jy[i, j, k])

            # Upper y-boundary
            for j in range(Ny_in - npml_y, Ny_in - 1):
                psi_Eyx[i, j, k] = bEx[i] * psi_Eyx[i, j, k] + cEx[i] * (Hz[i + 1, j, k] - Hz[i, j, k]) / dx_in
                psi_Eyz[i, j, k] = bEz[k] * psi_Eyz[i, j, k] + cEz[k] * (Hx[i, j, k + 1] - Hx[i, j, k]) / dz_in
                curlHx = (Hx[i, j, k + 1] - Hx[i, j, k]) / dz_in
                curlHz = (Hz[i + 1, j, k] - Hz[i, j, k]) / dx_in
                Ey[i, j, k] = CaY[i, j, k] * Ey[i, j, k] + CbY[i, j, k] * (
                    (curlHx - curlHz) + psi_Eyx[i, j, k] + psi_Eyz[i, j, k] - Jy[i, j, k])

    # --- Ez component: PML in z-direction ---
    for j in range(Ny_in - 1):
        for i in range(Nx_in - 1):
            # Lower z-boundary
            for k in range(npml_z):
                psi_Ezx[i, j, k] = bEx[i] * psi_Ezx[i, j, k] + cEx[i] * (Hy[i + 1, j, k] - Hy[i, j, k]) / dx_in
                psi_Ezy[i, j, k] = bEy[j] * psi_Ezy[i, j, k] + cEy[j] * (Hx[i, j + 1, k] - Hx[i, j, k]) / dy_in
                curlHy = (Hy[i + 1, j, k] - Hy[i, j, k]) / dx_in
                curlHx = (Hx[i, j + 1, k] - Hx[i, j, k]) / dy_in
                Ez[i, j, k] = CaZ[i, j, k] * Ez[i, j, k] + CbZ[i, j, k] * (
                    (curlHy - curlHx) + psi_Ezx[i, j, k] + psi_Ezy[i, j, k] - Jz[i, j, k])

            # Upper z-boundary
            for k in range(Nz_in - npml_z, Nz_in - 1):
                psi_Ezx[i, j, k] = bEx[i] * psi_Ezx[i, j, k] + cEx[i] * (Hy[i + 1, j, k] - Hy[i, j, k]) / dx_in
                psi_Ezy[i, j, k] = bEy[j] * psi_Ezy[i, j, k] + cEy[j] * (Hx[i, j + 1, k] - Hx[i, j, k]) / dy_in
                curlHy = (Hy[i + 1, j, k] - Hy[i, j, k]) / dx_in
                curlHx = (Hx[i, j + 1, k] - Hx[i, j, k]) / dy_in
                Ez[i, j, k] = CaZ[i, j, k] * Ez[i, j, k] + CbZ[i, j, k] * (
                    (curlHy - curlHx) + psi_Ezx[i, j, k] + psi_Ezy[i, j, k] - Jz[i, j, k])


# ---------------------------------------------------------------------------
# H-field CPML update
# ---------------------------------------------------------------------------
def UpdateCPML_H(Hx, Hy, Hz, Ex, Ey, Ez, Jx, Jy, Jz,
                 Nx_in, Ny_in, Nz_in, dx_in, dy_in, dz_in, dt_in):
    """Apply CPML corrections to H-field components in PML regions.

    Parameters
    ----------
    Hx, Hy, Hz : ndarray (Nx, Ny, Nz), complex
        Magnetic field components (modified in-place).
    Ex, Ey, Ez : ndarray (Nx, Ny, Nz), complex
        Electric field components (read-only).
    Jx, Jy, Jz : ndarray (Nx, Ny, Nz), float
        Current density (unused in standard Faraday update, kept for API).
    Nx_in, Ny_in, Nz_in : int
        Grid dimensions.
    dx_in, dy_in, dz_in, dt_in : float
        Grid spacings and time step.
    """
    global psi_Hxy, psi_Hxz, psi_Hyx, psi_Hyz, psi_Hzx, psi_Hzy

    # --- Hx component ---
    for k in range(Nz_in):
        for j in range(Ny_in):
            # Lower x-boundary
            for i in range(npml_x):
                j1 = min(j + 1, Ny_in - 1)
                k1 = min(k + 1, Nz_in - 1)
                psi_Hxy[i, j, k] = bHy[j] * psi_Hxy[i, j, k] + cHy[j] * (Ez[i, j1, k] - Ez[i, j, k]) / dy_in
                psi_Hxz[i, j, k] = bHz[k] * psi_Hxz[i, j, k] + cHz[k] * (Ey[i, j, k1] - Ey[i, j, k]) / dz_in
                curlEz = (Ez[i, j1, k] - Ez[i, j, k]) / dy_in
                curlEy = (Ey[i, j, k1] - Ey[i, j, k]) / dz_in
                Hx[i, j, k] = DaX[i, j, k] * Hx[i, j, k] - DbX[i, j, k] * (
                    (curlEz - curlEy) + psi_Hxy[i, j, k] + psi_Hxz[i, j, k])

            # Upper x-boundary
            for i in range(Nx_in - npml_x, Nx_in - 1):
                j1 = min(j + 1, Ny_in - 1)
                k1 = min(k + 1, Nz_in - 1)
                psi_Hxy[i, j, k] = bHy[j] * psi_Hxy[i, j, k] + cHy[j] * (Ez[i, j1, k] - Ez[i, j, k]) / dy_in
                psi_Hxz[i, j, k] = bHz[k] * psi_Hxz[i, j, k] + cHz[k] * (Ey[i, j, k1] - Ey[i, j, k]) / dz_in
                curlEz = (Ez[i, j1, k] - Ez[i, j, k]) / dy_in
                curlEy = (Ey[i, j, k1] - Ey[i, j, k]) / dz_in
                Hx[i, j, k] = DaX[i, j, k] * Hx[i, j, k] - DbX[i, j, k] * (
                    (curlEz - curlEy) + psi_Hxy[i, j, k] + psi_Hxz[i, j, k])

    # --- Hy component ---
    for k in range(Nz_in):
        for i in range(Nx_in):
            # Lower y-boundary
            for j in range(npml_y):
                i1 = min(i + 1, Nx_in - 1)
                k1 = min(k + 1, Nz_in - 1)
                psi_Hyx[i, j, k] = bHx[i] * psi_Hyx[i, j, k] + cHx[i] * (Ez[i1, j, k] - Ez[i, j, k]) / dx_in
                psi_Hyz[i, j, k] = bHz[k] * psi_Hyz[i, j, k] + cHz[k] * (Ex[i, j, k1] - Ex[i, j, k]) / dz_in
                curlEx = (Ex[i, j, k1] - Ex[i, j, k]) / dz_in
                curlEz = (Ez[i1, j, k] - Ez[i, j, k]) / dx_in
                Hy[i, j, k] = DaY[i, j, k] * Hy[i, j, k] - DbY[i, j, k] * (
                    (curlEx - curlEz) + psi_Hyx[i, j, k] + psi_Hyz[i, j, k])

            # Upper y-boundary
            for j in range(Ny_in - npml_y, Ny_in - 1):
                i1 = min(i + 1, Nx_in - 1)
                k1 = min(k + 1, Nz_in - 1)
                psi_Hyx[i, j, k] = bHx[i] * psi_Hyx[i, j, k] + cHx[i] * (Ez[i1, j, k] - Ez[i, j, k]) / dx_in
                psi_Hyz[i, j, k] = bHz[k] * psi_Hyz[i, j, k] + cHz[k] * (Ex[i, j, k1] - Ex[i, j, k]) / dz_in
                curlEx = (Ex[i, j, k1] - Ex[i, j, k]) / dz_in
                curlEz = (Ez[i1, j, k] - Ez[i, j, k]) / dx_in
                Hy[i, j, k] = DaY[i, j, k] * Hy[i, j, k] - DbY[i, j, k] * (
                    (curlEx - curlEz) + psi_Hyx[i, j, k] + psi_Hyz[i, j, k])

    # --- Hz component ---
    for j in range(Ny_in):
        for i in range(Nx_in):
            # Lower z-boundary
            for k in range(npml_z):
                i1 = min(i + 1, Nx_in - 1)
                j1 = min(j + 1, Ny_in - 1)
                psi_Hzx[i, j, k] = bHx[i] * psi_Hzx[i, j, k] + cHx[i] * (Ey[i1, j, k] - Ey[i, j, k]) / dx_in
                psi_Hzy[i, j, k] = bHy[j] * psi_Hzy[i, j, k] + cHy[j] * (Ex[i, j1, k] - Ex[i, j, k]) / dy_in
                curlEy = (Ey[i1, j, k] - Ey[i, j, k]) / dx_in
                curlEx = (Ex[i, j1, k] - Ex[i, j, k]) / dy_in
                Hz[i, j, k] = DaZ[i, j, k] * Hz[i, j, k] - DbZ[i, j, k] * (
                    (curlEy - curlEx) + psi_Hzx[i, j, k] + psi_Hzy[i, j, k])

            # Upper z-boundary
            for k in range(Nz_in - npml_z, Nz_in):
                i1 = min(i + 1, Nx_in - 1)
                j1 = min(j + 1, Ny_in - 1)
                psi_Hzx[i, j, k] = bHx[i] * psi_Hzx[i, j, k] + cHx[i] * (Ey[i1, j, k] - Ey[i, j, k]) / dx_in
                psi_Hzy[i, j, k] = bHy[j] * psi_Hzy[i, j, k] + cHy[j] * (Ex[i, j1, k] - Ex[i, j, k]) / dy_in
                curlEy = (Ey[i1, j, k] - Ey[i, j, k]) / dx_in
                curlEx = (Ex[i, j1, k] - Ex[i, j, k]) / dy_in
                Hz[i, j, k] = DaZ[i, j, k] * Hz[i, j, k] - DbZ[i, j, k] * (
                    (curlEy - curlEx) + psi_Hzx[i, j, k] + psi_Hzy[i, j, k])