"""
Host material polarization calculations for quantum wire simulations.

This module calculates the host material polarization response
for propagation simulations.

Author: Rahul R. Sah
"""

import os

import numpy as np
import pyfftw
from numba import jit
from scipy.constants import c as c0_SI, e as e0, epsilon_0 as eps0, m_e as me0

pyfftw.interfaces.cache.enable()

# Physical constants
eps0_val = eps0
c0 = c0_SI
me0_val = me0
pi = np.pi
twopi = 2.0 * np.pi
ii = 1j  # Imaginary unit


# ── JIT kernels (module-level — Numba requirement) ──────────────────


@jit(nopython=True)
def _CalcNextP_jit(E_size1, E_size2, osc, P1, P2, E, dt, gam, w, B, eps0_val):
    """JIT-compiled version of CalcNextP."""
    CalcNextP = np.zeros((E_size1, E_size2, osc), dtype=np.complex128)
    f1 = np.zeros(osc)
    f2 = np.zeros(osc)
    f3 = np.zeros(osc)

    for n in range(osc):
        f1[n] = -(1.0 - gam[n] * dt) / (gam[n] * dt + 1.0)
        f2[n] = (2.0 - w[n]**2 * dt**2) / (gam[n] * dt + 1.0)
        f3[n] = (B[n] * w[n]**2 * dt**2) / (gam[n] * dt + 1.0) * eps0_val

    for n in range(osc):
        for j in range(E_size2):
            for i in range(E_size1):
                CalcNextP[i, j, n] = f1[n] * P1[i, j, n] + f2[n] * P2[i, j, n] + f3[n] * E[i, j]

    return CalcNextP


@jit(nopython=True)
def _CalcMonoP_jit(E_size1, E_size2, osc, E, chi1_real, eps0_val):
    """JIT-compiled version of CalcMonoP."""
    CalcMonoP = np.zeros((E_size1, E_size2, osc), dtype=np.complex128)

    for n in range(osc):
        for j in range(E_size2):
            for i in range(E_size1):
                CalcMonoP[i, j, n] = eps0_val * E[i, j] * chi1_real[n]

    return CalcMonoP


# ── Stateless utilities (module-level) ──────────────────────────────


def MakeTransverse(Ex, Ey, qx, qy, qsq):
    """
    Make electric field transverse.

    Projects the electric field onto the transverse direction
    perpendicular to the momentum vector.  Modifies Ex, Ey in-place.
    """
    Ny = Ex.shape[1]

    for j in range(Ny):
        dot_product = qx * Ex[:, j] + qy[j] * Ey[:, j]
        Ex[:, j] = Ex[:, j] - qx * dot_product / qsq[:, j]
        Ey[:, j] = Ey[:, j] - qy[j] * dot_product / qsq[:, j]


def IFFT(f):
    """
    Inverse Fast Fourier Transform wrapper.

    Performs 2D IFFT on the input array.  Modifies f in-place.
    """
    f[:, :] = pyfftw.interfaces.numpy_fft.ifft2(f)


# ── HostMaterial class ──────────────────────────────────────────────


class HostMaterial:
    """Host material polarization state and calculations.

    Encapsulates all mutable state that was previously at module level:
    material parameters (oscillator model), time-stepped polarization
    history arrays, and pre-computed dispersion arrays.

    Usage (mirrors Fortran workflow)::

        hm = HostMaterial()
        hm.SetHostMaterial(True, 'AlAs', 1e-6)
        hm.InitializeHost(Nx, Ny, n0, qsq, True)
        hm.CalcPHost(Ex, Ey, dt, m, epsb, Px, Py)
    """

    def __init__(self):
        """Initialize with defaults (no material selected yet)."""
        # Oscillator model parameters
        self._osc = 2
        self._q = -e0
        self._w = None
        self._gam = None
        self._B = None
        self._C = None
        self._chi1 = None
        self._Nf = None
        self._A0 = 0.0
        self._chi3 = 0.0 + 0.0j

        # Dielectric constants
        self._epsr_0 = 10.0
        self._epsr_infty = 8.2

        # Frequency / wavelength
        self._w1 = 0.0
        self._w2 = 1e20
        self._w0 = 0.0
        self._lambda0 = 0.0

        # Grid info
        self._N1 = 0
        self._N2 = 0

        # Material name
        self._material = 'AlAs'

        # Time-stepped polarization arrays
        self._Px_before = None
        self._Py_before = None
        self._Px_now = None
        self._Py_now = None
        self._Px_after = None
        self._Py_after = None

        # Pre-computed dispersion arrays
        self._omega_q = None
        self._EpsrWq = None

    # ── Material parameter setup ────────────────────────────────────

    def SetHostMaterial(self, host, mat, lam, epsr=0.0, n0=1.0):
        """
        Set host material parameters.

        Parameters
        ----------
        host : bool
            Whether to use host material dispersion
        mat : str
            Material name ('AlAs', 'fsil', 'GaAs', 'none')
        lam : float
            Wavelength (m)
        epsr : float
            Dielectric constant (Fortran intent(inout) artifact)
        n0 : float
            Refractive index (Fortran intent(inout) artifact)

        Returns
        -------
        epsr : float
            Updated dielectric constant
        n0 : float
            Updated refractive index
        """
        self._lambda0 = lam

        mat_trimmed = mat.strip()
        if mat_trimmed == 'AlAs':
            self.SetParamsAlAs()
        elif mat_trimmed == 'fsil':
            self.SetParamsSilica()
        elif mat_trimmed == 'GaAs':
            self.SetParamsGaAs()
        elif mat_trimmed == 'none':
            self.SetParamsNone()
        else:
            print(f"ERROR: Host Material = {mat} Is Not Included In phost.f90 Code")
            raise ValueError(f"Unknown material: {mat}")

        self._material = mat_trimmed

        if self._B is not None and self._C is not None:
            self._chi1 = np.zeros(self._osc, dtype=complex)
            self._chi1 = self._B * lam**2 / (lam**2 - self._C)
        else:
            self._chi1 = np.zeros(self._osc, dtype=complex)

        self._w0 = twopi * c0 / lam

        if host:
            epsr_val = np.real(self.nw2_no_gam(self._w0))
            n0_val = np.real(np.sqrt(epsr_val))
            epsr = epsr_val
            n0 = n0_val
            self.WriteHostDispersion()

        # Fortran always executes epsr = n0**2 at the end
        epsr = n0**2
        return epsr, n0

    def InitializeHost(self, Nx, Ny, n0, qsq, host):
        """
        Initialize host material arrays.

        Allocates and initializes arrays for host material calculations.

        Parameters
        ----------
        Nx, Ny : int
            Grid dimensions
        n0 : float
            Refractive index
        qsq : ndarray
            Squared momentum array, shape (Nx, Ny)
        host : bool
            Whether to use host material dispersion
        """
        self._omega_q = np.zeros((Nx, Ny), dtype=complex)
        self._EpsrWq = np.zeros((Nx, Ny), dtype=complex)

        if host:
            self._Px_before = np.zeros((Nx, Ny, self._osc), dtype=complex)
            self._Px_now = np.zeros((Nx, Ny, self._osc), dtype=complex)
            self._Px_after = np.zeros((Nx, Ny, self._osc), dtype=complex)
            self._Py_before = np.zeros((Nx, Ny, self._osc), dtype=complex)
            self._Py_now = np.zeros((Nx, Ny, self._osc), dtype=complex)
            self._Py_after = np.zeros((Nx, Ny, self._osc), dtype=complex)

            self._omega_q = np.zeros((Nx, Ny), dtype=complex)
            self.CalcWq(np.sqrt(qsq))
            self.CalcEpsrWq(np.sqrt(qsq))
        else:
            self._omega_q = np.sqrt(np.real(qsq)) * c0 / n0
            self._EpsrWq = np.full((Nx, Ny), n0**2, dtype=complex)

    def SetParamsSilica(self):
        """Set material parameters for silica (Sellmeier coefficients)."""
        self._osc = 3
        self._B = np.zeros(self._osc)
        self._C = np.zeros(self._osc)
        self._w = np.zeros(self._osc)
        self._gam = np.zeros(self._osc)
        self._Nf = np.zeros(self._osc)

        self._A0 = 1.0
        self._B[0] = 0.696166300
        self._B[1] = 0.407942600
        self._B[2] = 0.897479400
        self._C[0] = 4.67914826e-3 * 1e-12
        self._C[1] = 1.35120631e-2 * 1e-12
        self._C[2] = 97.9340025 * 1e-12

        self._w = twopi * c0 / np.sqrt(self._C)
        self._gam = self._w / 25.0

        self._Nf = self._B * self._w**2 * eps0_val * me0_val / self._q**2

        self._epsr_0 = self._A0 + np.sum(self._B * self._lambda0**2 / (self._lambda0**2 - self._C))
        self._epsr_infty = self._A0 + np.sum(self._B * self._lambda0**2 / (self._lambda0**2 - self._C))

    def SetParamsGaAs(self):
        """Set material parameters for GaAs (Sellmeier coefficients)."""
        self._osc = 3
        self._B = np.zeros(self._osc)
        self._C = np.zeros(self._osc)
        self._w = np.zeros(self._osc)
        self._gam = np.zeros(self._osc)
        self._Nf = np.zeros(self._osc)

        self._A0 = 4.37251400
        self._B[0] = 5.46674200
        self._B[1] = 0.02429960
        self._B[2] = 1.95752200
        self._C[0] = 0.4431307**2 * 1e-12
        self._C[1] = 0.8746453**2 * 1e-12
        self._C[2] = 36.916600**2 * 1e-12

        self._w = twopi * c0 / np.sqrt(self._C)
        self._gam = self._w / 10.0

        self._Nf = self._B * self._w**2 * eps0_val * me0_val / self._q**2

        self._epsr_0 = self._A0 + np.sum(self._B * self._lambda0**2 / (self._lambda0**2 - self._C))
        self._epsr_infty = self._A0 + np.sum(self._B * self._lambda0**2 / (self._lambda0**2 - self._C))

    def SetParamsAlAs(self):
        """Set material parameters for AlAs (Sellmeier coefficients)."""
        self._osc = 2
        self._B = np.zeros(self._osc)
        self._C = np.zeros(self._osc)
        self._w = np.zeros(self._osc)
        self._gam = np.zeros(self._osc)
        self._Nf = np.zeros(self._osc)

        self._A0 = 2.0792
        self._B[0] = 6.0840
        self._B[1] = 1.9000
        self._C[0] = 0.2822**2 * 1e-12
        self._C[1] = 27.620**2 * 1e-12

        self._w = twopi * c0 / np.sqrt(self._C)
        self._gam[:] = 0.0

        self._Nf = self._B * self._w**2 * eps0_val * me0_val / self._q**2

        self._w1 = twopi * c0 / (2.2e-6)
        self._w2 = twopi * c0 / (0.56e-6)

        self._epsr_0 = 10.0
        self._epsr_infty = 8.2

    def SetParamsNone(self):
        """Set material parameters for vacuum (no dispersion)."""
        self._osc = 1
        self._B = np.zeros(self._osc)
        self._C = np.zeros(self._osc)
        self._w = np.zeros(self._osc)
        self._gam = np.zeros(self._osc)
        self._Nf = np.zeros(self._osc)

        self._A0 = 1.0
        self._B[0] = 0.0
        self._C[0] = 1.0

        self._w = twopi * c0 / np.sqrt(self._C)
        self._gam[:] = 0.0
        self._Nf[:] = 0.0

        self._epsr_0 = 1.0
        self._epsr_infty = 1.0

    # ── Polarization time-stepping ──────────────────────────────────

    def CalcPHost(self, Ex, Ey, dt, m, epsb, Px, Py):
        """
        Calculate host material polarization.

        Computes the host material polarization response using a
        time-stepping scheme.  Modifies Px, Py in-place.
        """
        self._Px_before = self._Px_now.copy() if self._Px_now is not None else np.zeros((Ex.shape[0], Ex.shape[1], self._osc), dtype=complex)
        self._Py_before = self._Py_now.copy() if self._Py_now is not None else np.zeros((Ey.shape[0], Ey.shape[1], self._osc), dtype=complex)

        self._Px_now = self._Px_after.copy() if self._Px_after is not None else np.zeros((Ex.shape[0], Ex.shape[1], self._osc), dtype=complex)
        self._Py_now = self._Py_after.copy() if self._Py_after is not None else np.zeros((Ey.shape[0], Ey.shape[1], self._osc), dtype=complex)

        self._Px_after = self.CalcNextP(self._Px_before, self._Px_now, Ex, dt)
        self._Py_after = self.CalcNextP(self._Py_before, self._Py_now, Ey, dt)

        epsb = self._A0
        Px[:, :] = np.sum(self._Px_after, axis=2)
        Py[:, :] = np.sum(self._Py_after, axis=2)

    def CalcPHostOld(self, Ex, Ey, dt, m, epsb, Px, Py):
        """
        Calculate host material polarization (old version).

        Behavior depends on time step index m:
        - m > 2:  use previous values
        - m >= 2: initialize with monochromatic polarization
        - m < 2:  initialize with monochromatic, set next to zero

        Modifies Px, Py in-place.
        """
        self._Px_before = self._Px_now.copy() if self._Px_now is not None else np.zeros((Ex.shape[0], Ex.shape[1], self._osc), dtype=complex)
        self._Py_before = self._Py_now.copy() if self._Py_now is not None else np.zeros((Ey.shape[0], Ey.shape[1], self._osc), dtype=complex)

        if m > 2:
            self._Px_now = self._Px_after.copy() if self._Px_after is not None else np.zeros((Ex.shape[0], Ex.shape[1], self._osc), dtype=complex)
            self._Px_after = self.CalcNextP(self._Px_before, self._Px_now, Ex, dt)

            self._Py_now = self._Py_after.copy() if self._Py_after is not None else np.zeros((Ey.shape[0], Ey.shape[1], self._osc), dtype=complex)
            self._Py_after = self.CalcNextP(self._Py_before, self._Py_now, Ey, dt)

            epsb = self._A0
        elif m >= 2:
            self._Px_now = self.CalcMonoP(Ex)
            self._Px_after = self.CalcNextP(self._Px_before, self._Px_now, Ex, dt)

            self._Py_now = self.CalcMonoP(Ey)
            self._Py_after = self.CalcNextP(self._Py_before, self._Py_now, Ey, dt)

            epsb = self._A0
        else:
            self._Px_now = self.CalcMonoP(Ex)
            self._Px_after = np.zeros_like(self._Px_now)

            self._Py_now = self.CalcMonoP(Ey)
            self._Py_after = np.zeros_like(self._Py_now)

            epsb = np.real(self.nw2_no_gam(self._w0))

        Px[:, :] = np.sum(self._Px_after, axis=2)
        Py[:, :] = np.sum(self._Py_after, axis=2)

    def CalcNextP(self, P1, P2, E, dt):
        """
        Calculate next polarization value.

        Uses finite difference scheme for the oscillator model:
        P_next = f1 * P1 + f2 * P2 + f3 * E
        """
        E_size1, E_size2 = E.shape

        try:
            return _CalcNextP_jit(E_size1, E_size2, self._osc, P1, P2, E, dt,
                                  self._gam, self._w, self._B, eps0_val)
        except Exception:
            # Fallback to pure Python
            CalcNextP_result = np.zeros((E_size1, E_size2, self._osc), dtype=complex)
            f1 = np.zeros(self._osc)
            f2 = np.zeros(self._osc)
            f3 = np.zeros(self._osc)

            for n in range(self._osc):
                f1[n] = -(1.0 - self._gam[n] * dt) / (self._gam[n] * dt + 1.0)
                f2[n] = (2.0 - self._w[n]**2 * dt**2) / (self._gam[n] * dt + 1.0)
                f3[n] = (self._B[n] * self._w[n]**2 * dt**2) / (self._gam[n] * dt + 1.0) * eps0_val

            for n in range(self._osc):
                for j in range(E_size2):
                    for i in range(E_size1):
                        CalcNextP_result[i, j, n] = f1[n] * P1[i, j, n] + f2[n] * P2[i, j, n] + f3[n] * E[i, j]

            return CalcNextP_result

    def CalcMonoP(self, E):
        """
        Calculate monochromatic polarization.

        P = eps0 * E * real(chi1)
        """
        E_size1, E_size2 = E.shape

        try:
            chi1_real = np.real(self._chi1)
            return _CalcMonoP_jit(E_size1, E_size2, self._osc, E, chi1_real, eps0_val)
        except Exception:
            # Fallback to pure Python
            CalcMonoP_result = np.zeros((E_size1, E_size2, self._osc), dtype=complex)

            for n in range(self._osc):
                for j in range(E_size2):
                    for i in range(E_size1):
                        CalcMonoP_result[i, j, n] = eps0_val * E[i, j] * np.real(self._chi1[n])

            return CalcMonoP_result

    def SetInitialP(self, Ex, Ey, qx, qy, qsq, dt, Px, Py, epsb):
        """
        Set initial polarization values.

        Initializes the polarization arrays for the first time step.
        Modifies Ex, Ey, Px, Py in-place.
        """
        for n in range(self._osc):
            self._Px_after[:, :, n] = eps0_val * Ex[:, :] * self._B[n] * self._w[n]**2 / (self._w[n]**2 - self._omega_q[:, :]**2)
            self._Py_after[:, :, n] = eps0_val * Ey[:, :] * self._B[n] * self._w[n]**2 / (self._w[n]**2 - self._omega_q[:, :]**2)

        for n in range(self._osc):
            self._Px_now[:, :, n] = self._Px_after[:, :, n] * np.exp(-ii * self._omega_q[:, :] * (-dt))
            self._Py_now[:, :, n] = self._Py_after[:, :, n] * np.exp(-ii * self._omega_q[:, :] * (-dt))

        MakeTransverse(Ex, Ey, qx, qy, qsq)
        for n in range(self._osc):
            MakeTransverse(self._Px_now[:, :, n], self._Py_now[:, :, n], qx, qy, qsq)
            MakeTransverse(self._Px_after[:, :, n], self._Py_after[:, :, n], qx, qy, qsq)

        for n in range(self._osc):
            IFFT(self._Px_now[:, :, n])
            IFFT(self._Py_now[:, :, n])
            IFFT(self._Px_after[:, :, n])
            IFFT(self._Py_after[:, :, n])

        self._Px_now = np.real(self._Px_now)
        self._Py_now = np.real(self._Py_now)
        self._Px_after = np.real(self._Px_after)
        self._Py_after = np.real(self._Py_after)

        epsb = self._A0
        Px[:, :] = np.sum(self._Px_after, axis=2)
        Py[:, :] = np.sum(self._Py_after, axis=2)

    # ── Dispersion calculations ─────────────────────────────────────

    def CalcWq(self, q):
        """
        Calculate frequency from momentum using the dispersion relation.

        Writes output to 'fields/host/w.q.dat'.
        """
        n0 = np.sqrt(self.nw2_no_gam(self._w0))
        np0 = self.nwp_no_gam(self._w0)
        x = n0 - np0 * self._w0

        self._omega_q = (-x + np.sqrt(x**2 + 4 * np0 * q * c0)) / (2 * np0)

        os.makedirs('fields/host', exist_ok=True)
        with open('fields/host/w.q.dat', 'w', encoding='utf-8') as f:
            j_max = max(q.shape[1] // 2, 1)
            i_max = max(q.shape[0] // 2, 1)
            for j in range(j_max):
                for i in range(i_max):
                    f.write(f"{np.real(q[i, j]) * 1e-7} {np.real(self._omega_q[i, j]) * 1e-15} {np.imag(self._omega_q[i, j]) * 1e-15}\n")

    def CalcEpsrWq(self, q):
        """Calculate dielectric constant as function of frequency."""
        aw = np.zeros(2)
        bw = np.zeros(2)
        self.DetermineCoeffs(aw, bw)

        for j in range(self._omega_q.shape[1]):
            for i in range(self._omega_q.shape[0]):
                self._EpsrWq[i, j] = self.CalcEpsrWq_ij(np.abs(self._omega_q[i, j]), aw, bw)

    def CalcEpsrWq_ij(self, w_ij, aw, bw):
        """Calculate dielectric constant for a single frequency."""
        if w_ij < self._w1:
            return self._epsr_0 + aw[0] * w_ij**2 + aw[1] * w_ij**3
        elif w_ij > self._w2:
            return self._epsr_infty + bw[0] / w_ij**2 + bw[1] / w_ij**3
        else:
            return self.nw2_no_gam(w_ij)

    def DetermineCoeffs(self, aw, bw):
        """
        Determine expansion coefficients for low/high frequency
        approximations.  Modifies aw, bw in-place.
        """
        e1 = self.nw2_no_gam(self._w1)
        e2 = self.nw2_no_gam(self._w2)

        ep1 = self.epsrwp_no_gam(self._w1)
        ep2 = self.epsrwp_no_gam(self._w2)

        aw[0] = + 3.0 / self._w1**2 * (e1 - self._epsr_0) - ep1 / self._w1
        aw[1] = - 2.0 / self._w1**3 * (e1 - self._epsr_0) + ep1 / self._w1**2

        bw[0] = + 3.0 * self._w2**2 * (e2 - self._epsr_infty) + ep2 * self._w2**3
        bw[1] = - 2.0 * self._w2**3 * (e2 - self._epsr_infty) - ep2 * self._w2**4

    def Epsr_q(self, q):
        """Return copy of the dielectric constant array."""
        return self._EpsrWq.copy()

    def Epsr_qij(self, i, j):
        """Return dielectric constant at indices (i, j)."""
        return self._EpsrWq[i, j]

    def FDTD_Dispersion(self, qx, qy, dx, dy, dt, n0):
        """Calculate FDTD dispersion relation."""
        self._omega_q = np.zeros((len(qx), len(qy)), dtype=complex)

        for j in range(len(qy)):
            for i in range(len(qx)):
                self._omega_q[i, j] = np.sqrt(np.sin(qx[i] * dx / 2.0)**2 / dx**2 +
                                               np.sin(qy[j] * dy / 2.0)**2 / dy**2)
                self._omega_q[i, j] = 2.0 / dt * np.arcsin((c0 / n0) * dt * np.real(self._omega_q[i, j]))

    def wq(self, i, j):
        """Return frequency at indices (i, j)."""
        return self._omega_q[i, j]

    # ── Dielectric constant models ──────────────────────────────────

    def nw2_no_gam(self, wL):
        """Dielectric constant without damping: n^2(w)."""
        if self._B is None or self._w is None:
            return self._epsr_infty

        result = self._A0
        for n in range(self._osc):
            result = result + self._B[n] * self._w[n]**2 / (self._w[n]**2 - wL**2)

        return result

    def nw2(self, wL):
        """Dielectric constant with damping: n^2(w)."""
        if self._B is None or self._w is None or self._gam is None:
            return self._epsr_infty

        result = self._A0
        for n in range(self._osc):
            result = result + self._B[n] * self._w[n]**2 / (self._w[n]**2 - ii * 2.0 * self._gam[n] * wL - wL**2)

        return result

    def nwp_no_gam(self, wL):
        """Derivative of refractive index without damping: dn/dw."""
        if self._B is None or self._w is None:
            return 0.0

        nw = np.sqrt(self.nw2_no_gam(wL))
        result = 0.0
        for n in range(self._osc):
            result = result + self._B[n] * self._w[n]**2 * wL / ((self._w[n]**2 - wL**2)**2)

        return result / nw

    def nwp(self, wL):
        """Derivative of refractive index with damping: dn/dw."""
        if self._B is None or self._w is None or self._gam is None:
            return 0.0

        nw = np.sqrt(self.nw2(wL))
        result = 0.0
        for n in range(self._osc):
            result = result + self._B[n] * self._w[n]**2 * (ii * self._gam[n] + wL) / ((self._w[n]**2 - ii * 2.0 * self._gam[n] * wL - wL**2)**2)

        return result / nw

    def nl2_no_gam(self, lam):
        """Dielectric constant from wavelength without damping: n^2(lam)."""
        if self._B is None or self._C is None:
            return self._epsr_infty

        result = self._A0
        for n in range(self._osc):
            result = result + self._B[n] * lam**2 / (lam**2 - self._C[n])

        return result

    def nl2(self, lam):
        """Dielectric constant from wavelength with damping: n^2(lam)."""
        if self._B is None or self._w is None or self._gam is None:
            return self._epsr_infty

        wL = twopi * c0 / (lam + 1e-100)
        result = self._A0
        for n in range(self._osc):
            result = result + self._B[n] * self._w[n]**2 / (self._w[n]**2 - ii * 2.0 * self._gam[n] * wL - wL**2)

        return result

    def epsrwp_no_gam(self, wL):
        """Derivative of dielectric constant without damping: depsr/dw."""
        if self._B is None or self._w is None:
            return 0.0

        result = 0.0
        for n in range(self._osc):
            result = result + self._B[n] * self._w[n]**2 * (2 * wL) / ((self._w[n]**2 - wL**2)**2)

        return result

    # ── I/O ─────────────────────────────────────────────────────────

    def WriteHostDispersion(self):
        """Write host material dispersion data to files."""
        if self._w is None or self._B is None:
            return

        Nxx = 10000

        # Frequency domain
        x0 = 0.0
        xf = np.max(self._w) * 3.0
        dx = (xf - x0) / Nxx

        os.makedirs('fields/host', exist_ok=True)
        os.makedirs('fields/host/nogam', exist_ok=True)

        f_nw_real = open('fields/host/n.w.real.dat', 'w', encoding='utf-8')
        f_nw_imag = open('fields/host/n.w.imag.dat', 'w', encoding='utf-8')
        f_epsrw_real = open('fields/host/epsr.w.real.dat', 'w', encoding='utf-8')
        f_epsrw_imag = open('fields/host/epsr.w.imag.dat', 'w', encoding='utf-8')
        f_nw_real_ng = open('fields/host/nogam/n.w.real.dat', 'w', encoding='utf-8')
        f_nw_imag_ng = open('fields/host/nogam/n.w.imag.dat', 'w', encoding='utf-8')
        f_epsrw_real_ng = open('fields/host/nogam/epsr.w.real.dat', 'w', encoding='utf-8')
        f_epsrw_imag_ng = open('fields/host/nogam/epsr.w.imag.dat', 'w', encoding='utf-8')
        f_q2w_real = open('fields/host/q2.w.real.dat', 'w', encoding='utf-8')
        f_q2w_imag = open('fields/host/q2.w.imag.dat', 'w', encoding='utf-8')

        for l in range(1, Nxx + 1):
            x = x0 + l * dx
            n2 = self.nw2(x)
            n2X = self.nw2_no_gam(x)
            n = np.sqrt(n2)
            nX = np.sqrt(n2X)

            f_nw_real.write(f'{x} {np.real(n)}\n')
            f_nw_imag.write(f'{x} {np.imag(n)}\n')
            f_epsrw_real.write(f'{x} {np.real(n2)}\n')
            f_epsrw_imag.write(f'{x} {np.imag(n2)}\n')
            exp_factor = np.exp(-((np.abs(n2X) - 9) / 8)**12)
            f_nw_real_ng.write(f'{x} {np.real(nX) * exp_factor}\n')
            f_nw_imag_ng.write(f'{x} {np.imag(nX) * exp_factor}\n')
            f_epsrw_real_ng.write(f'{x} {np.real(n2X) * exp_factor}\n')
            f_epsrw_imag_ng.write(f'{x} {np.imag(n2X) * exp_factor}\n')

        f_nw_real.close()
        f_nw_imag.close()
        f_epsrw_real.close()
        f_epsrw_imag.close()
        f_nw_real_ng.close()
        f_nw_imag_ng.close()
        f_epsrw_real_ng.close()
        f_epsrw_imag_ng.close()
        f_q2w_real.close()
        f_q2w_imag.close()

        # Wavelength domain
        x0 = 0.0
        xf = twopi * c0 / (np.min(self._w) + 1e-100) * 3.0
        dx = (xf - x0) / Nxx

        f_nl_real = open('fields/host/n.l.real.dat', 'w', encoding='utf-8')
        f_nl_imag = open('fields/host/n.l.imag.dat', 'w', encoding='utf-8')
        f_epsrl_real = open('fields/host/epsr.l.real.dat', 'w', encoding='utf-8')
        f_epsrl_imag = open('fields/host/epsr.l.imag.dat', 'w', encoding='utf-8')
        f_nl_real_ng = open('fields/host/nogam/n.l.real.dat', 'w', encoding='utf-8')
        f_nl_imag_ng = open('fields/host/nogam/n.l.imag.dat', 'w', encoding='utf-8')
        f_epsrl_real_ng = open('fields/host/nogam/epsr.l.real.dat', 'w', encoding='utf-8')
        f_epsrl_imag_ng = open('fields/host/nogam/epsr.l.imag.dat', 'w', encoding='utf-8')

        for l in range(1, Nxx + 1):
            x = x0 + (l - 1) * dx
            n2 = self.nl2(x)
            n2X = self.nl2_no_gam(x)
            n = np.sqrt(n2)
            nX = np.sqrt(n2X)

            x_um = x * 1e6

            f_nl_real.write(f'{x_um} {np.real(n)}\n')
            f_nl_imag.write(f'{x_um} {np.imag(n)}\n')
            f_epsrl_real.write(f'{x_um} {np.real(n2)}\n')
            f_epsrl_imag.write(f'{x_um} {np.imag(n2)}\n')
            exp_factor = np.exp(-((np.abs(n2X) - 9) / 8)**12)
            f_nl_real_ng.write(f'{x_um} {np.real(nX) * exp_factor}\n')
            f_nl_imag_ng.write(f'{x_um} {np.imag(nX) * exp_factor}\n')
            f_epsrl_real_ng.write(f'{x_um} {np.real(n2X) * exp_factor}\n')
            f_epsrl_imag_ng.write(f'{x_um} {np.imag(n2X) * exp_factor}\n')

        f_nl_real.close()
        f_nl_imag.close()
        f_epsrl_real.close()
        f_epsrl_imag.close()
        f_nl_real_ng.close()
        f_nl_imag_ng.close()
        f_epsrl_real_ng.close()
        f_epsrl_imag_ng.close()


# ── Backward-compatible module-level API ────────────────────────────
#
# Singleton instance + wrapper functions so that existing code like
#   from pulsesuite.PSTD3D import phost
#   phost.SetParamsAlAs()
#   phost._osc
# continues to work.

_instance = HostMaterial()

# Attribute names that live on the instance (for __getattr__/__setattr__)
_INSTANCE_ATTRS = frozenset([
    '_osc', '_q', '_w', '_gam', '_B', '_C', '_chi1', '_Nf', '_A0', '_chi3',
    '_epsr_0', '_epsr_infty', '_w1', '_w2', '_w0', '_lambda0',
    '_N1', '_N2', '_material',
    '_Px_before', '_Py_before', '_Px_now', '_Py_now', '_Px_after', '_Py_after',
    '_omega_q', '_EpsrWq',
])

# Allow ``phost._osc = 2`` and ``phost._osc`` at module level
# (tests do this heavily).
import sys as _sys


class _ModuleProxy(_sys.modules[__name__].__class__):
    """Module subclass that proxies attribute read/write to _instance."""

    def __setattr__(self, name, value):
        if name in _INSTANCE_ATTRS:
            setattr(_instance, name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in _INSTANCE_ATTRS:
            return getattr(_instance, name)
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

_sys.modules[__name__].__class__ = _ModuleProxy


# ── Wrapper functions delegating to _instance ───────────────────────

def SetHostMaterial(host, mat, lam, epsr=0.0, n0=1.0):
    return _instance.SetHostMaterial(host, mat, lam, epsr, n0)

def InitializeHost(Nx, Ny, n0, qsq, host):
    _instance.InitializeHost(Nx, Ny, n0, qsq, host)

def SetParamsAlAs():
    _instance.SetParamsAlAs()

def SetParamsGaAs():
    _instance.SetParamsGaAs()

def SetParamsSilica():
    _instance.SetParamsSilica()

def SetParamsNone():
    _instance.SetParamsNone()

def CalcPHost(Ex, Ey, dt, m, epsb, Px, Py):
    _instance.CalcPHost(Ex, Ey, dt, m, epsb, Px, Py)

def CalcPHostOld(Ex, Ey, dt, m, epsb, Px, Py):
    _instance.CalcPHostOld(Ex, Ey, dt, m, epsb, Px, Py)

def CalcNextP(P1, P2, E, dt):
    return _instance.CalcNextP(P1, P2, E, dt)

def CalcMonoP(E):
    return _instance.CalcMonoP(E)

def SetInitialP(Ex, Ey, qx, qy, qsq, dt, Px, Py, epsb):
    _instance.SetInitialP(Ex, Ey, qx, qy, qsq, dt, Px, Py, epsb)

def CalcWq(q):
    _instance.CalcWq(q)

def CalcEpsrWq(q):
    _instance.CalcEpsrWq(q)

def CalcEpsrWq_ij(w_ij, aw, bw):
    return _instance.CalcEpsrWq_ij(w_ij, aw, bw)

def DetermineCoeffs(aw, bw):
    _instance.DetermineCoeffs(aw, bw)

def Epsr_q(q):
    return _instance.Epsr_q(q)

def Epsr_qij(i, j):
    return _instance.Epsr_qij(i, j)

def FDTD_Dispersion(qx, qy, dx, dy, dt, n0):
    _instance.FDTD_Dispersion(qx, qy, dx, dy, dt, n0)

def wq(i, j):
    return _instance.wq(i, j)

def nw2_no_gam(wL):
    return _instance.nw2_no_gam(wL)

def nw2(wL):
    return _instance.nw2(wL)

def nwp_no_gam(wL):
    return _instance.nwp_no_gam(wL)

def nwp(wL):
    return _instance.nwp(wL)

def nl2_no_gam(lam):
    return _instance.nl2_no_gam(lam)

def nl2(lam):
    return _instance.nl2(lam)

def epsrwp_no_gam(wL):
    return _instance.epsrwp_no_gam(wL)

def WriteHostDispersion():
    _instance.WriteHostDispersion()
