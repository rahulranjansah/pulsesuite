"""SBEspythonic.py - Python port of SBEs.f90 module"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Union

# Type aliases
_dp = np.float64
_dc = np.complex128
_dp_array = NDArray[_dp]
_dc_array = NDArray[_dc]

# Robust imports with fallback
try:
    from .constants import *
    from .usefulsubspythonic import FFTG, iFFTG, printIT, printIT2D, printITR
    from .qwopticspythonic import Prop2QW, QW2Prop, QWPolarization3, QWRho5, WriteSBESolns, InitializeQWOptics
    from .coulombpythonic import InitializeCoulomb, CalcCoulombArrays, MBCE, MBCH
    from .phononspythonic import InitializePhonons, MBPE, MBPH
    from .emissionpythonic import InitializeEmission, SpontEmission
    from .dcfieldpythonic import InitializeDC, CalcDCE2, CalcDCH2, CalcI0, GetEDrift
    from .dephasingpythonic import InitializeDephasing, CalcGammaE, CalcGammaH, OffDiagDephasing2
    from .helperspythonic import GetSpaceArray, GetKArray, magsq, locator
except ImportError:
    try:
        from constants import *
        from usefulsubspythonic import FFTG, iFFTG, printIT, printIT2D, printITR
        from qwopticspythonic import Prop2QW, QW2Prop, QWPolarization3, QWRho5, WriteSBESolns, InitializeQWOptics
        from coulombpythonic import InitializeCoulomb, CalcCoulombArrays, MBCE, MBCH
        from phononspythonic import InitializePhonons, MBPE, MBPH
        from emissionpythonic import InitializeEmission, SpontEmission
        from dcfieldpythonic import InitializeDC, CalcDCE2, CalcDCH2, CalcI0, GetEDrift
        from dephasingpythonic import InitializeDephasing, CalcGammaE, CalcGammaH, OffDiagDephasing2
        from helperspythonic import GetSpaceArray, GetKArray, magsq, locator
    except ImportError:
        print("Warning: Some required modules not found.")
        twopi = 2.0 * np.pi
        c0 = 299792458.0
        eps0 = 8.854187817e-12
        e0 = 1.602176634e-19
        hbar = 1.054571817e-34
        ii = 1.0j
        eV = e0
        me0 = 9.10938356e-31

# Numba fallback
try:
    from numba import njit, prange
except ImportError:
    def njit(cache=True, parallel=False):
        def decorator(func):
            return func
        return decorator
    def prange(*args, **kwargs):
        return range(*args)


class SBEs:
    """Main SBE solver class implementing semiconductor Bloch equations."""

    def __init__(self):
        # Physical parameters
        self.L = 100e-9
        self.Delta0 = 5e-9
        self.gap = 1.5 * eV
        self.me = 0.07 * me0
        self.mh = 0.45 * me0
        self.HO = 100e-3 * eV
        self.gam_e = 1e12
        self.gam_h = 1e12
        self.gam_eh = 1e12
        self.wL = 0.0
        self.epsr = 9.1
        self.Oph = 36e-3 * eV / hbar
        self.Gph = 3e-3 * eV / hbar
        self.Edc = 0.0

        # Boolean flags
        self.Optics = True
        self.Excitons = True
        self.EHs = True
        self.Screened = True
        self.Phonon = True
        self.DCTrans = False
        self.LF = True
        self.FreePot = False
        self.DiagDph = True
        self.OffDiagDph = True
        self.OBE = False
        self.Recomb = False
        self.ReadDC = False
        self.PLSpec = False
        self.ignorewire = False
        self.debug1 = False
        self.Xqwparams = False

        # Arrays (allocated during initialization)
        self.YY1 = None
        self.YY2 = None
        self.YY3 = None
        self.CC1 = None
        self.CC2 = None
        self.CC3 = None
        self.DD1 = None
        self.DD2 = None
        self.DD3 = None
        self.qwPx = None
        self.qwPy = None
        self.qwPz = None
        self.Id = None
        self.Ia = None
        self.Ee = None
        self.Eh = None
        self.r = None
        self.Qr = None
        self.QE = None
        self.kr = None
        self.I0 = None
        self.ErI0 = None
        self.hw = None
        self.PLS = None

        # Other parameters
        self.dcv = 0.0
        self.ehint = 1.0
        self.Emax0 = 0.0
        self.alphae = 0.0
        self.alphah = 0.0
        self.qc = 0.0
        self.area = 1e-16
        self.t = 0.0
        self.wph = 0.0
        self.chiw = 0.0
        self.vhh0 = 0.0
        self.ETHz = 0.0
        self.Nr = 0
        self.Nk = 0
        self.small = 1e-200
        self.NK0 = 0
        self.NQ0 = 0
        self.nqq = 0
        self.nqq10 = 0
        self.kkp = None
        self.wireoff = True
        self.xxx = 1
        self.jjj = 1
        self.jmax = 1000
        self.ntmax = 100000
        self.uw = 820
        self.start = False
        self.dkr = 0.0
        self.dr = 0.0
        self.Nr1 = 0
        self.Nr2 = 0
        self.EPEnergy = 0.0
        self.EPEnergyW = 0.0

        # Solver instances
        self.coulomb_solver = None
        self.phonon_solver = None
        self.emission_solver = None
        self.dc_solver = None
        self.dephasing_solver = None


    def QWCalculator(self, Exx: _dc_array, Eyy: _dc_array, Ezz: _dc_array,
                    Vrr: _dc_array, rr: _dp_array, q: _dp_array, dt: float,
                    w: int, Pxx: _dc_array, Pyy: _dc_array, Pzz: _dc_array,
                    Rho: _dc_array, DoQWP: bool, DoQWDl: bool) -> None:
        """
        Time-evolves the source terms of the quantum wire for Maxwell's equations.
        Solves the 1D semiconductor Bloch equations.
        """
        # Initialize source terms for propagator
        Pxx[:] = 0.0
        Pyy[:] = 0.0
        Pzz[:] = 0.0
        Rho[:] = 0.0

        # Book keeping - only once per time step for first wire
        if w == 1:
            if self.jjj == self.jmax:
                self.jjj = 0
            self.xxx += 1
            self.jjj += 1
            self.t += dt

            if self.jjj == self.jmax:
                self.PLS[:] = 0.0

        # Record results only 1 out of every jmax times
        WriteFields = (self.jjj == self.jmax)

        # Only do QW calculation if field was once large enough
        if (self.wireoff and
            np.max(np.sqrt(magsq(Exx) + magsq(Eyy) + magsq(Ezz))) < 1e-3 * self.Emax0):
            return
        else:
            self.wireoff = False
            DoQWP = self.Optics
            DoQWDl = self.LF

        # Calculate ETHz for DC field
        tau = self.t - 4e-12
        self.ETHz = (self.Edc * np.exp(-tau**2 / 1e-24) *
                    np.sin(twopi * tau / 1e-12) *
                    np.exp(-tau**12 / (2e-12)**12))

        # Allocate QW polarization arrays if needed
        if self.qwPx is None:
            self.qwPx = np.zeros((len(rr), self.CC1.shape[2]), dtype=_dc)
            self.qwPy = np.zeros((len(rr), self.CC1.shape[2]), dtype=_dc)
            self.qwPz = np.zeros((len(rr), self.CC1.shape[2]), dtype=_dc)

        # Create QW fields from propagation fields
        Edc0 = 0.0
        Ex = np.zeros(self.Nr, dtype=_dc)
        Ey = np.zeros(self.Nr, dtype=_dc)
        Ez = np.zeros(self.Nr, dtype=_dc)
        Vr = np.zeros(self.Nr, dtype=_dc)

        Prop2QW(rr, Exx, Eyy, Ezz, Vrr, Edc0, self.r,
                Ex, Ey, Ez, Vr, self.t, self.xxx)

        if self.debug1:
            # Simple linear response for debugging
            Px = eps0 * 1.0 * Ex
            Py = eps0 * 1.0 * Ey
            Pz = eps0 * 1.0 * Ez
            re = np.zeros_like(self.r, dtype=_dc)
            rh = np.zeros_like(self.r, dtype=_dc)
        else:
            # Solve SBEs for the w'th quantum wire
            Px = np.zeros(self.Nr, dtype=_dc)
            Py = np.zeros(self.Nr, dtype=_dc)
            Pz = np.zeros(self.Nr, dtype=_dc)
            re = np.zeros(self.Nr, dtype=_dc)
            rh = np.zeros(self.Nr, dtype=_dc)

            self.SBECalculator(Ex, Ey, Ez, Vr, dt, Px, Py, Pz, re, rh, WriteFields, w)

        # FFT QW Qr-fields and interpolate to propagation space
        RhoE = np.zeros_like(rr, dtype=_dc)
        RhoH = np.zeros_like(rr, dtype=_dc)

        QW2Prop(self.r, self.Qr, Ex, Ey, Ez, Vr, Px, Py, Pz, re, rh,
                rr, Pxx, Pyy, Pzz, RhoE, RhoH, w, self.xxx, WriteFields, self.LF)

        Rho[:] = RhoH - RhoE

        if WriteFields:
            WritePropFields(rr, Exx, Eyy, Ezz, Vrr, Pxx, Pyy, Pzz,
                           RhoE, RhoH, 'r', w, self.xxx)

        if self.ignorewire:
            Pxx[:] = 0.0
            Pyy[:] = 0.0
            Pzz[:] = 0.0
            Rho[:] = 0.0


    def SBECalculator(self, Ex: _dc_array, Ey: _dc_array, Ez: _dc_array,
                     Vr: _dc_array, dt: float, Px: _dc_array, Py: _dc_array,
                     Pz: _dc_array, Re: _dc_array, Rh: _dc_array,
                     WriteFields: bool, w: int) -> None:
        """
        Solves the 1D semiconductor Bloch equations and calculates
        the source terms Px, Py, Re, and Rh for the w'th quantum wire.
        """
        # Initialize source terms
        Px[:] = 0.0
        Py[:] = 0.0
        Pz[:] = 0.0
        Re[:] = 0.0
        Rh[:] = 0.0

        # Set up identity and anti-identity matrices
        self.Ia = np.ones((self.Nk, self.Nk), dtype=_int_array)
        self.Id = np.zeros((self.Nk, self.Nk), dtype=_int_array)
        for k in range(self.Nk):
            self.Ia[k, k] = 0
            self.Id[k, k] = 1

        # Get current state arrays
        P1, P2, C1, C2, D1, D2 = self.Checkout(w)

        # Prepare arrays for SBEs
        Heh = np.zeros((self.Nk, self.Nk), dtype=_dc)
        Hee = np.zeros((self.Nk, self.Nk), dtype=_dc)
        Hhh = np.zeros((self.Nk, self.Nk), dtype=_dc)
        VC = np.zeros((self.Nk, self.Nk, 3), dtype=_dp)
        E1D = np.zeros((self.Nk, self.Nk), dtype=_dp)
        GamE = np.zeros(self.Nk, dtype=_dp)
        GamH = np.zeros(self.Nk, dtype=_dp)
        OffG = np.zeros((self.Nk, self.Nk, 3), dtype=_dc)
        Rsp = np.zeros(self.Nk, dtype=_dp)

        self.Preparation(P2, C2, D2, Ex, Ey, Ez, Vr, Heh, Hee, Hhh,
                        VC, E1D, GamE, GamH, OffG, Rsp)

        # Calculate time derivatives
        dpdt2 = self.dpdt(C2, D2, P2, Heh, Hee, Hhh, GamE, GamH, OffG[:, :, 0])
        dCdt2 = self.dCdt(C2, D2, P2, Heh, Hee, Hhh, GamE, GamH, OffG[:, :, 1])
        dDdt2 = self.dDdt(C2, D2, P2, Heh, Hee, Hhh, GamE, GamH, OffG[:, :, 2])

        # Time evolve by leapfrog
        C3 = C1 + dCdt2 * dt * 2.0
        D3 = D1 + dDdt2 * dt * 2.0
        P3 = P1 + dpdt2 * dt * 2.0

        # Apply relaxation if needed
        if self.EHs or self.Phonon:
            ne3 = np.diag(C3)
            nh3 = np.diag(D3)
            self.Relaxation(ne3, nh3, VC, E1D, Rsp, dt, w, WriteFields)
            for k in range(self.Nk):
                C3[k, k] = ne3[k]
                D3[k, k] = nh3[k]

        # Apply DC transport if needed
        if self.DCTrans:
            self.Transport(C3, self.ETHz, 0.0, dt, self.DCTrans, self.LF)
            self.Transport(D3, self.ETHz, 0.0, dt, self.DCTrans, self.LF)
            self.Transport(P3, self.ETHz, 0.0, dt, self.DCTrans, self.LF)

        # Normalize carrier densities
        ne3 = np.abs(np.diag(C3))
        nh3 = np.abs(np.diag(D3))
        total = (np.sum(ne3) + np.sum(nh3)) / 2.0
        ne3 = ne3 * total / (np.sum(ne3) + self.small)
        nh3 = nh3 * total / (np.sum(nh3) + self.small)
        for k in range(self.Nk):
            C3[k, k] = ne3[k]
            D3[k, k] = nh3[k]

        # Reshuffle for stability (convert leapfrog to implicit Euler)
        P2 = (P1 + P3) / 2.0
        C2 = (C1 + C3) / 2.0
        D2 = (D1 + D3) / 2.0

        ne2 = np.abs(np.diag(C2))
        nh2 = np.abs(np.diag(D2))

        if WriteFields:
            WriteSBESolns(self.kr, ne3, nh3, C3, D3, P3,
                         np.diag(Hee) / eV, np.diag(Hhh) / eV, w, self.xxx)

        # Calculate QW polarization
        if self.Optics:
            QWPolarization3(self.r, self.kr, P3, self.ehint, self.area,
                           self.L, Px, Py, Pz, self.xxx)

        # Calculate charge densities
        if self.LF:
            QWRho5(self.Qr, self.kr, self.r, self.L, self.kkp, P3, C3, D3,
                   ne3, nh3, Re, Rh, self.xxx, self.jjj)
            Re[:] = 2 * e0 * Re / self.area * self.ehint
            Rh[:] = 2 * e0 * Rh / self.area * self.ehint

        # Calculate current
        if self.I0 is not None:
            self.I0[w] = CalcI0(ne2, nh2, self.Ee, self.Eh, VC, self.dkr, self.kr, self.dc_solver)

        # FFT fields back to real space for output
        iFFTG(Ex)
        iFFTG(Ey)
        iFFTG(Ez)
        iFFTG(Px)
        iFFTG(Py)
        iFFTG(Pz)

        # Write field data
        if self.uw + w < 1000:  # Safety check
            with open(f'dataQW/EP.{w:02d}.t.dat', 'a') as f:
                f.write(f"{self.xxx*dt:.6e} {Ex[self.Nr//2].real:.6e} {Ey[self.Nr//2].real:.6e} {Ez[self.Nr//2].real:.6e} "
                       f"{Px[self.Nr//2].real:.6e} {Py[self.Nr//2].real:.6e} {Pz[self.Nr//2].real:.6e}\n")

        # FFT back to k-space
        FFTG(Ex)
        FFTG(Ey)
        FFTG(Ez)
        FFTG(Px)
        FFTG(Py)
        FFTG(Pz)

        # Update state arrays
        self.Checkin(P1, P2, P3, C1, C2, C3, D1, D2, D3, w)


    def dpdt(self, C: _dc_array, D: _dc_array, P: _dc_array,
             Heh: _dc_array, Hee: _dc_array, Hhh: _dc_array,
             GamE: _dp_array, GamH: _dp_array, OffP: _dc_array) -> _dc_array:
        """Calculates the SBE for coherence between electrons & holes."""
        dpdt = np.zeros((self.Nk, self.Nk), dtype=_dc)

        Pt = P.T
        Heht = Heh.T
        Hhht = Hhh.T

        for ke in range(self.Nk):
            for kh in range(self.Nk):
                dpdt[kh, ke] = (np.sum(Hhh[kh, :] * P[:, ke] + Hee[ke, :] * P[kh, :]) -
                               np.sum(Heh[:, kh] * C[:, ke] + Heh[ke, :] * D[:, kh]) +
                               Heh[ke, kh] -
                               ii * hbar * (GamE[ke] + GamH[kh]) * P[kh, ke] +
                               OffP[kh, ke])

        return dpdt / (ii * hbar)

    def dCdt(self, Cee: _dc_array, Dhh: _dc_array, Phe: _dc_array,
             Heh: _dc_array, Hee: _dc_array, Hhh: _dc_array,
             GamE: _dp_array, GamH: _dp_array, OffE: _dc_array) -> _dc_array:
        """Calculates the SBE for coherence between electrons."""
        dCdt = np.zeros((self.Nk, self.Nk), dtype=_dc)

        Hhe = Heh.T.conj()
        Peh = Phe.T.conj()

        for k2 in range(self.Nk):
            for k1 in range(self.Nk):
                dCdt[k1, k2] = (np.sum(Hee[k2, :] * Cee[k1, :] - Hee[:, k1] * Cee[:, k2]) +
                               np.sum(Heh[k2, :] * Peh[k1, :] - Hhe[:, k1] * Phe[:, k2]))

        return dCdt / (ii * hbar)

    def dDdt(self, Cee: _dc_array, Dhh: _dc_array, Phe: _dc_array,
             Heh: _dc_array, Hee: _dc_array, Hhh: _dc_array,
             GamE: _dp_array, GamH: _dp_array, OffH: _dc_array) -> _dc_array:
        """Calculates the SBE for coherence between holes."""
        dDdt = np.zeros((self.Nk, self.Nk), dtype=_dc)

        Peh = Phe.T.conj()
        Hhe = Heh.T.conj()

        for k2 in range(self.Nk):
            for k1 in range(self.Nk):
                dDdt[k1, k2] = (np.sum(Hhh[k2, :] * Dhh[k1, :] - Hhh[:, k1] * Dhh[:, k2]) +
                               np.sum(Heh[:, k2] * Peh[:, k1] - Hhe[k1, :] * Phe[k2, :]))

        return dDdt / (ii * hbar)

    def CalcMeh(self, Ex: _dc_array, Ey: _dc_array, Ez: _dc_array,
                Meh: _dc_array) -> None:
        """Calculates dipole matrix elements."""
        Meh[:] = 0.0

        for kh in range(self.Nk):
            for ke in range(self.Nk):
                q = self.kkp[ke, kh]
                if 0 <= q < len(Ex):
                    Meh[ke, kh] = (-self.ehint * Xcv(ke, kh) * Ex[q] -
                                   self.ehint * Ycv(ke, kh) * Ey[q] -
                                   self.ehint * Zcv(ke, kh) * Ez[q])

    def CalcWnn(self, q0: float, Vr: _dc_array, Wnn: _dc_array) -> None:
        """Calculates monopole matrix elements."""
        Wnn[:] = 0.0

        for k2 in range(self.Nk):
            for k1 in range(self.Nk):
                q = self.kkp[k1, k2]
                if 0 <= q < len(Vr):
                    Wnn[k1, k2] = q0 * Vr[q]


    def Preparation(self, P: _dc_array, C: _dc_array, D: _dc_array,
                   Ex: _dc_array, Ey: _dc_array, Ez: _dc_array, Vr: _dc_array,
                   Heh: _dc_array, Hee: _dc_array, Hhh: _dc_array,
                   VC: _dp_array, E1D: _dp_array, GamE: _dp_array, GamH: _dp_array,
                   OffG: _dc_array, Rsp: _dp_array) -> None:
        """Prepares arrays needed for SBEs."""
        # Initialize arrays
        Meh = np.zeros((self.Nk, self.Nk), dtype=_dc)
        Wee = np.zeros((self.Nk, self.Nk), dtype=_dc)
        Whh = np.zeros((self.Nk, self.Nk), dtype=_dc)
        ne = np.zeros(self.Nk, dtype=_dc)
        nh = np.zeros(self.Nk, dtype=_dc)

        VC[:] = 0.0
        E1D[:] = 0.0
        GamE[:] = self.gam_e
        GamH[:] = self.gam_h
        OffG[:] = 0.0
        Rsp[:] = 0.0

        # Get carrier populations
        for k in range(self.Nk):
            ne[k] = C[k, k]
            nh[k] = D[k, k]

        # Calculate dipole coupling matrix
        if self.Optics:
            self.CalcMeh(Ex, Ey, Ez, Meh)

        # Calculate monopole matrix elements if needed
        if self.Optics and self.LF:
            self.CalcWnn(-e0, Vr, Wee)
            self.CalcWnn(+e0, Vr.conj(), Whh)

        # Screen Coulomb arrays
        if self.coulomb_solver:
            VC[:, :, :] = self.coulomb_solver.get_screened_potentials(ne, nh)

        # Calculate Hamiltonian matrix elements
        self.CalcH(Meh, Wee, Whh, C, D, P, VC, Heh, Hee, Hhh)

        # Calculate dephasing rates
        if self.DiagDph and self.dephasing_solver:
            CalcGammaE(self.kr, ne, nh, VC, GamE, self.dephasing_solver)
            CalcGammaH(self.kr, ne, nh, VC, GamH, self.dephasing_solver)

        # Calculate off-diagonal dephasing
        if self.OffDiagDph and self.dephasing_solver:
            g = np.array([self.gam_eh, self.gam_e, self.gam_h])
            OffDiagDephasing2(ne, nh, P, self.kr, self.Ee, self.Eh, g, VC,
                             self.t, OffG[:, :, 0], self.dephasing_solver)

        # Calculate spontaneous emission rates
        if self.Recomb and self.emission_solver:
            SpontEmission(ne, nh, self.Ee, self.Eh, self.gap, self.gam_eh,
                         VC, Rsp, self.emission_solver)

    def CalcH(self, Meh: _dc_array, Wee: _dc_array, Whh: _dc_array,
              C: _dc_array, D: _dc_array, p: _dc_array, VC: _dp_array,
              Heh: _dc_array, Hee: _dc_array, Hhh: _dc_array) -> None:
        """Calculates Hamiltonian matrix elements."""
        # Initialize Hamiltonian
        Heh[:] = 0.0
        Hee[:] = 0.0
        Hhh[:] = 0.0

        # Set up single particle energies
        for k in range(self.Nk):
            Hee[k, k] = self.Ee[k] + self.gap
            Hhh[k, k] = self.Eh[k]

        # Set up potential matrix elements
        Heh[:] = Meh[:]

        if self.FreePot:
            Hee[:] += Wee[:]
            Hhh[:] += Whh[:]

        # Add exciton effects if enabled
        if self.Excitons and self.LF:
            self._add_exciton_effects(C, D, p, VC, Heh, Hee, Hhh)

    def _add_exciton_effects(self, C: _dc_array, D: _dc_array, p: _dc_array,
                            VC: _dp_array, Heh: _dc_array, Hee: _dc_array,
                            Hhh: _dc_array) -> None:
        """Adds exciton effects to Hamiltonian."""
        # Set up Vq calculation
        V = np.zeros((2*self.Nk+1, 3), dtype=_dp)
        for q in range(1, self.Nk):
            V[+q, :] = VC[1+q, 1, :]
            V[-q, :] = VC[1+q, 1, :]

        Ct = C.T
        Dt = D.T
        pt = p.T

        # Calculate H^{e,h}_{k_1,k_2}
        for k2 in range(self.Nk):
            for k1 in range(self.Nk):
                for q in range(max(1-k1, 1-k2), min(self.Nk-k1, self.Nk-k2)+1):
                    if -self.Nk <= q <= self.Nk:
                        noq0 = 1 if q != 0 else 0
                        Heh[k1, k2] += V[q, 0] * pt[k1+q, k2+q] * noq0

        # Calculate H^{e,e}_{k_1,k_2}
        for k2 in range(self.Nk):
            for k1 in range(self.Nk):
                for q in range(max(1-k1, 1-k2), min(self.Nk-k1, self.Nk-k2)+1):
                    if -self.Nk <= q <= self.Nk:
                        noq0 = 1 if q != 0 else 0
                        Hee[k1, k2] -= V[q, 1] * Ct[k1+q, k2+q] * noq0

                q = k1 - k2
                for k in range(max(1, 1-q), min(self.Nk, self.Nk-q)+1):
                    if -self.Nk <= q <= self.Nk:
                        noq0 = 1 if q != 0 else 0
                        Hee[k1, k2] += (V[q, 1] * C[k, k+q] - V[q, 2] * D[k+q, k]) * noq0

        # Calculate H^{h,h}_{k_1,k_2}
        for k2 in range(self.Nk):
            for k1 in range(self.Nk):
                for q in range(max(1-k1, 1-k2), min(self.Nk-k1, self.Nk-k2)+1):
                    if -self.Nk <= q <= self.Nk:
                        noq0 = 1 if q != 0 else 0
                        Hhh[k1, k2] -= V[q, 2] * Dt[k1+q, k2+q] * noq0

                q = k1 - k2
                for k in range(max(1, 1-q), min(self.Nk, self.Nk-q)+1):
                    if -self.Nk <= q <= self.Nk:
                        noq0 = 1 if q != 0 else 0
                        Hhh[k1, k2] += (V[q, 2] * D[k, k+q] - V[q, 2] * C[k+q, k]) * noq0


    def Relaxation(self, ne: _dc_array, nh: _dc_array, VC: _dp_array,
                   E1D: _dp_array, Rsp: _dp_array, dt: float, w: int,
                   writefields: bool) -> None:
        """Calculates carrier relaxation."""
        if self.EHs or self.Phonon:
            WinE = np.zeros(self.Nk, dtype=_dp)
            WoutE = np.zeros(self.Nk, dtype=_dp)
            WinH = np.zeros(self.Nk, dtype=_dp)
            WoutH = np.zeros(self.Nk, dtype=_dp)

            if self.EHs:
                MBCE(ne.real, nh.real, self.kr, self.Ee, self.Eh, VC,
                     self.gam_eh, self.gam_e, WinE, WoutE, self.coulomb_solver)
                MBCH(ne.real, nh.real, self.kr, self.Ee, self.Eh, VC,
                     self.gam_eh, self.gam_h, WinH, WoutH, self.coulomb_solver)

            if self.Phonon:
                MBPE(ne.real, VC, E1D, WinE, WoutE, self.phonon_solver)
                MBPH(nh.real, VC, E1D, WinH, WoutH, self.phonon_solver)

            if writefields:
                printITR(WinE * (1 - ne.real), self.kr, self.xxx, f'Wire/Win/Win.e.{w:02d}.k.')
                printITR(WinH * (1 - nh.real), self.kr, self.xxx, f'Wire/Win/Win.h.{w:02d}.k.')
                printITR(WoutE * ne.real, self.kr, self.xxx, f'Wire/Wout/Wout.e.{w:02d}.k.')
                printITR(WoutH * nh.real, self.kr, self.xxx, f'Wire/Wout/Wout.h.{w:02d}.k.')

            # Update carrier densities
            ne[:] = (np.exp(-(WinE + WoutE) * dt) *
                    ((-1 + np.exp(+(WinE + WoutE) * dt) + ne) * WinE + ne * WoutE) /
                    (WinE + WoutE + self.small))
            nh[:] = (np.exp(-(WinH + WoutH) * dt) *
                    ((-1 + np.exp(+(WinH + WoutH) * dt) + nh) * WinH + nh * WoutH) /
                    (WinH + WoutH + self.small))

        if self.Recomb:
            ne[:] *= np.exp(-Rsp * nh * dt)
            nh[:] *= np.exp(-Rsp * ne * dt)

    def Transport(self, C: _dc_array, ETHz: float, Ey: float, dt: float,
                  DCTrans: bool, LF: bool) -> None:
        """Applies DC transport effects."""
        if DCTrans and self.dc_solver:
            # Apply DC field effects
            C[:] += CalcDCE2(DCTrans, self.kr, self.kkp, ETHz, self.me,
                            self.gam_e, self.Oph, 0.0, C, self.Ee, VC,
                            self.xxx, self.jjj, self.dc_solver) * dt

    def Checkout(self, P1: _dc_array, P2: _dc_array, C1: _dc_array,
                 C2: _dc_array, D1: _dc_array, D2: _dc_array, w: int) -> Tuple:
        """Gets current state arrays."""
        P1[:] = self.YY2[:, :, w]
        P2[:] = self.YY3[:, :, w]
        C1[:] = self.CC2[:, :, w]
        C2[:] = self.CC3[:, :, w]
        D1[:] = self.DD2[:, :, w]
        D2[:] = self.DD3[:, :, w]
        return P1, P2, C1, C2, D1, D2

    def Checkin(self, P1: _dc_array, P2: _dc_array, P3: _dc_array,
                C1: _dc_array, C2: _dc_array, C3: _dc_array,
                D1: _dc_array, D2: _dc_array, D3: _dc_array, w: int) -> None:
        """Updates state arrays."""
        self.YY1[:, :, w] = P1
        self.YY2[:, :, w] = P2
        self.YY3[:, :, w] = P3
        self.CC1[:, :, w] = C1
        self.CC2[:, :, w] = C2
        self.CC3[:, :, w] = C3
        self.DD1[:, :, w] = D1
        self.DD2[:, :, w] = D2
        self.DD3[:, :, w] = D3


    def InitializeSBE(self, q: _dp_array, rr: _dp_array, r0: float,
                     Emaxxx: float, lam: float, Nw: int, QW: bool) -> None:
        """Initializes the SBE module for calculations."""
        if not QW:
            return

        # Read parameters from files
        self.ReadQWParams()
        self.ReadMBParams()

        self.Emax0 = Emaxxx
        self.t = 0.0

        # Calculate grid parameters
        kmax = np.sqrt(1.2 * self.gap * 2.0 * self.me / hbar**2)
        self.dkr = twopi / (2 * self.L)
        self.Nk = int(np.floor(kmax / self.dkr) * 2 + 1)
        self.Nr = self.Nk * 2
        self.Nk -= 1  # Adjust for FFT


        # Allocate arrays
        self.r = np.zeros(self.Nr, dtype=_dp)
        self.Qr = np.zeros(self.Nr, dtype=_dp)
        self.QE = np.zeros(self.Nr - 1, dtype=_dp)
        self.kr = np.zeros(self.Nk, dtype=_dp)
        self.Ee = np.zeros(self.Nk, dtype=_dp)
        self.Eh = np.zeros(self.Nk, dtype=_dp)

        # Create arrays for Coulomb solver (same length)
        self.r_coulomb = np.zeros(self.Nk, dtype=_dp)
        self.kr_coulomb = np.zeros(self.Nk, dtype=_dp)
        self.I0 = np.zeros(Nw, dtype=_dp)
        self.ErI0 = np.zeros(Nw, dtype=_dp)

        # Allocate coherence arrays
        self.YY1 = np.zeros((self.Nk, self.Nk, Nw), dtype=_dc)
        self.YY2 = np.zeros((self.Nk, self.Nk, Nw), dtype=_dc)
        self.YY3 = np.zeros((self.Nk, self.Nk, Nw), dtype=_dc)
        self.CC1 = np.zeros((self.Nk, self.Nk, Nw), dtype=_dc)
        self.CC2 = np.zeros((self.Nk, self.Nk, Nw), dtype=_dc)
        self.CC3 = np.zeros((self.Nk, self.Nk, Nw), dtype=_dc)
        self.DD1 = np.zeros((self.Nk, self.Nk, Nw), dtype=_dc)
        self.DD2 = np.zeros((self.Nk, self.Nk, Nw), dtype=_dc)
        self.DD3 = np.zeros((self.Nk, self.Nk, Nw), dtype=_dc)
        self.Id = np.zeros((self.Nk, self.Nk), dtype=np.int32)
        self.Ia = np.ones((self.Nk, self.Nk), dtype=np.int32)

        # Set up identity matrices
        for k in range(self.Nk):
            self.Id[k, k] = 1
            self.Ia[k, k] = 0

        # Calculate spatial and momentum arrays
        self.GetArrays(self.r, self.Qr, self.kr)
        self.dr = (self.r[2] - self.r[1]) * (self.Nr - 1) / self.Nr

        # Create Coulomb arrays (same length)
        self.GetCoulombArrays(self.r_coulomb, self.kr_coulomb)

        # Calculate material constants
        self.dcv = np.sqrt((e0 * hbar)**2 / (6 * me0 * self.gap) * (me0 / self.me - 1.0))
        self.alphae = np.sqrt(self.me * self.HO) / hbar
        self.alphah = np.sqrt(self.mh * self.HO) / hbar
        self.ehint = np.sqrt(2 * self.alphae * self.alphah / (self.alphae**2 + self.alphah**2))
        self.gam_eh = (self.gam_e + self.gam_h) / 2.0
        self.qc = 2 * self.alphae * self.alphah / (self.alphae + self.alphah)
        self.area = np.sqrt(2 * np.pi) / np.sqrt(self.alphae**2 + self.alphah**2) * self.Delta0

        # Calculate band structure
        self.Ee = hbar**2 * self.kr**2 / (2.0 * self.me)
        self.Eh = hbar**2 * self.kr**2 / (2.0 * self.mh)

        # Initialize coherence arrays
        for k in range(self.Nk):
            self.CC1[k, k, :] = FermiDistr(self.Ee[k] + self.gap / 2.0)
            self.DD1[k, k, :] = FermiDistr(self.Ee[k] + self.gap / 2.0)

        self.CC2[:] = self.CC1
        self.CC3[:] = self.CC1
        self.DD2[:] = self.DD1
        self.DD3[:] = self.DD1
        self.YY2[:] = 0.0
        self.YY3[:] = 0.0

        # Initialize QW optics
        InitializeQWOptics(rr, self.L, self.dcv, self.kr, self.Qr,
                          self.Ee, self.Eh, self.ehint, self.area, self.gap)

        # Make kkp matrix
        self.MakeKKP()

        # Initialize other modules
        self.coulomb_solver = InitializeCoulomb(self.r_coulomb, self.kr_coulomb, self.L, self.Delta0,
                                               self.me, self.mh, self.Ee, self.Eh,
                                               self.gam_e, self.gam_h, self.alphae,
                                               self.alphah, self.epsr, self.Qr,
                                               self.kkp, self.Screened)

        self.phonon_solver = InitializePhonons(self.kr, self.Ee, self.Eh, self.L,
                                              self.epsr, self.Gph, self.Oph)

        self.dc_solver = InitializeDC(self.kr, self.me, self.mh)

        self.dephasing_solver = InitializeDephasing(self.kr, self.me, self.mh)

        if self.Recomb:
            self.emission_solver = InitializeEmission(self.kr, self.Ee, self.Eh,
                                                     abs(self.dcv), self.epsr,
                                                     self.gam_eh, self.ehint)

        # Determine QW boundaries
        self.Nr1 = locator(rr - r0, self.r[0])
        self.Nr2 = locator(rr - r0, self.r[self.Nr-1])

        self.t = 0.0
        self.wph = (self.gap + hbar * self.Oph - hbar * self.wL) / hbar

        self.chiw = self.QWChi1(lam, self.dkr, self.Ee + self.gap, self.Eh,
                               self.area, self.gam_eh, self.dcv)

        # Set OBE mode if requested
        if self.OBE:
            self.Optics = True
            self.Excitons = False
            self.EHs = False
            self.Phonon = False
            self.Recomb = False
            self.LF = False
            self.DCTrans = False

        self.start = True

    def ReadQWParams(self) -> None:
        """Reads QW parameters from file."""
        try:
            with open('params/qw.params', 'r') as f:
                # Helper function to parse Fortran-style scientific notation
                def parse_fortran_float(s):
                    # Remove comments and whitespace
                    s = s.split('!')[0].strip()
                    # Replace 'd' with 'e' for scientific notation
                    s = s.replace('d', 'e')
                    return float(s)

                self.L = parse_fortran_float(f.readline())  # Already in meters
                self.Delta0 = parse_fortran_float(f.readline())  # Already in meters
                self.gap = parse_fortran_float(f.readline()) * eV  # Convert eV to Joules
                self.me = parse_fortran_float(f.readline()) * me0  # Convert to kg
                self.mh = parse_fortran_float(f.readline()) * me0  # Convert to kg
                self.HO = parse_fortran_float(f.readline()) * eV  # Convert eV to Joules
                self.gam_e = parse_fortran_float(f.readline())
                self.gam_h = parse_fortran_float(f.readline())
                self.gam_eh = parse_fortran_float(f.readline())
                self.epsr = parse_fortran_float(f.readline())
                self.Oph = parse_fortran_float(f.readline()) * eV / hbar
                self.Gph = parse_fortran_float(f.readline()) * eV / hbar
                self.Edc = parse_fortran_float(f.readline())
                self.jmax = int(parse_fortran_float(f.readline()))
                self.ntmax = int(parse_fortran_float(f.readline()))
        except FileNotFoundError:
            print("Warning: params/qw.params not found, using defaults")

    def ReadMBParams(self) -> None:
        """Reads many-body parameters from file."""
        try:
            with open('params/mb.params', 'r') as f:
                # Helper function to parse Fortran boolean values
                def parse_fortran_bool(s):
                    # Remove comments and whitespace
                    s = s.split('!')[0].strip().lower()
                    return s == '.true.'

                self.Optics = parse_fortran_bool(f.readline())
                self.Excitons = parse_fortran_bool(f.readline())
                self.EHs = parse_fortran_bool(f.readline())
                self.Screened = parse_fortran_bool(f.readline())
                self.Phonon = parse_fortran_bool(f.readline())
                self.DCTrans = parse_fortran_bool(f.readline())
                self.LF = parse_fortran_bool(f.readline())
                self.FreePot = parse_fortran_bool(f.readline())
                self.DiagDph = parse_fortran_bool(f.readline())
                self.OffDiagDph = parse_fortran_bool(f.readline())
                self.Recomb = parse_fortran_bool(f.readline())
                self.PLSpec = parse_fortran_bool(f.readline())
                self.ignorewire = parse_fortran_bool(f.readline())
                self.Xqwparams = parse_fortran_bool(f.readline())
                LorentzDelta = parse_fortran_bool(f.readline())
        except FileNotFoundError:
            print("Warning: params/mb.params not found, using defaults")

    def GetArrays(self, x: _dp_array, qx: _dp_array, kx: _dp_array) -> None:
        """Calculates spatial and momentum arrays."""
        dnk = (self.Nk - 1) // 2
        self.NK0 = dnk + 1

        for k in range(self.Nk):
            kx[k] = self.dkr * (-dnk + k - 1.5)

        x[:] = GetSpaceArray(self.Nr, 2 * self.L)
        qx[:] = GetKArray(self.Nr, 2 * self.L)
        qx[:] = np.roll(qx, self.Nr // 2)
        qx[0] = -qx[0]

        self.NQ0 = self.GetArray0Index(qx)

    def GetArray0Index(self, qx: _dp_array) -> int:
        """Finds index of zero frequency."""
        return np.argmin(np.abs(qx))

    def GetCoulombArrays(self, x: _dp_array, kx: _dp_array) -> None:
        """Calculates spatial and momentum arrays for Coulomb solver (same length)."""
        dnk = (self.Nk - 1) // 2
        self.NK0 = dnk + 1

        for k in range(self.Nk):
            kx[k] = self.dkr * (-dnk + k - 1.5)

        x[:] = GetSpaceArray(self.Nk, 2 * self.L)

    def MakeKKP(self) -> None:
        """Makes kkp matrix for momentum mapping."""
        self.kkp = np.zeros((self.Nk, self.Nk), dtype=np.int32)

        for k in range(self.Nk):
            for kp in range(self.Nk):
                q = self.kr[k] - self.kr[kp]
                self.kkp[k, kp] = int(np.round(q / self.dkr)) + self.NQ0

    def QWChi1(self, lam: float, dky: float, Ee: _dp_array, Eh: _dp_array,
               area: float, geh: float, dcv: complex) -> complex:
        """Calculates linear susceptibility."""
        # Simplified implementation
        return 0.0


# Global instance for backward compatibility
_sbe_instance = None

def InitializeSBE(q: _dp_array, rr: _dp_array, r0: float, Emaxxx: float,
                  lam: float, Nw: int, QW: bool) -> None:
    """Interface function for InitializeSBE."""
    global _sbe_instance
    _sbe_instance = SBEs()
    _sbe_instance.InitializeSBE(q, rr, r0, Emaxxx, lam, Nw, QW)

def QWCalculator(Exx: _dc_array, Eyy: _dc_array, Ezz: _dc_array,
                Vrr: _dc_array, rr: _dp_array, q: _dp_array, dt: float,
                w: int, Pxx: _dc_array, Pyy: _dc_array, Pzz: _dc_array,
                Rho: _dc_array, DoQWP: bool, DoQWDl: bool) -> None:
    """Interface function for QWCalculator."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    _sbe_instance.QWCalculator(Exx, Eyy, Ezz, Vrr, rr, q, dt, w,
                              Pxx, Pyy, Pzz, Rho, DoQWP, DoQWDl)

def SBECalculator(Ex: _dc_array, Ey: _dc_array, Ez: _dc_array, Vr: _dc_array,
                 dt: float, Px: _dc_array, Py: _dc_array, Pz: _dc_array,
                 Re: _dc_array, Rh: _dc_array, WriteFields: bool, w: int) -> None:
    """Interface function for SBECalculator."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    _sbe_instance.SBECalculator(Ex, Ey, Ez, Vr, dt, Px, Py, Pz, Re, Rh, WriteFields, w)

def Relaxation(ne: _dc_array, nh: _dc_array, VC: _dp_array, E1D: _dp_array,
               Rsp: _dp_array, dt: float, w: int, writefields: bool) -> None:
    """Interface function for Relaxation."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    _sbe_instance.Relaxation(ne, nh, VC, E1D, Rsp, dt, w, writefields)

def dpdt(C: _dc_array, D: _dc_array, p: _dc_array, Heh: _dc_array,
         Hee: _dc_array, Hhh: _dc_array, GamE: _dp_array, GamH: _dp_array,
         OffP: _dc_array) -> _dc_array:
    """Interface function for dpdt."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    return _sbe_instance.dpdt(C, D, p, Heh, Hee, Hhh, GamE, GamH, OffP)

def dCdt(Cee: _dc_array, Dhh: _dc_array, Phe: _dc_array, Heh: _dc_array,
         Hee: _dc_array, Hhh: _dc_array, GamE: _dp_array, GamH: _dp_array,
         OffE: _dc_array) -> _dc_array:
    """Interface function for dCdt."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    return _sbe_instance.dCdt(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffE)

def dDdt(Cee: _dc_array, Dhh: _dc_array, Phe: _dc_array, Heh: _dc_array,
         Hee: _dc_array, Hhh: _dc_array, GamE: _dp_array, GamH: _dp_array,
         OffH: _dc_array) -> _dc_array:
    """Interface function for dDdt."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    return _sbe_instance.dDdt(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffH)

def CalcMeh(Ex: _dc_array, Ey: _dc_array, Ez: _dc_array, Meh: _dc_array) -> None:
    """Interface function for CalcMeh."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    _sbe_instance.CalcMeh(Ex, Ey, Ez, Meh)

def CalcWnn(q0: float, Vr: _dc_array, Wnn: _dc_array) -> None:
    """Interface function for CalcWnn."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    _sbe_instance.CalcWnn(q0, Vr, Wnn)

def Preparation(P: _dc_array, C: _dc_array, D: _dc_array, Ex: _dc_array,
               Ey: _dc_array, Ez: _dc_array, Vr: _dc_array, Heh: _dc_array,
               Hee: _dc_array, Hhh: _dc_array, VC: _dp_array, E1D: _dp_array,
               GamE: _dp_array, GamH: _dp_array, OffG: _dc_array,
               Rsp: _dp_array) -> None:
    """Interface function for Preparation."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    _sbe_instance.Preparation(P, C, D, Ex, Ey, Ez, Vr, Heh, Hee, Hhh,
                             VC, E1D, GamE, GamH, OffG, Rsp)

def CalcH(Meh: _dc_array, Wee: _dc_array, Whh: _dc_array, C: _dc_array,
          D: _dc_array, p: _dc_array, VC: _dp_array, Heh: _dc_array,
          Hee: _dc_array, Hhh: _dc_array) -> None:
    """Interface function for CalcH."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    _sbe_instance.CalcH(Meh, Wee, Whh, C, D, p, VC, Heh, Hee, Hhh)

def Checkout(P1: _dc_array, P2: _dc_array, C1: _dc_array, C2: _dc_array,
            D1: _dc_array, D2: _dc_array, w: int) -> Tuple:
    """Interface function for Checkout."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    return _sbe_instance.Checkout(P1, P2, C1, C2, D1, D2, w)

def Checkin(P1: _dc_array, P2: _dc_array, P3: _dc_array, C1: _dc_array,
           C2: _dc_array, C3: _dc_array, D1: _dc_array, D2: _dc_array,
           D3: _dc_array, w: int) -> None:
    """Interface function for Checkin."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    _sbe_instance.Checkin(P1, P2, P3, C1, C2, C3, D1, D2, D3, w)

def QWArea() -> float:
    """Returns QW area."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    return _sbe_instance.area

def ShutOffOptics() -> None:
    """Shuts off optics calculations."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    _sbe_instance.Optics = False

def chiqw() -> complex:
    """Returns linear susceptibility."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    return _sbe_instance.chiw

def getqc() -> float:
    """Returns qc parameter."""
    if _sbe_instance is None:
        raise RuntimeError("InitializeSBE must be called first")
    return _sbe_instance.qc

# Additional utility functions
def FermiDistr(En: Union[float, _dp_array]) -> Union[complex, _dc_array]:
    """Fermi distribution function."""
    if isinstance(En, (int, float)):
        if En < 0:
            return 1.0
        else:
            return 0.0
    else:
        result = np.zeros_like(En, dtype=_dc)
        result[En < 0] = 1.0
        return result

def example_usage():
    """Example usage of the SBE module."""
    # Create arrays
    Nk = 100
    Nw = 1
    Nr = 200

    q = np.linspace(-1e7, 1e7, Nk)
    rr = np.linspace(-1e-6, 1e-6, Nr)

    # Initialize SBE
    InitializeSBE(q, rr, 0.0, 1e6, 800e-9, Nw, True)

    # Example field arrays
    Exx = np.zeros(Nr, dtype=_dc)
    Eyy = np.zeros(Nr, dtype=_dc)
    Ezz = np.zeros(Nr, dtype=_dc)
    Vrr = np.zeros(Nr, dtype=_dc)
    Pxx = np.zeros(Nr, dtype=_dc)
    Pyy = np.zeros(Nr, dtype=_dc)
    Pzz = np.zeros(Nr, dtype=_dc)
    Rho = np.zeros(Nr, dtype=_dc)

    # Run calculation
    DoQWP = True
    DoQWDl = True
    QWCalculator(Exx, Eyy, Ezz, Vrr, rr, q, 1e-15, 1,
                Pxx, Pyy, Pzz, Rho, DoQWP, DoQWDl)

    print("SBE calculation completed successfully!")

if __name__ == "__main__":
    example_usage()
