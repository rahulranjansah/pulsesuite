import configparser
from pathlib import Path
import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from numba import njit, prange
from typing import Annotated, Optional, Tuple

from constants import Constants
from guardrails.guardrails import with_guardrails
from helpers import Transforms


w2l = Transforms.w2l
l2w = Transforms.l2w

c0 = Constants.c0
eps0 = Constants.eps0
pi = Constants.pi
twopi = Constants.twopi


# # get rid of these they are present in helpers in the hardisk
# def l2w(lam: float) -> float:
#     """Convert wavelength (m) to angular frequency (rad/s)."""
#     return twopi * c0 / lam


# def w2l(w: float) -> float:
#     """Convert angular frequency (rad/s) to wavelength (m)."""
#     return twopi * c0 / w

def n0_sellmeier(A: float,
                 B: np.ndarray,
                 C: np.ndarray,
                 lam: float
                 ) -> float:
    """
    Pure Sellmeier index formula:
        n0 = sqrt(sum(B * lam^2/(lam^2 - C)) + A)
    Clamped to 1.0 if NaN, <1.0, or >1e100.
    """
    val = np.sqrt(np.sum(B * lam**2 / (lam**2 - C)) + A)
    if np.isnan(val) or val < 1.0 or val > 1e100:
        return 1.0
    return float(val)

##############


class MaterialError(Exception):
    """Custom exception for material property errors."""
    pass


class MaterialProperties:
    """
    Port of the Fortran `materialproperties` module.

    Preserves all original routines and naming conventions.
    """

    # Error codes
    MAT_ERR_NOERROR     = 0
    MAT_ERR_NOFILE      = 1
    MAT_ERR_NOTFOUND    = 2
    MAT_ERR_FILE_FORMAT = 3
    MAT_ERR_OUTOFRANGE  = 4
    MAT_ERR_BADVALUE    = 5
    MAT_ERR_DEFAULTUSED = 6

    def __init__(self,
                 databasefile: str = "materials_py.ini"):
        """
        Initialize with a local database and a system default.

        Parameters
        ----------
        databasefile : str
            Path to the user materials database.
        """
        self._local_path = Path(databasefile)
        self._system_path = Path(Path("PKGDATADIR") / "materials_py.ini")

        # Load configs (but postpone exceptions until access)
        self._cfg_local = configparser.ConfigParser()
        self._cfg_system = configparser.ConfigParser()
        if self._local_path.exists():
            self._cfg_local.read(self._local_path)
        if self._system_path.exists():
            self._cfg_system.read(self._system_path)

    def _read_tag_str(self, mat: str, param: str) -> str:
        """
        Read a string parameter from local then system database.

        Raises MaterialError if neither exists or not found.
        """
        section = mat.upper()
        # Try local
        if self._local_path.exists() and self._cfg_local.has_option(section, param):
            return self._cfg_local.get(section, param)
        # Try system
        if self._system_path.exists() and self._cfg_system.has_option(section, param):
            return self._cfg_system.get(section, param)
        # Neither found
        raise MaterialError(f"[{section}] {param} not found in any database.")

    def _read_tag_array(self, mat: str, param: str) -> np.ndarray:
        """
        Read a comma-separated list of floats from INI and return as NumPy array.

        Raises MaterialError on parse errors or missing.
        """
        s = self._read_tag_str(mat, param)
        try:
            arr = np.fromstring(s, sep=',', dtype=np.float64)
        except ValueError as e:
            raise MaterialError(f"Failed to parse array '{param}' for material {mat}: {e}")
        return arr

    def _read_tag_val(self,
                      mat: str,
                      param: str
                      ) -> Tuple[float, int]:
        """
        Read a single real from the INI (for ReadDbsTagVal).
        Returns (value, err_code).
        """
        try:
            s = self._read_tag_str(mat, param)
            return float(s), self.MAT_ERR_NOERROR
        except MaterialError:
            return 0.0, self.MAT_ERR_NOTFOUND

    def _discrete_val(self,
                      mat: str,
                      param: str,
                      lam: Annotated[float, np.float64]
                      ) -> Tuple[float, int]:
        """
        Fallback lookup: choose nearest value within tolerances.

        Returns
        -------
        val : float
        err : int
            One of the MAT_ERR_* codes.
        """
        vals = self._read_tag_array(mat, param)
        lams = self._read_tag_array(mat, param + '-wavelength')
        if vals.size != lams.size:
            raise MaterialError(f"Mismatched array sizes for {param} in {mat}")

        rel = np.abs(lams - lam) / lam
        idx = np.argmin(rel)
        if rel[idx] < 0.01:
            return float(vals[idx]), self.MAT_ERR_NOERROR
        if rel[idx] < 0.05:
            return float(vals[idx]), self.MAT_ERR_OUTOFRANGE
        if np.min(lams) < 0.0:
            default_idx = np.argmin(lams)
            return float(vals[default_idx]), self.MAT_ERR_DEFAULTUSED
        return 0.0, self.MAT_ERR_NOTFOUND

    # Example public method port: alpha
    @with_guardrails
    def alpha(self,
              mat: str,
              lam: Annotated[float, np.float64],
              err: Optional[int] = None
              ) -> Annotated[float, np.float64]:
        """
        Linear absorption coefficient. Mirrors Fortran `alpha`.

        Parameters
        ----------
        mat : str
        lam : float, np.float64
        err : int, optional

        Returns
        -------
        float
        """
        try:
            A = self._read_tag_array(mat, 'Absorption')
            lams = self._read_tag_array(mat, 'Absorption-lams')
        except MaterialError:
            # fallback to discrete
            val, code = self._discrete_val(mat, 'alpha', lam)
            if err is not None:
                return val  # user inspects code separately
            else:
                raise

        if A.size != lams.size:
            raise MaterialError(f"Format error in alpha for {mat}")

        if lam < lams.min() or lam > lams.max():
            if err is not None:
                return 1e100
            else:
                raise MaterialError(f"Wavelength out of range for alpha in {mat}")

        # Spline interpolation
        cs = CubicSpline(lams, A)
        val = float(cs(lam))
        if val < 0:
            val = 0.0
            if err is None:
                warnings.warn(f"Negative alpha for {mat} at {lam}, clamped to zero.")
        return val

    @with_guardrails
    def beta(self, mat: str, lam: Annotated[float,np.float64], err: Optional[int]=None) -> float:
        return self._discrete_val(mat,'beta',lam)[0]

    def _sellmeier_coeff(self, mat: str, lam: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Read Sellmeier A, B, C from the INI and return (A, B, C).
        Warn if lam outside Sellmeier-limits.
        """
        # — scalar A
        try:
            A_str = self._read_tag_str(mat, 'Sellmeier-A')
            A = float(A_str)
        except MaterialError:
            A = 1.0

        # — arrays B and C
        B = self._read_tag_array(mat, 'Sellmeier-B')
        C = self._read_tag_array(mat, 'Sellmeier-C')
        if B.shape != C.shape:
            raise MaterialError(f"Sellmeier-B/C size mismatch for material '{mat}'")

        # — optional limits
        try:
            lim_str = self._read_tag_str(mat, 'Sellmeier-limits')
            low, high = [float(x) for x in lim_str.split(',')]
        except MaterialError:
            low, high = -np.inf, np.inf

        if not (low <= lam <= high):
            warnings.warn(f"Wavelength {lam:.2e} m is outside Sellmeier limits [{low:.2e}, {high:.2e}] for '{mat}'")

        return A, B, C

    @with_guardrails
    def dn_dl(self,
              mat: str,
              lam: float,
              err: Optional[int] = None
              ) -> float:
        A, B, C = self._sellmeier_coeff(mat, lam)
        # fast path: call the jitted reducer
        val = float(_sum_dn_dl(A, B, C, lam))
        return val

    @with_guardrails
    def ddn_dll(self,
               mat: str,
               lam: float,
               err: Optional[int] = None
               ) -> float:
        A, B, C = self._sellmeier_coeff(mat, lam)
        val = float(_sum_ddn_dll(A, B, C, lam))
        return val

    # @with_guardrails
    # def n0(self, mat: str, lam: Annotated[float,np.float64], err: Optional[int]=None) -> float:
    #     try:
    #         A,B,C = self._sellmeier_coeff(mat, lam)
    #         val = np.sqrt(np.sum(B*lam**2/(lam**2-C))+A)
    #         return float(val)
    #     except (MaterialError, NotImplementedError):
    #         return self._discrete_val(mat,'n0',lam)[0]
    @with_guardrails
    def n0(self,
           mat: str,
           lam: Annotated[float, np.float64],
           err: Optional[int] = None
           ) -> float:
        """
        Refractive index: tries Sellmeier, falls back to discrete lookup.
        """
        try:
            # read A, B, C from INI
            A, B, C = self._sellmeier_coeff(mat, lam)
            # pure Sellmeier formula
            val = n0_sellmeier(A, B, C, lam)
        except MaterialError:
            # fallback
            val, code = self._discrete_val(mat, 'n0', lam)
            if err is None and code != self.MAT_ERR_NOERROR:
                mat_error_handler(code, 'n0', mat, lam)
        else:
            # if we read Sellmeier but lam was out of limits, warn via err handler
            # (you could have _sellmeier_coeff set a flag for out-of-range)
            # for now we assume _sellmeier_coeff did its own warning
            pass

        return val

    @with_guardrails
    def index_of_refraction(self,
                            mat: str,
                            lam: Annotated[float, np.float64],
                            err: Optional[int] = None
                            ) -> float:
        """
        Alias for n0(): more Pythonic name for the refractive index.
        """
        return self.n0(mat, lam, err)

    @with_guardrails
    def n2I(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        val,code=self._discrete_val(mat,'n2I',lam)
        if code==self.MAT_ERR_NOTFOUND:
            f,cd=self._discrete_val(mat,'n2F',lam)
            if cd!=self.MAT_ERR_NOTFOUND:
                val=f/(eps0*c0*self.n0(mat,lam))
        return val

    @with_guardrails
    def n2F(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        val,code=self._discrete_val(mat,'n2F',lam)
        if code==self.MAT_ERR_NOTFOUND:
            i,cd=self._discrete_val(mat,'n2I',lam)
            if cd!=self.MAT_ERR_NOTFOUND:
                val=i*(eps0*c0*self.n0(mat,lam))
        return val

    @with_guardrails
    def Vp(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        return c0/self.n0(mat,lam)

    @with_guardrails
    def k0(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        return twopi*self.n0(mat,lam)/lam

    @with_guardrails
    def Vg(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        return 1.0/self.k1(mat,lam)

    @with_guardrails
    def k1(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        try:
            A,B,C=self._sellmeier_coeff(mat,lam)
            D=C/(twopi*c0)**2; w=twopi*c0/lam
            return float((_n0_w(A,B,D,w)+w*_dn_dw(A,B,D,w))/c0)
        except MaterialError:
            return self._discrete_val(mat,'k1',lam)[0]

    @with_guardrails
    def k2(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        try:
            A,B,C=self._sellmeier_coeff(mat,lam)
            D=C/(twopi*c0)**2; w=twopi*c0/lam
            return float((2*_dn_dw(A,B,D,w)+w*_ddn_dww(A,B,D,w))/c0)
        except MaterialError:
            return self._discrete_val(mat,'k2',lam)[0]

    @with_guardrails
    def k3(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        try:
            A,B,C=self._sellmeier_coeff(mat,lam)
            D=C/(twopi*c0)**2; w=twopi*c0/lam
            return float((3*_ddn_dww(A,B,D,w)+w*_dddn_dwww(A,B,D,w))/c0)
        except MaterialError:
            return self._discrete_val(mat,'k3',lam)[0]

    @with_guardrails
    def k4(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        try:
            A,B,C=self._sellmeier_coeff(mat,lam)
            D=C/(twopi*c0)**2; w=twopi*c0/lam
            return float((4*_dddn_dwww(A,B,D,w)+w*_ddddn_dwwww(A,B,D,w))/c0)
        except MaterialError:
            return self._discrete_val(mat,'k4',lam)[0]

    @with_guardrails
    def k5(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        try:
            A,B,C=self._sellmeier_coeff(mat,lam)
            D=C/(twopi*c0)**2; w=twopi*c0/lam
            return float((5*_ddddn_dwwww(A,B,D,w)+w*_dddddG_dwwwww(A,B,D,w))/c0)
        except MaterialError:
            return self._discrete_val(mat,'k5',lam)[0]

    @with_guardrails
    def k1_l(self, mat: str, lam: float, err: Optional[int]=None) -> float:
        A, B, C = self._sellmeier_coeff(mat, lam)
        # use the pure‐math helpers:
        val = ( n0_sellmeier(A, B, C, lam)
                - lam * dn_dl(A, B, C, lam) ) / c0
        return val  # plus your err‐handling as before


    @with_guardrails
    def k2_l(self, mat: str, lam: float, err: Optional[int]=None) -> float:
        A, B, C = self._sellmeier_coeff(mat, lam)
        val = lam**3 / (2.0 * pi * c0**2) * ddn_dll(A, B, C, lam)
        return val  # plus err‐handling


    # ##Fix this thing  sellmeier_coeff should return A,B,C
    # @with_guardrails
    # def GetKW(self,
    #           mat: str,
    #           W: Annotated[np.ndarray, np.float64],
    #           err: Optional[int] = None
    #           ) -> np.ndarray:
    #     """
    #     Fortran GetKW: Kw[i] = 2π * n0(λ[i]) / λ[i], with λ = w2l(W).
    #     """
    #     lams = w2l(W)   # numpy-vectored: twopi*c0/W
    #     mid  = lams.size // 2
    #     A, B, C = self._sellmeier_coeff(mat, lams[mid])
    #     if code != self.MAT_ERR_NOTFOUND:
    #         Kw = twopi * n0_sellmeier(A, B, C, np.abs(lams)) / lams
    #     else:
    #         Kw = np.zeros_like(W)
    #     if err is not None:
    #         return Kw
    #     if code != self.MAT_ERR_NOERROR:
    #         mat_error_handler(code, 'GetKW', mat, lams[mid])
    #     return Kw

    @with_guardrails
    def GetKW(self,
              mat: str,
              W: Annotated[np.ndarray, np.float64],
              err: Optional[int] = None
              ) -> np.ndarray:
        """
        Fortran GetKW: Kw[i] = 2π * n0(λ[i]) / λ[i], with λ = w2l(W).
        """
        lams = w2l(W)   # numpy-vectored: twopi*c0/W
        mid  = lams.size // 2
        A, B, C = self._sellmeier_coeff(mat, lams[mid])
        Kw = np.zeros_like(W)
        for i, lam in enumerate(lams):
            Kw[i] = twopi * n0_sellmeier(A, B, C, abs(lam)) / lam
        if err is not None:
            return Kw
        return Kw

    @with_guardrails
    def Tr(self,
           mat: str,
           lam: Annotated[float, np.float64],
           err: Optional[int] = None
           ) -> float:
        """
        Raman response: discrete lookup with default=0 if not found.
        """
        val, code = self._discrete_val(mat, 'Raman-tr', lam)
        if code == self.MAT_ERR_NOTFOUND:
            code = self.MAT_ERR_DEFAULTUSED
            val = 0.0
        if err is not None:
            return val
        if code != self.MAT_ERR_NOERROR:
            mat_error_handler(code, 'Raman-tr', mat, lam)
        return val


    # Plasma getters
    @with_guardrails
    def GetPlasmaElectronMass(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        return self._discrete_val(mat,'Plasma-mass',lam)[0]

    @with_guardrails
    def GetPlasmaBandGap(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        return self._discrete_val(mat,'Plasma-band_gap',lam)[0]

    @with_guardrails
    def GetPlasmaTrappingTime(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        return self._discrete_val(mat,'Plasma-trap_time',lam)[0]

    @with_guardrails
    def GetPlasmaCollisionTime(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        return self._discrete_val(mat,'Plasma-collision_time',lam)[0]

    @with_guardrails
    def GetPlasmaMaximumDensity(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        return self._discrete_val(mat,'Plasma-max_density',lam)[0]

    @with_guardrails
    def GetPlasmaOrder(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        return self._discrete_val(mat,'Plasma-multi_order',lam)[0]

    @with_guardrails
    def GetPlasmaCrossSection(self,mat:str,lam:Annotated[float,np.float64],err:Optional[int]=None)->float:
        return self._discrete_val(mat,'Plasma-multi_cross_section',lam)[0]


# CLI stubs (Python style would be to use argparse) (I used stub based over argparse)

_default_mp = MaterialProperties()

def MaterialOptions(opt: str) -> bool:
    """
    Parse one command-line option.
    Returns True if handled (i.e. started with --material-datafile=).
    """
    prefix = "--material-datafile="
    if opt.startswith(prefix):
        fn = opt[len(prefix):]
        p = Path(fn)
        if p.exists():
            # update the local INI path and reload
            _default_mp._local_path = p
            _default_mp._cfg_local.read(p)
        else:
            warnings.warn(f"Cannot find material database file: {fn}")
        return True
    return False

def MatHelp():
    """
    Print help for the material-datafile flag.
    """
    print("Material Options:")
    print("  --material-datafile=filename")
    print(f"     Select the local materials database file. "
          f"The system file   ({_default_mp._system_path!r}) will still be searched.")


def mat_error_handler(err:int,param:str,mat:str,lam:float):
    """Map error codes to warnings or exceptions."""
    if err==MaterialProperties.MAT_ERR_NOERROR: return
    msg=f"Error {err} on {param} for {mat} at {lam}"
    if err in (MaterialProperties.MAT_ERR_NOFILE,MaterialProperties.MAT_ERR_NOTFOUND):
        raise MaterialError(msg)
    warnings.warn(msg)

# Numba-elemental dispersion kernels
def _G(A: float, B: np.ndarray, D: np.ndarray, w: float) -> np.ndarray:
    """
    Elemental dispersion G(A,B,D,w) = B / (1 – D w²), vectorized.
    """
    return B / (1.0 - D * w * w)

def _dG_dw(A: float, B: np.ndarray, D: np.ndarray, w: float) -> np.ndarray:
    """
    ∂G/∂w = 2 D/B · w · G², with B==0 → 0
    """
    Gv = _G(A, B, D, w)
    return np.where(B == 0.0, 0.0, 2.0 * D / B * w * Gv * Gv)

@njit(parallel=True)
def _G_elem(B,D,w):
    return B/(1-D*w*w)

@njit(parallel=True)
def _dG_dw_elem(B,D,w):
    return 0.0 if B==0.0 else 2.0*D/B*w*_G_elem(B,D,w)**2

@njit(parallel=True)
def _ddG_dww_elem(B,D,w):
    g=_G_elem(B,D,w); dg=_dG_dw_elem(B,D,w)
    return 0.0 if g!=g else 2.0*D/B*(g*g+2.0*w*g*dg)

@njit(parallel=True)
def _dddG_dwww_elem(B,D,w):
    g=_G_elem(B,D,w); dg=_dG_dw_elem(B,D,w); ddg=_ddG_dww_elem(B,D,w)
    return 0.0 if B==0.0 else 4.0*D/B*(2.0*g*dg + w*(dg*dg+g*ddg))

@njit(parallel=True)
def _ddddG_dwwww_elem(B,D,w):
    dg=_dG_dw_elem(B,D,w); ddg=_ddG_dww_elem(B,D,w); dddg=_dddG_dwww_elem(B,D,w)
    return 4.0*D/B*(3.0*dg*dg+3.0*g*ddg+3.0*w*dg*ddg+w*g*dddg)

@njit(parallel=True)
def _dddddG_dwwwww_elem(B,D,w):
    dg=_dG_dw_elem(B,D,w); ddg=_ddG_dww_elem(B,D,w); dddg=_dddG_dwww_elem(B,D,w); ddddg=_ddddG_dwwww_elem(B,D,w)
    return 4.0*D/B*(12.0*dg*ddg+4.0*g*dddg+3.0*w*ddg*ddg+4.0*w*dg*dddg+w*g*ddddg)

@njit(parallel=True)
def _dddddG_dwwwww(A,B,D,w):
    # reuse element func
    return _sum_kernel(None,B,D,w,_dddddG_dwwwww_elem)


@njit(parallel=True)
def _sum_kernel(arr,B,D,w,func):
    s=0.0
    for i in prange(B.shape[0]): s+=func(B[i],D[i],w)
    return s

@njit(parallel=True)
def _n0_w(A,B,D,w): return np.sqrt(A+_sum_kernel(None,B,D,w,_G_elem))

@njit(parallel=True)
def _dn_dw(A,B,D,w): return _sum_kernel(None,B,D,w,_dG_dw_elem)/(2.0*_n0_w(A,B,D,w))

@njit(parallel=True)
def _ddn_dww(A,B,D,w):
    return (_sum_kernel(None,B,D,w,_ddG_dww_elem)/2.0 - _dn_dw(A,B,D,w)**2)/_n0_w(A,B,D,w)

@njit(parallel=True)
def _dddn_dwww(A,B,D,w):
    return (_sum_kernel(None,B,D,w,_dddG_dwww_elem)/2.0 - 3.0*_dn_dw(A,B,D,w)*_ddn_dww(A,B,D,w))/_n0_w(A,B,D,w)

@njit(parallel=True)
def _ddddn_dwwww(A,B,D,w):
    return (_sum_kernel(None,B,D,w,_ddddG_dwwww_elem)/2.0 - 3.0*_ddn_dww(A,B,D,w)**2 - 4.0*_dn_dw(A,B,D,w)*_dddn_dwww(A,B,D,w))/_n0_w(A,B,D,w)

@njit(parallel=True)
def _ddddd_n_dwwwww(A,B,D,w):
    return (_dddddG_dwwwww(A,B,D,w)/2.0 - 10.0*_ddn_dww(A,B,D,w)*_dddn_dwww(A,B,D,w) - 5.0*_dn_dw(A,B,D,w)*_ddddn_dwwww(A,B,D,w))/_n0_w(A,B,D,w)



@njit(parallel=True)
def _sum_dn_dl(A: float, B: np.ndarray, C: np.ndarray, lam: float) -> float:
    # compute Σ [Bᵢ*Cᵢ/(lam²−Cᵢ)²]
    s_bc = 0.0
    # also build Σ [Bᵢ*lam²/(lam²−Cᵢ)] for n₀
    s_g  = 0.0
    lam2 = lam*lam
    for i in prange(B.shape[0]):
        denom = lam2 - C[i]
        s_bc += B[i]*C[i]/(denom*denom)
        s_g  += B[i]*lam2/denom
    n0 = np.sqrt(s_g + A)
    return -lam / n0 * s_bc

@njit(parallel=True)
def _sum_ddn_dll(A: float, B: np.ndarray, C: np.ndarray, lam: float) -> float:
    # first get dn/dl and n₀
    dndl = _sum_dn_dl(A, B, C, lam)
    n0   = np.sqrt( (B*lam*lam/(lam*lam - C)).sum() + A )
    lam2 = lam*lam
    # compute Σ [Bᵢ*Cᵢ/(lam²−Cᵢ)³]
    s3 = 0.0
    for i in prange(B.shape[0]):
        denom = lam2 - C[i]
        s3   += B[i]*C[i]/(denom*denom*denom)
    return dndl/lam - dndl*dndl/n0 + 4.0*lam2/n0 * s3

@with_guardrails
def verify_G(mat: str, lam: Annotated[float, np.float64]) -> None:
    """
    Run finite-difference checks on the internal dispersion kernels,
    mirroring the Fortran verify_G subroutine.
    """
    mp = _default_mp

    # 1) grab Sellmeier coeffs A, B, C
    A, B, C = mp._sellmeier_coeff(mat, lam)
    # 2) build D array as in Fortran: D = C / (2π c0)^2
    D = C / (2 * np.pi * c0)**2

    diff = 1e-5
    w  = l2w(lam)
    w1 = w * (1 - diff)
    w2 = w * (1 + diff)
    lam1, lam2 = w2l(w1), w2l(w2)
    dw = w2 - w1

    # --- Table 1: G-kernels vs finite diff ---
    print(f"{'What':5s} {'Symbolic':20s} {'Finite Diff.':20s} {'Fractional Err.':20s}")
    # G1
    n1 = np.sum(_dG_dw(A, B, D, w))
    n2 = (np.sum(_G(A, B, D, w2)) - np.sum(_G(A, B, D, w1))) / dw
    print(f"{'G1':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")
    # G2
    n1 = np.sum(_ddG_dww_elem(A, B, D, w))
    n2 = (np.sum(_dG_dw(A, B, D, w2)) - np.sum(_dG_dw(A, B, D, w1))) / dw
    print(f"{'G2':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")
    # G3
    n1 = np.sum(_dddG_dwww_elem(A, B, D, w))
    n2 = (np.sum(_ddG_dww_elem(A, B, D, w2)) - np.sum(_ddG_dww_elem(A, B, D, w1))) / dw
    print(f"{'G3':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")
    # G4
    n1 = np.sum(_ddddG_dwwww_elem(A, B, D, w))
    n2 = (np.sum(_dddG_dwww_elem(A, B, D, w2)) - np.sum(_dddG_dwww_elem(A, B, D, w1))) / dw
    print(f"{'G4':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")
    # G5
    n1 = np.sum(_dddddG_dwwwww_elem(A, B, D, w))
    n2 = (np.sum(_ddddG_dwwww_elem(A, B, D, w2)) - np.sum(_ddddG_dwwww_elem(A, B, D, w1))) / dw
    print(f"{'G5':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")

    print()

    # --- Table 2: dn/dw etc. vs finite diff ---
    print(f"{'What':5s} {'Symbolic':20s} {'Finite Diff.':20s} {'Fractional Err.':20s}")
    # n1 = dn/dw
    n1 = _dn_dw(A, B, D, w)
    n2 = (_n0_w(A, B, D, w2) - _n0_w(A, B, D, w1)) / dw
    print(f"{'n1':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")
    # n2 = d²n/dω²
    n1 = _ddn_dww(A, B, D, w)
    n2 = (_dn_dw(A, B, D, w2) - _dn_dw(A, B, D, w1)) / dw
    print(f"{'n2':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")
    # n3 = d³n/dω³
    n1 = _dddn_dwww(A, B, D, w)
    n2 = (_ddn_dww(A, B, D, w2) - _ddn_dww(A, B, D, w1)) / dw
    print(f"{'n3':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")
    # n4 = d⁴n/dω⁴
    n1 = _ddddn_dwwww(A, B, D, w)
    n2 = (_dddn_dwww(A, B, D, w2) - _dddn_dwww(A, B, D, w1)) / dw
    print(f"{'n4':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")
    # n5 = d⁵n/dω⁵
    n1 = _ddddd_n_dwwwww(A, B, D, w)
    n2 = (_ddddn_dwwww(A, B, D, w2) - _ddddn_dwwww(A, B, D, w1)) / dw
    print(f"{'n5':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")

    print()

    # --- Table 3: new (w-domain) vs old (λ-domain) ---
    print(f"{'What':5s} {'New':20s} {'Old':20s} {'Fractional Err.':20s}")
    # n0
    n1 = _n0_w(A, B, D, w)
    n2 = n0_sellmeier(A, B, C, lam)
    print(f"{'n0':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")
    # k1
    n1 = mp.k1(mat, lam)
    n2 = mp.k1_l(mat, lam)
    print(f"{'k1':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")
    # k2
    n1 = mp.k2(mat, lam)
    n2 = mp.k2_l(mat, lam)
    print(f"{'k2':5s} {n1:20.10e} {n2:20.10e} {(n1 - n2)/n1:20.10e}")

