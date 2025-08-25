"""
helpers.py

Object-oriented conversion of the Fortran `helpers` module into Python classes.
Provides vectorized utilities optimized for HPC workloads, with optional Numba JIT compilation.
"""
# Standard imports
import numpy as np
from math import factorial as _fact

# Optional performance imports
try:
    from numba import njit, prange
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False

# Local imports
from constants import Constants

eps0 = Constants.eps0
c0 = Constants.c0
pi = Constants.pi

# Helper decorator factory
def _jit(fn):
    """
    Decorator: apply Numba JIT if available, otherwise return original function.

    Parameters
    ----------
    fn : Callable
        Function to decorate.

    Returns
    -------
    Callable
        JIT-compiled function if Numba is present, else original.
    """
    return njit(fn) if _USE_NUMBA else fn

# Derived constant
_EC2 = 2.0 * eps0 * c0

# Intensity and Field Conversions
class Intensity:
    @staticmethod
    def amp_to_inten(e: float, n0: float = 1.0) -> float:
        """
        Convert real electric field amplitude to intensity in a medium.

        Parameters
        ----------
        e : float
            Electric field amplitude.
        n0 : float, optional
            Refractive index (default is 1.0).

        Returns
        -------
        float
            Intensity corresponding to the amplitude.
        """
        return n0 * _EC2 * e**2

    @staticmethod
    def fld_to_inten(e: np.ndarray, n0: float = 1.0) -> float:
        """
        Convert complex field amplitude to intensity in a medium.

        Parameters
        ----------
        e : array_like of complex
            Complex field amplitude(s).
        n0 : float, optional
            Refractive index (default is 1.0).

        Returns
        -------
        float
            Intensity corresponding to the field amplitude.
        """
        return n0 * _EC2 * np.abs(e)**2

    @staticmethod
    def inten_to_amp(inten: float, n0: float = 1.0) -> float:
        """
        Convert intensity back to real electric field amplitude.

        Parameters
        ----------
        inten : float
            Intensity value.
        n0 : float, optional
            Refractive index (default is 1.0).

        Returns
        -------
        float
            Electric field amplitude corresponding to the intensity.
        """
        return np.sqrt(inten / (_EC2 * n0))

class MathOps:
    @staticmethod
    def arg_val(Z: complex) -> float:
        """
        Compute the argument (angle) of a complex number.

        Parameters
        ----------
        Z : complex
            Input complex number.

        Returns
        -------
        float
            Phase angle in radians.
        """
        return np.angle(Z)

    @staticmethod
    def sech(t: np.ndarray) -> np.ndarray:
        """
        Compute the hyperbolic secant elementwise: sech(t) = 1 / cosh(t).

        Parameters
        ----------
        t : array_like
            Input array.

        Returns
        -------
        ndarray
            sech of each element.
        """
        return 1.0 / np.cosh(t)

    @staticmethod
    def gauss(x: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian function elementwise: exp(-x**2).

        Parameters
        ----------
        x : array_like
            Input array.

        Returns
        -------
        ndarray
            Gaussian of each element.
        """
        return np.exp(-x**2)

    @staticmethod
    def magsq(Z: np.ndarray) -> np.ndarray:
        """
        Compute magnitude squared elementwise: |Z|^2.

        Parameters
        ----------
        Z : array_like of complex
            Input complex array.

        Returns
        -------
        ndarray
            Magnitude squared of each element.
        """
        return np.real(Z)**2 + np.imag(Z)**2

    @staticmethod
    def constrain(x: np.ndarray, low: float, high: float) -> np.ndarray:
        """
        Constrain values in an array between two limits.

        Parameters
        ----------
        x : array_like
            Input array.
        low : float
            Lower bound.
        high : float
            Upper bound.

        Returns
        -------
        ndarray
            Array with values clipped to [low, high].
        """
        return np.clip(x, low, high)

class Transforms:
    @staticmethod
    def l2f(lam: float) -> float:
        """
        Convert wavelength to frequency: f = c0 / λ.

        Parameters
        ----------
        lam : float
            Wavelength in meters.

        Returns
        -------
        float
            Frequency in Hz.
        """
        return c0 / lam

    @staticmethod
    def l2w(lam: float) -> float:
        """
        Convert wavelength to angular frequency: ω = 2πc0 / λ.

        Parameters
        ----------
        lam : float
            Wavelength in meters.

        Returns
        -------
        float
            Angular frequency in rad/s.
        """
        return 2.0 * pi * c0 / lam

    @staticmethod
    def w2l(w: float) -> float:
        """
        Convert angular frequency to wavelength: λ = 2πc0 / ω.

        Parameters
        ----------
        w : float
            Angular frequency in rad/s.

        Returns
        -------
        float
            Wavelength in meters.
        """
        return 2.0 * pi * c0 / w

class Grid:
    @staticmethod
    def get_space_array(N: int, length: float) -> np.ndarray:
        """
        Generate an N-point spatial array centered at zero.

        Parameters
        ----------
        N : int
            Number of points.
        length : float
            Total spatial length.

        Returns
        -------
        ndarray
            Points from -length/2 to length/2 inclusive.
        """
        return np.linspace(-length/2.0, length/2.0, N)

    @staticmethod
    def get_k_array(N: int, length: float) -> np.ndarray:
        """
        Generate FFT wavevector array for FFT-based transforms.

        Parameters
        ----------
        N : int
            Number of points.
        length : float
            Total spatial length.

        Returns
        -------
        ndarray
            Wavevector values suitable for FFT output.
        """
        # dk = 2.0 * np.pi / length
        # k = np.empty(N, dtype=float)
        # half = N // 2
        # for i in range(N):
        #     if i < half:
        #         k[i] = i * dk
        #     elif i == half:
        #         k[i] = -half * dk
        #     else:
        #         k[i] = (i - N) * dk
        # return k
        dk = 2*np.pi/length
        i = np.arange(N)
        half = N//2

        # build [0..half-1, -half, -(half-1)..-1]
        k = np.where(
            i < half,
            i,
            np.where(i == half, -half, i - N)
        )
        return k * dk

class Interpolator:
    @staticmethod
    def linear(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        """
        Perform 1D linear interpolation for real arrays.

        Parameters
        ----------
        x : float
            Point to interpolate.
        xp : ndarray
            Known x-coordinates.
        fp : ndarray
            Known y-values at xp.

        Returns
        -------
        float
            Interpolated value at x.
        """
        return np.interp(x, xp, fp)

    @staticmethod
    def linear_complex(x: float, xp: np.ndarray, fp: np.ndarray) -> complex:
        """
        Perform 1D linear interpolation for complex arrays.

        Parameters
        ----------
        x : float
            Point to interpolate.
        xp : ndarray
            Known x-coordinates.
        fp : ndarray of complex
            Known complex values at xp.

        Returns
        -------
        complex
            Interpolated complex value at x.
        """
        r = np.interp(x, xp, np.real(fp))
        i = np.interp(x, xp, np.imag(fp))
        return r + 1j * i

    @staticmethod
    def bilinear(x: float, y: float,
                 xp: np.ndarray, yp: np.ndarray,
                 fp: np.ndarray) -> float:
        """
        Perform 2D bilinear interpolation for real arrays.

        Parameters
        ----------
        x, y : float
            Coordinates to interpolate.
        xp : ndarray
            Known x-coordinates.
        yp : ndarray
            Known y-coordinates.
        fp : ndarray
            Known values on grid xp×yp.

        Returns
        -------
        float
            Interpolated value at (x, y).
        """
        i0 = yp.searchsorted(y) - 1
        i1 = i0 + 1
        f1 = np.interp(x, xp, fp[:, i0])
        f2 = np.interp(x, xp, fp[:, i1])
        y0, y1 = yp[i0], yp[i1]
        return (f1 * (y1 - y) + f2 * (y - y0)) / (y1 - y0)

    @staticmethod
    def bilinear_complex(x: float, y: float,
                          xp: np.ndarray, yp: np.ndarray,
                          fp: np.ndarray) -> complex:
        """
        Perform 2D bilinear interpolation for complex arrays.

        Parameters
        ----------
        x, y : float
            Coordinates to interpolate.
        xp : ndarray
            Known x-coordinates.
        yp : ndarray
            Known y-coordinates.
        fp : ndarray of complex
            Known complex values on grid xp×yp.

        Returns
        -------
        complex
            Interpolated complex value at (x, y).
        """
        i0 = yp.searchsorted(y) - 1
        i1 = i0 + 1
        c1 = Interpolator.linear_complex(x, xp, fp[:, i0])
        c2 = Interpolator.linear_complex(x, xp, fp[:, i1])
        y0, y1 = yp[i0], yp[i1]
        return (c1 * (y1 - y) + c2 * (y - y0)) / (y1 - y0)

    @staticmethod
    def trilinear(x: float, y: float, z: float,
                  xp: np.ndarray, yp: np.ndarray, zp: np.ndarray,
                  fp: np.ndarray) -> float:
        """
        Perform 3D trilinear interpolation for real arrays.

        Parameters
        ----------
        x, y, z : float
            Coordinates to interpolate.
        xp, yp, zp : ndarray
            Known coordinate arrays.
        fp : ndarray
            Known values on grid xp×yp×zp.

        Returns
        -------
        float
            Interpolated value at (x, y, z).
        """
        j = zp.searchsorted(z)
        f1 = Interpolator.bilinear(x, y, xp, yp, fp[:, :, j-1])
        f2 = Interpolator.bilinear(x, y, xp, yp, fp[:, :, j])
        z0, z1 = zp[j-1], zp[j]
        return (f1 * (z1 - z) + f2 * (z - z0)) / (z1 - z0)

    @staticmethod
    def trilinear_complex(x: float, y: float, z: float,
                           xp: np.ndarray, yp: np.ndarray, zp: np.ndarray,
                           fp: np.ndarray) -> complex:
        """
        Perform 3D trilinear interpolation for complex arrays.

        Parameters
        ----------
        x, y, z : float
            Coordinates to interpolate.
        xp, yp, zp : ndarray
            Known coordinate arrays.
        fp : ndarray of complex
            Known complex values on grid xp×yp×zp.

        Returns
        -------
        complex
            Interpolated complex value at (x, y, z).
        """
        j = zp.searchsorted(z)
        c1 = Interpolator.bilinear_complex(x, y, xp, yp, fp[:, :, j-1])
        c2 = Interpolator.bilinear_complex(x, y, xp, yp, fp[:, :, j])
        z0, z1 = zp[j-1], zp[j]
        return (c1 * (z1 - z) + c2 * (z - z0)) / (z1 - z0)

class Stencils:
    @staticmethod
    @_jit
    def locator(x: np.ndarray, u: float) -> int:
        """
        Locate index i such that x[i] <= u < x[i+1].

        Parameters
        ----------
        x : ndarray
            Sorted input array.
        u : float
            Value to locate in x.

        Returns
        -------
        int
            Index i satisfying x[i] <= u < x[i+1].
        """
        lo, hi = 0, x.size - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if u < x[mid]:
                hi = mid
            else:
                lo = mid
        return lo

    @staticmethod
    def gradient(f: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute numerical derivative using central differences.

        Parameters
        ----------
        f : ndarray
            Function values.
        dt : float
            Time step.

        Returns
        -------
        ndarray
            Array of derivative values.
        """
        return np.gradient(f, dt, edge_order=2)


    @staticmethod
    @_jit
    def dfdt_index_real(f: np.ndarray, dt: float, k: int) -> float:
        """
        Five-point stencil derivative at a single index (real).

        Parameters
        ----------
        f : ndarray
            Function values.
        dt : float
            Time step.
        k : int
            Index at which to compute derivative.

        Returns
        -------
        float
            Derivative at index k.
        """
        n = f.size
        if 2 < k < n - 2:
            return (-f[k+2] + 8*f[k+1] - 8*f[k-1] + f[k-2]) / (12*dt)
        elif k == 0:
            return f[1] / (2*dt)
        elif k == 1:
            return (f[2] - f[0]) / (2*dt)
        elif k == n - 1:
            return -f[n-2] / (2*dt)
        elif k == n - 2:
            return (f[n-1] - f[n-3]) / (2*dt)
        else:
            return 0.0


    @staticmethod
    @_jit
    def dfdt_index_complex(f: np.ndarray, dt: float, k: int) -> complex:
        """
        Five-point stencil derivative at a single index (complex).

        Parameters
        ----------
        f : ndarray of complex
            Function values.
        dt : float
            Time step.
        k : int
            Index at which to compute derivative.

        Returns
        -------
        complex
            Derivative at index k.
        """
        real_der = Stencils.dfdt_index_real(f.real, dt, k)
        imag_der = Stencils.dfdt_index_real(f.imag, dt, k)
        return real_der + 1j*imag_der


    @staticmethod
    def dfdt1d_real(f: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply five-point stencil derivative over all indices (real).

        Parameters
        ----------
        f : ndarray
            Function values.
        dt : float
            Time step.

        Returns
        -------
        ndarray
            Array of derivatives.
        """
        n = f.size
        out = np.empty_like(f)
        out[2:n-2] = (-f[4:] + 8*f[3:n-1] - 8*f[1:n-3] + f[:n-4])/(12*dt)
        out[0]  = (f[1] - f[0])/(dt)
        out[1]  = (f[2] - f[0])/(2*dt)
        out[-2] = (f[-1] - f[-3])/(2*dt)
        out[-1] = (f[-1] - f[-2])/(dt)
        return out

    @staticmethod
    def dfdt1d_complex(f: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply five-point stencil derivative over all indices (complex).

        Parameters
        ----------
        f : ndarray of complex
            Function values.
        dt : float
            Time step.

        Returns
        -------
        ndarray of complex
            Array of derivatives.
        """
        real = Stencils.dfdt1d_real(f.real, dt)
        imag = Stencils.dfdt1d_real(f.imag, dt)
        return real + 1j*imag

class Smoothers:
    @staticmethod
    @_jit
    def lax(u: np.ndarray, i: int, j: int, k: int) -> complex:
        """
        Lax smoothing: average of neighbor points in 3D grid.

        Parameters
        ----------
        u : ndarray
            3D array of values.
        i, j, k : int
            Grid indices.

        Returns
        -------
        complex
            Averaged value.
        """
        return (u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] +
                u[i,j+1,k] + u[i,j,k-1] + u[i,j,k+1]) / 6.0

    @staticmethod
    @_jit
    def no_lax(u: np.ndarray, i: int, j: int, k: int) -> complex:
        """
        Identity smoother: returns central point, matching Lax signature.

        Parameters
        ----------
        u : ndarray
            3D array of values.
        i, j, k : int
            Grid indices.

        Returns
        -------
        complex
            Central grid value.
        """
        return u[i,j,k]

class Utils:
    @staticmethod
    def unwrap_phase(phase: np.ndarray) -> np.ndarray:
        """
        Unwrap phase array to remove jumps of size 2π.

        Parameters
        ----------
        phase : ndarray
            Wrapped phase values.

        Returns
        -------
        ndarray
            Unwrapped phase array.
        """
        # return np.unwrap(phase)
        out = np.array(phase, dtype=float)
        two_pi = 2 * np.pi
        for i in range(1, out.size):
            if out[i] < out[i-1]:
                out[i:] += two_pi
        return out

    @staticmethod
    def factorial(n: int) -> int:
        """
        Compute factorial of an integer.

        Parameters
        ----------
        n : int
            Input integer.

        Returns
        -------
        int
            Factorial of n.
        """
        return _fact(n)

# - See libpulsesuite/future/helpers.txt for full task list.