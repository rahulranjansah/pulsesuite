"""
Physical and mathematical constants for scientific computing, matching Fortran's constants.F90.

Optimized for high-performance and HPC use. All constants use NumPy types for speed and compatibility.
Fortran-specific file path constants (stdin, stdout, stderr) are omitted as Python uses sys.stdin, sys.stdout, sys.stderr.
"""
import numpy as np

class Constants:
    """
    Provides physical and mathematical constants for scientific computing.

    All constants are class attributes and use NumPy types for high performance.

    Attributes
    ----------
    pi : float
        The value of pi.
    const_e : float
        The value of Euler's number e.
    pio2 : float
        pi/2.
    twopi : float
        2*pi.
    sqrt2 : float
        Square root of 2.
    euler : float
        Euler-Mascheroni constant.
    ii : complex
        Imaginary unit (sqrt(-1)).
    c0 : float
        Speed of light in m/s.
    eps0 : float
        Vacuum permittivity (F/m).
    mu0 : float
        Vacuum permeability (H/m).
    hplank : float
        Planck constant (J*s).
    hbar : float
        Reduced Planck constant (J*s).
    e0 : float
        Elementary charge (C).
    eV : float
        Electron volt (J).
    me0 : float
        Electron rest mass (kg).
    efrmt, pfrmt, efrmt2x, efrmtA, pfrmtA, ifrmtA, Aefrmt : str
        Format strings for numeric display.
    """
    pi = np.float64(3.141592653589793238462643383279502884197)
    const_e = np.float64(2.7182818284590452353602874713527)
    pio2 = np.float64(1.57079632679489661923132169163975144209858)
    twopi = np.float64(6.283185307179586476925286766559005768394)
    sqrt2 = np.float64(1.41421356237309504880168872420969807856967)
    euler = np.float64(0.5772156649015328606065120900824024310422)
    ii = np.complex128(0.0 + 1.0j)
    c0 = np.float64(299792458.0)
    eps0 = np.float64(8.8541878176203898505365630317107e-12)
    mu0 = np.float64(1.2566370614359172953850573533118e-6)
    hplank = np.float64(6.62606876e-34)
    hbar = np.float64(1.05457159e-34)
    e0 = np.float64(1.60217733e-19)
    eV = np.float64(1.60217733e-19)
    me0 = np.float64(9.109534e-31)
    # Format strings
    efrmt = '(SP,1PE23.15E3,1X)'
    pfrmt = '(ES25.14E3)'
    efrmt2x = f'(2{efrmt})'
    efrmtA = f'({efrmt},A)'
    pfrmtA = f'({pfrmt},A)'
    ifrmtA = '((I25),A)'
    Aefrmt = f'(A,{efrmt})'

    # Fortran-specific file paths are not needed in Python.
    # stdin = '/dev/stdin'
    # stdout = '/dev/stdout'
    # stderr = '/dev/stderr'

    @classmethod
    def as_dict(cls):
        """
        Return all constants as a dictionary.

        Returns
        -------
        dict
            Dictionary of all constant names and values.
        """
        return {k: getattr(cls, k) for k in dir(cls) if not k.startswith('__') and not callable(getattr(cls, k))}



pi = np.float64(3.141592653589793238462643383279502884197)
const_e = np.float64(2.7182818284590452353602874713527)
pio2 = np.float64(1.57079632679489661923132169163975144209858)
twopi = np.float64(6.283185307179586476925286766559005768394)
sqrt2 = np.float64(1.41421356237309504880168872420969807856967)
euler = np.float64(0.5772156649015328606065120900824024310422)
ii = np.complex128(0.0 + 1.0j)
c0 = np.float64(299792458.0)
eps0 = np.float64(8.8541878176203898505365630317107e-12)
mu0 = np.float64(1.2566370614359172953850573533118e-6)
hplank = np.float64(6.62606876e-34)
hbar = np.float64(1.05457159e-34)
e0 = np.float64(1.60217733e-19)
eV = np.float64(1.60217733e-19)
me0 = np.float64(9.109534e-31)
