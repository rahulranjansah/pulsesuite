import numpy as np
from typing import List, Optional, Tuple
from calcintlength import IntLengthCalculator
from numerictypes import NumericTypes

class Strings:
    """
    High-performance string utilities for scientific computing.

    Provides conversion between numbers and strings, word wrapping, trimming, and case conversion.
    All methods use camelCase. Context-enabled for future extensibility.
    """
    @staticmethod
    def int2str(i: int, n: Optional[int] = None) -> str:
        """
        Convert integer to string, optionally with fixed width.

        Parameters
        ----------
        i : int
            Integer to convert.
        n : int, optional
            Fixed width for output string.

        Returns
        -------
        str
            String representation of integer.
        """
        s = str(i)
        if n is not None:
            return s.rjust(n)
        return s

    @staticmethod
    def bool2str(x: bool, n: Optional[int] = None) -> str:
        """
        Convert boolean to string ('T' or 'F'), optionally with fixed width.

        Parameters
        ----------
        x : bool
            Boolean value.
        n : int, optional
            Fixed width for output string.

        Returns
        -------
        str
            String representation ('T' or 'F').
        """
        s = 'T' if x else 'F'
        if n is not None:
            return s.rjust(n)
        return s

    @staticmethod
    def dbl2str(x: float, p: int = 5, e: int = 3, frmt: str = 'ES') -> str:
        """
        Convert double to string using scientific format.

        Parameters
        ----------
        x : float
            Value to convert.
        p : int
            Precision.
        e : int
            Exponent length.
        frmt : str
            Format specifier ('ES', 'EN', 'E', 'D').

        Returns
        -------
        str
            String representation.
        """
        if frmt == 'ES':
            return f"{x:.{p}E}"
        elif frmt == 'EN':
            return f"{x:.{p}E}"
        elif frmt == 'E':
            return f"{x:.{p}E}"
        elif frmt == 'D':
            return f"{x:.{p}E}".replace('E', 'D')
        else:
            raise ValueError(f"Unknown format: {frmt}")

    @staticmethod
    def sgl2str(x: float, p: int = 5, e: int = 3, frmt: str = 'ES') -> str:
        """
        Convert single precision float to string using scientific format.
        """
        return Strings.dbl2str(float(x), p, e, frmt)

    @staticmethod
    def wordwrap(s: str, n: int = 80) -> List[str]:
        """
        Wrap text to lines of maximum length n.

        Parameters
        ----------
        s : str
            Input string.
        n : int
            Maximum line length.

        Returns
        -------
        List[str]
            List of wrapped lines.
        """
        import textwrap
        return textwrap.wrap(s, width=n)

    @staticmethod
    def trimb(x: str) -> str:
        """
        Trim spaces off both ends of a string.
        """
        return x.strip()

    @staticmethod
    def toupper(s: str) -> str:
        """
        Convert string to upper case.
        """
        return s.upper()

    @staticmethod
    def tolower(s: str) -> str:
        """
        Convert string to lower case.
        """
        return s.lower()

    @staticmethod
    def cmplx2str(z: complex, p: int = 5, e: int = 3) -> str:
        """
        Convert complex number to string using dbl2str for real and imaginary parts.
        """
        re_str = Strings.dbl2str(z.real, p, e)
        im_str = Strings.dbl2str(z.imag, p, e)
        return f"({re_str}, {im_str})"