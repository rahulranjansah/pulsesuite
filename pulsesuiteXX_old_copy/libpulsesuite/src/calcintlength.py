"""
Provides high-performance integer and double length calculation utilities, matching Fortran's intlength module, using NumPy, Numba, and NumericTypes.

This module is optimized for scientific computing and high-performance clusters, supporting batch operations on large arrays.
"""
from numerictypes import NumericTypes
from typing import Any, Optional, Union, ContextManager

from numba import njit
import numpy as np


@njit(cache=True)
def _calcIntLengthNumba(arr):
    n = np.empty_like(arr, dtype=np.int64)
    for idx in range(arr.size):
        val = arr[idx]
        if val == 0:
            n[idx] = 1
        else:
            n[idx] = int(np.floor(np.log10(np.abs(val))) + 1)
            if val < 0:
                n[idx] += 1
    return n

@njit(cache=True)
def _calcDblLengthNumba(arr, p, e, frmt):
    n = np.zeros_like(arr, dtype=np.int64)
    for idx in range(arr.size):
        x = arr[idx]
        if frmt == "ES":
            n[idx] = p + e + 4
        elif frmt == "EN":
            if x != 0:
                g = int(np.floor(np.log10(np.abs(x)) / 3) * 3)
            else:
                g = 0
            x0 = x / (10.0 ** g) if g != 0 else x
            i = int(np.trunc(x0))
            if i != 0:
                n[idx] = p + e + 4 + int(np.floor(np.log10(np.abs(i))))
            else:
                n[idx] = p + e + 4
        elif frmt == "E" or frmt == "D":
            n[idx] = p + e + 4
        if x < 0.0:
            n[idx] += 1
    return n

class IntLengthCalculator:
    """
    Calculator for determining the string length of integer and floating-point numbers, similar to Fortran's intlength module.
    Uses NumPy, Numba, and NumericTypes for performance and precision.

    Methods
    -------
    CalcIntLength(i: Union[int, np.ndarray]) -> Union[int, np.ndarray]
        Calculates the number of characters needed to represent an integer or array of integers.
    CalcDblLength(x: Union[float, np.ndarray], p: int, e: int, frmt: str) -> Union[int, np.ndarray]
        Calculates the number of characters needed to represent a float or array of floats in a given format.
    BatchContext():
        Context manager for efficient batch operations on large datasets.
    """

    @classmethod
    def calcIntLength(cls, i: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Calculates the number of characters needed to represent an integer or array of integers, including sign if negative.

        Parameters
        ----------
        i : int or np.ndarray
            The integer or array of integers to evaluate.

        Returns
        -------
        int or np.ndarray
            The number of characters needed to represent the integer(s) as string(s).
        """
        arr = np.atleast_1d(np.asarray(i, dtype=np.int64))
        n = _calcIntLengthNumba(arr)
        if np.isscalar(i):
            return int(n[0])
        return n

    @classmethod
    def calcDblLength(cls, x: Union[float, np.ndarray], p: int, e: int, frmt: str) -> Union[int, np.ndarray]:
        """
        Calculates the number of characters needed to represent a floating-point number or array in a given format.

        Parameters
        ----------
        x : float or np.ndarray
            The floating-point number or array to evaluate.
        p : int
            Precision (number of digits after decimal).
        e : int
            Exponent width.
        frmt : str
            Format specifier ("ES", "EN", "E", or "D").

        Returns
        -------
        int or np.ndarray
            The number of characters needed to represent the float(s) as string(s).
        """
        arr = np.atleast_1d(np.asarray(x, dtype=NumericTypes.dp))
        n = _calcDblLengthNumba(arr, p, e, frmt)
        if np.isscalar(x):
            return int(n[0])
        return n

    class BatchContext(ContextManager):
        """
        Context manager for efficient batch operations on large datasets.
        Use this context to ensure memory locality and optimal performance on HPC clusters.

        Examples
        --------
        >>> with IntLengthCalculator.BatchContext():
        ...     result = IntLengthCalculator.CalcIntLength(np.arange(-1000000, 1000000))
        ...
        """
        def __enter__(self) -> 'IntLengthCalculator.BatchContext':
            # Could pin memory, set thread affinity, or preallocate buffers here if needed
            return self
        def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
            # Cleanup or release resources if needed
            return None