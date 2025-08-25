"""
Numeric type aliases for high-performance scientific computing, matching Fortran kind parameters, using NumPy.
"""
from typing import Type
import numpy as np

class NumericTypes:
    """
    Provides high-performance numeric type aliases matching Fortran kind parameters using NumPy.

    Attributes:
        i8b (Type[np.int64]): 64-bit integer type.
        i4b (Type[np.int32]): 32-bit integer type.
        i2b (Type[np.int16]): 16-bit integer type.
        i1b (Type[np.int8]): 8-bit integer type.
        sp (Type[np.float32]): Single precision (32-bit) floating point type.
        dp (Type[np.float64]): Double precision (64-bit) floating point type.
        qp (Type[np.floating]): Quad precision (128-bit) floating point type, or double if unavailable.
    """
    i8b: Type[np.int64] = np.int64
    i4b: Type[np.int32] = np.int32
    i2b: Type[np.int16] = np.int16
    i1b: Type[np.int8] = np.int8
    sp: Type[np.float32] = np.float32
    dp: Type[np.float64] = np.float64
    try:
        qp: Type[np.floating] = np.float128
    except AttributeError:
        qp: Type[np.floating] = np.float64  # fallback if float128 is not available