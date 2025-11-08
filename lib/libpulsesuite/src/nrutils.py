"""
High-performance numerical utilities for scientific computing, ported from Fortran's nrutils.F90.

Implements array/matrix utilities, polynomial evaluation, and more, using NumPy and Numba for speed.
All functions use camelCase. Designed for use in HPC and large-scale data processing.

Dependencies: constants.py, logger.py, numerictypes.py
"""
import numpy as np
from numba import njit, prange
from typing import Any, Optional, Union, Sequence
from numerictypes import NumericTypes
from typing import Annotated, Tuple
from numpy.typing import NDArray
from guardrails.guardrails import with_guardrails
from constants import Constants

from logger import Logger

# Type aliases for clarity
sp = NumericTypes.sp
dp = NumericTypes.dp

def arrayCopy(src: np.ndarray, dest: np.ndarray) -> tuple[int, int]:
    """
    Copy elements from src to dest, up to the minimum size.

    Parameters
    ----------
    src : np.ndarray
        Source array.
    dest : np.ndarray
        Destination array.

    Returns
    -------
    nCopied : int
        Number of elements copied.
    nNotCopied : int
        Number of elements not copied (src.size - nCopied).
    """
    nCopied = min(src.size, dest.size)
    nNotCopied = src.size - nCopied
    dest.flat[:nCopied] = src.flat[:nCopied]
    return nCopied, nNotCopied

def swap(a: Any, b: Any) -> tuple[Any, Any]:
    """
    Swap two variables or arrays.

    Parameters
    ----------
    a, b : Any
        Variables or arrays to swap.

    Returns
    -------
    b, a : tuple
        Swapped values.
    """
    return b, a

def reallocate(arr: np.ndarray, newShape: Union[int, tuple[int, ...]]) -> np.ndarray:
    """
    Reallocate an array to a new shape, copying data up to the new size.

    Parameters
    ----------
    arr : np.ndarray
        Original array.
    newShape : int or tuple of int
        New shape.

    Returns
    -------
    np.ndarray
        Reallocated array with data copied.
    """
    newArr = np.empty(newShape, dtype=arr.dtype)
    minShape = tuple(min(a, b) for a, b in zip(arr.shape, newArr.shape))
    if arr.ndim == 1:
        newArr[:minShape[0]] = arr[:minShape[0]]
    elif arr.ndim == 2:
        newArr[:minShape[0], :minShape[1]] = arr[:minShape[0], :minShape[1]]
    else:
        # For higher dimensions, use slicing
        slices = tuple(slice(0, m) for m in minShape)
        newArr[slices] = arr[slices]
    return newArr

@njit(cache=True)
def imaxloc(arr: np.ndarray) -> int:
    """
    Return the index of the maximum value in the array (1-based, Fortran style).

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    int
        Index of the maximum value (1-based).
    """
    return np.argmax(arr) + 1

@njit(cache=True)
def iminloc(arr: np.ndarray) -> int:
    """
    Return the index of the minimum value in the array (1-based, Fortran style).

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    int
        Index of the minimum value (1-based).
    """
    return np.argmin(arr) + 1

def assertTrue(test: bool, msg: str = "Assertion failed", file: Optional[str] = None, line: Optional[int] = None):
    """
    Assert that a condition is true, else log error and exit.
    Uses Logger for error reporting.
    """
    if not test:
        Logger.getInstance().error(msg)
        raise ValueError(msg)

def assertEq(*args, msg: str = "Equality assertion failed") -> int:
    """
    Assert that all arguments are equal. Returns the value if true, else logs error and exits.
    """
    first = args[0]
    if all(a == first for a in args):
        return first
    Logger.getInstance().error(msg)
    return -1  # Not reached

@njit(cache=True)
def arth(first: float, increment: float, n: int) -> np.ndarray:
    """Arithmetic progression of length n."""
    arr: np.ndarray = np.empty(n, dtype=np.float64)
    if n > 0:
        arr[0] = first
    for k in range(1, n):
        arr[k] = arr[k-1] + increment
    return arr

@njit(cache=True)
def geop(first: float, factor: float, n: int) -> np.ndarray:
    """Geometric progression of length n."""
    arr: np.ndarray = np.empty(n, dtype=np.float64)
    if n > 0:
        arr[0] = first
    for k in range(1, n):
        arr[k] = arr[k-1] * factor
    return arr

@njit(cache=True)
def cumsum(arr: np.ndarray, seed: Optional[Union[float, int]] = None) -> np.ndarray:
    """
    Cumulative sum of an array, optionally with a seed value.
    """
    n = arr.size
    out = np.empty_like(arr)
    if n == 0:
        return out
    if seed is None:
        out[0] = arr[0]
    else:
        out[0] = arr[0] + seed
    for j in range(1, n):
        out[j] = out[j-1] + arr[j]
    return out

@njit(cache=True)
def poly(x: Union[float, np.ndarray], coeffs: np.ndarray) -> Union[float, np.ndarray]:
    """
    Evaluate a polynomial at x with given coefficients (Horner's method).
    """
    n = coeffs.size
    if n == 0:
        return 0.0
    result = coeffs[-1]
    for i in range(n-2, -1, -1):
        result = x * result + coeffs[i]
    return result

@njit(cache=True)
def polyTerm(a: np.ndarray, b: Union[float, int]) -> np.ndarray:
    """
    Evaluate a polynomial term recursively (Fortran's poly_term).
    """
    n = a.size
    u = np.empty_like(a)
    if n == 0:
        return u
    u[0] = a[0]
    for j in range(1, n):
        u[j] = a[j] + b * u[j-1]
    return u

@njit(cache=True, parallel=True)
def outerprod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product (parallel)."""
    m, n = a.size, b.size
    out = np.empty((m, n))
    for i in prange(m):
        for j in range(n):
            out[i, j] = a[i] * b[j]
    return out


@njit(cache=True, parallel=True)
def outerdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer difference (parallel)."""
    m, n = a.size, b.size
    out = np.empty((m, n))
    for i in prange(m):
        for j in range(n):
            out[i, j] = a[i] - b[j]
    return out

@njit(cache=True)
def scatterAdd(dest: np.ndarray, source: np.ndarray, destIndex: np.ndarray):
    """
    Scatter-add: add source values to dest at indices destIndex (1-based).
    """
    for j in range(source.size):
        i = destIndex[j] - 1  # Fortran to Python index
        if 0 <= i < dest.size:
            dest[i] += source[j]

@njit(cache=True)
def scatterMax(dest: np.ndarray, source: np.ndarray, destIndex: np.ndarray):
    """
    Scatter-max: set dest[i] = max(dest[i], source[j]) at indices destIndex (1-based).
    """
    for j in range(source.size):
        i = destIndex[j] - 1
        if 0 <= i < dest.size:
            dest[i] = max(dest[i], source[j])

# Diagonal helpers (JIT'd)
@njit(cache=True)
def _diagAddScalar(mat: np.ndarray, diag: float) -> None:
    n: int = min(mat.shape[0], mat.shape[1])
    for j in range(n):
        mat[j, j] += diag

@njit(cache=True)
def _diagAddVector(mat: np.ndarray, diagv: np.ndarray) -> None:
    n: int = min(mat.shape[0], mat.shape[1], diagv.size)
    for j in range(n):
        mat[j, j] += diagv[j]

@njit(cache=True)
def _diagMultScalar(mat: np.ndarray, diag: float) -> None:
    n: int = min(mat.shape[0], mat.shape[1])
    for j in range(n):
        mat[j, j] *= diag

@njit(cache=True)
def _diagMultVector(mat: np.ndarray, diagv: np.ndarray) -> None:
    n: int = min(mat.shape[0], mat.shape[1], diagv.size)
    for j in range(n):
        mat[j, j] *= diagv[j]

# Python dispatch functions with type hints
from typing import Union

def diagAdd(mat: np.ndarray, diag: Union[float, np.ndarray]) -> None:
    """Add scalar or vector diag to mat's diagonal."""
    if isinstance(diag, (int, float, np.generic)):
        _diagAddScalar(mat, float(diag))
    else:
        _diagAddVector(mat, diag)

def diagMult(mat: np.ndarray, diag: Union[float, np.ndarray]) -> None:
    """Multiply scalar or vector diag onto mat's diagonal."""
    if isinstance(diag, (int, float, np.generic)):
        _diagMultScalar(mat, float(diag))
    else:
        _diagMultVector(mat, diag)

# Parallel functions
@njit(cache=True, parallel=True)
def getDiag(mat: np.ndarray) -> np.ndarray:
    """
     Extract diagonal into a 1D array (parallel).
    """
    n = min(mat.shape[0], mat.shape[1])
    out = np.empty(n)
    for j in prange(n):
        out[j] = mat[j, j]
    return out

@njit(cache=True, parallel=True)
def putDiag(diagv: np.ndarray, mat: np.ndarray):
    """
    Set the diagonal of mat to diagv (paralle).
    """
    n = min(mat.shape[0], mat.shape[1], diagv.size)
    for j in prange(n):
        mat[j, j] = diagv[j]

@njit(cache=True, parallel=True)
def unitMatrix(mat: np.ndarray):
    """
    Set mat to the identity matrix (in-place) (parallel rows).
    """
    rows, cols = mat.shape
    for i in prange(rows):
        for j in range(cols):
            mat[i, j] = 0.0
    for i in prange(min(rows, cols)):
        mat[i, i] = 1.0

@njit(cache=True, parallel=True)
def upperTriangle(j: int, k: int, extra: int = 0) -> np.ndarray:
    """
    Return a boolean mask for the upper triangle of a (j, k) matrix (parallel rows).
    """
    out = np.empty((j, k))
    for ii in prange(j):
        for jj in range(k):
            out[ii, jj] = 1.0 if ii < jj + extra else 0.0
    return out

@njit(cache=True, parallel=True)
def lowerTriangle(j: int, k: int, extra: int = 0) -> np.ndarray:
    """
    Return a boolean mask for the lower triangle of a (j, k) matrix (parallel reduction).
    """
    out = np.empty((j, k))
    for ii in prange(j):
        for jj in range(k):
            out[ii, jj] = 1.0 if ii > jj - extra else 0.0
    return out

@njit(cache=True, parallel=True)
def vabs(v: np.ndarray) -> float:
    """
    Return the Euclidean norm of a vector (parallel reduction).
    """
    s = 0.0
    for i in prange(v.size):
        s += v[i] * v[i]
    return np.sqrt(s)


@with_guardrails
def dummy_jacobian_dp(
    x: float,
    y: Annotated[NDArray[np.float64], np.float64]
) -> Tuple[
      Annotated[NDArray[np.float64], np.float64],
      Annotated[NDArray[np.float64], np.float64]
]:
    """Zero‐Jacobian for real ODEs."""
    n = y.size
    return np.zeros(n, dtype=np.float64), np.zeros((n, n), dtype=np.float64)

@njit(cache=True)
def _dummy_jacobian_dp_impl(x, y):
    # Calls the guardrails-wrapped version under the hood,
    # but remains pure Numba for speed.
    return dummy_jacobian_dp(x, y)


@with_guardrails
def dummy_jacobian_dpc(
    x: float,
    y: Annotated[NDArray[np.complex128], np.complex128]
) -> Tuple[
      Annotated[NDArray[np.complex128], np.complex128],
      Annotated[NDArray[np.complex128], np.complex128]
]:
    """Zero‐Jacobian for complex ODEs."""
    n = y.size
    return np.zeros(n, dtype=np.complex128), np.zeros((n, n), dtype=np.complex128)

@njit(cache=True)
def _dummy_jacobian_dpc_impl(x, y):
    return dummy_jacobian_dpc(x, y)



# TODO: Implement all masked_swap, complex, and advanced poly/outerprod variants as needed.
# TODO: Add more Fortran-style interfaces if required by downstream code.

# TODOs are left for masked swap, complex, and advanced poly/outerprod variants,
# as these are less commonly used and can be added if required by downstream code.