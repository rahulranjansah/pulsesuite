"""
High-performance numerical utilities for scientific computing, ported from Fortran's nrutils.F90.

1:1 port of the Numerical Recipes ``nrutils`` module. Every public Fortran
routine has a Python callable with the same camelCase name.  Vectorised with
NumPy; no Numba or guardrails required.

Fortran interfaces that dispatch on type/rank are replaced by a single Python
function that inspects shape and dtype at runtime.

Author: Rahul R. Sah
"""

import sys
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Fortran kind-parameter aliases  (SP = single, DP = double)
# ---------------------------------------------------------------------------
sp = np.float32
dp = np.float64


# ===================================================================
#  Array copy / swap utilities
#  Fortran: array_copy, swap_*, masked_swap_*
# ===================================================================

def arrayCopy(src: np.ndarray, dest: np.ndarray) -> Tuple[int, int]:
    """
    Copy elements from *src* into *dest* up to the smaller size.

    Mirrors Fortran's ``array_copy``.

    Parameters
    ----------
    src : ndarray
        Source array.
    dest : ndarray
        Destination array (modified in-place).

    Returns
    -------
    nCopied : int
        Number of elements actually copied.
    nNotCopied : int
        ``src.size - nCopied``.
    """
    nCopied = min(src.size, dest.size)
    nNotCopied = src.size - nCopied
    dest.flat[:nCopied] = src.flat[:nCopied]
    return nCopied, nNotCopied


def swap(a: Any, b: Any) -> Tuple[Any, Any]:
    """
    Swap two scalars, arrays, or any objects.

    Fortran's ``swap_*`` interfaces (scalar, vector, matrix) are handled
    by Python's generic assignment.

    Parameters
    ----------
    a, b : Any
        Values to swap.

    Returns
    -------
    (b, a) : tuple
        Swapped pair.
    """
    return b, a


def maskedSwap(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> None:
    """
    Swap elements of *a* and *b* where *mask* is True (in-place).

    Fortran's ``masked_swap_*`` interfaces.  Uses vectorised boolean
    indexing — no Python loop.

    Parameters
    ----------
    a, b : ndarray  (same shape, modified in-place)
    mask : ndarray of bool
        Boolean mask; swap only where True.
    """
    tmp = a[mask].copy()
    a[mask] = b[mask]
    b[mask] = tmp


def reallocate(arr: np.ndarray, newShape: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """
    Reallocate *arr* to *newShape*, copying the overlapping region.

    Fortran's ``reallocate_*`` family (1-D, 2-D, etc.).

    Parameters
    ----------
    arr : ndarray
        Original array.
    newShape : int or tuple of int
        Desired shape.

    Returns
    -------
    ndarray
        New array (same dtype) with data copied from the overlap.
    """
    newArr = np.empty(newShape, dtype=arr.dtype)
    slices = tuple(
        slice(0, min(o, n))
        for o, n in zip(arr.shape, newArr.shape)
    )
    newArr[slices] = arr[slices]
    return newArr


# ===================================================================
#  Index / location utilities (0-based)
#  Fortran: imaxloc, iminloc, ifirstloc
# ===================================================================

def imaxloc(arr: np.ndarray) -> int:
    """Index of the maximum value (0-based)."""
    return int(np.argmax(arr))


def iminloc(arr: np.ndarray) -> int:
    """Index of the minimum value (0-based)."""
    return int(np.argmin(arr))


def ifirstloc(mask: np.ndarray) -> int:
    """Index of the first True in a boolean array (0-based).

    Returns -1 when no True is found.
    """
    locs = np.flatnonzero(mask)
    if locs.size == 0:
        return -1
    return int(locs[0])


# ===================================================================
#  Assertion / error utilities
#  Fortran: nrerror, assert_eq2/3/4/n
# ===================================================================

def nrerror(msg: str) -> None:
    """
    Report a fatal error and stop (Fortran's ``nrerror``).

    Parameters
    ----------
    msg : str
        Error message printed to stderr.

    Raises
    ------
    SystemExit
    """
    print(f"nrerror: {msg}", file=sys.stderr)
    raise SystemExit(msg)


def assertTrue(
    test: bool,
    msg: str = "Assertion failed",
    file: Optional[str] = None,
    line: Optional[int] = None,
) -> None:
    """
    Assert *test* is ``True``; call ``nrerror`` on failure.

    Parameters
    ----------
    test : bool
    msg : str
    file, line : optional
        Source location for diagnostics.
    """
    if not test:
        loc = f"{file}:{line} : " if file is not None and line is not None else ""
        nrerror(f"{loc}{msg}")


def assertEq(*args: Any, msg: str = "Equality assertion failed") -> Any:
    """
    Assert all positional arguments are equal; return the common value.

    Replaces Fortran's ``assert_eq2``, ``assert_eq3``, ``assert_eq4``,
    and ``assert_eqn`` interfaces with a single variadic dispatcher.

    Parameters
    ----------
    *args
        Values that must be equal.
    msg : str
        Message on failure.

    Returns
    -------
    value
        The common value.
    """
    first = args[0]
    if all(a == first for a in args[1:]):
        return first
    nrerror(msg)


# ===================================================================
#  Progressions
#  Fortran: arth, geop, cumsum, cumprod
# ===================================================================

def arth(first: float, increment: float, n: int) -> np.ndarray:
    """
    Arithmetic progression of length *n*.

    $a_k = \\text{first} + k \\cdot \\text{increment},\\quad k = 0, \\dots, n-1$

    Fortran's ``arth`` (SP/DP/I4B interfaces).

    Parameters
    ----------
    first : float
        Starting value.
    increment : float
        Common difference.
    n : int
        Length.

    Returns
    -------
    ndarray of float64
    """
    return first + np.arange(n, dtype=np.float64) * increment


def geop(first: float, factor: float, n: int) -> np.ndarray:
    """
    Geometric progression of length *n*.

    $a_k = \\text{first} \\cdot \\text{factor}^k,\\quad k = 0, \\dots, n-1$

    Fortran's ``geop`` (SP/DP interfaces).

    Parameters
    ----------
    first : float
        Starting value.
    factor : float
        Common ratio.
    n : int
        Length.

    Returns
    -------
    ndarray of float64
    """
    return first * np.power(factor, np.arange(n, dtype=np.float64))


def cumsum(arr: np.ndarray, seed: Optional[Union[float, int]] = None) -> np.ndarray:
    """
    Cumulative sum, optionally offset by *seed*.

    $S_j = \\text{seed} + \\sum_{i=0}^{j} a_i$

    Fortran's ``cumsum`` (SP/DP interfaces).

    Parameters
    ----------
    arr : ndarray
    seed : float or int, optional
        Additive offset applied to the running total.

    Returns
    -------
    ndarray
    """
    out = np.cumsum(arr)
    if seed is not None:
        out = out + seed
    return out


def cumprod(arr: np.ndarray, seed: Optional[Union[float, int]] = None) -> np.ndarray:
    """
    Cumulative product, optionally scaled by *seed*.

    $P_j = \\text{seed} \\cdot \\prod_{i=0}^{j} a_i$

    Fortran's ``cumprod`` (SP/DP interfaces).

    Parameters
    ----------
    arr : ndarray
    seed : float or int, optional
        Multiplicative scale applied to all products.

    Returns
    -------
    ndarray
    """
    out = np.cumprod(arr)
    if seed is not None:
        out = out * seed
    return out


# ===================================================================
#  Polynomial evaluation
#  Fortran: poly, poly_term
# ===================================================================

def poly(x: Union[float, np.ndarray], coeffs: np.ndarray) -> Union[float, np.ndarray]:
    """
    Evaluate polynomial at *x* via Horner's method.

    $P(x) = c_0 + c_1 x + c_2 x^2 + \\cdots$

    Coefficients are ordered lowest-degree first (Fortran convention).
    Handles both real and complex coefficients/arguments.

    Fortran's ``poly`` (SP/DP/SPC/DPC interfaces).

    Parameters
    ----------
    x : float or ndarray
        Evaluation point(s).
    coeffs : ndarray
        Polynomial coefficients, lowest degree first.

    Returns
    -------
    float or ndarray
    """
    n = coeffs.size
    if n == 0:
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
    result = coeffs[-1]
    for i in range(n - 2, -1, -1):
        result = x * result + coeffs[i]
    return result


def polyTerm(a: np.ndarray, b: Union[float, int]) -> np.ndarray:
    """
    Recursive polynomial term (Fortran's ``poly_term``).

    $u_0 = a_0,\\quad u_j = a_j + b \\cdot u_{j-1}$

    Handles both real and complex arrays.

    Fortran's ``poly_term`` (SP/DP interfaces).

    Parameters
    ----------
    a : ndarray
        Coefficient array.
    b : float or int
        Multiplier.

    Returns
    -------
    ndarray
    """
    n = a.size
    if n == 0:
        return np.empty_like(a)
    u = np.empty_like(a)
    u[0] = a[0]
    for j in range(1, n):
        u[j] = a[j] + b * u[j - 1]
    return u


# ===================================================================
#  Outer operations  (vectorised via broadcasting / BLAS)
#  Fortran: outerprod, outerdiff, outersum, outerand
# ===================================================================

def outerprod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Outer product: $M_{ij} = a_i \\cdot b_j$.

    Uses ``np.outer`` (BLAS-backed for large arrays).
    Handles real and complex vectors.

    Fortran's ``outerprod`` (SP/DP interfaces).

    Parameters
    ----------
    a, b : ndarray (1-D)

    Returns
    -------
    ndarray, shape ``(a.size, b.size)``
    """
    return np.outer(a, b)


def outerdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Outer difference: $M_{ij} = a_i - b_j$.

    Vectorised via broadcasting.

    Fortran's ``outerdiff`` (SP/DP interfaces).

    Parameters
    ----------
    a, b : ndarray (1-D)

    Returns
    -------
    ndarray, shape ``(a.size, b.size)``
    """
    return a[:, np.newaxis] - b[np.newaxis, :]


def outersum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Outer sum: $M_{ij} = a_i + b_j$.

    Vectorised via broadcasting.

    Fortran's ``outersum`` (SP/DP interfaces).

    Parameters
    ----------
    a, b : ndarray (1-D)

    Returns
    -------
    ndarray, shape ``(a.size, b.size)``
    """
    return a[:, np.newaxis] + b[np.newaxis, :]


def outerand(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Outer logical AND: $M_{ij} = a_i \\wedge b_j$.

    Fortran's ``outerand``.

    Parameters
    ----------
    a, b : ndarray of bool (1-D)

    Returns
    -------
    ndarray of bool, shape ``(a.size, b.size)``
    """
    return a[:, np.newaxis] & b[np.newaxis, :]


# ===================================================================
#  Scatter operations  (0-based indexing)
#  Fortran: scatter_add, scatter_max
# ===================================================================

def scatterAdd(dest: np.ndarray, source: np.ndarray, destIndex: np.ndarray) -> None:
    """
    Scatter-add: ``dest[i] += source[j]`` at 0-based indices.

    Uses ``np.add.at`` for unbuffered accumulation (correct with
    duplicate indices).

    Fortran's ``scatter_add`` (SP/DP interfaces).

    Parameters
    ----------
    dest : ndarray  (modified in-place)
    source : ndarray
    destIndex : ndarray of int
        0-based target indices.
    """
    valid = (destIndex >= 0) & (destIndex < dest.size)
    np.add.at(dest, destIndex[valid], source[valid])


def scatterMax(dest: np.ndarray, source: np.ndarray, destIndex: np.ndarray) -> None:
    """
    Scatter-max: ``dest[i] = max(dest[i], source[j])`` at 0-based indices.

    Uses ``np.maximum.at`` for unbuffered reduction.

    Fortran's ``scatter_max`` (SP/DP interfaces).

    Parameters
    ----------
    dest : ndarray  (modified in-place)
    source : ndarray
    destIndex : ndarray of int
        0-based target indices.
    """
    valid = (destIndex >= 0) & (destIndex < dest.size)
    np.maximum.at(dest, destIndex[valid], source[valid])


# ===================================================================
#  Diagonal helpers
#  Fortran: diagadd, diagmult  (scalar & vector interfaces)
# ===================================================================

def _diagAddScalar(mat: np.ndarray, diag: float) -> None:
    """Add scalar to diagonal of *mat* in-place."""
    n = min(mat.shape[0], mat.shape[1])
    idx = np.arange(n)
    mat[idx, idx] += diag


def _diagAddVector(mat: np.ndarray, diagv: np.ndarray) -> None:
    """Add vector to diagonal of *mat* in-place."""
    n = min(mat.shape[0], mat.shape[1], diagv.size)
    idx = np.arange(n)
    mat[idx, idx] += diagv[:n]


def _diagMultScalar(mat: np.ndarray, diag: float) -> None:
    """Multiply diagonal of *mat* by scalar in-place."""
    n = min(mat.shape[0], mat.shape[1])
    idx = np.arange(n)
    mat[idx, idx] *= diag


def _diagMultVector(mat: np.ndarray, diagv: np.ndarray) -> None:
    """Multiply diagonal of *mat* by vector in-place."""
    n = min(mat.shape[0], mat.shape[1], diagv.size)
    idx = np.arange(n)
    mat[idx, idx] *= diagv[:n]


def diagAdd(mat: np.ndarray, diag: Union[float, np.ndarray]) -> None:
    """
    Add scalar or vector *diag* to the diagonal of *mat* in-place.

    Dispatcher replacing Fortran's ``diagadd`` interface (scalar / vector).

    Parameters
    ----------
    mat : ndarray, shape (M, N)  — modified in-place
    diag : float or ndarray
    """
    if isinstance(diag, (int, float, np.generic)):
        _diagAddScalar(mat, float(diag))
    else:
        _diagAddVector(mat, np.asarray(diag))


def diagMult(mat: np.ndarray, diag: Union[float, np.ndarray]) -> None:
    """
    Multiply the diagonal of *mat* by scalar or vector *diag* in-place.

    Dispatcher replacing Fortran's ``diagmult`` interface (scalar / vector).

    Parameters
    ----------
    mat : ndarray, shape (M, N)  — modified in-place
    diag : float or ndarray
    """
    if isinstance(diag, (int, float, np.generic)):
        _diagMultScalar(mat, float(diag))
    else:
        _diagMultVector(mat, np.asarray(diag))


def getDiag(mat: np.ndarray) -> np.ndarray:
    """
    Extract diagonal of *mat* into a new 1-D array.

    Fortran's ``get_diag`` (SP/DP interfaces).

    Parameters
    ----------
    mat : ndarray, shape (M, N)

    Returns
    -------
    ndarray, shape ``(min(M, N),)``
    """
    return np.diag(mat).copy()


def putDiag(diagv: np.ndarray, mat: np.ndarray) -> None:
    """
    Set the diagonal of *mat* to *diagv* in-place.

    Fortran's ``put_diag`` (SP/DP interfaces).

    Parameters
    ----------
    diagv : ndarray
    mat : ndarray, shape (M, N)  — modified in-place
    """
    n = min(mat.shape[0], mat.shape[1], diagv.size)
    idx = np.arange(n)
    mat[idx, idx] = diagv[:n]


# ===================================================================
#  Matrix utilities
#  Fortran: unit_matrix, upper_triangle, lower_triangle
# ===================================================================

def unitMatrix(mat: np.ndarray) -> None:
    """
    Set *mat* to the identity matrix in-place ($I_{ij} = \\delta_{ij}$).

    Fortran's ``unit_matrix``.

    Parameters
    ----------
    mat : ndarray, shape (M, N)  — modified in-place
    """
    mat[:] = 0.0
    np.fill_diagonal(mat, 1.0)


def upperTriangle(j: int, k: int, extra: int = 0) -> np.ndarray:
    """
    Float mask for the upper triangle of a ``(j, k)`` matrix.

    Entry $(r, c)$ is 1.0 where $r < c + \\text{extra}$, else 0.0.

    Fortran's ``upper_triangle``.

    Parameters
    ----------
    j : int — rows
    k : int — columns
    extra : int, optional
        Diagonal offset (default 0).

    Returns
    -------
    ndarray of float64, shape ``(j, k)``
    """
    return np.triu(np.ones((j, k), dtype=np.float64), 1 - extra)


def lowerTriangle(j: int, k: int, extra: int = 0) -> np.ndarray:
    """
    Float mask for the lower triangle of a ``(j, k)`` matrix.

    Entry $(r, c)$ is 1.0 where $r > c - \\text{extra}$, else 0.0.

    Fortran's ``lower_triangle``.

    Parameters
    ----------
    j : int — rows
    k : int — columns
    extra : int, optional
        Diagonal offset (default 0).

    Returns
    -------
    ndarray of float64, shape ``(j, k)``
    """
    return np.tril(np.ones((j, k), dtype=np.float64), -1 + extra)


# ===================================================================
#  Vector utilities
#  Fortran: vabs
# ===================================================================

def vabs(v: np.ndarray) -> float:
    """
    Euclidean norm $\\|v\\|_2 = \\sqrt{\\sum_i v_i^2}$.

    Uses ``np.linalg.norm`` (BLAS-backed).

    Fortran's ``vabs``.

    Parameters
    ----------
    v : ndarray

    Returns
    -------
    float
    """
    return float(np.linalg.norm(v))


# ===================================================================
#  Dummy Jacobians for ODE integrators (project-specific additions)
# ===================================================================

def dummy_jacobian_dp(
    x: float,
    y: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Zero-Jacobian placeholder for real (float64) ODE systems.

    Returns $(\\mathbf{0},\\; \\mathbf{0})$ with dimensions matching *y*.

    Parameters
    ----------
    x : float
        Independent variable (unused).
    y : ndarray of float64, shape (n,)
        State vector.

    Returns
    -------
    dfdx : ndarray of float64, shape (n,)
    dfdy : ndarray of float64, shape (n, n)
    """
    n = y.size
    return np.zeros(n, dtype=np.float64), np.zeros((n, n), dtype=np.float64)


def dummy_jacobian_dpc(
    x: float,
    y: NDArray[np.complex128],
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Zero-Jacobian placeholder for complex (complex128) ODE systems.

    Returns $(\\mathbf{0},\\; \\mathbf{0})$ with dimensions matching *y*.

    Parameters
    ----------
    x : float
        Independent variable (unused).
    y : ndarray of complex128, shape (n,)
        State vector.

    Returns
    -------
    dfdx : ndarray of complex128, shape (n,)
    dfdy : ndarray of complex128, shape (n, n)
    """
    n = y.size
    return np.zeros(n, dtype=np.complex128), np.zeros((n, n), dtype=np.complex128)


# Backward-compatible aliases (used by integrator tests)
_dummy_jacobian_dp_impl = dummy_jacobian_dp
_dummy_jacobian_dpc_impl = dummy_jacobian_dpc
