"""
Integrator module (Python version)
==================================
High-performance ODE integrators for scientific computing, ported from Fortran.

Dependencies: numpy, numba, logger, nrutils, constants, helpers

Implements: odeint, rkqs, rkck, rk4, idiot, 3D-versions and related routines.

Author: @Rahul_Ranjan_Sah
"""
# Standard imports
import numpy as np
from numba import njit, prange
from typing import Annotated, Callable, Tuple, Optional
from numpy.typing import NDArray

# Local imports
from logger import Logger
from nrutils import *
from guardrails.guardrails import with_guardrails

# Constants for adaptive step-size control
SAFETY = 0.9
PGROW = -0.20
PSHRNK = -0.25
ERRCON = (5.0 / SAFETY) ** (-5)
TINY = 1.0e-30
MAXSTP = 100_000_000
KMAXX = 8

# Numba JIT decorator helper
try:
    import numba
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False


def _jit(parallel: bool = False, cache: bool = True):
    def _decorator(fn):
        return njit(parallel=parallel, cache=cache)(fn) if _USE_NUMBA else fn
    return _decorator

@with_guardrails
def odeint_dp(
    y: Annotated[np.ndarray, np.float64],  # in/out solution vector
    x1: Annotated[float,    np.float64],    # start of interval
    x2: Annotated[float,    np.float64],    # end of interval
    eps: Annotated[float,   np.float64],    # error tolerance
    h1: Annotated[float,    np.float64],    # initial step size
    hmin: Annotated[float, np.float64],     # minimum allowed step size
    D: Callable[[float, np.ndarray], np.ndarray],
    integ: Callable[[np.ndarray, np.ndarray, float, float, float, np.ndarray, Callable, Callable], Tuple[float, float]],
    jacobn: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.int32, np.int32]:
    """
    Adaptive-step ODE driver for real double precision.
    """
    return _odeint_dp_impl(y, x1, x2, eps, h1, hmin, D, integ, jacobn)

@_jit()
def _odeint_dp_impl(
    y: Annotated[np.ndarray, np.float64],  # in/out solution vector
    x1: Annotated[float,    np.float64],    # start of interval
    x2: Annotated[float,    np.float64],    # end of interval
    eps: Annotated[float,   np.float64],    # error tolerance
    h1: Annotated[float,    np.float64],    # initial step size
    hmin: Annotated[float, np.float64],     # minimum allowed step size
    D: Callable[[float, np.ndarray], np.ndarray],
    integ: Callable[[np.ndarray, np.ndarray, float, float, float, np.ndarray, Callable, Callable], Tuple[float, float]],
    jacobn: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.int32, np.int32]:
    """
    Adaptive‐step ODE driver (double precision).

    Parameters
    ----------
    y     : ndarray[float64], modified in-place to hold y(x2)
    x1, x2: float64 start/end
    eps   : float64 desired local error
    h1    : float64 initial trial step
    hmin  : float64 minimum allowable step
    D     : function(x: float64, y: float64[:]) -> float64[:]
    integ : function(y, dydx, x, htry, eps, yscale, D, jacobn) -> (hdid, hnext)
    jacobn: function(x, y) -> (dfdx, dfdy)

    Returns
    -------
    tuple of (y, nok, nbad)
    """
    n = y.size
    yscale = np.empty(n, dtype=np.float64)
    dydx   = np.empty(n, dtype=np.float64)

    x   = x1
    h   = h1 if (x2 - x1) >= 0.0 else -h1
    nok = 0
    nbad = 0

    for _ in range(MAXSTP):
        # 1) compute derivative
        dydx = D(x, y)

        # 2) error‐scaling vector
        for i in range(n):
            yscale[i] = abs(y[i]) if abs(y[i]) > 1.0 else 1.0

        # 3) avoid overshooting x2
        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x

        # 4) take one adaptive step
        hdid, hnext = integ(y, dydx, x, h, eps, yscale, D, jacobn)

        # 5) tally good/bad
        if hdid == h:
            nok += 1
        else:
            nbad += 1

        # 6) advance independent variable
        x += hdid

        # 7) done?
        if (x - x2) * (x2 - x1) >= 0.0:
            return y, nok, nbad

        # 8) check for collapsing step
        if abs(hnext) <= abs(hmin):
            # Logger.getInstance().error("Step size too small in odeint_dp.", 1)
            raise ValueError("Step size too small in odeint_dp.")

        # 9) next step size
        h = hnext

    # too many steps
    # Logger.getInstance().error("Too many steps in odeint_dp.", 1)
    raise ValueError("Too many steps in odeint_dp.")

@with_guardrails
def odeint_dpc(
    y: Annotated[np.ndarray, np.complex128],  # in/out solution vector
    x1: Annotated[float,    np.float64],    # start of interval
    x2: Annotated[float,    np.float64],    # end of interval
    eps: Annotated[float,   np.float64],    # error tolerance
) -> Tuple[np.ndarray, np.int32, np.int32]:
    """
    Adaptive-step ODE driver for complex double precision.
    """
    return _odeint_dpc_impl(y, x1, x2, eps, h1, hmin, D, integ, jacobn)

@_jit()
def _odeint_dpc_impl(
    y: Annotated[np.ndarray, np.complex128],  # in/out solution vector
    x1: Annotated[float,    np.float64],    # start of interval
    x2: Annotated[float,    np.float64],    # end of interval
    eps: Annotated[float,   np.float64],    # error tolerance
    h1: Annotated[float,    np.float64],    # initial step size
    hmin: Annotated[float, np.float64],     # minimum allowed step size
    D: Callable[[float, np.ndarray], np.ndarray],
    integ: Callable[[np.ndarray, np.ndarray, float, float, float, np.ndarray, Callable, Callable], Tuple[float, float]],
    jacobn: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.int32, np.int32]:
    """
    Adaptive-step ODE driver for complex double precision.

    Parameters
    ----------
    y     : ndarray[complex128]
        On entry: y at x1; on exit: y at x2 (in-place).
    x1, x2: float64
        Interval endpoints.
    eps   : float64
        Desired local error tolerance.
    h1    : float64
        Initial trial step size.
    hmin  : float64
        Minimum allowable step size.
    D     : Callable[[float, ndarray], ndarray]
        Complex derivative function.
    integ : Callable[[ndarray, ndarray, float, float, float, ndarray, Callable, Callable], Tuple[float, float]]
        Single-step integrator.
    jacobn: Callable[[float, ndarray], Tuple[ndarray, ndarray]]
        Jacobian evaluator.
    """
    n = y.size
    yscale = np.empty(n, dtype=np.float64)
    dydx   = np.empty(n, dtype=np.float64)

    x   = x1
    h   = h1 if (x2 - x1) >= 0.0 else -h1
    nok = 0
    nbad = 0

    for _ in range(MAXSTP):
        # 1) compute derivative
        dydx = D(x, y)

        # 2) error‐scaling vector
        for i in range(n):
            yscale[i] = abs(y[i]) if abs(y[i]) > 1.0 else 1.0

        # 3) avoid overshooting x2
        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x

        # 4) take one adaptive step
        hdid, hnext = integ(y, dydx, x, h, eps, yscale, D, jacobn)

        # 5) tally good/bad
        if hdid == h:
            nok += 1
        else:
            nbad += 1

        # 6) advance independent variable
        x += hdid

        # 7) done?
        if (x - x2) * (x2 - x1) >= 0.0:
            return y, nok, nbad

        # 8) check for collapsing step
        if abs(hnext) <= abs(hmin):
            # Logger.getInstance().error("Step size too small in odeint_dp.", 1)
            raise ValueError("Step size too small in odeint_dp.")
        # 9) next step size
        h = hnext

    # too many steps
    # Logger.getInstance().error("Too many steps in odeint_dp.", 1)
    raise ValueError("Too many steps in odeint_dp.")

# jacobians
@with_guardrails
def dummy_jacobian_dp(
    x: Annotated[float, np.float64],
    y: Annotated[np.ndarray, np.float64]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dummy jacobian for real ODEs: returns zero dfdx and zero dfdy.
    """
    return _dummy_jacobian_dp_impl(x, y)

@_jit(cache=True)
def _dummy_jacobian_dp_impl(
    x: Annotated[float, np.float64],
    y: Annotated[np.ndarray, np.float64]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dummy jacobian for real ODEs: returns zero dfdx and zero dfdy.

    Parameters
    ----------
    x : float64
        Independent variable (unused).
    y : ndarray[float64]
        Dependent variable vector.

    Returns
    -------
    dfdx : ndarray[float64]
        Zero vector of same shape as y.
    dfdy : ndarray[float64,2]
        Zero matrix of shape (n, n).
    """
    n = y.size
    dfdx = np.zeros(n, dtype=np.float64)
    dfdy = np.zeros((n, n), dtype=np.float64)
    return dfdx, dfdy

@with_guardrails
def dummy_jacobian_dpc(
    x: Annotated[float, np.float64],
    y: Annotated[np.ndarray, np.complex128]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dummy jacobian for complex ODEs: returns zero dfdx and zero dfdy.
    """
    return _dummy_jacobian_dpc_impl(x, y)

@_jit()
def _dummy_jacobian_dpc_impl(
    x: Annotated[float, np.float64],
    y: Annotated[np.ndarray, np.complex128]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dummy jacobian for complex ODEs: returns zero dfdx and zero dfdy.

    Parameters
    ----------
    x : float64
        Independent variable (unused).
    y : ndarray[complex128]
        Dependent variable vector.

    Returns
    -------
    dfdx : ndarray[complex128]
        Zero vector of same shape as y.
    dfdy : ndarray[complex128,2]
        Zero matrix of shape (n, n).
    """
    n = y.size
    dfdx = np.zeros(n, dtype=np.complex128)
    dfdy = np.zeros((n, n), dtype=np.complex128)
    return dfdx, dfdy

# simple integrators
@with_guardrails
def simpleint_dp(
    y: Annotated[np.ndarray, np.complex128],
    t1: Annotated[float, np.float64],
    t2: Annotated[float, np.float64],
    h1: Annotated[float, np.float64],
    D: Callable[[float, np.ndarray], np.ndarray],
    simple: Callable[[np.ndarray, np.ndarray, float, float, Callable], Tuple[np.ndarray, float]]
) -> Tuple[np.ndarray, np.int32]:
    """
    Extremely simple integrator (Euler step) for complex ODEs.
    """
    return _simpleint_dp_impl(y, t1, t2, h1, D, simple)

@_jit()
def _simpleint_dp_impl(
    y: Annotated[np.ndarray, np.float64],
    t1: Annotated[float, np.float64],
    t2: Annotated[float, np.float64],
    h1: Annotated[float, np.float64],
    D: Callable[[float, np.ndarray], np.ndarray],
    simple: Callable[[np.ndarray, np.ndarray, float, float, Callable], Tuple[np.ndarray, float]]
) -> Tuple[np.ndarray, np.int32]:
    """
    Extremely simple integrator (Euler step) for real ODEs.
    """
    t = t1
    h = h1 if (t2 - t1) >= 0.0 else -h1
    nok = np.int32(0)
    while True:
        if (t + h - t2) * (t + h - t1) > 0.0:
            h = t2 - t
        dydt = D(t, y)
        y, t = simple(y, dydt, t, h, D)
        nok += 1
        if (t - t2) * (t2 - t1) >= 0.0:
            break
    return y, nok

@with_guardrails
def simpleint_dpc(
    y: Annotated[np.ndarray, np.complex128],
    t1: Annotated[float, np.float64],
    t2: Annotated[float, np.float64],
    h1: Annotated[float, np.float64],
    D: Callable[[float, np.ndarray], np.ndarray],
    simple: Callable[[np.ndarray, np.ndarray, float, float, Callable], Tuple[np.ndarray, float]]
) -> Tuple[np.ndarray, np.int32]:
    """
    Extremely simple integrator (Euler step) for complex ODEs.
    """
    return _simpleint_dpc_impl(y, t1, t2, h1, D, simple)

@_jit()
def _simpleint_dpc_impl(
    y: Annotated[np.ndarray, np.complex128],
    t1: Annotated[float, np.float64],
    t2: Annotated[float, np.float64],
    h1: Annotated[float, np.float64],
    D: Callable[[float, np.ndarray], np.ndarray],
    simple: Callable[[np.ndarray, np.ndarray, float, float, Callable], Tuple[np.ndarray, float]]
) -> Tuple[np.ndarray, np.int32]:
    """
    Extremely simple integrator (Euler step) for complex ODEs.
    """
    t = t1
    h = h1 if (t2 - t1) >= 0.0 else -h1
    nok = np.int32(0)
    while True:
        if (t + h - t2) * (t + h - t1) > 0.0:
            h = t2 - t
        dydt = D(t, y)
        y, t = simple(y, dydt, t, h, D)
        nok += 1
        if (t - t2) * (t2 - t1) >= 0.0:
            break
    return y, nok


# #rkqs
# @_jit(cache=True)
# def rkqs_dp(self,
#     y: Annotated[np.ndarray, np.float64],
#     dydt: Annotated[np.ndarray, np.float64],
#     t: Annotated[float, np.float64],
#     htry: Annotated[float, np.float64],
#     eps: Annotated[float, np.float64],
#     yscale: Annotated[np.ndarray, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray],
#     jacobn: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]
# ) -> Tuple[np.ndarray, float, float, float]:
#     """
#     Adaptive Runge-Kutta step control (Cash-Karp) for real ODEs.

#     Parameters
#     ----------
#     y     : ndarray[float64]
#         Current solution vector (modified in-place to next step).
#     dydt  : ndarray[float64]
#         Derivative at current t.
#     t     : float64
#         Independent variable (modified in-place).
#     htry  : float64
#         Initial trial step size.
#     eps   : float64
#         Desired local error tolerance.
#     yscale: ndarray[float64]
#         Scaling vector for each component.
#     D     : Callable[[float, ndarray], ndarray]
#         Derivative function f(t, y).
#     jacobn: Callable[[float, ndarray], Tuple[ndarray, ndarray]]
#         Jacobian evaluator (unused here).

#     Returns
#     -------
#     y     : ndarray[float64]
#         Updated solution at t + hdid.
#     t     : float64
#         Updated independent variable.
#     hdid  : float64
#         Actual step size taken.
#     hnext : float64
#         Suggested next step size.
#     """
#     n = y.size
#     yerr = np.empty(n, dtype=np.float64)
#     ytmp = np.empty(n, dtype=np.float64)
#     h = htry
#     errmax = 0.0

#     while True:
#         ytmp, yerr = rkck(y, dydt, t, h, D)
#         errmax = np.max(yerr / yscale) / eps
#         if errmax <= 1.0:
#             break
#         htmp = SAFETY * h * errmax**PSHRNK
#         h = htmp if h >= 0.0 else -htmp
#         h = max(htmp, 0.1 * h) if htry >= 0.0 else min(htmp, 0.1 * h)
#         tnew = t + h
#         if tnew == t:
#             Logger.getInstance().error("Stepsize underflow in rkqs_dp", 1)
#     hnext = SAFETY * h * errmax**PGROW if errmax > ERRCON else 5.0 * h
#     t += h
#     hdid = h
#     y[:] = ytmp
#     return y, t, hdid, hnext

# @_jit(cache=True)
# def rkqs_dpc(self,
#     y: Annotated[np.ndarray, np.complex128],
#     dydt: Annotated[np.ndarray, np.complex128],
#     t: Annotated[float, np.float64],
#     htry: Annotated[float, np.float64],
#     eps: Annotated[float, np.float64],
#     yscale: Annotated[np.ndarray, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray],
#     jacobn: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]
# ) -> Tuple[np.ndarray, float, float, float]:
#     """
#     Adaptive Runge-Kutta step control (Cash-Karp) for complex ODEs.

#     Parameters and returns mirror rkqs_dp but for complex128 data.
#     """
#     n = y.size
#     yerr = np.empty(n, dtype=np.complex128)
#     ytmp = np.empty(n, dtype=np.complex128)
#     h = htry
#     errmax = 0.0

#     while True:
#         ytmp, yerr = rkck(y, dydt, t, h, D)
#         errmax = np.max(np.abs(yerr) / yscale) / eps
#         if errmax <= 1.0:
#             break
#         htmp = SAFETY * h * errmax**PSHRNK
#         h = htmp if h >= 0.0 else -htmp
#         h = max(htmp, 0.1 * h) if htry >= 0.0 else min(htmp, 0.1 * h)
#         tnew = t + h
#         if tnew == t:
#             Logger.getInstance().error("Stepsize underflow in rkqs_dpc", 1)
#     hnext = SAFETY * h * errmax**PGROW if errmax > ERRCON else 5.0 * h
#     t += h
#     hdid = h
#     y[:] = ytmp
#     return y, t, hdid, hnext

# #rkck
# @_jit(cache=True)
# def rkck_dp(self,
#     y: Annotated[np.ndarray, np.float64],
#     dydt: Annotated[np.ndarray, np.float64],
#     t: Annotated[float, np.float64],
#     h: Annotated[float, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray]
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Cash-Karp step: computes yout and error estimate yerr for real ODEs.

#     Parameters
#     ----------
#     y    : ndarray[float64]
#         Current solution vector.
#     dydt : ndarray[float64]
#         Derivative at current t.
#     t    : float64
#         Independent variable.
#     h    : float64
#         Step size.
#     D    : Callable[[float, ndarray], ndarray]
#         Derivative function f(t, y).

#     Returns
#     -------
#     yout : ndarray[float64]
#         Fourth-order solution estimate at t+h.
#     yerr : ndarray[float64]
#         Error estimate between fourth and fifth order.
#     """
#     n = y.size
#     # Allocate temporary arrays
#     ytmp = np.empty(n, dtype=np.float64)
#     ak2  = np.empty(n, dtype=np.float64)
#     ak3  = np.empty(n, dtype=np.float64)
#     ak4  = np.empty(n, dtype=np.float64)
#     ak5  = np.empty(n, dtype=np.float64)
#     ak6  = np.empty(n, dtype=np.float64)

#     # Coefficients from Fortran cash-karp
#     a2, a3, a4, a5, a6 = 1/5, 3/10, 3/5, 1.0, 7/8
#     b21 = 1/5
#     b31, b32 = 3/40, 9/40
#     b41, b42, b43 = 3/10, -9/10, 6/5
#     b51, b52, b53, b54 = -11/54, 5/2, -70/27, 35/27
#     b61, b62, b63, b64, b65 = 1631/55296, 175/512, 575/13824, 44275/110592, 253/4096
#     c1, c3, c4, c6 = 37/378, 250/621, 125/594, 512/1771
#     dc1, dc3, dc4, dc5, dc6 = 2825/27648, 18575/48384, 13525/55296, 277/14336, 1/4

#     # Stage 1 already have dydt
#     # Stage 2
#     for i in range(n): ytmp[i] = y[i] + b21 * h * dydt[i]
#     ak2 = D(t + a2*h, ytmp)

#     # Stage 3
#     for i in range(n): ytmp[i] = y[i] + h * (b31 * dydt[i] + b32 * ak2[i])
#     ak3 = D(t + a3*h, ytmp)

#     # Stage 4
#     for i in range(n): ytmp[i] = y[i] + h * (b41 * dydt[i] + b42 * ak2[i] + b43 * ak3[i])
#     ak4 = D(t + a4*h, ytmp)

#     # Stage 5
#     for i in range(n): ytmp[i] = y[i] + h * (b51 * dydt[i] + b52 * ak2[i] + b53 * ak3[i] + b54 * ak4[i])
#     ak5 = D(t + a5*h, ytmp)

#     # Stage 6
#     for i in range(n): ytmp[i] = y[i] + h * (b61 * dydt[i] + b62 * ak2[i] + b63 * ak3[i] + b64 * ak4[i] + b65 * ak5[i])
#     ak6 = D(t + a6*h, ytmp)

#     # Combine for output and error
#     yout = np.empty(n, dtype=np.float64)
#     yerr = np.empty(n, dtype=np.float64)
#     for i in range(n):
#         yout[i] = y[i] + h * (c1 * dydt[i] + c3 * ak3[i] + c4 * ak4[i] + c6 * ak6[i])
#         yerr[i] = h * (dc1 * dydt[i] + dc3 * ak3[i] + dc4 * ak4[i] + dc5 * ak5[i] + dc6 * ak6[i])

#     return yout, yerr

# @_jit(cache=True)
# def rkck_dpc(self,
#     y: Annotated[np.ndarray, np.complex128],
#     dydt: Annotated[np.ndarray, np.complex128],
#     t: Annotated[float, np.float64],
#     h: Annotated[float, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray]
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Cash-Karp step: computes yout and error estimate yerr for complex ODEs.
#     """
#     n = y.size
#     ytmp = np.empty(n, dtype=np.complex128)
#     ak2  = np.empty(n, dtype=np.complex128)
#     ak3  = np.empty(n, dtype=np.complex128)
#     ak4  = np.empty(n, dtype=np.complex128)
#     ak5  = np.empty(n, dtype=np.complex128)
#     ak6  = np.empty(n, dtype=np.complex128)

#     # reuse real coefficients
#     a2, a3, a4, a5, a6 = 1/5, 3/10, 3/5, 1.0, 7/8
#     b21 = 1/5
#     b31, b32 = 3/40, 9/40
#     b41, b42, b43 = 3/10, -9/10, 6/5
#     b51, b52, b53, b54 = -11/54, 5/2, -70/27, 35/27
#     b61, b62, b63, b64, b65 = 1631/55296, 175/512, 575/13824, 44275/110592, 253/4096
#     c1, c3, c4, c6 = 37/378, 250/621, 125/594, 512/1771
#     dc1, dc3, dc4, dc5, dc6 = 2825/27648, 18575/48384, 13525/55296, 277/14336, 1/4

#     for i in range(n): ytmp[i] = y[i] + b21 * h * dydt[i]
#     ak2 = D(t + a2*h, ytmp)
#     for i in range(n): ytmp[i] = y[i] + h * (b31 * dydt[i] + b32 * ak2[i])
#     ak3 = D(t + a3*h, ytmp)
#     for i in range(n): ytmp[i] = y[i] + h * (b41 * dydt[i] + b42 * ak2[i] + b43 * ak3[i])
#     ak4 = D(t + a4*h, ytmp)
#     for i in range(n): ytmp[i] = y[i] + h * (b51 * dydt[i] + b52 * ak2[i] + b53 * ak3[i] + b54 * ak4[i])
#     ak5 = D(t + a5*h, ytmp)
#     for i in range(n): ytmp[i] = y[i] + h * (b61 * dydt[i] + b62 * ak2[i] + b63 * ak3[i] + b64 * ak4[i] + b65 * ak5[i])
#     ak6 = D(t + a6*h, ytmp)

#     yout = np.empty(n, dtype=np.complex128)
#     yerr = np.empty(n, dtype=np.complex128)
#     for i in range(n):
#         yout[i] = y[i] + h * (c1 * dydt[i] + c3 * ak3[i] + c4 * ak4[i] + c6 * ak6[i])
#         yerr[i] = h * (dc1 * dydt[i] + dc3 * ak3[i] + dc4 * ak4[i] + dc5 * ak5[i] + dc6 * ak6[i])

#     return yout, yerr

# #rk4
# @_jit(cache=True)
# def rk4_dp(self,
#     y: Annotated[np.ndarray, np.float64],
#     dydt: Annotated[np.ndarray, np.float64],
#     t: Annotated[float, np.float64],
#     h: Annotated[float, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray]
# ) -> Tuple[np.ndarray, float]:
#     """
#     Classical RK4 step for real ODEs.

#     Parameters
#     ----------
#     y    : ndarray[float64]
#         Current solution vector.
#     dydt : ndarray[float64]
#         Derivative at current t.
#     t    : float64
#         Independent variable.
#     h    : float64
#         Step size.
#     D    : Callable[[float, ndarray], ndarray]
#         Derivative function f(t, y).

#     Returns
#     -------
#     yout : ndarray[float64]
#         Updated solution at t+h.
#     tnew : float64
#         Updated independent variable.
#     """
#     n = y.size
#     ytmp = np.empty(n, dtype=np.float64)
#     dyt  = np.empty(n, dtype=np.float64)
#     dym  = np.empty(n, dtype=np.float64)

#     # Stage 1 already have dydt
#     for i in range(n): ytmp[i] = y[i] + 0.5 * h * dydt[i]
#     dyt = D(t + 0.5*h, ytmp)

#     for i in range(n): ytmp[i] = y[i] + 0.5 * h * dyt[i]
#     dym = D(t + 0.5*h, ytmp)

#     for i in range(n): ytmp[i] = y[i] + h * dym[i]
#     dym = dym + dyt

#     dyt = D(t + h, ytmp)
#     yout = np.empty(n, dtype=np.float64)
#     for i in range(n):
#         yout[i] = y[i] + h/6 * (dydt[i] + dyt[i] + 2*dym[i])

#     return yout, t + h

# @_jit(cache=True)
# def rk4_dpc(self,
#     y: Annotated[np.ndarray, np.complex128],
#     dydt: Annotated[np.ndarray, np.complex128],
#     t: Annotated[float, np.float64],
#     h: Annotated[float, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray]
# ) -> Tuple[np.ndarray, float]:
#     """
#     Classical RK4 step for complex ODEs.

#     Parameters and returns mirror rk4_dp but for complex128 data.
#     """
#     n = y.size
#     ytmp = np.empty(n, dtype=np.complex128)
#     dyt  = np.empty(n, dtype=np.complex128)
#     dym  = np.empty(n, dtype=np.complex128)

#     for i in range(n): ytmp[i] = y[i] + 0.5 * h * dydt[i]
#     dyt = D(t + 0.5*h, ytmp)

#     for i in range(n): ytmp[i] = y[i] + 0.5 * h * dyt[i]
#     dym = D(t + 0.5*h, ytmp)

#     for i in range(n): ytmp[i] = y[i] + h * dym[i]
#     dym = dym + dyt

#     dyt = D(t + h, ytmp)
#     yout = np.empty(n, dtype=np.complex128)
#     for i in range(n):
#         yout[i] = y[i] + h/6 * (dydt[i] + dyt[i] + 2*dym[i])

#     return yout, t + h

# #simple euler step-equations for ODEs
# @_jit(cache=True)
# def idiot_dp(self,
#     y: Annotated[np.ndarray, np.float64],
#     dydt: Annotated[np.ndarray, np.float64],
#     t: Annotated[float, np.float64],
#     h: Annotated[float, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray]
# ) -> Tuple[np.ndarray, float]:
#     """
#     Simple Euler step for real ODEs: y += dydt * h; t += h.

#     Parameters
#     ----------
#     y    : ndarray[float64]
#         Solution vector (modified in-place).
#     dydt : ndarray[float64]
#         Derivative at current t.
#     t    : float64
#         Independent variable.
#     h    : float64
#         Step size.
#     D    : Callable[[float, ndarray], ndarray]
#         Derivative function (unused here).

#     Returns
#     -------
#     yout : ndarray[float64]
#         Updated solution vector.
#     tnew : float64
#         Updated independent variable.
#     """
#     for i in range(y.size):
#         y[i] = y[i] + dydt[i] * h
#     t = t + h
#     return y, t

# @_jit(cache=True)
# def idiot_dpc(self,
#     y: Annotated[np.ndarray, np.complex128],
#     dydt: Annotated[np.ndarray, np.complex128],
#     t: Annotated[float, np.float64],
#     h: Annotated[float, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray]
# ) -> Tuple[np.ndarray, float]:
#     """
#     Simple Euler step for complex ODEs: y += dydt * h; t += h.

#     Same parameters and returns as idiot_dp but complex128.
#     """
#     for i in range(y.size):
#         y[i] = y[i] + dydt[i] * h
#     t = t + h
#     return y, t

# # Bulirsch-Stoer methods
# @njit(cache=True)
# def bsstep(self,
#     y: Annotated[np.ndarray, np.float64],
#     dydx: Annotated[np.ndarray, np.float64],
#     x: Annotated[float, np.float64],
#     htry: Annotated[float, np.float64],
#     eps: Annotated[float, np.float64],
#     yscale: Annotated[np.ndarray, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray],
#     jacobn: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]
# ) -> Tuple[np.ndarray, float, float]:
#     """
#     Bulirsch-Stoer step for real ODEs (bsstep).

#     Parameters
#     ----------
#     y      : ndarray[float64]
#         Current solution vector (in-out).
#     dydx   : ndarray[float64]
#         Derivative at current x.
#     x      : float64
#         Independent variable (in-out).
#     htry   : float64
#         Initial trial step size.
#     eps    : float64
#         Desired local error tolerance.
#     yscale : ndarray[float64]
#         Scaling vector for error control.
#     D      : Callable[[float, ndarray], ndarray]
#         Derivative function f(x, y).
#     jacobn : Callable[[float, ndarray], Tuple[ndarray, ndarray]]
#         Jacobian evaluator (unused).

#     Returns
#     -------
#     yout  : ndarray[float64]
#         Updated solution at x + hdid.
#     hdid  : float64
#         Actual step size taken.
#     hnext : float64
#         Suggested next step size.

#     Notes
#     -----
#     This routine implements the modified midpoint and Richardson extrapolation
#     sequence (`mmid` and `pzextr`), along with step-size control as in the Fortran code.
#     """
#     # Sequence lengths
#     nseq = np.array([2,4,6,8,10,12,14,16,18], dtype=np.int32)
#     first = True
#     epsold = -1.0
#     kmax = KMAXX
#     kopt = 2
#     xnew = -1e30
#     hnext = htry
#     # Work arrays
#     ysave = np.empty_like(y)
#     yseq  = np.empty_like(y)
#     yerr  = np.empty_like(y)
#     err   = np.empty(KMAXX, dtype=np.float64)

#     # initialize on first call or eps change
#     if eps != epsold:
#         eps1 = 0.25 * eps  # SAFE1 = 0.25
#         # compute extrapolation table coefficients... (omitted)
#         epsold = eps
#         # select optimal kopt (omitted)

#     h = htry
#     ysave[:] = y[:]
#     # main sequence
#     for k in range(1, kmax+1):
#         xnew = x + h
#         if xnew == x:
#             Logger.getInstance().error("Step size underflow in bsstep", 1)
#         # modified midpoint
#         yseq = mmid(ysave, dydx, x, h, nseq[k-1], D)
#         # extrapolate
#         y, yerr = pzextr(k, (h/nseq[k-1])**2, yseq, y, yerr)
#         # error estimate and decision logic (omitted for brevity)
#     # finalize hdid and hnext (omitted)
#     hdid = h
#     hnext = h  # placeholder
#     x = xnew
#     return y, hdid, hnext


# #modified Bulirsch-Store methods with midpoints
# @_jit(cache=True)
# def mmid(self,
#     y: Annotated[np.ndarray, np.float64],
#     dydx: Annotated[np.ndarray, np.float64],
#     xs: Annotated[float, np.float64],
#     htot: Annotated[float, np.float64],
#     nstep: Annotated[int, None],
#     D: Callable[[float, np.ndarray], np.ndarray]
# ) -> np.ndarray:
#     """
#     Modified midpoint method for Bulirsch-Stoer sequence.

#     Parameters
#     ----------
#     y    : ndarray[float64]
#         Initial solution vector.
#     dydx : ndarray[float64]
#         Derivative at xs.
#     xs   : float64
#         Starting independent variable value.
#     htot : float64
#         Total step size.
#     nstep: int
#         Number of substeps.
#     D    : Callable[[float, ndarray], ndarray]
#         Derivative function f(x, y).

#     Returns
#     -------
#     yout : ndarray[float64]
#         Estimate after htot total step.
#     """
#     n = y.size
#     h = htot / nstep
#     ym = y.copy()
#     yn = y + h * dydx
#     x = xs + h
#     yout = D(x, yn)
#     for _ in range(2, nstep+1):
#         swap = ym + 2.0 * h * yout
#         ym = yn
#         yn = swap
#         x += h
#         yout = D(x, yn)
#     yout = 0.5 * (ym + yn + h * yout)
#     return yout

# @_jit(cache=True)
# def pzextr(self,
#     iest: Annotated[int, None],
#     xest: Annotated[float, np.float64],
#     yest: Annotated[np.ndarray, np.float64],
#     yz: Annotated[np.ndarray, np.float64],
#     dy: Annotated[np.ndarray, np.float64]
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Polynomial extrapolation (Richardson) for Bulirsch-Stoer.

#     Parameters
#     ----------
#     iest: int
#         Sequence index.
#     xest: float64
#         Extrapolation parameter (h/n)^2.
#     yest: ndarray[float64]
#         Current midpoint estimate.
#     yz  : ndarray[float64]
#         Accumulated extrapolated values (in-out).
#     dy  : ndarray[float64]
#         Extrapolation increments (in-out).

#     Returns
#     -------
#     yz_new : ndarray[float64]
#         Updated extrapolated values.
#     dy_new : ndarray[float64]
#         New increment values.
#     """
#     nv = yest.size
#     # static arrays for x and d are emulated by module-level storage
#     # Here we use Python lists for simplicity
#     if not hasattr(pzextr, "xarr"):
#         pzextr.xarr = np.zeros(KMAXX+1, dtype=np.float64)
#         pzextr.d = np.zeros((KMAXX+1, KMAXX+1), dtype=np.float64)
#     x = pzextr.xarr
#     d = pzextr.d
#     x[iest] = xest
#     yz_out = yest.copy()
#     dy_out = yest.copy()
#     if iest == 1:
#         d[1,1:1+nv] = yest
#     else:
#         c = yest.copy()
#         for k1 in range(1, iest):
#             f1 = xest / (x[iest-k1] - xest)
#             f2 = x[iest-k1] / (x[iest-k1] - xest)
#             q = d[k1, :nv].copy()
#             d[k1, :nv] = dy_out
#             d2 = c - q
#             dy_out = f1 * d2
#             c = f2 * d2
#             yz_out = yz_out + dy_out
#         d[iest, :nv] = dy_out
#     return yz_out, dy_out

# @_jit(cache=True)
# def simpr(self,
#     y: Annotated[np.ndarray, np.float64],
#     dydx: Annotated[np.ndarray, np.float64],
#     dfdx: Annotated[np.ndarray, np.float64],
#     dfdy: Annotated[np.ndarray, np.float64],
#     xs: Annotated[float, np.float64],
#     htot: Annotated[float, np.float64],
#     nstep: Annotated[int, None],
#     derivs: Callable[[float, np.ndarray], np.ndarray]
# ) -> np.ndarray:
#     """
#     Semi-implicit predictor-corrector integrator.

#     Parameters
#     ----------
#     y    : ndarray[float64]
#         Initial solution.
#     dydx : ndarray[float64]
#         First derivative at xs.
#     dfdx : ndarray[float64]
#         Derivative of f w.r.t x at xs.
#     dfdy : ndarray[float64]
#         Jacobian matrix of f w.r.t y at xs.
#     xs   : float64
#         Starting independent variable.
#     htot : float64
#         Total step size.
#     nstep: int
#         Number of substeps.
#     derivs: Callable[[float, ndarray], ndarray]
#         Derivative function f(x, y).

#     Returns
#     -------
#     yout : ndarray[float64]
#         Solution at xs + htot.
#     """
#     ndum = y.size
#     h = htot / nstep
#     x = xs
#     # Setup matrix a = I - h*dfdy
#     a = -h * dfdy.copy()
#     diagadd(a, 1.0)
#     indx = np.empty(ndum, dtype=np.int32)
#     d = ludcmp(a, indx)
#     # initial corrector
#     yout = h * (dydx + h * dfdx)
#     lubksb(a, indx, yout)
#     del_vec = yout.copy()
#     ytemp = y + del_vec
#     x += h
#     yout = derivs(x, ytemp)
#     for _ in range(2, nstep+1):
#         yout = h * yout - del_vec
#         lubksb(a, indx, yout)
#         del_vec = del_vec + 2.0 * yout
#         ytemp = ytemp + del_vec
#         x += h
#         yout = derivs(x, ytemp)
#     yout = h * yout - del_vec
#     lubksb(a, indx, yout)
#     return ytemp + yout

# # stiff equations
# @_jit
# def stifbs(self,
#     y: Annotated[NDArray[np.float64], np.float64],
#     dydx: Annotated[NDArray[np.float64], np.float64],
#     x: Annotated[float, np.float64],
#     htry: Annotated[float, np.float64],
#     eps: Annotated[float, np.float64],
#     yscal: Annotated[NDArray[np.float64], np.float64],
#     derivs: Callable[
#         [float, NDArray[np.float64]],              # ← list of args
#         NDArray[np.float64]                        # ← return type
#     ],
#     jacobn: Callable[
#         [float, NDArray[np.float64]],              # ← args
#         Tuple[NDArray[np.float64], NDArray[np.float64]]  # ← result
#     ]
# ) -> Tuple[Annotated[float, np.float64],
#         Annotated[float, np.float64],
#         Annotated[float, np.float64]]:
#     """
#     Bulirsch–Stoer stiff‐ODE stepper (Fortran `stifbs`).

#     Parameters
#     ----------
#     y      : float64[:]  – on input, current solution; on output, advanced by hdid
#     dydx   : float64[:]  – derivative at x
#     x      : float64     – independent variable (returned updated)
#     htry   : float64     – initial trial step size
#     eps    : float64     – desired accuracy
#     yscal  : float64[:]  – error‐scaling array
#     derivs : (x, y) → dydx
#     jacobn : (x, y) → (dfdx, dfdy)

#     Returns
#     -------
#     xnew   : float64     – x + hdid
#     hdid   : float64     – actual step length used
#     hnext  : float64     – estimated next step size
#     """
#     # --- constants & sequences ---
#     imax, kmaxx = 8, 7
#     safe1, safe2   = 0.25, 0.7
#     redmax, redmin = 1e-5, 0.7
#     tiny, scalmx   = 1e-30, 0.1
#     nseq = np.array([2, 6, 10, 14, 22, 34, 50, 70], dtype=np.int64)

#     # --- precompute coefficient matrix ---
#     eps1 = safe1 * eps
#     a = cumsum(nseq.astype(np.float64), seed=1.0)    # length 8
#     a2 = a[1:]                                       # a(2:8) in Fortran
#     ap = outerprod(arth(3.0, 2.0, kmaxx), a2 - a[0] + 1.0)
#     od = outerdiff(a2, a2)
#     mask = upperTriangle(kmaxx, kmaxx).astype(bool)
#     alf = np.ones((kmaxx, kmaxx), dtype=np.float64)
#     alf[mask] = eps1 ** (od[mask] / ap[mask])

#     # --- initialization ---
#     ndum = y.size
#     ysav = y.copy()
#     dfdx, dfdy = jacobn(x, y)

#     h = htry
#     kopt = kmaxx
#     success = False
#     err   = np.zeros(kmaxx, dtype=np.float64)
#     yerr  = np.zeros(ndum, dtype=np.float64)
#     yseq  = np.zeros(ndum, dtype=np.float64)
#     first = True

#     # --- main adapt‐loop ---
#     while not success:
#         for k in range(1, kopt+1):
#             xnew = x + h
#             if xnew == x:
#                 Logger.getInstance().error("step size underflow in stifbs")

#             # perform the modified midpoint ('simpr') and Richardson extrapolation ('pzextr')
#             simpr(ysav, dydx, dfdx, dfdy, x, h, int(nseq[k-1]), yseq, derivs)
#             xest = (h / nseq[k-1])**2
#             pzextr(k, xest, yseq, y, yerr)

#             if k > 1:
#                 errmax = np.max(np.abs(yerr / yscal))
#                 errmax = max(tiny, errmax) / eps
#                 km = k - 2
#                 err[km] = (errmax / safe1)**(1.0 / (2*km + 1))

#             # check acceptance
#             if k > 1 and (k >= kopt-1 or first):
#                 if errmax < 1.0:
#                     success = True
#                     break
#                 # compute reduction factor `red`
#                 if k == kopt+1 or k == kopt+2:
#                     red = safe2 / err[km]
#                     break
#                 if k == kopt:
#                     if alf[kopt-2, kopt-1] < err[km]:
#                         red = 1.0 / err[km]
#                         break
#                 if kopt == kmaxx and alf[km, kmaxx-2] < err[km]:
#                     red = alf[km, kmaxx-2] * safe2 / err[km]
#                     break
#                 if alf[km, kopt-1] < err[km]:
#                     red = alf[km, kopt-1] / err[km]
#                     break
#         if not success:
#             red = max(min(red, redmin), redmax)
#             h *= red
#             first = False

#     # --- finalize step & predict next h ---
#     x = xnew
#     hdid = h

#     # re‐estimate optimal sequence index
#     km_len = k - 1
#     q = a[1:km_len+1] * np.maximum(err[:km_len], scalmx)
#     p_min = int(np.argmin(q)) + 1            # Fortran minloc + 1
#     kopt  = p_min + 1

#     scale  = max(err[kopt-2], scalmx)
#     hnext  = h / scale
#     wrkmin = scale * a[kopt-1]

#     if kopt >= k and kopt != kmaxx and not first:
#         fact = max(scale / alf[kopt-2, kopt-1], scalmx)
#         if a[kopt] * fact <= wrkmin:
#             hnext = h / fact
#             kopt += 1

#     return x, hdid, hnext

# #LU-Decomposition matrices
# @_jit(cache=True)
# def ludcmp(self,
#     a: Annotated[np.ndarray, np.float64],
#     indx: Annotated[np.ndarray, np.int32]
# ) -> float:
#     """
#     LU decomposition with partial pivoting, in place.

#     Parameters
#     ----------
#     a    : ndarray[float64, 2D]
#         Input matrix to decompose; replaced by LU factors.
#     indx : ndarray[int32]
#         Output pivot indices.

#     Returns
#     -------
#     d : float64
#         +1 or -1 depending on number of row interchanges.

#     Raises
#     ------
#     RuntimeError
#         If matrix is singular.
#     """
#     n = assert_eq(a.shape[0], a.shape[1], indx.size, 'ludcmp')
#     vv = np.empty(n, dtype=np.float64)
#     d = np.float64(1.0)
#     # scaling factors
#     for i in range(n):
#         vv[i] = 1.0 / np.max(np.abs(a[i, :]))
#         if vv[i] == 0.0:
#             nrerror('singular matrix in ludcmp', __file__, 0)
#     # main loop
#     for j in range(n):
#         imax = j
#         big = 0.0
#         for i in range(j, n):
#             temp = vv[i] * abs(a[i, j])
#             if temp > big:
#                 big = temp
#                 imax = i
#         if j != imax:
#             swap(a[imax, :], a[j, :])
#             d = -d
#             vv[imax] = vv[j]
#         indx[j] = imax
#         if a[j, j] == 0.0:
#             a[j, j] = TINY
#         for i in range(j+1, n):
#             a[i, j] /= a[j, j]
#         for i in range(j+1, n):
#             for k in range(j+1, n):
#                 a[i, k] -= a[i, j] * a[j, k]
#     return d

# @_jit(cache=True)
# def lubksb(self,
#     a: Annotated[np.ndarray, np.float64],
#     indx: Annotated[np.ndarray, np.int32],
#     b: Annotated[np.ndarray, np.float64]
# ) -> None:
#     """
#     Solve Ly = b, then Ux = y using LU factors from ludcmp.

#     Parameters
#     ----------
#     a    : ndarray[float64, 2D]
#         LU-decomposed matrix from ludcmp.
#     indx : ndarray[int32]
#         Pivot indices from ludcmp.
#     b    : ndarray[float64]
#         Right-hand side vector; replaced by solution.
#     """
#     n = assert_eq(a.shape[0], a.shape[1], indx.size, 'lubksb')
#     ii = 0
#     for i in range(n):
#         ll = indx[i]
#         sum_ = b[ll]
#         b[ll] = b[i]
#         if ii != 0:
#             for j in range(ii-1, i):
#                 sum_ -= a[i, j] * b[j]
#         elif sum_ != 0.0:
#             ii = i + 1
#         b[i] = sum_
#     for i in range(n-1, -1, -1):
#         sum_ = b[i]
#         for j in range(i+1, n):
#             sum_ -= a[i, j] * b[j]
#         b[i] = sum_ / a[i, i]

# # This is the 3D versions of the NR code for UPPE applications, NO JACOBIANS
# @_jit(cache=True)
# def odeint_3D_dpc(self,
#     y: Annotated[np.ndarray, np.complex128],
#     x1: Annotated[float, np.float64],
#     x2: Annotated[float, np.float64],
#     eps: Annotated[float, np.float64],
#     h1: Annotated[float, np.float64],
#     hmin: Annotated[float, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray],
#     integ: Callable[[np.ndarray, np.ndarray, float, float, float, Callable], Tuple[float, float]]
# ) -> Tuple[np.ndarray, np.int32, np.int32]:
#     """
#     Adaptive-step driver for 3D complex fields.

#     Parameters
#     ----------
#     y     : ndarray[complex128], shape (nx, ny, nz)
#         Input: field at x1; Output: at x2 (in-place).
#     x1, x2: float64
#         Interval endpoints.
#     eps   : float64
#         Desired local error tolerance.
#     h1    : float64
#         Initial step size (in-out to return last used hprev).
#     hmin  : float64
#         Minimum allowable step size.
#     D     : Callable[[float, ndarray], ndarray]
#         Derivative function D(t, y).
#     integ : Callable[[ndarray, ndarray, float, float, ndarray, float, float], Tuple[float, float]]
#         Integrator subroutine returning (hdid, hnext).

#     Returns
#     -------
#     y    : ndarray[complex128]
#         Final field at x2.
#     nok  : int32
#         Number of successful steps.
#     nbad : int32
#         Number of rejected steps.
#     """
#     # shapes and buffers
#     yscale = np.empty_like(y, dtype=np.float64)
#     dydx   = np.empty_like(y)
#     x      = x1
#     h      = h1 if (x2 - x1) >= 0.0 else -h1
#     nok    = np.int32(0)
#     nbad   = np.int32(0)
#     hprev  = h
#     # main loop
#     for _ in range(MAXSTP):
#         dydx = D(x, y)
#         baseline = np.max(np.abs(y)) * 1e-4
#         # yscale = max(baseline, |y| + |h*dydx|)
#         np.maximum(baseline, np.abs(y) + np.abs(h * dydx), out=yscale)
#         # prevent overshoot
#         if (x + h - x2) * (x + h - x1) > 0.0:
#             h = x2 - x
#         # step
#         hdid, hnext = integ(y, dydx, x, h, eps, yscale, D)
#         if hdid == h:
#             nok += 1
#         else:
#             nbad += 1
#         # advance
#         x += hdid
#         print("dz=", h)  # progress debug
#         if (x - x2) * (x2 - x1) >= 0.0:
#             h1 = hprev
#             return y, nok, nbad
#         if abs(hnext) <= abs(hmin):
#             Logger.getInstance().error("Step size too small in odeint_3D_dpc.", 1)
#         hprev = hdid
#         h = hnext
#     Logger.getInstance().error("Too many steps in odeint_3D_dpc.", 1)
#     return y, nok, nbad

# @_jit(cache=True)
# def rkqs_3D_dpc(self,
#     y: Annotated[np.ndarray, np.complex128],
#     dydt: Annotated[np.ndarray, np.complex128],
#     t: Annotated[float, np.float64],
#     htry: Annotated[float, np.float64],
#     eps: Annotated[float, np.float64],
#     yscale: Annotated[np.ndarray, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray]
# ) -> Tuple[np.ndarray, float, float]:
#     """
#     Adaptive RKQS for 3D complex ODEs.
#     """
#     h = htry
#     while True:
#         ytmp, yerr = rkck(y, dydt, t, h, D)
#         errmax = np.max(np.abs(yerr) / yscale) / eps
#         if errmax <= 1.0:
#             break
#         htmp = SAFETY * h * errmax**PSHRNK
#         h = max(htmp, 0.1 * h) if h >= 0.0 else min(htmp, 0.1 * h)
#         tnew = t + h
#         if tnew == t:
#             Logger.getInstance().error("Stepsize underflow in rkqs_3D_dpc.", 1)
#     hnext = SAFETY * h * errmax**PGROW if errmax > ERRCON else 5.0 * h
#     t += h
#     y[:] = ytmp
#     return y, h, hnext

# @_jit(cache=True)
# def rkck_3D_dpc(self,
#     y: Annotated[np.ndarray, np.complex128],
#     dydt: Annotated[np.ndarray, np.complex128],
#     t: Annotated[float, np.float64],
#     h: Annotated[float, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray]
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Cash-Karp RK step for 3D complex arrays.
#     """
#     # reuse existing rkck implementation by flattening dims
#     flat_y = y.ravel()
#     flat_dydt = dydt.ravel()
#     yout_flat, yerr_flat = rkck(flat_y, flat_dydt, t, h, D)
#     return yout_flat.reshape(y.shape), yerr_flat.reshape(y.shape)

# @_jit(cache=True)
# def rk4_3D_dpc(self,
#     y: Annotated[np.ndarray, np.complex128],
#     dydt: Annotated[np.ndarray, np.complex128],
#     t: Annotated[float, np.float64],
#     h: Annotated[float, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray]
# ) -> Tuple[np.ndarray, float]:
#     """
#     Classical RK4 for 3D complex ODEs.
#     """
#     flat_y = y.ravel()
#     flat_dydt = dydt.ravel()
#     yout_flat, tnew = rk4_dp(flat_y, flat_dydt, t, h, D)
#     return yout_flat.reshape(y.shape), tnew

# @_jit(cache=True)
# def odeint_3D_dpc_TOM(self,
#     y: Annotated[np.ndarray, np.complex128],
#     x1: Annotated[float, np.float64],
#     x2: Annotated[float, np.float64],
#     eps: Annotated[np.complex128, np.complex128],
#     h1: Annotated[float, np.float64],
#     hmin: Annotated[float, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray],
#     integ: Callable[[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, Callable], Tuple[float, float]]
# ) -> Tuple[np.ndarray, np.int32, np.int32]:
#     """
#     Adaptive-step driver for 3D complex fields considering real and imaginary error scales.

#     Parameters
#     ----------
#     y         : ndarray[complex128]
#         Field array, shape (nx,ny,nz); modified in-place from x1 to x2.
#     x1, x2    : float64
#         Interval endpoints.
#     eps       : complex128
#         Desired complex error tolerance (real & imag).
#     h1        : float64
#         Initial step size (in-out to return last used hprev).
#     hmin      : float64
#         Minimum allowable step size.
#     D         : Callable[[float, ndarray], ndarray]
#         Derivative function D(t, y).
#     integ     : Callable[[ndarray, ndarray, float, float, ndarray, ndarray, Callable], Tuple[float,float]]
#         Integrator returning (hdid, hnext) using separate real/imag scaling.

#     Returns
#     -------
#     y    : ndarray[complex128]
#         Final field at x2.
#     nok  : int32
#         Number of successful steps.
#     nbad : int32
#         Number of rejected steps.
#     """
#     # allocate scaling arrays
#     yscaleREAL = np.empty_like(y, dtype=np.float64)
#     yscaleIMAG = np.empty_like(y, dtype=np.float64)
#     dydx       = np.empty_like(y)

#     x     = x1
#     h     = h1 if (x2 - x1) >= 0.0 else -h1
#     nok   = np.int32(0)
#     nbad  = np.int32(0)
#     hprev = h

#     for _ in range(MAXSTP):
#         # compute derivative
#         dydx = D(x, y)
#         # baseline scale
#         baseline = np.max(np.abs(y)) * 1e-6
#         # separate real/imag scales
#         np.maximum(baseline, np.abs(y.real) + np.abs((h * dydx).real), out=yscaleREAL)
#         np.maximum(baseline, np.abs(y.imag) + np.abs((h * dydx).imag), out=yscaleIMAG)
#         # prevent overshoot
#         if (x + h - x2) * (x + h - x1) > 0.0:
#             h = x2 - x
#         # adaptive step
#         hdid, hnext = integ(y, dydx, x, h, eps, yscaleREAL, yscaleIMAG, D)
#         if hdid == h:
#             nok += 1
#         else:
#             nbad += 1
#         x += hdid
#         if (x - x2) * (x2 - x1) >= 0.0:
#             h1 = hprev
#             return y, nok, nbad
#         if abs(hnext) <= abs(hmin):
#             Logger.getInstance().error("Step size too small in odeint_3D_dpc_TOM.", 1)
#         hprev = hdid
#         h = hnext

#     Logger.getInstance().error("Too many steps in odeint_3D_dpc_TOM.", 1)
#     return y, nok, nbad

# @_jit(cache=True)
# def rkqs_3D_dpc_TOM(
#     y: Annotated[np.ndarray, np.complex128],
#     dydt: Annotated[np.ndarray, np.complex128],
#     t: Annotated[float, np.float64],
#     htry: Annotated[float, np.float64],
#     eps: Annotated[np.complex128, np.complex128],
#     yscaleREAL: Annotated[np.ndarray, np.float64],
#     yscaleIMAG: Annotated[np.ndarray, np.float64],
#     D: Callable[[float, np.ndarray], np.ndarray]
# ) -> Tuple[np.ndarray, float, float]:
#     """
#     Adaptive Cash-Karp RK step for 3D complex ODEs with real/imag scaling.

#     Parameters and returns mirror odeint_3D_dpc_TOM but for a single step.
#     """
#     h = htry
#     while True:
#         # compute trial step and error
#         ytmp, yerr = rkck(y, dydt, t, h, D)
#         # real and imag error metrics
#         errREAL = np.max(np.abs(yerr.real) / yscaleREAL) / np.real(eps)
#         errIMAG = np.max(np.abs(yerr.imag) / yscaleIMAG) / np.imag(eps)
#         errmax = max(errREAL, errIMAG)
#         if errmax <= 1.0:
#             break
#         # shrink step
#         htmp = SAFETY * h * errmax**PSHRNK
#         h = htmp if h >= 0.0 else -htmp
#         h = max(htmp, 0.1 * h) if h >= 0.0 else min(htmp, 0.1 * h)
#         tnew = t + h
#         if tnew == t:
#             Logger.getInstance().error("Stepsize underflow in rkqs_3D_dpc_TOM.", 1)
#     # next step estimate
#     hnext = SAFETY * h * errmax**PGROW if errmax > ERRCON else 5.0 * h
#     t += h
#     y[:] = ytmp
#     return y, h, hnext


# # user suppiled callables are supposed to be parallelized for speed ups
