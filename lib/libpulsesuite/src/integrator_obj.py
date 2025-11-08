"""
Integrator module (Python version)
==================================
High-performance ODE integrators for scientific computing, ported from Fortran.

Dependencies: numpy, numba, guardrails, nrutils

Implements: Adaptive-step integrators for real (float64) and complex (complex128) ODEs,
including Cash–Karp RKQS single-step routines and dummy Jacobians.

Author: @Rahul_Ranjan_Sah
"""

# Standard imports
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from numba import njit
from typing import Annotated, Callable, Tuple
from numpy.typing import NDArray

# Local imports
from guardrails.guardrails import with_guardrails
from nrutils import (
    dummy_jacobian_dp, dummy_jacobian_dpc,
    diagAdd, diagMult,
    arth, outerprod, outerdiff, upperTriangle, swap,
    assertEq
)
# from logger import Logger

# Constants for adaptive step-size control
SAFETY = 0.9
PGROW = -0.20
PSHRNK = -0.25
ERRCON = (5.0 / SAFETY) ** (-5)
TINY = 1.0e-30
MAXSTP = 100_000_000
KMAXX = 8

# Determine whether Numba is available
try:
    import numba
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False

# Helper to optionally compile with Numba
def _jit(parallel: bool = False, cache: bool = True):
    def _decorator(fn):
        return njit(parallel=parallel, cache=cache)(fn) if _USE_NUMBA else fn
    return _decorator


# -----------------------------------------------------------------------------
# RK4 single-step routines
# -----------------------------------------------------------------------------
@_jit(cache=True)
def rk4_dp(
    y: np.ndarray, dydt: np.ndarray, t: float, h: float,
    D: Callable[[float, np.ndarray], np.ndarray]
) -> Tuple[np.ndarray, float]:
    n = y.size
    ytmp = np.empty(n, dtype=np.float64)
    d1 = dydt
    # stage 1
    for i in range(n): ytmp[i] = y[i] + 0.5*h*d1[i]
    d2 = D(t+0.5*h, ytmp)
    # stage 2
    for i in range(n): ytmp[i] = y[i] + 0.5*h*d2[i]
    d3 = D(t+0.5*h, ytmp)
    # stage 3
    for i in range(n): ytmp[i] = y[i] + h*d3[i]
    d4 = D(t+h, ytmp)
    # combine
    yout = np.empty(n, dtype=np.float64)
    for i in range(n):
        yout[i] = y[i] + h/6*(d1[i] + 2*d2[i] + 2*d3[i] + d4[i])
    return yout, t+h

@_jit(cache=True)
def rk4_dpc(
    y: np.ndarray, dydt: np.ndarray, t: float, h: float,
    D: Callable[[float, np.ndarray], np.ndarray]
) -> Tuple[np.ndarray, float]:
    n = y.size
    ytmp = np.empty(n, dtype=np.complex128)
    d1 = dydt
    for i in range(n): ytmp[i] = y[i] + 0.5*h*d1[i]
    d2 = D(t+0.5*h, ytmp)
    for i in range(n): ytmp[i] = y[i] + 0.5*h*d2[i]
    d3 = D(t+0.5*h, ytmp)
    for i in range(n): ytmp[i] = y[i] + h*d3[i]
    d4 = D(t+h, ytmp)
    yout = np.empty(n, dtype=np.complex128)
    for i in range(n):
        yout[i] = y[i] + h/6*(d1[i] + 2*d2[i] + 2*d3[i] + d4[i])
    return yout, t+h

# -----------------------------------------------------------------------------
# Euler single-step
# -----------------------------------------------------------------------------
@_jit()
def idiot_dp(y: np.ndarray, dydt: np.ndarray, t: float, h: float,
             D: Callable[[float, np.ndarray], np.ndarray]) -> Tuple[np.ndarray, float]:
    y += dydt * h
    return y, t + h

@_jit()
def idiot_dpc(y: np.ndarray, dydt: np.ndarray, t: float, h: float,
              D: Callable[[float, np.ndarray], np.ndarray]) -> Tuple[np.ndarray, float]:
    y += dydt * h
    return y, t + h

# -----------------------------------------------------------------------------
# Cash–Karp stage routines
# -----------------------------------------------------------------------------

@_jit(cache=True)
def rkck_dp(
    y: np.ndarray,
    dydt: np.ndarray,
    t: float,
    h: float,
    D: Callable[[float, np.ndarray], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cash–Karp single-stage for real ODEs: returns (yout, yerr).
    """
    n    = y.size
    ytmp = np.empty(n, dtype=np.float64)
    ak2  = np.empty(n, dtype=np.float64)
    ak3  = np.empty(n, dtype=np.float64)
    ak4  = np.empty(n, dtype=np.float64)
    ak5  = np.empty(n, dtype=np.float64)
    ak6  = np.empty(n, dtype=np.float64)
    # coefficients
    a2, a3, a4, a5, a6 = 1/5, 3/10, 3/5, 1.0, 7/8
    b21 = 1/5
    b31, b32 = 3/40, 9/40
    b41, b42, b43 = 3/10, -9/10, 6/5
    b51, b52, b53, b54 = -11/54, 5/2, -70/27, 35/27
    b61, b62, b63, b64, b65 = 1631/55296, 175/512, 575/13824, 44275/110592, 253/4096
    c1, c3, c4, c6 = 37/378, 250/621, 125/594, 512/1771
    dc1, dc3, dc4, dc5, dc6 = 2825/27648, 18575/48384, 13525/55296, 277/14336, 1/4
    # stage 2
    for i in range(n): ytmp[i] = y[i] + b21 * h * dydt[i]
    ak2 = D(t + a2*h, ytmp)
    # stage 3
    for i in range(n): ytmp[i] = y[i] + h*(b31*dydt[i] + b32*ak2[i])
    ak3 = D(t + a3*h, ytmp)
    # stage 4
    for i in range(n): ytmp[i] = y[i] + h*(b41*dydt[i] + b42*ak2[i] + b43*ak3[i])
    ak4 = D(t + a4*h, ytmp)
    # stage 5
    for i in range(n): ytmp[i] = y[i] + h*(b51*dydt[i] + b52*ak2[i] + b53*ak3[i] + b54*ak4[i])
    ak5 = D(t + a5*h, ytmp)
    # stage 6
    for i in range(n): ytmp[i] = y[i] + h*(b61*dydt[i] + b62*ak2[i] + b63*ak3[i] + b64*ak4[i] + b65*ak5[i])
    ak6 = D(t + a6*h, ytmp)
    # combine
    yout = np.empty(n, dtype=np.float64)
    yerr = np.empty(n, dtype=np.float64)
    for i in range(n):
        yout[i] = y[i] + h*(c1*dydt[i] + c3*ak3[i] + c4*ak4[i] + c6*ak6[i])
        yerr[i] = h*(dc1*dydt[i] + dc3*ak3[i] + dc4*ak4[i] + dc5*ak5[i] + dc6*ak6[i])
    return yout, yerr

@_jit(cache=True)
def rkck_dpc(
    y: np.ndarray,
    dydt: np.ndarray,
    t: float,
    h: float,
    D: Callable[[float, np.ndarray], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cash–Karp single-stage for complex ODEs: returns (yout, yerr).
    """
    n = y.size
    ytmp = np.empty(n, dtype=np.complex128)
    ak2  = np.empty(n, dtype=np.complex128)
    ak3  = np.empty(n, dtype=np.complex128)
    ak4  = np.empty(n, dtype=np.complex128)
    ak5  = np.empty(n, dtype=np.complex128)
    ak6  = np.empty(n, dtype=np.complex128)
    # reuse coefficients
    a2, a3, a4, a5, a6 = 1/5, 3/10, 3/5, 1.0, 7/8
    b21 = 1/5
    b31, b32 = 3/40, 9/40
    b41, b42, b43 = 3/10, -9/10, 6/5
    b51, b52, b53, b54 = -11/54, 5/2, -70/27, 35/27
    b61, b62, b63, b64, b65 = 1631/55296, 175/512, 575/13824, 44275/110592, 253/4096
    c1, c3, c4, c6 = 37/378, 250/621, 125/594, 512/1771
    dc1, dc3, dc4, dc5, dc6 = 2825/27648, 18575/48384, 13525/55296, 277/14336, 1/4
    for i in range(n): ytmp[i] = y[i] + b21*h*dydt[i]
    ak2 = D(t+a2*h, ytmp)
    for i in range(n): ytmp[i] = y[i] + h*(b31*dydt[i]+b32*ak2[i])
    ak3 = D(t+a3*h, ytmp)
    for i in range(n): ytmp[i] = y[i] + h*(b41*dydt[i]+b42*ak2[i]+b43*ak3[i])
    ak4 = D(t+a4*h, ytmp)
    for i in range(n): ytmp[i] = y[i] + h*(b51*dydt[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i])
    ak5 = D(t+a5*h, ytmp)
    for i in range(n): ytmp[i] = y[i] + h*(b61*dydt[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i])
    ak6 = D(t+a6*h, ytmp)
    yout = np.empty(n, dtype=np.complex128)
    yerr = np.empty(n, dtype=np.complex128)
    for i in range(n):
        yout[i] = y[i] + h*(c1*dydt[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i])
        yerr[i] = h*(dc1*dydt[i]+dc3*ak3[i]+dc4*ak4[i]+dc5*ak5[i]+dc6*ak6[i])
    return yout, yerr

# -----------------------------------------------------------------------------
# Cash–Karp adaptive RK single-step routines
# -----------------------------------------------------------------------------

@_jit(cache=True)
def rkqs_dp(
    y: np.ndarray,
    dydt: np.ndarray,
    t: float,
    htry: float,
    eps: float,
    yscale: np.ndarray,
    D: Callable
) -> Tuple[np.ndarray, float, float, float]:
    """
    Adaptive Runge–Kutta step control (Cash–Karp) for real ODEs.
    Returns (y, tnew, hdid, hnext).
    """
    n = y.size
    yerr = np.empty(n, dtype=np.float64)
    ytmp = np.empty(n, dtype=np.float64)
    h = htry
    errmax = 0.0
    while True:
        ytmp, yerr = rkck_dp(y, dydt, t, h, D)
        errmax = np.max(yerr / yscale) / eps
        if errmax <= 1.0:
            break
        htmp = SAFETY * h * errmax**PSHRNK
        h = htmp if h >= 0.0 else -htmp
        h = max(htmp, 0.1 * abs(h))
        if t + h == t:
            # Logger.getInstance().error("Stepsize underflow in rkqs_dp", 1)
            raise ValueError("Stepsize underflow in rkqs_dp")
        h = h
    hnext = SAFETY * h * errmax**PGROW if errmax > ERRCON else 5.0 * h
    t = t + h
    hdid = h
    y[:] = ytmp
    return y, t, hdid, hnext

@_jit(cache=True)
def rkqs_dpc(
    y: np.ndarray,
    dydt: np.ndarray,
    t: float,
    htry: float,
    eps: float,
    yscale: np.ndarray,
    D: Callable
) -> Tuple[np.ndarray, float, float, float]:
    """
    Adaptive Runge–Kutta step control (Cash–Karp) for complex ODEs.
    Returns (y, tnew, hdid, hnext).
    """
    n = y.size
    yerr = np.empty(n, dtype=np.complex128)
    ytmp = np.empty(n, dtype=np.complex128)
    h = htry
    errmax = 0.0
    while True:
        ytmp, yerr = rkck_dpc(y, dydt, t, h, D)
        errmax = np.max(np.abs(yerr) / yscale) / eps
        errmax = np.abs(errmax)
        if errmax <= 1.0:
            break
        htmp = SAFETY * np.abs(h) * errmax**PSHRNK
        h = htmp if h.real >= 0.0 else -htmp
        h = max(htmp, 0.1 * abs(h))
        if t + h == t:
            # Logger.getInstance().error("Stepsize underflow in rkqs_dpc", 1)
            raise ValueError("Stepsize underflow in rkqs_dpc")
        h = h
    hnext = SAFETY * h * errmax**PGROW if errmax > ERRCON else 5.0 * h
    t = t + h
    hdid = h
    y[:] = ytmp
    return y, t, hdid, hnext

# -----------------------------------------------------------------------------
# Modified midpoint for Bulirsch–Stoer
# -----------------------------------------------------------------------------
@_jit(cache=True)
def mmid(y: np.ndarray, dydx: np.ndarray, xs: float, htot: float,
         nstep: int, D: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    n = y.size
    h = htot / nstep
    ym = y.copy()
    yn = y + h * dydx
    x = xs + h
    yout = D(x, yn)
    for _ in range(2, nstep+1):
        tmp = ym + 2*h * yout
        ym, yn = yn, tmp
        x += h
        yout = D(x, yn)
    return 0.5 * (ym + yn + h * yout)

# -----------------------------------------------------------------------------
# Richardson extrapolation for Bulirsch–Stoer
# -----------------------------------------------------------------------------
@_jit(cache=True)
def pzextr(iest, xest, yest, yz, dy):
    xarr = np.zeros(KMAXX+1, dtype=np.float64)
    d = np.zeros((KMAXX+1, KMAXX+1, yest.size), dtype=np.float64)
    xarr[iest] = xest
    if iest == 1:
        d[1, 1, :] = yest
        dy = yest.copy()
        yz = yest.copy()
    else:
        c = yest.copy()
        for k in range(1, iest):
            dk = iest - k
            f1 = xest / (xarr[dk] - xest)
            f2 = xarr[dk] / (xarr[dk] - xest)
            temp = d[k, 1, :].copy()
            d[k, 1, :] = dy
            dy = f1 * (dy - temp)
            c = f2 * (c - temp)
            yz += dy
        d[iest, 1, :] = dy
    return yz, dy

# -----------------------------------------------------------------------------
# Bulirsch–Stoer step
# -----------------------------------------------------------------------------
@_jit()
def bsstep(y: np.ndarray, dydx: np.ndarray, x: float, htry: float, eps: float,
            yscale: np.ndarray, D: Callable[[float, np.ndarray], np.ndarray],
            jacobn: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, float, float]:
    """
    Bulirsch–Stoer step using modified midpoint (mmid) and Richardson extrapolation (pzextr).
    """
    # sequence and work arrays
    nseq = np.array([2,4,6,8,10,12,14,16,18], dtype=np.int32)
    ysave = y.copy(); err = np.empty(KMAXX, dtype=np.float64)
    # attempt extrapolation
    for k in range(1, KMAXX+1):
        m = nseq[k-1]
        yseq = mmid(ysave, dydx, x, htry, m, D)
        yout, yerr = pzextr(k, (htry/m)**2, yseq, y, np.empty_like(y))
        if np.max(np.abs(yerr)/yscale) < eps:
            return yout, htry, htry
    raise ValueError("Bulirsch–Stoer failed to converge.")


# -----------------------------------------------------------------------------
# Semi-implicit predictor-corrector (Bulirsch–Stoer helper)
# -----------------------------------------------------------------------------
def simpr(y: np.ndarray,
         dydx: np.ndarray,
         dfdx: np.ndarray,
         dfdy: np.ndarray,
         xs: float,
         htot: float,
         nstep: int,
         derivs: Callable[[float, np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Semi-implicit predictor-corrector step for Bulirsch–Stoer.
    Builds A = I – h*dfdy, factors it once, then performs predictor+corrector
    across nstep sub-intervals.
    """
    ndum = y.size
    h = htot / nstep
    x = xs

    # Form A = I - h*dfdy
    A = -h * dfdy.copy()
    np.fill_diagonal(A, A.diagonal() + 1.0)

    # LU-factor once
    lu, piv = lu_factor(A, overwrite_a=False)

    # Predictor
    yout = h * (dydx + h * dfdx)
    yout = lu_solve((lu, piv), yout)
    del_vec = yout.copy()
    ytemp = y + del_vec
    x += h
    yout = derivs(x, ytemp)

    # Corrector loop
    for _ in range(2, nstep + 1):
        yout = h * yout - del_vec
        yout = lu_solve((lu, piv), yout)
        del_vec += 2.0 * yout
        ytemp += del_vec
        x += h
        yout = derivs(x, ytemp)

    # Final correction
    yout = h * yout - del_vec
    yout = lu_solve((lu, piv), yout)
    return ytemp + yout


# -----------------------------------------------------------------------------
# Stiff Bulirsch–Stoer stepper (stifbs)
# -----------------------------------------------------------------------------
def stifbs(y: np.ndarray,
           dydx: np.ndarray,
           x: float,
           htry: float,
           eps: float,
           yscal: np.ndarray,
           derivs: Callable[[float, np.ndarray], np.ndarray],
           jacobn: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]
) -> Tuple[float, float, float]:
    """
    Single step of the stiff Bulirsch–Stoer algorithm.
    Returns (x_new, h_used, h_next).
    """
    # Constants
    imax, kmaxx = 8, 7
    safe1, safe2 = 0.25, 0.7
    redmax, redmin = 1e-5, 0.7
    tiny, scalmx = 1e-30, 0.1
    nseq = np.array([2, 6, 10, 14, 22, 34, 50, 70], dtype=np.int32)

    # Jacobian at the start
    dfdx, dfdy = jacobn(x, y)
    ysav = y.copy()

    # Extrapolation coefficients
    eps1 = safe1 * eps
    a = np.cumsum(nseq.astype(np.float64))
    a2 = a[1:]
    ap = outerprod(arth(3.0, 2.0, kmaxx), a2 - a[0] + 1.0)
    od = outerdiff(a2, a2)
    mask = upperTriangle(kmaxx, kmaxx).astype(bool)
    alf = np.ones((kmaxx, kmaxx), dtype=np.float64)
    alf[mask] = eps1 ** (od[mask] / ap[mask])

    # Work arrays
    err = np.zeros(kmaxx, dtype=np.float64)
    yerr = np.zeros_like(y)
    yseq = np.zeros_like(y)
    kopt = kmaxx

    # Adaptation loop
    while True:
        for k in range(1, kopt + 1):
            xnew = x + htry
            if xnew == x:
                raise ValueError("Step size underflow in stifbs.")

            # midpoint + extrapolate
            yseq = simpr(ysav, dydx, dfdx, dfdy, x, htry, int(nseq[k-1]), derivs)
            _, yerr = pzextr(k, (htry / nseq[k-1])**2, yseq, y, yerr)

            if k > 1:
                errmax = np.max(np.abs(yerr / yscal))
                errmax = max(tiny, errmax) / eps

            if k > 1 and errmax < 1.0:
                # Accept step
                x = xnew
                hdid = htry
                # next step size estimate
                scale = max(errmax, scalmx)
                hnext = htry / scale
                return x, hdid, hnext

        # If no k succeeded, reduce h and retry
        red = max(min(errmax**(-1/(2*kopt+1)) * safe2, redmin), redmax)
        htry *= red
# # -----------------------------------------------------------------------------
# # Semi-implicit predictor-corrector (Bulirsch–Stoer helper)
# # -----------------------------------------------------------------------------
# @_jit(cache=True)
# def simpr(y: np.ndarray,
#          dydx: np.ndarray,
#          dfdx: np.ndarray,
#          dfdy: np.ndarray,
#          xs: float,
#          htot: float,
#          nstep: int,
#          derivs: Callable[[float, np.ndarray], np.ndarray]
# ) -> np.ndarray:
#     ndum = y.size
#     h = htot / nstep
#     x = xs
#     a = -h * dfdy.copy()
#     for i in range(a.shape[0]):
#         a[i, i] += 1.0
#     indx = np.empty(ndum, dtype=np.int32)
#     _ = ludcmp(a, indx)
#     # predictor
#     yout = h * (dydx + h * dfdx)
#     lubksb(a, indx, yout)
#     del_vec = yout.copy()
#     ytemp = y + del_vec
#     x += h
#     yout = derivs(x, ytemp)
#     # corrector steps
#     for _ in range(2, nstep+1):
#         yout = h * yout - del_vec
#         lubksb(a, indx, yout)
#         del_vec += 2.0 * yout
#         ytemp += del_vec
#         x += h
#         yout = derivs(x, ytemp)
#     # final correction
#     yout = h * yout - del_vec
#     lubksb(a, indx, yout)
#     return ytemp + yout

# # -----------------------------------------------------------------------------
# # Stiff Bulirsch–Stoer stepper (stifbs)
# # -----------------------------------------------------------------------------
# @_jit()
# def stifbs(y: NDArray[np.float64],
#            dydx: NDArray[np.float64],
#            x: float,
#            htry: float,
#            eps: float,
#            yscal: NDArray[np.float64],
#            derivs: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
#            jacobn: Callable[[float, NDArray[np.float64]], Tuple[NDArray[np.float64], NDArray[np.float64]]]
# ) -> Tuple[float, float, float]:
#     # constants
#     imax, kmaxx = 8, 7
#     safe1, safe2 = 0.25, 0.7
#     redmax, redmin = 1e-5, 0.7
#     tiny, scalmx = 1e-30, 0.1
#     nseq = np.array([2,6,10,14,22,34,50,70], dtype=np.int32)
#     # Jacobs
#     dfdx, dfdy = jacobn(x, y)
#     ysav = y.copy()
#     # extrap coefficients
#     eps1 = safe1 * eps
#     a = np.cumsum(nseq.astype(np.float64), dtype=np.float64)
#     a2 = a[1:]
#     ap = outerprod(arth(3.0, 2.0, kmaxx), a2 - a[0] + 1.0)
#     od = outerdiff(a2, a2)
#     mask = upperTriangle(kmaxx, kmaxx).astype(bool)
#     alf = np.ones((kmaxx, kmaxx), dtype=np.float64)
#     alf[mask] = eps1 ** (od[mask] / ap[mask])
#     # work arrays
#     err = np.zeros(kmaxx, dtype=np.float64)
#     yerr = np.zeros(y.size, dtype=np.float64)
#     yseq = np.zeros_like(y)
#     first = True
#     kopt = kmaxx
#     # adaptation loop
#     while True:
#         for k in range(1, kopt+1):
#             xnew = x + htry
#             if xnew == x:
#                 raise ValueError("Step size underflow in stifbs.")
#             # midpoint + extrapolate
#             yseq = simpr(ysav, dydx, dfdx, dfdy, x, htry, int(nseq[k-1]), derivs)
#             _, yerr = pzextr(k, (htry/nseq[k-1])**2, yseq, y, yerr)
#             if k > 1:
#                 errmax = np.max(np.abs(yerr/yscal))
#                 errmax = max(tiny, errmax) / eps
#             if k > 1 and errmax < 1.0:
#                 # accept
#                 x = xnew
#                 hdid = htry
#                 # estimate next h
#                 scale = max(errmax, scalmx)
#                 hnext = htry / scale
#                 return x, hdid, hnext
#         # reduce and retry
#         red = max(min(errmax**(-1/(2*kopt+1)) * safe2, redmin), redmax)
#         htry *= red
#         first = False



# -----------------------------------------------------------------------------
# LU decomposition and back-substitution
# -----------------------------------------------------------------------------
# @_jit(cache=True)
# def ludcmp(a: np.ndarray, indx: np.ndarray) -> float:
#     n = a.shape[0]
#     assert a.shape[0] == a.shape[1] == indx.size
#     vv = np.empty(n, dtype=np.float64)
#     d = np.float64(1.0)
#     for i in range(n):
#         big = 0.0
#         for j in range(n):
#             big = max(big, abs(a[i, j]))
#         vv[i] = 1.0 / big
#         if vv[i] == 0.0:
#             raise ValueError('singular matrix in ludcmp')
#     for j in range(n):
#         imax = j
#         big = 0.0
#         for i in range(j, n):
#             temp = vv[i] * abs(a[i, j])
#             if temp > big:
#                 big = temp; imax = i
#         if j != imax:
#             tmp = a[imax, :].copy()
#             a[imax, :] = a[j, :]
#             a[j, :] = tmp
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
# def lubksb(a: np.ndarray, indx: np.ndarray, b: np.ndarray) -> None:
#     n = a.shape[0]
#     assert a.shape[0] == a.shape[1] == indx.size
#     ii = 0
#     for i in range(n):
#         ll = indx[i]
#         sum_ = b[ll]
#         b[ll] = b[i]
#         b[i] = sum_
#     for i in range(n-1, -1, -1):
#         sum_ = b[i]
#         for j in range(i+1, n):
#             sum_ -= a[i, j] * b[j]
#         b[i] = sum_ / a[i, i]


def ludcmp(a: np.ndarray, indx: np.ndarray) -> float:
    """
    Wrapper around LAPACK’s DGETRF via SciPy.

    Overwrites `a` in place with the combined L and U factors,
    fills `indx` with the pivot indices (0-based),
    and returns +1.0 or –1.0 depending on the parity of row swaps.
    """
    # lu: combined L & U in one matrix, piv: pivot indices
    lu, piv = lu_factor(a, overwrite_a=True)
    a[:, :] = lu
    indx[:] = piv

    # compute parity: odd number of interchanges → –1.0
    swaps = np.count_nonzero(piv != np.arange(piv.size))
    return -1.0 if (swaps % 2) else +1.0

def lubksb(a: np.ndarray, indx: np.ndarray, b: np.ndarray) -> None:
    """
    Wrapper around LAPACK’s DGETRS via SciPy.

    Solves A·x = b in two steps using the LU factors in `a`
    and pivot array `indx`.  Overwrites `b` with the solution.
    """
    x = lu_solve((a, indx), b)
    b[:] = x

# -----------------------------------------------------------------------------
# 3D complex integrators
# -----------------------------------------------------------------------------
@_jit(cache=True)
def odeint_3D_dpc(y,x1,x2,eps,h1,hmin,D,integ):
    yscale=np.empty_like(y.real); dydx=np.empty_like(y)
    x,h,nok,nbad,prev=h1, h1 if x2>=x1 else -h1,0,0, None
    x, h1 = x1, h
    for _ in range(MAXSTP):
        dydx=D(x,y)
        baseline=np.max(abs(y))*1e-4
        np.maximum(baseline, abs(y)+abs(h*dydx), out=yscale)
        if (x+h-x2)*(x+h-x1)>0: h=x2-x
        hdid,hnext=integ(y,dydx,x,h,eps,yscale,D)
        nok+=hdid==h; nbad+=hdid!=h
        x+=hdid
        if (x-x2)*(x2-x1)>=0: return y,nok,nbad
        if abs(hnext)<=abs(hmin): raise ValueError('Step too small 3D')
        prev, h = hdid, hnext
    raise ValueError('Too many steps 3D')

# reuse stepper wrappers for 3D
rkck_3D_dpc = lambda y,dydt,t,h,D: (
    *rkck_dpc(y.ravel(),dydt.ravel(),t,h,D),)

rk4_3D_dpc = lambda y,dydt,t,h,D: (
    *rk4_dp(y.ravel(),dydt.ravel(),t,h,D),)

# Adaptive control and variant with real/imag scaling
@_jit(cache=True)
def odeint_3D_dpc_TOM(y,x1,x2,eps,h1,hmin,D,integ):
    yr, yi = np.empty_like(y.real), np.empty_like(y.real)
    dydx=np.empty_like(y)
    x,h,nok,nbad,prev=x1,(h1 if x2>=x1 else -h1),0,0,None
    for _ in range(MAXSTP):
        dydx=D(x,y)
        base=np.max(abs(y))*1e-6
        np.maximum(base,abs(y.real)+abs((h*dydx).real), out=yr)
        np.maximum(base,abs(y.imag)+abs((h*dydx).imag), out=yi)
        if (x+h-x2)*(x+h-x1)>0: h=x2-x
        hdid,hnext=integ(y,dydx,x,h,eps,yr,yi,D)
        nok+=hdid==h; nbad+=hdid!=h
        x+=hdid
        if (x-x2)*(x2-x1)>=0: return y,nok,nbad
        if abs(hnext)<=abs(hmin): raise ValueError('Step too small 3D_TOM')
        prev, h = hdid, hnext
    raise ValueError('Too many steps 3D_TOM')

# -----------------------------------------------------------------------------
# Internal drivers (fast path)
# -----------------------------------------------------------------------------

@_jit()
def _odeint_dp_impl(
    y: np.ndarray,
    x1: float,
    x2: float,
    eps: float,
    h1: float,
    hmin: float,
    D: Callable,
    stepper: Callable,
    jacobian: Callable
) -> Tuple[np.ndarray, np.int32, np.int32]:
    """
    Internal adaptive-step driver for real double precision.
    """
    n = y.size
    yscale = np.empty(n, dtype=np.float64)
    x = x1
    h = h1 if (x2 - x1) >= 0.0 else -h1
    nok = np.int32(0)
    nbad = np.int32(0)
    for _ in range(MAXSTP):
        dydx = D(x, y)
        for i in range(n):
            yscale[i] = abs(y[i]) if abs(y[i]) > 1.0 else 1.0
        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x
        hdid, hnext = stepper(y, dydx, x, h, eps, yscale, D, jacobian)
        if hdid == h:
            nok += 1
        else:
            nbad += 1
        x += hdid
        if (x - x2) * (x2 - x1) >= 0.0:
            return y, nok, nbad
        if abs(hnext) <= abs(hmin):
            raise ValueError("Step size too small in odeint_dp.")
        h = hnext
    raise ValueError("Too many steps in odeint_dp.")

@_jit()
def _odeint_dpc_impl(
    y: np.ndarray,
    x1: float,
    x2: float,
    eps: float,
    h1: float,
    hmin: float,
    D: Callable,
    stepper: Callable,
    jacobian: Callable
) -> Tuple[np.ndarray, np.int32, np.int32]:
    """
    Internal adaptive-step driver for complex double precision.
    """
    return _odeint_dp_impl(y, x1, x2, eps, h1, hmin, D, stepper, jacobian)

# -----------------------------------------------------------------------------
# Class-based API
# -----------------------------------------------------------------------------

class Integrator:
    """
    Adaptive ODE integrator for real (float64) problems.
    """
    def __init__(
        self,
        deriv: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
        stepper: Callable,
        jacobian: Callable[[float, NDArray[np.float64]], Tuple[NDArray[np.float64], NDArray[np.float64]]],
        h1: float,
        hmin: float
    ):
        self._D = deriv
        self._stepper = stepper
        self._jacobian = jacobian
        self._h1 = h1
        self._hmin = hmin

    @with_guardrails
    def solve_dp(
        self,
        y0: Annotated[NDArray[np.float64], np.float64],
        x1: Annotated[float, np.float64],
        x2: Annotated[float, np.float64],
        eps: Annotated[float, np.float64]
    ) -> Tuple[NDArray[np.float64], np.int32, np.int32]:
        return _odeint_dp_impl(
            y0.copy(), x1, x2, eps,
            self._h1, self._hmin,
            self._D, self._stepper, self._jacobian
        )

class ComplexIntegrator(Integrator):
    """
    Adaptive ODE integrator for complex (complex128) problems.
    """
    def __init__(
        self,
        deriv: Callable[[float, NDArray[np.complex128]], NDArray[np.complex128]],
        stepper: Callable,
        jacobian: Callable[[float, NDArray[np.complex128]], Tuple[NDArray[np.complex128], NDArray[np.complex128]]],
        h1: float,
        hmin: float
    ):
        super().__init__(deriv, stepper, jacobian, h1, hmin)

    @with_guardrails
    def solve_dpc(
        self,
        y0: Annotated[NDArray[np.complex128], np.complex128],
        x1: Annotated[float, np.float64],
        x2: Annotated[float, np.float64],
        eps: Annotated[float, np.float64]
    ) -> Tuple[NDArray[np.complex128], np.int32, np.int32]:
        return _odeint_dpc_impl(
            y0.copy(), x1, x2, eps,
            self._h1, self._hmin,
            self._D, self._stepper, self._jacobian
        )

class Complex3DIntegrator:
    def __init__(self, deriv, stepper, h1, hmin):
        self._D, self._stepper, self._h1, self._hmin = deriv, stepper, h1, hmin

    @with_guardrails
    def solve_dpc_3D(self, y0: NDArray[np.complex128], x1: float, x2: float, eps: float):
        return odeint_3D_dpc(y0.copy(), x1, x2, eps,
                             self._h1, self._hmin,
                             self._D, self._stepper)

    @with_guardrails
    def solve_dpc_3D_TOM(self, y0: NDArray[np.complex128], x1: float, x2: float, eps: complex):
        return odeint_3D_dpc_TOM(y0.copy(), x1, x2, eps,
                                 self._h1, self._hmin,
                                 self._D, self._stepper)
