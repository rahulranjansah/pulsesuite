"""
ODE integrators ported from integrator.F90.

1:1 port of the Fortran ``integrator`` module.  Every public Fortran
subroutine has a Python callable with the same name.  Fortran interfaces
that dispatch on type/rank are kept as separate named functions.

Vectorised with NumPy; no Numba or guardrails required.

Implements:
    - Cash-Karp Runge-Kutta step (rkck)
    - Quality-controlled adaptive RK step (rkqs)
    - Classic 4th-order Runge-Kutta (rk4)
    - Forward Euler (idiot)
    - Adaptive ODE driver (odeint)
    - Simple fixed-step driver (simpleint)
    - Modified midpoint (mmid)
    - Polynomial extrapolation (pzextr)
    - Bulirsch-Stoer adaptive step (bsstep)
    - Semi-implicit midpoint for stiff systems (simpr)
    - Stiff Bulirsch-Stoer (stifbs)
    - LU decomposition with partial pivoting (ludcmp)
    - LU back-substitution (lubksb)

Author: Rahul R. Sah
"""

import sys

import numpy as np

from .nrutils import (
    arth,
    assertEq,
    cumsum,
    diagAdd,
    nrerror,
    outerdiff,
    outerprod,
    upperTriangle,
)

sp = np.float32
dp = np.float64

# ─── Module constants (from integrator.F90) ──────────────────────────
SAFETY = 0.9
PGROW = -0.20
PSHRNK = -0.25
ERRCON = (5.0 / SAFETY) ** (-5)
TINY = 1.0e-30
MAXSTP = 100_000_000
KMAXX = 8


# ─── Cash-Karp Butcher tableau ───────────────────────────────────────
_a2 = 1.0 / 5.0
_a3 = 3.0 / 10.0
_a4 = 3.0 / 5.0
_a5 = 1.0
_a6 = 7.0 / 8.0

_b21 = 1.0 / 5.0
_b31 = 3.0 / 40.0
_b32 = 9.0 / 40.0
_b41 = 3.0 / 10.0
_b42 = -9.0 / 10.0
_b43 = 6.0 / 5.0
_b51 = -11.0 / 54.0
_b52 = 5.0 / 2.0
_b53 = -70.0 / 27.0
_b54 = 35.0 / 27.0
_b61 = 1631.0 / 55296.0
_b62 = 175.0 / 512.0
_b63 = 575.0 / 13824.0
_b64 = 44275.0 / 110592.0
_b65 = 253.0 / 4096.0

_c1 = 37.0 / 378.0
_c3 = 250.0 / 621.0
_c4 = 125.0 / 594.0
_c6 = 512.0 / 1771.0

# Error coefficients: difference between 5th-order (c) and 4th-order weights.
# The Fortran source stores the 4th-order weights directly; the actual
# error estimate is  yerr = h * (c5th - c4th) . k_i
_dc1 = _c1 - 2825.0 / 27648.0
_dc3 = _c3 - 18575.0 / 48384.0
_dc4 = _c4 - 13525.0 / 55296.0
_dc5 = 0.0 - 277.0 / 14336.0       # c5 = 0 in the 5th-order formula
_dc6 = _c6 - 1.0 / 4.0


# ═════════════════════════════════════════════════════════════════════
#  Cash-Karp Runge-Kutta step  (rkck)
#  Returns (yout, yerr).  Does NOT modify y.
# ═════════════════════════════════════════════════════════════════════

def _rkck(y, dydt, t, h, D):
    """Common Cash-Karp implementation, dtype/shape agnostic."""
    ytmp = y + _b21 * h * dydt
    ak2 = D(t + _a2 * h, ytmp)

    ytmp = y + h * (_b31 * dydt + _b32 * ak2)
    ak3 = D(t + _a3 * h, ytmp)

    ytmp = y + h * (_b41 * dydt + _b42 * ak2 + _b43 * ak3)
    ak4 = D(t + _a4 * h, ytmp)

    ytmp = y + h * (_b51 * dydt + _b52 * ak2 + _b53 * ak3 + _b54 * ak4)
    ak5 = D(t + _a5 * h, ytmp)

    ytmp = y + h * (_b61 * dydt + _b62 * ak2 + _b63 * ak3
                    + _b64 * ak4 + _b65 * ak5)
    ak6 = D(t + _a6 * h, ytmp)

    yout = y + h * (_c1 * dydt + _c3 * ak3 + _c4 * ak4 + _c6 * ak6)
    yerr = h * (_dc1 * dydt + _dc3 * ak3 + _dc4 * ak4
                + _dc5 * ak5 + _dc6 * ak6)
    return yout, yerr


def rkck_dp(y, dydt, t, h, D):
    """Cash-Karp RK step for real (float64) ODEs.  Returns (yout, yerr)."""
    return _rkck(y, dydt, t, h, D)


def rkck_dpc(y, dydt, t, h, D):
    """Cash-Karp RK step for complex (complex128) ODEs.  Returns (yout, yerr)."""
    return _rkck(y, dydt, t, h, D)


def rkck_3D_dpc(y, dydt, t, h, D):
    """Cash-Karp RK step for 3D complex arrays.  Returns (yout, yerr)."""
    return _rkck(y, dydt, t, h, D)


# ═════════════════════════════════════════════════════════════════════
#  Classic 4th-order Runge-Kutta  (rk4)
#  Modifies y in-place.  Returns t_new.
# ═════════════════════════════════════════════════════════════════════

def _rk4(y, dydt, t, h, D):
    """Common RK4 implementation."""
    ytmp = y + (h / 2.0) * dydt
    dyt = D(t + h / 2.0, ytmp)

    ytmp = y + (h / 2.0) * dyt
    dym = D(t + h / 2.0, ytmp)

    ytmp = y + h * dym
    dym = dym + dyt

    dyt = D(t + h, ytmp)

    y[:] = y + (h / 6.0) * (dydt + dyt + 2.0 * dym)
    return t + h


def rk4_dp(y, dydt, t, h, D):
    """Classic RK4 for real ODEs.  Modifies y in-place.  Returns t_new."""
    return _rk4(y, dydt, t, h, D)


def rk4_dpc(y, dydt, t, h, D):
    """Classic RK4 for complex ODEs.  Modifies y in-place.  Returns t_new."""
    return _rk4(y, dydt, t, h, D)


def rk4_3D_dpc(y, dydt, t, h, D):
    """Classic RK4 for 3D complex arrays.  Modifies y in-place.  Returns t_new."""
    return _rk4(y, dydt, t, h, D)


# ═════════════════════════════════════════════════════════════════════
#  Forward Euler  (idiot)
#  Modifies y in-place.  Returns t_new.
# ═════════════════════════════════════════════════════════════════════

def idiot_dp(y, dydt, t, h, D):
    """Forward Euler for real ODEs.  Modifies y in-place.  Returns t_new."""
    y[:] = y + dydt * h
    return t + h


def idiot_dpc(y, dydt, t, h, D):
    """Forward Euler for complex ODEs.  Modifies y in-place.  Returns t_new."""
    y[:] = y + dydt * h
    return t + h


# ═════════════════════════════════════════════════════════════════════
#  Adaptive Cash-Karp quality-controlled step  (rkqs)
#  Modifies y in-place.  Returns (t_new, hdid, hnext).
# ═════════════════════════════════════════════════════════════════════

def rkqs_dp(y, dydt, t, htry, eps, yscale, D, jacobn):
    """
    Quality-controlled RK step for real ODEs.

    Modifies y in-place.
    Returns (t_new, hdid, hnext).
    """
    h = htry

    while True:
        ytmp, yerr = rkck_dp(y, dydt, t, h, D)

        errmax = np.max(yerr / yscale) / eps

        if errmax <= 1.0:
            break

        htmp = SAFETY * h * errmax ** PSHRNK

        if h >= 0.0:
            h = max(htmp, 0.1 * h)
        else:
            h = min(htmp, 0.1 * h)

        tnew = t + h
        if tnew == t:
            raise RuntimeError("Stepsize underflow in rkqs")

    if errmax > ERRCON:
        hnext = SAFETY * h * errmax ** PGROW
    else:
        hnext = 5.0 * h

    t_new = t + h
    hdid = h
    y[:] = ytmp
    return t_new, hdid, hnext


def rkqs_dpc(y, dydt, t, htry, eps, yscale, D, jacobn):
    """
    Quality-controlled RK step for complex ODEs.

    Modifies y in-place.
    Returns (t_new, hdid, hnext).
    """
    h = htry

    while True:
        ytmp, yerr = rkck_dpc(y, dydt, t, h, D)

        errmax = np.max(np.abs(yerr) / yscale) / eps

        if errmax <= 1.0:
            break

        htmp = SAFETY * h * errmax ** PSHRNK

        if h >= 0.0:
            h = max(htmp, 0.1 * h)
        else:
            h = min(htmp, 0.1 * h)

        tnew = t + h
        if tnew == t:
            raise RuntimeError("Stepsize underflow in rkqs")

    if errmax > ERRCON:
        hnext = SAFETY * h * errmax ** PGROW
    else:
        hnext = 5.0 * h

    t_new = t + h
    hdid = h
    y[:] = ytmp
    return t_new, hdid, hnext


def rkqs_3D_dpc(y, dydt, t, htry, eps, yscale, D):
    """
    Quality-controlled RK step for 3D complex arrays (UPPE, no Jacobian).

    Modifies y in-place.
    Returns (t_new, hdid, hnext).
    """
    h = htry

    while True:
        ytmp, yerr = rkck_3D_dpc(y, dydt, t, h, D)

        errmax = np.max(np.abs(yerr) / yscale) / eps

        if errmax <= 1.0:
            break

        htmp = SAFETY * h * errmax ** PSHRNK

        if h >= 0.0:
            h = max(htmp, 0.1 * h)
        else:
            h = min(htmp, 0.1 * h)

        tnew = t + h
        if tnew == t:
            raise RuntimeError("Stepsize underflow in rkqs 3D")

    if errmax > ERRCON:
        hnext = SAFETY * h * errmax ** PGROW
    else:
        hnext = 5.0 * h

    t_new = t + h
    hdid = h
    y[...] = ytmp
    return t_new, hdid, hnext


def rkqs_3D_dpc_TOM(y, dydt, t, htry, eps, yscaleREAL, yscaleIMAG, D):
    """
    Quality-controlled RK step for 3D complex arrays with separate
    real/imaginary error tolerances.  eps is complex: real(eps) for
    real part tolerance, imag(eps) for imaginary part tolerance.

    Modifies y in-place.
    Returns (t_new, hdid, hnext).
    """
    h = htry

    while True:
        ytmp, yerr = rkck_3D_dpc(y, dydt, t, h, D)

        errmax = max(
            np.max(np.abs(yerr.real) / yscaleREAL) / eps.real,
            np.max(np.abs(yerr.imag) / yscaleIMAG) / eps.imag,
        )

        if errmax <= 1.0:
            break

        htmp = SAFETY * h * errmax ** PSHRNK

        if h >= 0.0:
            h = max(htmp, 0.1 * h)
        else:
            h = min(htmp, 0.1 * h)

        tnew = t + h
        if tnew == t:
            raise RuntimeError("Stepsize underflow in rkqs 3D TOM")

    if errmax > ERRCON:
        hnext = SAFETY * h * errmax ** PGROW
    else:
        hnext = 5.0 * h

    t_new = t + h
    hdid = h
    y[...] = ytmp
    return t_new, hdid, hnext


# ═════════════════════════════════════════════════════════════════════
#  Adaptive step-size ODE drivers  (odeint)
#  Modifies y in-place.  Returns (nok, nbad).
# ═════════════════════════════════════════════════════════════════════

def odeint_dp(y, x1, x2, eps, h1, hmin, D, integ, jacobn):
    """
    Adaptive ODE driver for real ODEs.

    Parameters
    ----------
    y : ndarray, float64 — state vector (modified in-place)
    x1, x2 : float — integration bounds
    eps : float — error tolerance
    h1 : float — initial step size
    hmin : float — minimum step size
    D : callable(t, y) -> dy — derivative function
    integ : callable — stepper (rkqs_dp, bsstep, stifbs)
    jacobn : callable — Jacobian function

    Returns
    -------
    nok : int — number of good steps
    nbad : int — number of bad (retried) steps
    """
    x = x1
    h = np.copysign(h1, x2 - x1)
    nok = 0
    nbad = 0

    for nstp in range(1, MAXSTP + 1):
        dydx = D(x, y)
        yscale = np.maximum(np.abs(y), 1.0)

        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x

        x, hdid, hnext = integ(y, dydx, x, h, eps, yscale, D, jacobn)

        if hdid == h:
            nok += 1
        else:
            nbad += 1

        if (x - x2) * (x2 - x1) >= 0.0:
            return nok, nbad

        if abs(hnext) <= abs(hmin):
            print(nstp, x, x2, h, nok, nbad, file=sys.stderr)
            raise RuntimeError("Step size too small in odeint.")

        h = hnext

    raise RuntimeError("Too many steps in odeint.")


def odeint_dpc(y, x1, x2, eps, h1, hmin, D, integ, jacobn):
    """
    Adaptive ODE driver for complex ODEs.

    Same interface as odeint_dp but y is complex128.
    Returns (nok, nbad).
    """
    x = x1
    h = np.copysign(h1, x2 - x1)
    nok = 0
    nbad = 0

    for nstp in range(1, MAXSTP + 1):
        dydx = D(x, y)
        yscale = np.maximum(np.abs(y), 1.0)

        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x

        x, hdid, hnext = integ(y, dydx, x, h, eps, yscale, D, jacobn)

        if hdid == h:
            nok += 1
        else:
            nbad += 1

        if (x - x2) * (x2 - x1) >= 0.0:
            return nok, nbad

        if abs(hnext) <= abs(hmin):
            print(nstp, x, x2, h, nok, nbad, file=sys.stderr)
            raise RuntimeError("Step size too small in odeint.")

        h = hnext

    raise RuntimeError("Too many steps in odeint.")


def odeint_3D_dpc(y, x1, x2, eps, h1, hmin, D, integ):
    """
    Adaptive ODE driver for 3D complex arrays (UPPE, no Jacobian).

    h1 is returned modified (remembers last used step size for the
    bridge between pulsesuite "Steps").

    Returns (nok, nbad, h1_out).
    """
    x = x1
    h = np.copysign(h1, x2 - x1)
    nok = 0
    nbad = 0
    hprev = h

    for nstp in range(1, MAXSTP + 1):
        dydx = D(x, y)

        yscale_baseline = np.max(np.abs(y)) * 1.0e-4
        yscale = np.maximum(yscale_baseline, np.abs(y) + np.abs(h * dydx))

        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x

        x, hdid, hnext = integ(y, dydx, x, h, eps, yscale, D)
        # print(f'dz= {h}')  # debug output from Fortran source

        if hdid == h:
            nok += 1
        else:
            nbad += 1

        if (x - x2) * (x2 - x1) >= 0.0:
            return nok, nbad, hprev

        if abs(hnext) <= abs(hmin):
            print(nstp, x, x2, h, nok, nbad, file=sys.stderr)
            raise RuntimeError("Step size too small in odeint.")

        hprev = hdid
        h = hnext

    raise RuntimeError("Too many steps in odeint.")


def odeint_3D_dpc_TOM(y, x1, x2, eps, h1, hmin, D, integ):
    """
    Adaptive ODE driver for 3D complex arrays with separate real/imaginary
    error tolerances.  eps is complex.

    Returns (nok, nbad, h1_out).
    """
    x = x1
    h = np.copysign(h1, x2 - x1)
    nok = 0
    nbad = 0
    hprev = h

    for nstp in range(1, MAXSTP + 1):
        dydx = D(x, y)

        yscale_baseline = np.max(np.abs(y)) * 1.0e-6
        yscaleREAL = np.maximum(
            yscale_baseline,
            np.abs(y.real) + np.abs((h * dydx).real),
        )
        yscaleIMAG = np.maximum(
            yscale_baseline,
            np.abs(y.imag) + np.abs((h * dydx).imag),
        )

        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x

        x, hdid, hnext = integ(
            y, dydx, x, h, eps, yscaleREAL, yscaleIMAG, D,
        )

        if hdid == h:
            nok += 1
        else:
            nbad += 1

        if (x - x2) * (x2 - x1) >= 0.0:
            return nok, nbad, hprev

        if abs(hnext) <= abs(hmin):
            print(nstp, x, x2, h, nok, nbad, file=sys.stderr)
            raise RuntimeError("Step size too small in odeint.")

        hprev = hdid
        h = hnext

    raise RuntimeError("Too many steps in odeint.")


# ═════════════════════════════════════════════════════════════════════
#  Simple fixed-step ODE driver  (simpleint)
#  Modifies y in-place.  Returns nok.
# ═════════════════════════════════════════════════════════════════════

def simpleint_dp(y, t1, t2, h1, D, simple):
    """
    Simple fixed-step driver for real ODEs.  Y1 = DY * DX + Y0.

    Parameters
    ----------
    y : ndarray, float64 (modified in-place)
    t1, t2 : float — integration bounds
    h1 : float — step size
    D : callable(t, y) -> dy
    simple : callable — single-step method (rk4_dp, idiot_dp)

    Returns
    -------
    nok : int — number of steps taken
    """
    t = t1
    h = np.copysign(h1, t2 - t1)
    nok = 0

    while True:
        if (t + h - t2) * (t + h - t1) > 0.0:
            h = t2 - t

        t = simple(y, D(t, y), t, h, D)
        nok += 1

        if (t - t2) * (t2 - t1) >= 0.0:
            break

    return nok


def simpleint_dpc(y, t1, t2, h1, D, simple):
    """
    Simple fixed-step driver for complex ODEs.

    Returns nok.
    """
    t = t1
    h = np.copysign(h1, t2 - t1)
    nok = 0

    while True:
        if (t + h - t2) * (t + h - t1) > 0.0:
            h = t2 - t

        t = simple(y, D(t, y), t, h, D)
        nok += 1

        if (t - t2) * (t2 - t1) >= 0.0:
            break

    return nok


# ═════════════════════════════════════════════════════════════════════
#  Modified midpoint  (mmid)
# ═════════════════════════════════════════════════════════════════════

def mmid(y, dydx, xs, htot, nstep, D):
    """
    Modified midpoint method.

    Takes nstep sub-steps of size h = htot/nstep using the modified
    midpoint (Stoermer) rule.  Does NOT modify y.

    Returns yout.
    """
    h = htot / nstep

    ym = y.copy()
    yn = y + h * dydx
    x = xs + h

    yout = D(x, yn)

    for n in range(2, nstep + 1):
        swap_tmp = ym + 2.0 * h * yout
        ym = yn.copy()
        yn = swap_tmp
        x = x + h
        yout = D(x, yn)

    yout = 0.5 * (ym + yn + h * yout)
    return yout


# ═════════════════════════════════════════════════════════════════════
#  Polynomial extrapolation  (pzextr)
#  Uses explicit state arrays passed by the caller.
# ═════════════════════════════════════════════════════════════════════

def pzextr(iest, xest, yest, yz, dy, x_save, d_save):
    """
    Polynomial extrapolation step (Neville's algorithm).

    Parameters
    ----------
    iest : int — 1-based step index
    xest : float — x-estimate (typically (h/nseq)**2)
    yest : ndarray — y-estimate from midpoint/simpr
    yz : ndarray — output y (modified in-place)
    dy : ndarray — output error (modified in-place)
    x_save : ndarray, shape (KMAXX,) — persistent x storage
    d_save : ndarray, shape (nv, KMAXX) — persistent d storage
    """
    nv = yest.size
    x_save[iest - 1] = xest

    yz[:] = yest
    dy[:] = yest

    if iest == 1:
        d_save[:nv, 0] = yest
    else:
        c = yest.copy()
        for k1 in range(1, iest):
            # x_save index for iest-k1 (1-based) → [iest-k1-1] (0-based)
            x_prev = x_save[iest - k1 - 1]
            delta = x_prev - xest
            f1 = xest / delta
            f2 = x_prev / delta
            q = d_save[:nv, k1 - 1].copy()
            d_save[:nv, k1 - 1] = dy.copy()
            d2 = c - q
            dy[:] = f1 * d2
            c[:] = f2 * d2
            yz[:] += dy
        d_save[:nv, iest - 1] = dy.copy()


# ═════════════════════════════════════════════════════════════════════
#  Bulirsch-Stoer adaptive step  (bsstep)
#  Modifies y in-place.  Returns (t_new, hdid, hnext).
# ═════════════════════════════════════════════════════════════════════

# Persistent state across bsstep calls (Fortran SAVE variables)
_bs = {
    'first': True,
    'kmax': 0,
    'kopt': 0,
    'epsold': -1.0,
    'xnew': 0.0,
    'a': np.zeros(KMAXX + 2),
    'alf': np.zeros((KMAXX + 1, KMAXX + 1)),
}

_NSEQ_BS = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])

_IMAXX = KMAXX + 1
_SAFE1_BS = 0.25
_SAFE2_BS = 0.70
_REDMAX_BS = 1.0e-5
_REDMIN_BS = 0.70
_SCALMX_BS = 0.10


def bsstep(y, dydx, x, htry, eps, yscale, D, jacobn):
    """
    Bulirsch-Stoer adaptive step using modified midpoint (mmid) and
    polynomial extrapolation (pzextr).

    Modifies y in-place.
    Returns (t_new, hdid, hnext).
    """
    nv = y.size
    bs = _bs

    # Recompute work coefficients if eps changed
    if eps != bs['epsold']:
        bs['epsold'] = eps
        bs['xnew'] = -1.0e29
        bs['_hnext'] = -1.0e29

        eps1 = _SAFE1_BS * eps

        # a(1) = nseq(1) + 1;  a(k+1) = a(k) + nseq(k+1)  (Fortran 1-based)
        a = bs['a']
        a[0] = _NSEQ_BS[0] + 1
        for k in range(1, KMAXX + 1):
            a[k] = a[k - 1] + _NSEQ_BS[k]

        # Compute alpha coefficients
        alf = bs['alf']
        for iq in range(2, KMAXX + 1):      # Fortran iq=2..KMAXX
            for k in range(1, iq):            # Fortran k=1..iq-1
                alf[k - 1, iq - 1] = eps1 ** (
                    (a[k] - a[iq])
                    / ((a[iq] - a[0] - 1.0) * (2.0 * k + 1.0))
                )

        # Find optimal order
        kopt = KMAXX
        for ko in range(2, KMAXX):
            if a[ko] > a[ko - 1] * alf[ko - 2, ko - 1]:
                kopt = ko
                break
        bs['kopt'] = kopt
        bs['kmax'] = kopt

    kmax = bs['kmax']
    kopt = bs['kopt']
    a = bs['a']
    alf = bs['alf']

    h = htry
    ysav = y.copy()

    if x != bs['xnew'] or h != bs.get('_hnext', -1.0e29):
        bs['first'] = True
        kopt = kmax
        bs['kopt'] = kopt

    reduct = False

    # Extrapolation state arrays (fresh each call)
    x_pz = np.zeros(KMAXX)
    d_pz = np.zeros((nv, KMAXX))
    yerr = np.zeros(nv, dtype=y.dtype)
    err = np.zeros(KMAXX)

    exitflag = False

    while True:
        for k in range(1, kmax + 1):       # Fortran k=1..kmax
            bs['xnew'] = x + h
            if bs['xnew'] == x:
                raise RuntimeError("Step size underflow in bsstep")

            yseq = mmid(ysav, dydx, x, h, _NSEQ_BS[k - 1], D)
            xest = (h / _NSEQ_BS[k - 1]) ** 2

            pzextr(k, xest, yseq, y, yerr, x_pz, d_pz)

            if k != 1:
                errmax = max(TINY, np.max(np.abs(yerr / yscale))) / eps
                km = k - 1
                err[km - 1] = (errmax / _SAFE1_BS) ** (
                    1.0 / (2.0 * km + 1.0)
                )

            if k != 1 and (k >= kopt - 1 or bs['first']):
                if errmax < 1.0:
                    exitflag = True
                    break

                if k == kmax or k == kopt + 1:
                    red = _SAFE2_BS / err[km - 1]
                    break
                elif k == kopt and alf[kopt - 2, kopt - 1] < err[km - 1]:
                    red = 1.0 / err[km - 1]
                    break
                elif kopt == kmax and alf[km - 1, kmax - 2] < err[km - 1]:
                    red = alf[km - 1, kmax - 2] * _SAFE2_BS / err[km - 1]
                    break
                elif alf[km - 1, kopt - 1] < err[km - 1]:
                    red = alf[km - 1, kopt - 2] / err[km - 1]
                    break

        if exitflag:
            break

        red = max(min(red, _REDMIN_BS), _REDMAX_BS)
        h = h * red
        reduct = True

    # Accepted step
    x_new = bs['xnew']
    hdid = h
    bs['first'] = False

    wrkmin = 1.0e35
    kopt_new = 1
    for kk in range(1, km + 1):
        fact = max(err[kk - 1], _SCALMX_BS)
        work = fact * a[kk]
        if work < wrkmin:
            scale = fact
            wrkmin = work
            kopt_new = kk + 1

    hnext = h / scale

    if kopt_new >= k and kopt_new != kmax and not reduct:
        fact = max(scale / alf[kopt_new - 2, kopt_new - 1], _SCALMX_BS)
        if a[kopt_new] * fact <= wrkmin:
            hnext = h / fact
            kopt_new = kopt_new + 1

    bs['kopt'] = kopt_new
    bs['_hnext'] = hnext

    return x_new, hdid, hnext


# ═════════════════════════════════════════════════════════════════════
#  LU decomposition with partial pivoting  (ludcmp)
# ═════════════════════════════════════════════════════════════════════

def ludcmp(a, indx):
    """
    LU decomposition with partial pivoting (Crout's algorithm).

    Modifies a in-place (overwritten with L and U factors).
    Fills indx with pivot indices (0-based).
    Returns d (+1.0 or -1.0 for even/odd row exchanges).
    """
    _TINY_LU = 1.0e-20
    n = assertEq(a.shape[0], a.shape[1], indx.size, msg='ludcmp')

    d = 1.0
    vv = np.max(np.abs(a), axis=1)
    if np.any(vv == 0.0):
        nrerror('singular matrix in ludcmp')
    vv = 1.0 / vv

    for j in range(n):
        imax = j + np.argmax(vv[j:] * np.abs(a[j:, j]))

        if j != imax:
            a[[imax, j], :] = a[[j, imax], :]
            d = -d
            vv[imax] = vv[j]

        indx[j] = imax

        if a[j, j] == 0.0:
            a[j, j] = _TINY_LU

        a[j + 1:, j] /= a[j, j]
        a[j + 1:, j + 1:] -= outerprod(a[j + 1:, j], a[j, j + 1:])

    return d


# ═════════════════════════════════════════════════════════════════════
#  LU back-substitution  (lubksb)
# ═════════════════════════════════════════════════════════════════════

def lubksb(a, indx, b):
    """
    LU back-substitution.

    Solves A*x = b given the LU factors in a and pivot indices in indx.
    Modifies b in-place with the solution.
    """
    n = assertEq(a.shape[0], a.shape[1], indx.size, msg='lubksb')

    ii = None
    for i in range(n):
        ll = indx[i]
        summ = b[ll]
        b[ll] = b[i]
        if ii is not None:
            summ -= np.dot(a[i, ii:i], b[ii:i])
        elif summ != 0.0:
            ii = i
        b[i] = summ

    for i in range(n - 1, -1, -1):
        b[i] = (b[i] - np.dot(a[i, i + 1:], b[i + 1:])) / a[i, i]


# ═════════════════════════════════════════════════════════════════════
#  Semi-implicit midpoint for stiff systems  (simpr)
# ═════════════════════════════════════════════════════════════════════

def simpr(y, dydx, dfdx, dfdy, xs, htot, nstep, derivs):
    """
    Semi-implicit midpoint rule for stiff ODEs.

    Uses LU decomposition of (I - h*dfdy) and performs nstep sub-steps.
    Does NOT modify y.

    Returns yout.
    """
    n = y.size
    h = htot / nstep

    # Build A = I - h * dfdy
    a = -h * dfdy.copy()
    diagAdd(a, 1.0)

    # LU-factor once
    indx = np.zeros(n, dtype=np.intp)
    ludcmp(a, indx)

    # Predictor
    yout = h * (dydx + h * dfdx)
    lubksb(a, indx, yout)
    delta = yout.copy()
    ytemp = y + delta
    x = xs + h
    yout = derivs(x, ytemp)

    # Corrector loop
    for nn in range(2, nstep + 1):
        yout = h * yout - delta
        lubksb(a, indx, yout)
        delta = delta + 2.0 * yout
        ytemp = ytemp + delta
        x = x + h
        yout = derivs(x, ytemp)

    # Final correction
    yout = h * yout - delta
    lubksb(a, indx, yout)
    return ytemp + yout


# ═════════════════════════════════════════════════════════════════════
#  Stiff Bulirsch-Stoer  (stifbs)
#  Modifies y in-place.  Returns (t_new, hdid, hnext).
# ═════════════════════════════════════════════════════════════════════

# Persistent state across stifbs calls (Fortran SAVE variables)
_stiff = {
    'first': True,
    'kmax': 0,
    'kopt': 0,
    'nvold': -1,
    'epsold': -1.0,
    'xnew': 0.0,
    'a': np.zeros(9),
    'alf': np.zeros((7, 7)),
}

_IMAX_STIFF = 8
_KMAXX_STIFF = 7
_SAFE1_STIFF = 0.25
_SAFE2_STIFF = 0.70
_REDMAX_STIFF = 1.0e-5
_REDMIN_STIFF = 0.70
_TINY_STIFF = 1.0e-30
_SCALMX_STIFF = 0.10
_NSEQ_STIFF = np.array([2, 6, 10, 14, 22, 34, 50, 70])


def stifbs(y, dydx, x, htry, eps, yscal, derivs, jacobn):
    """
    Single step of the stiff Bulirsch-Stoer algorithm.

    Uses semi-implicit midpoint (simpr) with polynomial extrapolation (pzextr).

    Modifies y in-place.
    Returns (t_new, hdid, hnext).
    """
    kmaxx = _KMAXX_STIFF
    nseq = _NSEQ_STIFF
    nv = y.size
    st = _stiff

    # Recompute work coefficients if eps or problem size changed
    if eps != st['epsold'] or st['nvold'] != nv:
        st['epsold'] = eps
        st['nvold'] = nv
        st['xnew'] = -1.0e29

        eps1 = _SAFE1_STIFF * eps

        # a = cumsum(nseq) + 1  (work function)
        a_work = cumsum(nseq.astype(np.float64), 1)
        st['a'][:_IMAX_STIFF] = a_work

        # Compute alpha coefficients using nrutils
        a2 = a_work[1:]  # a(2:) in Fortran, 7 elements
        mask = upperTriangle(kmaxx, kmaxx).astype(bool)
        od = outerdiff(a2, a2)
        ap = outerprod(arth(3.0, 2.0, kmaxx), a2 - a_work[0] + 1.0)
        alf = np.zeros((kmaxx, kmaxx))
        alf[mask] = eps1 ** (od[mask] / ap[mask])
        st['alf'] = alf

        # Recompute a with problem-size dependent seed
        a_work = cumsum(nseq.astype(np.float64), 1 + nv)
        st['a'][:_IMAX_STIFF] = a_work

        # Find optimal order
        kopt = kmaxx
        for ko in range(2, kmaxx):   # Fortran: kopt=2..kmaxx-1
            if a_work[ko] > a_work[ko - 1] * alf[ko - 2, ko - 1]:
                kopt = ko
                break
        st['kopt'] = kopt
        st['kmax'] = kopt

    kmax = st['kmax']
    kopt = st['kopt']
    a = st['a']
    alf = st['alf']

    h = htry
    ysav = y.copy()

    # Jacobian at current point
    dfdx, dfdy = jacobn(x, y)

    if h != st.get('_hnext', None) or x != st['xnew']:
        st['first'] = True
        kopt = kmax
        st['kopt'] = kopt

    reduct = False

    # Extrapolation state arrays (fresh each call)
    x_pz = np.zeros(kmaxx + 1)
    d_pz = np.zeros((nv, kmaxx + 1))
    yerr = np.zeros(nv, dtype=y.dtype)
    err = np.zeros(kmaxx)

    exitflag = False

    while True:
        for k in range(1, kmax + 1):       # Fortran k=1..kmax
            st['xnew'] = x + h
            if st['xnew'] == x:
                raise RuntimeError("Step size underflow in stifbs")

            yseq = simpr(ysav, dydx, dfdx, dfdy, x, h, nseq[k - 1], derivs)
            xest = (h / nseq[k - 1]) ** 2

            pzextr(k, xest, yseq, y, yerr, x_pz, d_pz)

            if k != 1:
                errmax = np.max(np.abs(yerr / yscal))
                errmax = max(_TINY_STIFF, errmax) / eps
                km = k - 1
                err[km - 1] = (errmax / _SAFE1_STIFF) ** (
                    1.0 / (2.0 * km + 1.0)
                )

            if k != 1 and (k >= kopt - 1 or st['first']):
                if errmax < 1.0:
                    exitflag = True
                    break

                if k == kmax or k == kopt + 1:
                    red = _SAFE2_STIFF / err[km - 1]
                    break
                elif k == kopt:
                    if alf[kopt - 2, kopt - 1] < err[km - 1]:
                        red = 1.0 / err[km - 1]
                        break
                elif kopt == kmax:
                    if alf[km - 1, kmax - 2] < err[km - 1]:
                        red = alf[km - 1, kmax - 2] * _SAFE2_STIFF / err[km - 1]
                        break
                elif alf[km - 1, kopt - 1] < err[km - 1]:
                    red = alf[km - 1, kopt - 2] / err[km - 1]
                    break
        else:
            # Inner loop completed without break — need to reduce
            if not exitflag:
                red = max(min(red, _REDMIN_STIFF), _REDMAX_STIFF)
                h = h * red
                reduct = True
                continue

        if exitflag:
            break

        red = max(min(red, _REDMIN_STIFF), _REDMAX_STIFF)
        h = h * red
        reduct = True

    # Accepted step
    x_new = st['xnew']
    hdid = h
    st['first'] = False
    st['_hnext'] = None  # will be set below

    # Find new optimal order (Fortran uses minloc-based approach)
    kopt_new = 1 + np.argmin(
        a[1:km + 1] * np.maximum(err[:km], _SCALMX_STIFF)
    )
    scale = max(err[kopt_new - 1], _SCALMX_STIFF)
    wrkmin = scale * a[kopt_new]
    hnext = h / scale

    if kopt_new >= k and kopt_new != kmax and not reduct:
        fact = max(scale / alf[kopt_new - 2, kopt_new - 1], _SCALMX_STIFF)
        if a[kopt_new] * fact <= wrkmin:
            hnext = h / fact
            kopt_new = kopt_new + 1

    st['kopt'] = kopt_new
    st['_hnext'] = hnext

    return x_new, hdid, hnext
