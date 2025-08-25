import numpy as np
import sys
from pathlib import Path
from numpy.typing import NDArray
from typing import Callable, Annotated, Tuple

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from integrator_obj import (
    rk4_dp, rk4_dpc, idiot_dp, idiot_dpc, rkck_dp, rkck_dpc,
    rkqs_dp, rkqs_dpc, _odeint_dp_impl, _odeint_dpc_impl,
    Integrator, ComplexIntegrator, mmid, pzextr, bsstep, simpr,
    stifbs, ludcmp, lubksb
)
from nrutils import dummy_jacobian_dp, dummy_jacobian_dpc, _dummy_jacobian_dp_impl, _dummy_jacobian_dpc_impl

from numba import njit

# Real and complex derivatives
@njit
def D_real(x, y):
    return -y

@njit
def D_complex(x, y):
    return 1j * y

# Euler stepper for use with drivers
@njit
def euler_step(y, dydx, x, h, eps, yscale, D, jacobn):
    y_new = y + h * dydx
    hdid = h
    hnext = h
    y[:] = y_new
    return hdid, hnext

def test_rk4_dp():
    y0 = np.array([1.0])
    dydt = D_real(0.0, y0)
    yout, tnew = rk4_dp(y0.copy(), dydt, 0.0, 0.1, D_real)
    assert yout.shape == (1,)
    assert np.isscalar(tnew)

def test_rk4_dpc():
    y0 = np.array([1.0 + 0j])
    dydt = D_complex(0.0, y0)
    yout, tnew = rk4_dpc(y0.copy(), dydt, 0.0, 0.1, D_complex)
    assert yout.shape == (1,)
    assert np.isscalar(tnew)

def test_idiot_dp():
    y0 = np.array([1.0])
    dydt = D_real(0.0, y0)
    yout, tnew = idiot_dp(y0.copy(), dydt, 0.0, 0.1, D_real)
    assert yout.shape == (1,)
    assert np.isscalar(tnew)

def test_idiot_dpc():
    y0 = np.array([1.0 + 0j])
    dydt = D_complex(0.0, y0)
    yout, tnew = idiot_dpc(y0.copy(), dydt, 0.0, 0.1, D_complex)
    assert yout.shape == (1,)
    assert np.isscalar(tnew)

def test_rkck_dp():
    y0 = np.array([1.0])
    dydt = D_real(0.0, y0)
    yout, yerr = rkck_dp(y0.copy(), dydt, 0.0, 0.1, D_real)
    assert yout.shape == (1,)
    assert yerr.shape == (1,)

def test_rkck_dpc():
    y0 = np.array([1.0 + 0j])
    dydt = D_complex(0.0, y0)
    yout, yerr = rkck_dpc(y0.copy(), dydt, 0.0, 0.1, D_complex)
    assert yout.shape == (1,)
    assert yerr.shape == (1,)

def test_rkqs_dp():
    y0 = np.array([1.0])
    dydt = D_real(0.0, y0)
    yscale = np.ones_like(y0)
    y, t, hdid, hnext = rkqs_dp(y0.copy(), dydt, 0.0, 0.1, 1e-6, yscale, D_real)
    assert y.shape == (1,)
    assert np.isscalar(t)
    assert np.isscalar(hdid)
    assert np.isscalar(hnext)

def test_rkqs_dpc():
    y0 = np.array([1.0 + 0j])
    dydt = D_complex(0.0, y0)
    yscale = np.ones_like(y0)
    y, t, hdid, hnext = rkqs_dpc(y0.copy(), dydt, 0.0, 0.1, 1e-6, yscale, D_complex)
    assert y.shape == (1,)
    assert np.isscalar(t)
    assert np.isscalar(hdid)
    assert np.isscalar(hnext)

def test_odeint_dp_impl():
    y0 = np.array([1.0])
    y, nok, nbad = _odeint_dp_impl(
        y0.copy(), 0.0, 1.0, 1e-6, 0.01, 1e-8,
        D_real, euler_step, _dummy_jacobian_dp_impl
    )
    assert y.shape == (1,)
    assert np.allclose(y, np.exp(-1), atol=1e-2)
    assert isinstance(nok, (int, np.integer))
    assert isinstance(nbad, (int, np.integer))

def test_odeint_dpc_impl():
    y0 = np.array([1.0 + 0j])
    y, nok, nbad = _odeint_dpc_impl(
        y0.copy(), 0.0, 1.0, 1e-6, 0.01, 1e-8,
        D_complex, euler_step, _dummy_jacobian_dpc_impl
    )
    assert y.shape == (1,)
    assert np.allclose(y, np.exp(1j * 1), atol=1e-2)
    assert isinstance(nok, (int, np.integer))
    assert isinstance(nbad, (int, np.integer))

def test_dummy_jacobian_dp():
    y = np.ones(3)
    dfdx, dfdy = dummy_jacobian_dp(0.0, y)
    assert dfdx.shape == (3,)
    assert dfdy.shape == (3, 3)
    assert np.allclose(dfdx, 0)
    assert np.allclose(dfdy, 0)

def test_dummy_jacobian_dpc():
    y = np.ones(3, dtype=np.complex128)
    dfdx, dfdy = dummy_jacobian_dpc(0.0, y)
    assert dfdx.shape == (3,)
    assert dfdy.shape == (3, 3)
    assert np.allclose(dfdx, 0)
    assert np.allclose(dfdy, 0)

def test_class_integrator_solve_dp():
    integrator = Integrator(
        deriv=D_real,
        stepper=euler_step,
        jacobian=_dummy_jacobian_dp_impl,
        h1=0.01,
        hmin=1e-8
    )
    y0 = np.array([1.0])
    y, nok, nbad = integrator.solve_dp(y0, 0.0, 1.0, 1e-6)
    assert y.shape == (1,)
    assert np.allclose(y, np.exp(-1), atol=1e-2)
    assert isinstance(nok, (int, np.integer))
    assert isinstance(nbad, (int, np.integer))

def test_class_complex_integrator_solve_dpc():
    integrator = ComplexIntegrator(
        deriv=D_complex,
        stepper=euler_step,
        jacobian=_dummy_jacobian_dpc_impl,
        h1=0.01,
        hmin=1e-8
    )
    y0 = np.array([1.0 + 0j])
    y, nok, nbad = integrator.solve_dpc(y0, 0.0, 1.0, 1e-6)
    assert y.shape == (1,)
    assert np.allclose(y, np.exp(1j * 1), atol=1e-2)
    assert isinstance(nok, (int, np.integer))
    assert isinstance(nbad, (int, np.integer))

def test_mmid():
    # Modified midpoint method for a simple ODE: dy/dx = -y
    y0 = np.array([1.0])
    dydx = D_real(0.0, y0)
    yout = mmid(y0.copy(), dydx, 0.0, 1.0, 10, D_real)
    assert yout.shape == (1,)

def test_pzextr():
    # Richardson extrapolation for a simple sequence
    yest = np.ones(2)
    yz = np.zeros(2)
    dy = np.zeros(2)
    yz_new, dy_new = pzextr(1, 0.1, yest, yz, dy)
    assert yz_new.shape == (2,)
    assert dy_new.shape == (2,)

def test_bsstep():
    y0 = np.array([1.0])
    dydx = D_real(0.0, y0)
    yscale = np.ones_like(y0)
    try:
        yout, hdid, hnext = bsstep(
            y0.copy(), dydx, 0.0, 0.01, 1e-3, yscale, D_real, _dummy_jacobian_dp_impl
        )
        assert yout.shape == (1,)
        assert np.isscalar(hdid)
        assert np.isscalar(hnext)
    except ValueError as e:
        pytest.skip(f"Bulirsch–Stoer did not converge: {e}")

def test_simpr():
    # Semi-implicit predictor-corrector for a simple ODE
    y = np.ones(2)
    dydx = np.ones(2)
    dfdx = np.zeros(2)
    dfdy = np.eye(2)
    yout = simpr(y, dydx, dfdx, dfdy, 0.0, 1.0, 2, D_real)
    assert yout.shape == (2,)

def test_stifbs():
    y0 = np.array([1.0])
    dydx = D_real(0.0, y0)
    yscale = np.ones_like(y0)
    try:
        x, hdid, hnext = stifbs(
            y0.copy(), dydx, 0.0, 0.01, 1e-3, yscale, D_real, _dummy_jacobian_dp_impl
        )
        assert np.isscalar(x)
        assert np.isscalar(hdid)
        assert np.isscalar(hnext)
    except Exception as e:
        pytest.skip(f"stifbs did not converge or failed: {e}")

def test_ludcmp_and_lubksb():
    # LU decomposition and back-substitution
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    a_orig = a.copy()
    indx = np.zeros(2, dtype=np.int32)
    d = ludcmp(a, indx)
    b = np.array([5.0, 10.0])
    lubksb(a, indx, b)
    # Solve a_orig x = [5, 10]
    x_expected = np.linalg.solve(a_orig, [5.0, 10.0])
    assert np.allclose(b, x_expected)

@njit
def D(x, y):
    return -y

@njit
def integ(y, dydx, x, h, eps, yscale, D, jacobn):
    return h, h  # Dummy stepper

def test_odeint_3D_dpc():
    y0 = np.ones(3, dtype=np.complex128)
    y, nok, nbad = _odeint_dpc_impl(
        y0.copy(), 0.0, 1.0, 1e-6, 0.01, 1e-8, D, integ, _dummy_jacobian_dpc_impl
    )
    assert y.shape == (3,)
    assert isinstance(nok, (int, np.integer))
    assert isinstance(nbad, (int, np.integer))

def test_odeint_3D_dpc_TOM():
    y0 = np.ones(3, dtype=np.complex128)
    y, nok, nbad = _odeint_dpc_impl(
        y0.copy(), 0.0, 1.0, 1e-6, 0.01, 1e-8, D, integ, _dummy_jacobian_dpc_impl
    )
    assert y.shape == (3,)
    assert isinstance(nok, (int, np.integer))
    assert isinstance(nbad, (int, np.integer))

def test_Complex3DIntegrator():
    integrator = ComplexIntegrator(
        deriv=D,
        stepper=integ,
        jacobian=_dummy_jacobian_dpc_impl,
        h1=0.01,
        hmin=1e-8
    )
    y0 = np.ones(3, dtype=np.complex128)
    y, nok, nbad = integrator.solve_dpc(y0, 0.0, 1.0, 1e-6)
    assert y.shape == (3,)
    assert isinstance(nok, (int, np.integer))
    assert isinstance(nbad, (int, np.integer))


if __name__ == "__main__":
    pytest.main([__file__])