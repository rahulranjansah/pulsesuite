"""tests/test_integrator.py – complete unit tests for Integrator
==================================================================
Run with:
    pytest tests/test_integrator.py -q
or
    python tests/test_integrator.py
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Add src to sys.path
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from integrator import Integrator

def D_real(x, y):
    return -y

def D_complex(x, y):
    return 1j * y

def D_3d(x, y):
    return -y

def jacobian_real(x, y):
    n = y.size
    return np.zeros(n), np.zeros((n, n))

def jacobian_complex(x, y):
    n = y.size
    return np.zeros(n, dtype=np.complex128), np.zeros((n, n), dtype=np.complex128)

def simple_euler(y, dydt, t, h, D):
    return y + dydt * h, t + h

@pytest.fixture
def integrator():
    return Integrator()

def test_odeint_dp(integrator):
    y0 = np.array([1.0])
    y, nok, nbad = integrator.odeint_dp(
        y0.copy(), 0.0, 1.0, 1e-6, 0.1, 1e-8,
        D_real, integrator.rkqs_dp, integrator.dummy_jacobian_dp)
    assert y.shape == (1,)
    assert np.allclose(y, np.exp(-1), atol=1e-5)

def test_odeint_dpc(integrator):
    y0 = np.array([1.0 + 0j])
    y, nok, nbad = integrator.odeint_dpc(
        y0.copy(), 0.0, 1.0, 1e-6, 0.1, 1e-8,
        D_complex, integrator.rkqs_dpc, integrator.dummy_jacobian_dpc)
    assert y.shape == (1,)
    assert np.allclose(y, np.exp(1j * 1), atol=1e-5)

def test_dummy_jacobian_dp(integrator):
    y = np.ones(3)
    dfdx, dfdy = integrator.dummy_jacobian_dp(0.0, y)
    assert dfdx.shape == (3,)
    assert dfdy.shape == (3, 3)
    assert np.allclose(dfdx, 0)
    assert np.allclose(dfdy, 0)

def test_dummy_jacobian_dpc(integrator):
    y = np.ones(3, dtype=np.complex128)
    dfdx, dfdy = integrator.dummy_jacobian_dpc(0.0, y)
    assert dfdx.shape == (3,)
    assert dfdy.shape == (3, 3)
    assert np.allclose(dfdx, 0)
    assert np.allclose(dfdy, 0)

def test_simpleint_dp(integrator):
    y0 = np.array([1.0])
    y, n = integrator.simpleint_dp(
        y0.copy(), 0.0, 1.0, 0.1, D_real, simple_euler)
    assert y.shape == (1,)
    assert np.allclose(y, np.exp(-1), atol=1e-2)

def test_simpleint_dpc(integrator):
    y0 = np.array([1.0 + 0j])
    y, n = integrator.simpleint_dpc(
        y0.copy(), 0.0, 1.0, 0.1, D_complex, simple_euler)
    assert y.shape == (1,)
    assert np.allclose(y, np.exp(1j * 1), atol=1e-2)

def test_rkqs_dp(integrator):
    y0 = np.array([1.0])
    dydt = D_real(0.0, y0)
    y, t, hdid, hnext = integrator.rkqs_dp(
        y0.copy(), dydt, 0.0, 0.1, 1e-6, np.ones(1), D_real, jacobian_real)
    assert y.shape == (1,)
    assert np.isscalar(t)

def test_rkqs_dpc(integrator):
    y0 = np.array([1.0 + 0j])
    dydt = D_complex(0.0, y0)
    y, t, hdid, hnext = integrator.rkqs_dpc(
        y0.copy(), dydt, 0.0, 0.1, 1e-6, np.ones(1), D_complex, jacobian_complex)
    assert y.shape == (1,)
    assert np.isscalar(t)

def test_rkck_dp(integrator):
    y0 = np.array([1.0])
    dydt = D_real(0.0, y0)
    yout, yerr = integrator.rkck_dp(y0.copy(), dydt, 0.0, 0.1, D_real)
    assert yout.shape == (1,)
    assert yerr.shape == (1,)

def test_rkck_dpc(integrator):
    y0 = np.array([1.0 + 0j])
    dydt = D_complex(0.0, y0)
    yout, yerr = integrator.rkck_dpc(y0.copy(), dydt, 0.0, 0.1, D_complex)
    assert yout.shape == (1,)
    assert yerr.shape == (1,)

def test_rk4_dp(integrator):
    y0 = np.array([1.0])
    dydt = D_real(0.0, y0)
    yout, tnew = integrator.rk4_dp(y0.copy(), dydt, 0.0, 0.1, D_real)
    assert yout.shape == (1,)
    assert np.isscalar(tnew)

def test_rk4_dpc(integrator):
    y0 = np.array([1.0 + 0j])
    dydt = D_complex(0.0, y0)
    yout, tnew = integrator.rk4_dpc(y0.copy(), dydt, 0.0, 0.1, D_complex)
    assert yout.shape == (1,)
    assert np.isscalar(tnew)

def test_idiot_dp(integrator):
    y0 = np.array([1.0])
    dydt = D_real(0.0, y0)
    yout, tnew = integrator.idiot_dp(y0.copy(), dydt, 0.0, 0.1, D_real)
    assert yout.shape == (1,)
    assert np.isscalar(tnew)

def test_idiot_dpc(integrator):
    y0 = np.array([1.0 + 0j])
    dydt = D_complex(0.0, y0)
    yout, tnew = integrator.idiot_dpc(y0.copy(), dydt, 0.0, 0.1, D_complex)
    assert yout.shape == (1,)
    assert np.isscalar(tnew)

def test_mmid(integrator):
    y0 = np.array([1.0])
    dydx = D_real(0.0, y0)
    yout = integrator.mmid(y0.copy(), dydx, 0.0, 1.0, 10, D_real)
    assert yout.shape == (1,)

def test_pzextr(integrator):
    yest = np.ones(2)
    yz = np.zeros(2)
    dy = np.zeros(2)
    yz_new, dy_new = integrator.pzextr(1, 0.1, yest, yz, dy)
    assert yz_new.shape == (2,)
    assert dy_new.shape == (2,)

def test_simpr(integrator):
    y = np.ones(2)
    dydx = np.ones(2)
    dfdx = np.zeros(2)
    dfdy = np.eye(2)
    yout = integrator.simpr(y, dydx, dfdx, dfdy, 0.0, 1.0, 2, D_real)
    assert yout.shape == (2,)

def test_ludcmp_and_lubksb(integrator):
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    a_orig = a.copy()
    indx = np.zeros(2, dtype=np.int32)
    d = integrator.ludcmp(a, indx)
    b = np.array([5.0, 10.0])
    integrator.lubksb(a, indx, b)
    # Solve a_orig x = [5, 10]
    x_expected = np.linalg.solve(a_orig, [5.0, 10.0])
    assert np.allclose(b, x_expected)

def test_odeint_3D_dpc(integrator):
    arr = np.ones((2,2,2), dtype=np.complex128)
    arr_final, nok, nbad = integrator.odeint_3D_dpc(
        arr.copy(), 0.0, 1.0, 1e-6, 0.1, 1e-8, D_3d, integrator.rkqs_3D_dpc)
    assert arr_final.shape == (2,2,2)

def test_rkqs_3D_dpc(integrator):
    arr = np.ones((2,2,2), dtype=np.complex128)
    dydt = D_3d(0.0, arr)
    y, h, hnext = integrator.rkqs_3D_dpc(arr.copy(), dydt, 0.0, 0.1, 1e-6, np.ones((2,2,2)), D_3d)
    assert y.shape == (2,2,2)

def test_rkck_3D_dpc(integrator):
    arr = np.ones((2,2,2), dtype=np.complex128)
    dydt = D_3d(0.0, arr)
    yout, yerr = integrator.rkck_3D_dpc(arr.copy(), dydt, 0.0, 0.1, D_3d)
    assert yout.shape == (2,2,2)
    assert yerr.shape == (2,2,2)

def test_rk4_3D_dpc(integrator):
    arr = np.ones((2,2,2), dtype=np.complex128)
    dydt = D_3d(0.0, arr)
    yout, tnew = integrator.rk4_3D_dpc(arr.copy(), dydt, 0.0, 0.1, D_3d)
    assert yout.shape == (2,2,2)
    assert np.isscalar(tnew)

def test_odeint_3D_dpc_TOM(integrator):
    arr = np.ones((2,2,2), dtype=np.complex128)
    eps = 1e-6 + 1e-6j
    arr_final, nok, nbad = integrator.odeint_3D_dpc_TOM(
        arr.copy(), 0.0, 1.0, eps, 0.1, 1e-8, D_3d, integrator.rkqs_3D_dpc_TOM)
    assert arr_final.shape == (2,2,2)

def test_rkqs_3D_dpc_TOM(integrator):
    arr = np.ones((2,2,2), dtype=np.complex128)
    dydt = D_3d(0.0, arr)
    eps = 1e-6 + 1e-6j
    y, h, hnext = integrator.rkqs_3D_dpc_TOM(
        arr.copy(), dydt, 0.0, 0.1, eps, np.ones((2,2,2)), np.ones((2,2,2)), D_3d)
    assert y.shape == (2,2,2)

if __name__ == "__main__":
    pytest.main([__file__])
