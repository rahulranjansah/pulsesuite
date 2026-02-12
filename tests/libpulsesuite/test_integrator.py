"""
Tests for pulsesuite.libpulsesuite.integrator — ported from integrator.F90.

Every assertion derives from a known analytical solution, conservation law,
or property of the underlying ODE.  The tests are the specification.

Tolerances (per aiprompts/test.txt):
    algebraic  : rtol=1e-12, atol=1e-12 (float64)
    PDE / long : rtol<=1e-6  (stability)
"""

import numpy as np
import pytest

from pulsesuite.libpulsesuite import integrator as I

RNG = np.random.default_rng(0)

RTOL = 1e-12
ATOL = 1e-12


# ─── helper ODEs with known analytical solutions ─────────────────────

def _decay(t, y):
    """dy/dt = -y  =>  y(t) = y0 * exp(-t)"""
    return -y


def _harmonic_real(t, y):
    """
    2D system:  y0' = y1, y1' = -y0
    =>  y0 = cos(t), y1 = sin(t)  for y0(0)=1, y1(0)=0
    """
    return np.array([y[1], -y[0]])


def _harmonic_complex(t, y):
    """dy/dt = i*omega*y  =>  y(t) = y0 * exp(i*omega*t)"""
    omega = 2.0 * np.pi
    return 1j * omega * y


def _stiff_decay(t, y):
    """dy/dt = -1000*y  =>  y(t) = y0*exp(-1000*t).  Stiff."""
    return -1000.0 * y


def _stiff_jacobian(x, y):
    """Jacobian for dy/dt = -1000*y"""
    n = y.size
    dfdx = np.zeros(n, dtype=np.float64)
    dfdy = -1000.0 * np.eye(n, dtype=np.float64)
    return dfdx, dfdy


def _lotka_volterra(t, y):
    """
    Lotka-Volterra predator-prey:
        dx/dt =  alpha*x - beta*x*y
        dy/dt = -gamma*y + delta*x*y
    Conservation: gamma*ln(x) + beta*ln(y) - delta*x - alpha*y ≈ const
    """
    alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0
    return np.array([
        alpha * y[0] - beta * y[0] * y[1],
        -gamma * y[1] + delta * y[0] * y[1],
    ])


def _lotka_volterra_invariant(y):
    """Conserved quantity for Lotka-Volterra: V = delta*x - gamma*ln(x) + beta*y - alpha*ln(y)."""
    alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0
    return delta * y[0] - gamma * np.log(y[0]) + beta * y[1] - alpha * np.log(y[1])


# ═════════════════════════════════════════════════════════════════════
#  Cash-Karp Runge-Kutta step  (rkck)
# ═════════════════════════════════════════════════════════════════════

class TestRkck:
    def test_dp_exponential_decay(self):
        """Single RKCK step on dy/dt=-y matches exp(-h) to high precision."""
        y = np.array([1.0])
        h = 0.1
        yout, yerr = I.rkck_dp(y, _decay(0.0, y), 0.0, h, _decay)
        np.testing.assert_allclose(yout[0], np.exp(-h), rtol=1e-9)

    def test_dpc_rotation(self):
        """Complex RKCK on dy/dt=i*w*y preserves |y|=1 (rotation)."""
        y = np.array([1.0 + 0j])
        h = 0.01
        yout, yerr = I.rkck_dpc(y, _harmonic_complex(0.0, y), 0.0, h, _harmonic_complex)
        np.testing.assert_allclose(np.abs(yout[0]), 1.0, rtol=1e-10)

    def test_3D_dpc_shape_preserved(self):
        """3D RKCK preserves array shape."""
        y = np.ones((2, 3, 4), dtype=np.complex128)
        dydt = 1j * y
        D = lambda t, y: 1j * y
        yout, yerr = I.rkck_3D_dpc(y, dydt, 0.0, 0.01, D)
        assert yout.shape == (2, 3, 4)
        assert yerr.shape == (2, 3, 4)

    def test_does_not_modify_input(self):
        """rkck must not modify the input y array."""
        y = np.array([1.0, 2.0, 3.0])
        y_orig = y.copy()
        I.rkck_dp(y, _decay(0.0, y), 0.0, 0.1, _decay)
        np.testing.assert_array_equal(y, y_orig)

    def test_error_estimate_small_for_smooth(self):
        """Error estimate is small for smooth ODE with small step."""
        y = np.array([1.0])
        h = 0.001
        _, yerr = I.rkck_dp(y, _decay(0.0, y), 0.0, h, _decay)
        assert np.abs(yerr[0]) < 1e-12

    def test_cash_karp_is_5th_order(self):
        """Richardson: halving h should reduce error by ~2^5 = 32."""
        y0 = np.array([1.0])
        h1, h2 = 0.1, 0.05
        yout1, _ = I.rkck_dp(y0, _decay(0.0, y0), 0.0, h1, _decay)
        yout2, _ = I.rkck_dp(y0, _decay(0.0, y0), 0.0, h2, _decay)
        err1 = abs(yout1[0] - np.exp(-h1))
        err2 = abs(yout2[0] - np.exp(-h2))
        if err2 > 0:
            ratio = err1 / err2
            assert ratio > 20  # should be ~32 for 5th order


# ═════════════════════════════════════════════════════════════════════
#  Classic RK4  (rk4)
# ═════════════════════════════════════════════════════════════════════

class TestRk4:
    def test_dp_single_step(self):
        """RK4 single step on exp decay."""
        y = np.array([1.0])
        t_new = I.rk4_dp(y, _decay(0.0, y), 0.0, 0.01, _decay)
        np.testing.assert_allclose(y[0], np.exp(-0.01), rtol=1e-10)
        assert t_new == pytest.approx(0.01)

    def test_dpc_rotation_preserves_norm(self):
        """RK4 on dy/dt=i*w*y preserves |y|."""
        y = np.array([1.0 + 0j])
        I.rk4_dpc(y, _harmonic_complex(0.0, y), 0.0, 0.001, _harmonic_complex)
        np.testing.assert_allclose(np.abs(y[0]), 1.0, rtol=1e-8)

    def test_3D_modifies_in_place(self):
        """3D RK4 modifies y in-place."""
        y = np.ones((2, 3, 4), dtype=np.complex128)
        D = lambda t, y: -y
        I.rk4_3D_dpc(y, D(0.0, y), 0.0, 0.01, D)
        assert np.all(y.real < 1.0)

    def test_rk4_is_4th_order(self):
        """Richardson: halving h reduces error by ~2^4 = 16."""
        h1, h2 = 0.1, 0.05
        y1 = np.array([1.0])
        I.rk4_dp(y1, _decay(0.0, y1), 0.0, h1, _decay)
        y2 = np.array([1.0])
        I.rk4_dp(y2, _decay(0.0, y2), 0.0, h2, _decay)
        err1 = abs(y1[0] - np.exp(-h1))
        err2 = abs(y2[0] - np.exp(-h2))
        if err2 > 0:
            ratio = err1 / err2
            assert ratio > 10  # should be ~16

    def test_harmonic_oscillator_energy_conservation(self):
        """E = y0^2 + y1^2 = const for harmonic oscillator."""
        y = np.array([1.0, 0.0])
        E0 = y[0] ** 2 + y[1] ** 2
        t = 0.0
        for _ in range(1000):
            t = I.rk4_dp(y, _harmonic_real(t, y), t, 0.01, _harmonic_real)
        E = y[0] ** 2 + y[1] ** 2
        np.testing.assert_allclose(E, E0, rtol=1e-6)


# ═════════════════════════════════════════════════════════════════════
#  Forward Euler  (idiot)
# ═════════════════════════════════════════════════════════════════════

class TestIdiot:
    def test_dp_euler_step(self):
        """Euler: y1 = y0 + h*f = 1 + h*(-1) = 1 - h."""
        y = np.array([1.0])
        h = 0.01
        t_new = I.idiot_dp(y, _decay(0.0, y), 0.0, h, _decay)
        np.testing.assert_allclose(y[0], 1.0 - h, rtol=RTOL)
        assert t_new == pytest.approx(h)

    def test_dpc_euler_step(self):
        """Euler on complex: y1 = y0 + h*i*w*y0."""
        omega = 2.0 * np.pi
        y = np.array([1.0 + 0j])
        h = 0.001
        t_new = I.idiot_dpc(y, _harmonic_complex(0.0, y), 0.0, h, _harmonic_complex)
        expected = 1.0 + h * 1j * omega
        np.testing.assert_allclose(y[0], expected, rtol=1e-10)

    def test_euler_is_1st_order(self):
        """Halving h reduces error by ~2 (1st order) on linear ODE."""
        # Use truly linear ODE dy/dt = c (constant) for exact 1st-order check
        c = -1.0
        def const_deriv(t, y):
            return np.array([c])
        h1, h2 = 0.1, 0.05
        y1 = np.array([1.0])
        I.idiot_dp(y1, const_deriv(0.0, y1), 0.0, h1, const_deriv)
        y2 = np.array([1.0])
        I.idiot_dp(y2, const_deriv(0.0, y2), 0.0, h2, const_deriv)
        # Euler is exact for constant dy/dt, so use exp decay for order test
        y3 = np.array([1.0])
        I.idiot_dp(y3, _decay(0.0, y3), 0.0, h1, _decay)
        y4 = np.array([1.0])
        I.idiot_dp(y4, _decay(0.0, y4), 0.0, h2, _decay)
        err1 = abs(y3[0] - np.exp(-h1))
        err2 = abs(y4[0] - np.exp(-h2))
        ratio = err1 / err2
        # For exp decay, ratio is approximately h1/h2 * (1 + O(h))
        assert ratio > 1.5  # at least better than constant


# ═════════════════════════════════════════════════════════════════════
#  Adaptive RKQS step  (rkqs)
# ═════════════════════════════════════════════════════════════════════

class TestRkqs:
    def test_dp_single_step(self):
        """rkqs single adaptive step on decay."""
        y = np.array([1.0])
        dydt = _decay(0.0, y)
        yscale = np.maximum(np.abs(y), 1.0)
        t_new, hdid, hnext = I.rkqs_dp(
            y, dydt, 0.0, 0.1, 1e-8, yscale, _decay, I.dummy_jacobian_dp,
        )
        np.testing.assert_allclose(y[0], np.exp(-hdid), rtol=1e-6)

    def test_dpc_single_step_rotation(self):
        """rkqs_dpc on rotation preserves norm."""
        y = np.array([1.0 + 0j])
        dydt = _harmonic_complex(0.0, y)
        yscale = np.maximum(np.abs(y), 1.0)
        t_new, hdid, hnext = I.rkqs_dpc(
            y, dydt, 0.0, 0.1, 1e-8, yscale, _harmonic_complex, I.dummy_jacobian_dpc,
        )
        np.testing.assert_allclose(np.abs(y[0]), 1.0, rtol=1e-6)

    def test_3D_dpc_step(self):
        """rkqs_3D_dpc works on 3D arrays."""
        y = np.ones((2, 3, 4), dtype=np.complex128)
        D = lambda t, y: -y
        dydt = D(0.0, y)
        yscale = np.maximum(np.abs(y), 1.0)
        t_new, hdid, hnext = I.rkqs_3D_dpc(
            y, dydt, 0.0, 0.1, 1e-6, yscale, D,
        )
        assert y.shape == (2, 3, 4)
        assert t_new > 0.0

    def test_step_rejected_for_bad_accuracy(self):
        """rkqs should take smaller hdid than htry if error too large."""
        y = np.array([100.0])  # large value → large derivative
        def stiff(t, y):
            return -100.0 * y
        dydt = stiff(0.0, y)
        yscale = np.maximum(np.abs(y), 1.0)
        t_new, hdid, hnext = I.rkqs_dp(
            y, dydt, 0.0, 1.0, 1e-10, yscale, stiff, I.dummy_jacobian_dp,
        )
        assert hdid < 1.0  # must have rejected the trial step


# ═════════════════════════════════════════════════════════════════════
#  Adaptive ODE drivers  (odeint)
# ═════════════════════════════════════════════════════════════════════

class TestOdeint:
    def test_dp_exp_decay(self):
        """odeint_dp: dy/dt = -y from 0 to 1 gives exp(-1)."""
        y = np.array([1.0])
        nok, nbad = I.odeint_dp(
            y, 0.0, 1.0, 1e-8, 0.1, 1e-15, _decay, I.rkqs_dp, I.dummy_jacobian_dp,
        )
        np.testing.assert_allclose(y[0], np.exp(-1.0), rtol=1e-6)
        assert nok > 0

    def test_dpc_full_rotation(self):
        """odeint_dpc: one full rotation exp(2pi*i) ≈ 1."""
        y = np.array([1.0 + 0j])
        nok, nbad = I.odeint_dpc(
            y, 0.0, 1.0, 1e-8, 0.01, 1e-15,
            _harmonic_complex, I.rkqs_dpc, I.dummy_jacobian_dpc,
        )
        np.testing.assert_allclose(y[0], 1.0 + 0j, atol=1e-5)

    def test_dp_harmonic_final_values(self):
        """odeint_dp on harmonic oscillator: y0(2pi)≈1, y1(2pi)≈0."""
        y = np.array([1.0, 0.0])
        I.odeint_dp(
            y, 0.0, 2 * np.pi, 1e-10, 0.01, 1e-15,
            _harmonic_real, I.bsstep, I.dummy_jacobian_dp,
        )
        np.testing.assert_allclose(y[0], 1.0, atol=1e-4)
        np.testing.assert_allclose(y[1], 0.0, atol=1e-4)

    def test_dp_backward_integration(self):
        """odeint_dp handles x1 > x2 (backward integration)."""
        y = np.array([np.exp(-1.0)])
        I.odeint_dp(
            y, 1.0, 0.0, 1e-8, 0.1, 1e-15,
            _decay, I.rkqs_dp, I.dummy_jacobian_dp,
        )
        np.testing.assert_allclose(y[0], 1.0, rtol=1e-5)

    def test_3D_dpc_driver(self):
        """odeint_3D_dpc on dy/dt = -y."""
        D = lambda t, y: -y
        y = np.ones((2, 2, 2), dtype=np.complex128)
        nok, nbad, h1_out = I.odeint_3D_dpc(
            y, 0.0, 0.1, 1e-6, 0.01, 1e-15, D, I.rkqs_3D_dpc,
        )
        expected = np.exp(-0.1)
        np.testing.assert_allclose(y.real, expected, rtol=1e-4)
        np.testing.assert_allclose(y.imag, 0.0, atol=1e-10)

    def test_3D_dpc_TOM_driver(self):
        """odeint_3D_dpc_TOM on dy/dt = -y with complex eps."""
        D = lambda t, y: -y
        y = np.ones((2, 2, 2), dtype=np.complex128)
        eps = complex(1e-6, 1e-6)
        nok, nbad, h1_out = I.odeint_3D_dpc_TOM(
            y, 0.0, 0.1, eps, 0.01, 1e-15, D, I.rkqs_3D_dpc_TOM,
        )
        expected = np.exp(-0.1)
        np.testing.assert_allclose(y.real, expected, rtol=1e-3)

    def test_lotka_volterra_invariant(self):
        """Lotka-Volterra conserved quantity preserved by bsstep."""
        y = np.array([1.0, 1.0])
        C0 = _lotka_volterra_invariant(y)
        I.odeint_dp(
            y, 0.0, 0.5, 1e-12, 0.001, 1e-30,
            _lotka_volterra, I.bsstep, I.dummy_jacobian_dp,
        )
        assert y[0] > 0.0 and y[1] > 0.0, "populations must stay positive"
        C1 = _lotka_volterra_invariant(y)
        np.testing.assert_allclose(C1, C0, rtol=1e-3)


# ═════════════════════════════════════════════════════════════════════
#  Simple fixed-step driver  (simpleint)
# ═════════════════════════════════════════════════════════════════════

class TestSimpleint:
    def test_dp_with_rk4(self):
        """simpleint_dp + rk4 on exp decay."""
        y = np.array([1.0])
        nok = I.simpleint_dp(y, 0.0, 1.0, 0.001, _decay, I.rk4_dp)
        np.testing.assert_allclose(y[0], np.exp(-1.0), rtol=1e-8)
        assert nok == 1000

    def test_dp_with_euler(self):
        """simpleint_dp + Euler on exp decay (low accuracy)."""
        y = np.array([1.0])
        nok = I.simpleint_dp(y, 0.0, 1.0, 0.0001, _decay, I.idiot_dp)
        np.testing.assert_allclose(y[0], np.exp(-1.0), rtol=1e-3)

    def test_dpc_with_rk4(self):
        """simpleint_dpc + rk4 on rotation."""
        y = np.array([1.0 + 0j])
        nok = I.simpleint_dpc(y, 0.0, 1.0, 0.001, _harmonic_complex, I.rk4_dpc)
        np.testing.assert_allclose(np.abs(y[0]), 1.0, rtol=1e-6)

    def test_backward_integration(self):
        """simpleint handles backward integration (t2 < t1)."""
        y = np.array([np.exp(-1.0)])
        nok = I.simpleint_dp(y, 1.0, 0.0, 0.001, _decay, I.rk4_dp)
        np.testing.assert_allclose(y[0], 1.0, rtol=1e-4)


# ═════════════════════════════════════════════════════════════════════
#  Modified midpoint  (mmid)
# ═════════════════════════════════════════════════════════════════════

class TestMmid:
    def test_convergence_order_2(self):
        """Modified midpoint is 2nd order: error ~ h^2."""
        y0 = np.array([1.0])
        htot = 0.1
        yout4 = I.mmid(y0, _decay(0.0, y0), 0.0, htot, 4, _decay)
        yout8 = I.mmid(y0, _decay(0.0, y0), 0.0, htot, 8, _decay)
        exact = np.exp(-htot)
        err4 = abs(yout4[0] - exact)
        err8 = abs(yout8[0] - exact)
        ratio = err4 / err8
        assert ratio > 3.0  # should be ~4 for 2nd order

    def test_does_not_modify_input(self):
        """mmid must not modify input y."""
        y = np.array([1.0, 2.0])
        y_orig = y.copy()
        I.mmid(y, np.array([-1.0, -2.0]), 0.0, 0.1, 4, _decay)
        np.testing.assert_array_equal(y, y_orig)


# ═════════════════════════════════════════════════════════════════════
#  Bulirsch-Stoer  (bsstep)
# ═════════════════════════════════════════════════════════════════════

class TestBsstep:
    def test_exp_decay(self):
        """bsstep via odeint on exp decay."""
        y = np.array([1.0])
        nok, nbad = I.odeint_dp(
            y, 0.0, 1.0, 1e-10, 0.1, 1e-15,
            _decay, I.bsstep, I.dummy_jacobian_dp,
        )
        np.testing.assert_allclose(y[0], np.exp(-1.0), rtol=1e-8)

    def test_harmonic_oscillator_period(self):
        """bsstep preserves harmonic oscillator over one period."""
        y = np.array([1.0, 0.0])
        I.odeint_dp(
            y, 0.0, 2 * np.pi, 1e-10, 0.1, 1e-15,
            _harmonic_real, I.bsstep, I.dummy_jacobian_dp,
        )
        np.testing.assert_allclose(y[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(y[1], 0.0, atol=1e-6)


# ═════════════════════════════════════════════════════════════════════
#  LU decomposition  (ludcmp, lubksb)
# ═════════════════════════════════════════════════════════════════════

class TestLU:
    def test_solve_2x2(self):
        """LU solve of [2 1; 1 3]x = [5; 7] gives x = [1.6, 1.8]."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        indx = np.zeros(2, dtype=np.intp)
        I.ludcmp(A, indx)
        I.lubksb(A, indx, b)
        np.testing.assert_allclose(b, [1.6, 1.8], rtol=RTOL, atol=ATOL)

    def test_solve_identity(self):
        """LU solve of I*x = b gives x = b."""
        n = 5
        A = np.eye(n)
        b = RNG.standard_normal(n)
        expected = b.copy()
        indx = np.zeros(n, dtype=np.intp)
        I.ludcmp(A, indx)
        I.lubksb(A, indx, b)
        np.testing.assert_allclose(b, expected, rtol=RTOL, atol=ATOL)

    def test_solve_random_system(self):
        """LU solve of random A*x = b matches np.linalg.solve."""
        n = 10
        A = RNG.standard_normal((n, n))
        A_orig = A.copy()
        b_orig = RNG.standard_normal(n)

        b = b_orig.copy()
        indx = np.zeros(n, dtype=np.intp)
        I.ludcmp(A, indx)
        I.lubksb(A, indx, b)

        expected = np.linalg.solve(A_orig, b_orig)
        np.testing.assert_allclose(b, expected, rtol=1e-10, atol=1e-10)

    def test_determinant_sign(self):
        """ludcmp returns d=+1 for no swaps, d=-1 for odd swaps."""
        A = np.eye(3)
        indx = np.zeros(3, dtype=np.intp)
        d = I.ludcmp(A, indx)
        assert d == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
    def test_lu_round_trip(self, n):
        """LU factorize then solve A*x=b recovers x for various sizes."""
        A = RNG.standard_normal((n, n)) + n * np.eye(n)
        A_orig = A.copy()
        x_true = RNG.standard_normal(n)
        b = A_orig @ x_true

        indx = np.zeros(n, dtype=np.intp)
        I.ludcmp(A, indx)
        I.lubksb(A, indx, b)
        np.testing.assert_allclose(b, x_true, rtol=1e-8, atol=1e-8)


# ═════════════════════════════════════════════════════════════════════
#  Semi-implicit midpoint  (simpr)
# ═════════════════════════════════════════════════════════════════════

class TestSimpr:
    def test_stiff_decay_single_step(self):
        """simpr handles stiff ODE dy/dt=-100*y with reasonable accuracy."""
        lam = -100.0
        def stiff100(t, y):
            return lam * y
        y = np.array([1.0])
        dydx = stiff100(0.0, y)
        dfdx = np.zeros(1)
        dfdy = lam * np.eye(1)
        htot = 0.01
        nstep = 10
        yout = I.simpr(y, dydx, dfdx, dfdy, 0.0, htot, nstep, stiff100)
        exact = np.exp(lam * htot)
        np.testing.assert_allclose(yout[0], exact, rtol=1e-2)

    def test_does_not_modify_input(self):
        """simpr must not modify y."""
        y = np.array([1.0, 2.0])
        y_orig = y.copy()
        dydx = _decay(0.0, y)
        dfdx = np.zeros(2)
        dfdy = -np.eye(2)
        I.simpr(y, dydx, dfdx, dfdy, 0.0, 0.1, 4, _decay)
        np.testing.assert_array_equal(y, y_orig)


# ═════════════════════════════════════════════════════════════════════
#  Stiff Bulirsch-Stoer  (stifbs)
# ═════════════════════════════════════════════════════════════════════

class TestStifbs:
    def test_exp_decay(self):
        """stifbs via odeint on simple exp decay."""
        y = np.array([1.0])
        nok, nbad = I.odeint_dp(
            y, 0.0, 1.0, 1e-8, 0.1, 1e-15,
            _decay, I.stifbs, I.dummy_jacobian_dp,
        )
        np.testing.assert_allclose(y[0], np.exp(-1.0), rtol=1e-6)

    def test_stiff_decay(self):
        """stifbs on stiff dy/dt=-1000*y with proper Jacobian."""
        y = np.array([1.0])
        nok, nbad = I.odeint_dp(
            y, 0.0, 0.01, 1e-6, 0.001, 1e-15,
            _stiff_decay, I.stifbs, _stiff_jacobian,
        )
        np.testing.assert_allclose(y[0], np.exp(-10.0), rtol=0.01)

    def test_2d_harmonic(self):
        """stifbs on 2D harmonic oscillator over one period."""
        y = np.array([1.0, 0.0])
        I.odeint_dp(
            y, 0.0, 2 * np.pi, 1e-8, 0.1, 1e-15,
            _harmonic_real, I.stifbs, I.dummy_jacobian_dp,
        )
        np.testing.assert_allclose(y[0], 1.0, atol=1e-4)
        np.testing.assert_allclose(y[1], 0.0, atol=1e-4)


# ═════════════════════════════════════════════════════════════════════
#  Module constants
# ═════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_safety(self):
        assert I.SAFETY == 0.9

    def test_errcon(self):
        """ERRCON = (5/SAFETY)^(-5)."""
        np.testing.assert_allclose(I.ERRCON, (5.0 / 0.9) ** (-5), rtol=RTOL)

    def test_pgrow_pshrnk(self):
        assert I.PGROW == -0.20
        assert I.PSHRNK == -0.25

    def test_maxstp(self):
        assert I.MAXSTP == 100_000_000


# ═════════════════════════════════════════════════════════════════════
#  Cash-Karp tableau verification
# ═════════════════════════════════════════════════════════════════════

class TestButcherTableau:
    def test_c_weights_sum_to_one(self):
        """5th-order weights must sum to 1 (consistency condition)."""
        c_sum = I._c1 + 0.0 + I._c3 + I._c4 + 0.0 + I._c6
        np.testing.assert_allclose(c_sum, 1.0, rtol=RTOL)

    def test_b_row_sums(self):
        """Each b-row sums to the corresponding a-node."""
        np.testing.assert_allclose(I._b21, I._a2, rtol=RTOL)
        np.testing.assert_allclose(I._b31 + I._b32, I._a3, rtol=RTOL)
        np.testing.assert_allclose(I._b41 + I._b42 + I._b43, I._a4, rtol=RTOL)
        np.testing.assert_allclose(I._b51 + I._b52 + I._b53 + I._b54, I._a5, rtol=RTOL)
        np.testing.assert_allclose(
            I._b61 + I._b62 + I._b63 + I._b64 + I._b65, I._a6, rtol=RTOL,
        )


# ═════════════════════════════════════════════════════════════════════
#  pzextr (polynomial extrapolation)
# ═════════════════════════════════════════════════════════════════════

class TestPzextr:
    def test_single_estimate(self):
        """With iest=1, yz = yest (no extrapolation yet)."""
        nv = 3
        yz = np.zeros(nv)
        dy = np.zeros(nv)
        x_save = np.zeros(10)
        d_save = np.zeros((nv, 10))
        yest = np.array([1.0, 2.0, 3.0])
        I.pzextr(1, 0.1, yest, yz, dy, x_save, d_save)
        np.testing.assert_array_equal(yz, yest)

    def test_two_point_extrapolation(self):
        """Two-point extrapolation recovers limit for y(h^2) = a + b*h^2."""
        # If y = 5.0 + 3.0*h^2, extrapolation to h=0 gives 5.0
        nv = 1
        yz = np.zeros(nv)
        dy = np.zeros(nv)
        x_save = np.zeros(10)
        d_save = np.zeros((nv, 10))

        h1 = 0.1
        yest1 = np.array([5.0 + 3.0 * h1 ** 2])
        I.pzextr(1, h1 ** 2, yest1, yz, dy, x_save, d_save)

        h2 = 0.05
        yest2 = np.array([5.0 + 3.0 * h2 ** 2])
        I.pzextr(2, h2 ** 2, yest2, yz, dy, x_save, d_save)

        np.testing.assert_allclose(yz[0], 5.0, rtol=1e-10)


# ═════════════════════════════════════════════════════════════════════
#  Physics validation: conservation laws
# ═════════════════════════════════════════════════════════════════════

class TestPhysics:
    def test_energy_conservation_shm(self):
        """Simple harmonic motion: E = 0.5*(v^2 + x^2) is conserved."""
        y = np.array([1.0, 0.0])  # x=1, v=0
        E0 = 0.5 * (y[0] ** 2 + y[1] ** 2)
        I.odeint_dp(
            y, 0.0, 2 * np.pi, 1e-10, 0.01, 1e-15,
            _harmonic_real, I.bsstep, I.dummy_jacobian_dp,
        )
        E = 0.5 * (y[0] ** 2 + y[1] ** 2)
        np.testing.assert_allclose(E, E0, rtol=1e-6)

    def test_norm_conservation_rotation(self):
        """|exp(i*omega*t)| = 1 for all t."""
        y = np.array([1.0 + 0j])
        I.odeint_dpc(
            y, 0.0, 5.0, 1e-8, 0.01, 1e-15,
            _harmonic_complex, I.rkqs_dpc, I.dummy_jacobian_dpc,
        )
        np.testing.assert_allclose(np.abs(y[0]), 1.0, rtol=1e-5)

    def test_exp_decay_monotonic(self):
        """Exponential decay is strictly monotone decreasing."""
        results = []
        y = np.array([1.0])
        for t_end in [0.1, 0.5, 1.0, 2.0]:
            y_test = np.array([1.0])
            I.odeint_dp(
                y_test, 0.0, t_end, 1e-8, 0.1, 1e-15,
                _decay, I.rkqs_dp, I.dummy_jacobian_dp,
            )
            results.append(y_test[0])
        for i in range(len(results) - 1):
            assert results[i] > results[i + 1]

    def test_superposition_linearity(self):
        """For linear ODE dy/dt=-y, solution is linear in y0."""
        y1 = np.array([2.0])
        y2 = np.array([3.0])
        I.odeint_dp(y1, 0.0, 1.0, 1e-10, 0.1, 1e-15, _decay, I.rkqs_dp, I.dummy_jacobian_dp)
        I.odeint_dp(y2, 0.0, 1.0, 1e-10, 0.1, 1e-15, _decay, I.rkqs_dp, I.dummy_jacobian_dp)
        np.testing.assert_allclose(y2[0] / y1[0], 3.0 / 2.0, rtol=1e-8)

    def test_time_reversal_symmetry(self):
        """Integrating forward then backward recovers initial state."""
        y = np.array([1.0, 0.0])
        y_init = y.copy()
        I.odeint_dp(
            y, 0.0, np.pi, 1e-10, 0.01, 1e-15,
            _harmonic_real, I.rkqs_dp, I.dummy_jacobian_dp,
        )
        I.odeint_dp(
            y, np.pi, 0.0, 1e-10, 0.01, 1e-15,
            _harmonic_real, I.rkqs_dp, I.dummy_jacobian_dp,
        )
        np.testing.assert_allclose(y, y_init, atol=1e-4)


# ═════════════════════════════════════════════════════════════════════
#  Parametric stepper comparison
# ═════════════════════════════════════════════════════════════════════

class TestStepperComparison:
    @pytest.mark.parametrize("stepper,tol", [
        (I.rkqs_dp, 1e-6),
        (I.bsstep, 1e-6),
        (I.stifbs, 1e-4),
    ])
    def test_all_steppers_solve_decay(self, stepper, tol):
        """All adaptive steppers converge on exp decay."""
        y = np.array([1.0])
        I.odeint_dp(
            y, 0.0, 1.0, 1e-8, 0.1, 1e-15,
            _decay, stepper, I.dummy_jacobian_dp,
        )
        np.testing.assert_allclose(y[0], np.exp(-1.0), rtol=tol)

    @pytest.mark.parametrize("simple", [I.rk4_dp, I.idiot_dp])
    def test_simple_steppers(self, simple):
        """All simple steppers can be used with simpleint."""
        y = np.array([1.0])
        h = 0.001 if simple is I.rk4_dp else 0.0001
        nok = I.simpleint_dp(y, 0.0, 1.0, h, _decay, simple)
        assert nok > 0
        assert y[0] < 1.0  # decayed
        assert y[0] > 0.0  # still positive
