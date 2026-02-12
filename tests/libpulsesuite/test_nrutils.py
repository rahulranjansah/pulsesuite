"""
Tests for nrutils — grounded in mathematical truth and physical reality.

Every assertion derives from a known algebraic identity, series formula,
or property of linear algebra.  The tests are the specification; if the
implementation disagrees, the implementation is wrong.

Tolerances (per aiprompts/test.txt):
    algebraic  :  rtol=1e-12, atol=1e-12  (float64)
"""

import math

import numpy as np
import pytest

from pulsesuite.libpulsesuite import nrutils

# ── reproducible RNG ──────────────────────────────────────────────────
RNG = np.random.default_rng(0)

# ── tolerance aliases ─────────────────────────────────────────────────
RTOL = 1e-12
ATOL = 1e-12


# ======================================================================
#  arrayCopy — copied elements must be bitwise-identical to the source
# ======================================================================

class TestArrayCopy:
    def test_full_copy(self):
        src = np.arange(10, dtype=np.float64)
        dest = np.empty(10, dtype=np.float64)
        n, m = nrutils.arrayCopy(src, dest)
        assert n == 10 and m == 0
        np.testing.assert_array_equal(dest, src)

    def test_dest_smaller_truncates(self):
        src = np.arange(10, dtype=np.float64)
        dest = np.empty(4, dtype=np.float64)
        n, m = nrutils.arrayCopy(src, dest)
        assert n == 4 and m == 6
        np.testing.assert_array_equal(dest, src[:4])

    def test_dest_larger_partial_fill(self):
        src = np.array([1.0, 2.0])
        dest = np.full(5, -1.0)
        n, m = nrutils.arrayCopy(src, dest)
        assert n == 2 and m == 0
        np.testing.assert_array_equal(dest[:2], src)

    def test_empty_arrays(self):
        src = np.array([], dtype=np.float64)
        dest = np.array([], dtype=np.float64)
        n, m = nrutils.arrayCopy(src, dest)
        assert n == 0 and m == 0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
    def test_dtype_preservation(self, dtype):
        src = np.ones(5, dtype=dtype)
        dest = np.empty(5, dtype=dtype)
        nrutils.arrayCopy(src, dest)
        np.testing.assert_array_equal(dest, src)


# ======================================================================
#  swap — mathematical involution: swap(swap(a, b)) == (a, b)
# ======================================================================

class TestSwap:
    def test_scalar_involution(self):
        a, b = 3.14, 2.72
        b2, a2 = nrutils.swap(a, b)
        assert a2 == a and b2 == b
        # apply twice → identity
        a3, b3 = nrutils.swap(b2, a2)
        assert a3 == a and b3 == b

    def test_array_swap(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        b2, a2 = nrutils.swap(a, b)
        np.testing.assert_array_equal(a2, a)
        np.testing.assert_array_equal(b2, b)


# ======================================================================
#  maskedSwap — only swaps at True positions; rest unchanged
# ======================================================================

class TestMaskedSwap:
    def test_partial_mask(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        mask = np.array([True, False, True, False])
        nrutils.maskedSwap(a, b, mask)
        np.testing.assert_array_equal(a, [10.0, 2.0, 30.0, 4.0])
        np.testing.assert_array_equal(b, [1.0, 20.0, 3.0, 40.0])

    def test_all_true_full_swap(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        nrutils.maskedSwap(a, b, np.ones(2, dtype=bool))
        np.testing.assert_array_equal(a, [3.0, 4.0])
        np.testing.assert_array_equal(b, [1.0, 2.0])

    def test_all_false_no_change(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        a0, b0 = a.copy(), b.copy()
        nrutils.maskedSwap(a, b, np.zeros(2, dtype=bool))
        np.testing.assert_array_equal(a, a0)
        np.testing.assert_array_equal(b, b0)


# ======================================================================
#  reallocate — data preservation in the overlapping region
# ======================================================================

class TestReallocate:
    def test_grow_1d(self):
        arr = np.arange(5, dtype=np.float64)
        new = nrutils.reallocate(arr, 10)
        assert new.shape == (10,)
        np.testing.assert_array_equal(new[:5], arr)

    def test_shrink_1d(self):
        arr = np.arange(10, dtype=np.float64)
        new = nrutils.reallocate(arr, 3)
        assert new.shape == (3,)
        np.testing.assert_array_equal(new, arr[:3])

    def test_grow_2d(self):
        arr = RNG.standard_normal((3, 4))
        new = nrutils.reallocate(arr, (5, 6))
        assert new.shape == (5, 6)
        np.testing.assert_array_equal(new[:3, :4], arr)

    def test_shrink_2d(self):
        arr = RNG.standard_normal((5, 6))
        new = nrutils.reallocate(arr, (2, 3))
        assert new.shape == (2, 3)
        np.testing.assert_array_equal(new, arr[:2, :3])


# ======================================================================
#  imaxloc / iminloc — 0-based indices
# ======================================================================

class TestMaxMinLoc:
    def test_simple(self):
        arr = np.array([1.0, 5.0, 3.0, 2.0])
        assert nrutils.imaxloc(arr) == 1   # 5.0 at index 1
        assert nrutils.iminloc(arr) == 0   # 1.0 at index 0

    def test_negative_values(self):
        arr = np.array([-10.0, -3.0, -7.0])
        assert nrutils.imaxloc(arr) == 1   # -3 at index 1
        assert nrutils.iminloc(arr) == 0   # -10 at index 0

    def test_single_element(self):
        arr = np.array([42.0])
        assert nrutils.imaxloc(arr) == 0
        assert nrutils.iminloc(arr) == 0


# ======================================================================
#  ifirstloc — first True location (0-based, -1 if not found)
# ======================================================================

class TestIfirstloc:
    def test_first_element_true(self):
        assert nrutils.ifirstloc(np.array([True, False, False])) == 0

    def test_last_element_true(self):
        assert nrutils.ifirstloc(np.array([False, False, True])) == 2

    def test_no_true(self):
        assert nrutils.ifirstloc(np.array([False, False, False])) == -1

    def test_all_true(self):
        assert nrutils.ifirstloc(np.array([True, True, True])) == 0


# ======================================================================
#  nrerror / assertTrue / assertEq — program halts on violation
# ======================================================================

class TestAssertions:
    def test_nrerror_raises_system_exit(self):
        with pytest.raises(SystemExit):
            nrutils.nrerror("fatal")

    def test_assertTrue_passes(self):
        nrutils.assertTrue(True, "should not fire")

    def test_assertTrue_fails(self):
        with pytest.raises(SystemExit):
            nrutils.assertTrue(False, "should fire")

    def test_assertEq_all_equal(self):
        assert nrutils.assertEq(7, 7, 7) == 7

    def test_assertEq_mixed_fails(self):
        with pytest.raises(SystemExit):
            nrutils.assertEq(1, 2, 1)

    def test_assertEq_pair(self):
        assert nrutils.assertEq(42, 42) == 42


# ======================================================================
#  arth — arithmetic progression
#  Identity: sum = n/2 · (2·first + (n-1)·increment)
# ======================================================================

class TestArth:
    def test_gauss_sum_formula(self):
        """Sum of first 100 natural numbers: n(n+1)/2 = 5050."""
        a = nrutils.arth(1.0, 1.0, 100)
        assert a.size == 100
        np.testing.assert_allclose(a.sum(), 5050.0, rtol=RTOL, atol=ATOL)

    def test_arithmetic_mean(self):
        """Mean of AP = (first + last) / 2."""
        first, inc, n = 3.0, 0.5, 200
        a = nrutils.arth(first, inc, n)
        last = first + (n - 1) * inc
        np.testing.assert_allclose(a.mean(), (first + last) / 2, rtol=RTOL, atol=ATOL)

    def test_constant_difference(self):
        """np.diff of an AP is constant = increment."""
        a = nrutils.arth(2.0, 7.0, 50)
        np.testing.assert_allclose(np.diff(a), 7.0, rtol=RTOL, atol=ATOL)

    def test_zero_length(self):
        a = nrutils.arth(1.0, 1.0, 0)
        assert a.size == 0

    def test_negative_increment(self):
        a = nrutils.arth(10.0, -2.0, 6)
        np.testing.assert_allclose(a, [10, 8, 6, 4, 2, 0], rtol=RTOL, atol=ATOL)


# ======================================================================
#  geop — geometric progression
#  Identity: sum = first · (factor^n − 1) / (factor − 1)  when factor ≠ 1
# ======================================================================

class TestGeop:
    def test_geometric_sum(self):
        """Sum of GP: a(r^n - 1)/(r - 1)."""
        first, factor, n = 1.0, 2.0, 10
        g = nrutils.geop(first, factor, n)
        expected_sum = first * (factor**n - 1) / (factor - 1)
        np.testing.assert_allclose(g.sum(), expected_sum, rtol=RTOL, atol=ATOL)

    def test_ratio_property(self):
        """Consecutive ratio in GP is constant = factor."""
        g = nrutils.geop(3.0, 1.5, 20)
        ratios = g[1:] / g[:-1]
        np.testing.assert_allclose(ratios, 1.5, rtol=RTOL, atol=ATOL)

    def test_powers_of_two(self):
        g = nrutils.geop(1.0, 2.0, 8)
        np.testing.assert_allclose(g, 2.0 ** np.arange(8), rtol=RTOL, atol=ATOL)

    def test_unit_factor(self):
        """factor=1 → constant sequence."""
        g = nrutils.geop(5.0, 1.0, 10)
        np.testing.assert_allclose(g, 5.0, rtol=RTOL, atol=ATOL)


# ======================================================================
#  cumsum / cumprod — cumulative operations
#  cumsum is the discrete antiderivative: diff(cumsum(a)) == a[1:]
# ======================================================================

class TestCumsum:
    def test_discrete_antiderivative(self):
        """diff(cumsum(a)) == a[1:]  (fundamental theorem of discrete calculus)."""
        a = RNG.standard_normal(50)
        cs = nrutils.cumsum(a)
        np.testing.assert_allclose(np.diff(cs), a[1:], rtol=RTOL, atol=ATOL)

    def test_last_element_is_total(self):
        """cumsum(a)[-1] == sum(a)."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cs = nrutils.cumsum(a)
        np.testing.assert_allclose(cs[-1], a.sum(), rtol=RTOL, atol=ATOL)

    def test_seed_offset(self):
        """With seed s, cumsum(a, s) == cumsum(a) + s."""
        a = np.array([1.0, 2.0, 3.0])
        s = 100.0
        np.testing.assert_allclose(
            nrutils.cumsum(a, seed=s),
            nrutils.cumsum(a) + s,
            rtol=RTOL, atol=ATOL,
        )

    def test_single_element(self):
        a = np.array([7.0])
        np.testing.assert_allclose(nrutils.cumsum(a), [7.0], rtol=RTOL, atol=ATOL)


class TestCumprod:
    def test_last_element_is_product(self):
        """cumprod(a)[-1] == prod(a)."""
        a = np.array([2.0, 3.0, 4.0])
        cp = nrutils.cumprod(a)
        np.testing.assert_allclose(cp[-1], np.prod(a), rtol=RTOL, atol=ATOL)

    def test_factorial(self):
        """cumprod([1,2,3,...,n])[-1] == n!"""
        a = np.arange(1, 11, dtype=np.float64)
        cp = nrutils.cumprod(a)
        # 10! = 3628800
        np.testing.assert_allclose(cp[-1], 3628800.0, rtol=RTOL, atol=ATOL)

    def test_seed_scaling(self):
        """With seed s, cumprod(a, s) == s * cumprod(a)."""
        a = np.array([2.0, 3.0, 5.0])
        s = 10.0
        np.testing.assert_allclose(
            nrutils.cumprod(a, seed=s),
            s * nrutils.cumprod(a),
            rtol=RTOL, atol=ATOL,
        )


# ======================================================================
#  poly — Horner's method polynomial evaluation
#  Verified against known Taylor series and algebraic identities
# ======================================================================

class TestPoly:
    def test_quadratic(self):
        """P(x) = 1 + 2x + 3x²   →   P(2) = 17."""
        coeffs = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(nrutils.poly(2.0, coeffs), 17.0, rtol=RTOL, atol=ATOL)

    def test_constant(self):
        """P(x) = c  for all x."""
        coeffs = np.array([42.0])
        np.testing.assert_allclose(nrutils.poly(99.0, coeffs), 42.0, rtol=RTOL, atol=ATOL)

    def test_empty_coefficients(self):
        """No coefficients → P(x) = 0."""
        assert nrutils.poly(5.0, np.array([])) == 0.0

    def test_exp_taylor_at_one(self):
        """e ≈ ∑ 1/k! for k=0..20  (Taylor series of e^x at x=1)."""
        coeffs = np.array([1.0 / math.factorial(k) for k in range(21)])
        result = nrutils.poly(1.0, coeffs)
        np.testing.assert_allclose(result, np.e, rtol=1e-12, atol=1e-14)

    def test_array_input(self):
        """Broadcast evaluation at multiple points."""
        coeffs = np.array([0.0, 1.0])  # P(x) = x
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(nrutils.poly(x, coeffs), x, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    def test_complex_support(self, dtype):
        """P(x) = x² evaluated at x = i  →  -1."""
        coeffs = np.array([0, 0, 1], dtype=dtype)
        result = nrutils.poly(1j, coeffs)
        np.testing.assert_allclose(result, -1.0 + 0j, rtol=RTOL, atol=ATOL)


# ======================================================================
#  polyTerm — recurrence u_j = a_j + b · u_{j-1}
# ======================================================================

class TestPolyTerm:
    def test_recurrence(self):
        a = np.array([1.0, 2.0, 3.0])
        b = 2.0
        u = nrutils.polyTerm(a, b)
        # u[0]=1, u[1]=2+2*1=4, u[2]=3+2*4=11
        np.testing.assert_allclose(u, [1.0, 4.0, 11.0], rtol=RTOL, atol=ATOL)

    def test_zero_multiplier(self):
        """b = 0  →  u == a (no recurrence coupling)."""
        a = np.array([5.0, 10.0, 15.0])
        u = nrutils.polyTerm(a, 0.0)
        np.testing.assert_allclose(u, a, rtol=RTOL, atol=ATOL)

    def test_empty(self):
        u = nrutils.polyTerm(np.array([]), 1.0)
        assert u.size == 0


# ======================================================================
#  outerprod — rank-1 matrix: M = a ⊗ b
#  Property: trace(a ⊗ b) = a · b  (when lengths equal)
# ======================================================================

class TestOuterprod:
    def test_trace_equals_dot(self):
        """tr(a ⊗ b) = a · b."""
        a = RNG.standard_normal(20)
        b = RNG.standard_normal(20)
        M = nrutils.outerprod(a, b)
        np.testing.assert_allclose(np.trace(M), np.dot(a, b), rtol=RTOL, atol=ATOL)

    def test_rank_one(self):
        """Rank of a ⊗ b is 1 (for non-zero a, b)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        M = nrutils.outerprod(a, b)
        assert M.shape == (3, 2)
        # second singular value should be zero
        s = np.linalg.svd(M, compute_uv=False)
        np.testing.assert_allclose(s[1:], 0.0, atol=1e-12)

    def test_against_numpy(self):
        a = RNG.standard_normal(50)
        b = RNG.standard_normal(30)
        np.testing.assert_allclose(
            nrutils.outerprod(a, b), np.outer(a, b), rtol=RTOL, atol=ATOL
        )


# ======================================================================
#  outerdiff — M_ij = a_i - b_j
#  Antisymmetry: outerdiff(a, b) = -outerdiff(b, a)^T
# ======================================================================

class TestOuterdiff:
    def test_antisymmetry(self):
        a = RNG.standard_normal(10)
        b = RNG.standard_normal(10)
        D1 = nrutils.outerdiff(a, b)
        D2 = nrutils.outerdiff(b, a)
        np.testing.assert_allclose(D1, -D2.T, rtol=RTOL, atol=ATOL)

    def test_self_difference_antisymmetric(self):
        """outerdiff(a, a) is antisymmetric."""
        a = np.array([1.0, 3.0, 5.0])
        D = nrutils.outerdiff(a, a)
        np.testing.assert_allclose(D, -D.T, rtol=RTOL, atol=ATOL)

    def test_known_values(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        expected = np.array([[-2.0, -3.0], [-1.0, -2.0]])
        np.testing.assert_allclose(nrutils.outerdiff(a, b), expected, rtol=RTOL, atol=ATOL)


# ======================================================================
#  outersum — M_ij = a_i + b_j
# ======================================================================

class TestOutersum:
    def test_symmetry_relation(self):
        """outersum(a, b) == outersum(b, a)^T."""
        a = RNG.standard_normal(8)
        b = RNG.standard_normal(12)
        S1 = nrutils.outersum(a, b)
        S2 = nrutils.outersum(b, a)
        np.testing.assert_allclose(S1, S2.T, rtol=RTOL, atol=ATOL)

    def test_row_column_sums(self):
        """Every row of outersum(a, b) is b + a_i."""
        a = np.array([10.0, 20.0])
        b = np.array([1.0, 2.0, 3.0])
        S = nrutils.outersum(a, b)
        np.testing.assert_allclose(S[0], b + 10.0, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(S[1], b + 20.0, rtol=RTOL, atol=ATOL)


# ======================================================================
#  outerand — boolean outer product
# ======================================================================

class TestOuterand:
    def test_truth_table(self):
        a = np.array([True, False])
        b = np.array([True, True, False])
        expected = np.array([[True, True, False], [False, False, False]])
        np.testing.assert_array_equal(nrutils.outerand(a, b), expected)

    def test_all_false(self):
        a = np.array([False, False])
        b = np.array([True, True])
        np.testing.assert_array_equal(nrutils.outerand(a, b), np.zeros((2, 2), dtype=bool))


# ======================================================================
#  scatterAdd — conservation of total: sum(dest_after) == sum(dest_before) + sum(source)
#  scatterMax — maximum propagation
# ======================================================================

class TestScatterAdd:
    def test_sum_conservation(self):
        """Total is conserved: Σdest_after = Σdest_before + Σsource."""
        dest = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        source = np.array([10.0, 20.0])
        indices = np.array([1, 3])  # 0-based
        before = dest.sum()
        nrutils.scatterAdd(dest, source, indices)
        np.testing.assert_allclose(dest.sum(), before + source.sum(), rtol=RTOL, atol=ATOL)

    def test_specific_positions(self):
        dest = np.zeros(5)
        source = np.array([10.0, 20.0])
        indices = np.array([1, 3])  # 0-based
        nrutils.scatterAdd(dest, source, indices)
        np.testing.assert_allclose(dest, [0, 10, 0, 20, 0], rtol=RTOL, atol=ATOL)

    def test_duplicate_indices_accumulate(self):
        """Duplicate indices accumulate: dest[0] += 10 + 20 = 30."""
        dest = np.zeros(3)
        source = np.array([10.0, 20.0])
        indices = np.array([0, 0])  # both target index 0 (0-based)
        nrutils.scatterAdd(dest, source, indices)
        np.testing.assert_allclose(dest[0], 30.0, rtol=RTOL, atol=ATOL)

    def test_out_of_bounds_ignored(self):
        dest = np.zeros(3)
        source = np.array([10.0, 20.0])
        indices = np.array([-1, 5])  # both out of bounds for 0-based [0..2]
        nrutils.scatterAdd(dest, source, indices)
        np.testing.assert_allclose(dest, [0, 0, 0], rtol=RTOL, atol=ATOL)


class TestScatterMax:
    def test_max_propagation(self):
        dest = np.array([1.0, 2.0, 3.0])
        source = np.array([5.0, 1.0])
        indices = np.array([0, 1])  # 0-based
        nrutils.scatterMax(dest, source, indices)
        np.testing.assert_allclose(dest, [5.0, 2.0, 3.0], rtol=RTOL, atol=ATOL)

    def test_no_decrease(self):
        """scatterMax never decreases dest values."""
        dest = np.array([10.0, 20.0, 30.0])
        source = np.array([5.0, 15.0])
        indices = np.array([0, 1])  # 0-based
        dest_before = dest.copy()
        nrutils.scatterMax(dest, source, indices)
        assert np.all(dest >= dest_before)


# ======================================================================
#  diagAdd / diagMult — trace properties
#  diagAdd(I, s) → tr = n + n·s = n·(1+s)
#  diagMult(I, s) → tr = n·s
# ======================================================================

class TestDiagAdd:
    def test_scalar_trace(self):
        """diagAdd(I, s): tr(I + sI) = n(1+s)."""
        n = 5
        mat = np.eye(n)
        nrutils.diagAdd(mat, 2.0)
        np.testing.assert_allclose(np.trace(mat), n * 3.0, rtol=RTOL, atol=ATOL)

    def test_vector_add(self):
        mat = np.eye(3)
        nrutils.diagAdd(mat, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(np.diag(mat), [2.0, 3.0, 4.0], rtol=RTOL, atol=ATOL)


class TestDiagMult:
    def test_scalar_trace(self):
        """diagMult(I, s): tr = n·s."""
        n = 4
        mat = np.eye(n)
        nrutils.diagMult(mat, 3.0)
        np.testing.assert_allclose(np.trace(mat), n * 3.0, rtol=RTOL, atol=ATOL)

    def test_vector_mult(self):
        mat = np.eye(3)
        nrutils.diagMult(mat, np.array([2.0, 3.0, 4.0]))
        np.testing.assert_allclose(np.diag(mat), [2.0, 3.0, 4.0], rtol=RTOL, atol=ATOL)


# ======================================================================
#  getDiag / putDiag — extraction then insertion is identity
# ======================================================================

class TestGetPutDiag:
    def test_roundtrip(self):
        """putDiag(getDiag(M), M') then getDiag(M') recovers original diagonal."""
        M = RNG.standard_normal((4, 4))
        d = nrutils.getDiag(M)
        M2 = np.zeros((4, 4))
        nrutils.putDiag(d, M2)
        np.testing.assert_allclose(nrutils.getDiag(M2), d, rtol=RTOL, atol=ATOL)

    def test_identity_diagonal(self):
        d = nrutils.getDiag(np.eye(5))
        np.testing.assert_allclose(d, np.ones(5), rtol=RTOL, atol=ATOL)


# ======================================================================
#  unitMatrix — identity matrix: A · I = A
# ======================================================================

class TestUnitMatrix:
    def test_idempotent_multiplication(self):
        """A · I = A (identity matrix property)."""
        A = RNG.standard_normal((4, 4))
        I = np.empty((4, 4))
        nrutils.unitMatrix(I)
        np.testing.assert_allclose(A @ I, A, rtol=RTOL, atol=ATOL)

    def test_trace(self):
        """tr(I) = n."""
        mat = np.empty((6, 6))
        nrutils.unitMatrix(mat)
        np.testing.assert_allclose(np.trace(mat), 6.0, rtol=RTOL, atol=ATOL)

    def test_rectangular(self):
        mat = np.empty((3, 5))
        nrutils.unitMatrix(mat)
        for i in range(3):
            for j_ in range(5):
                expected = 1.0 if i == j_ else 0.0
                assert mat[i, j_] == expected


# ======================================================================
#  upperTriangle / lowerTriangle — complementarity
#  upper(extra=0) + lower(extra=0) + I == ones  (for square)
# ======================================================================

class TestTriangles:
    def test_upper_lower_complement(self):
        """U + L + I = J  (matrix of ones) for square matrices."""
        n = 5
        U = nrutils.upperTriangle(n, n)
        L = nrutils.lowerTriangle(n, n)
        I = np.eye(n)
        np.testing.assert_allclose(U + L + I, np.ones((n, n)), rtol=RTOL, atol=ATOL)

    def test_upper_no_diagonal(self):
        """With extra=0, diagonal entries are 0."""
        U = nrutils.upperTriangle(4, 4)
        np.testing.assert_allclose(np.diag(U), 0.0, atol=ATOL)

    def test_upper_with_extra(self):
        """extra=1 includes the diagonal."""
        U = nrutils.upperTriangle(3, 3, extra=1)
        np.testing.assert_allclose(np.diag(U), 1.0, atol=ATOL)

    def test_lower_with_extra(self):
        """extra=1 includes the diagonal."""
        L = nrutils.lowerTriangle(3, 3, extra=1)
        np.testing.assert_allclose(np.diag(L), 1.0, atol=ATOL)

    def test_rectangular(self):
        U = nrutils.upperTriangle(3, 5)
        assert U.shape == (3, 5)
        L = nrutils.lowerTriangle(3, 5)
        assert L.shape == (3, 5)


# ======================================================================
#  vabs — Euclidean norm
#  Properties: positive definite, homogeneous, triangle inequality
# ======================================================================

class TestVabs:
    def test_pythagorean_triple(self):
        """‖(3, 4)‖ = 5  (Pythagorean triple)."""
        np.testing.assert_allclose(nrutils.vabs(np.array([3.0, 4.0])), 5.0, rtol=RTOL, atol=ATOL)

    def test_positive_definite(self):
        """‖v‖ = 0 iff v = 0."""
        assert nrutils.vabs(np.zeros(5)) == 0.0
        assert nrutils.vabs(np.array([1e-15])) > 0.0

    def test_homogeneity(self):
        """‖αv‖ = |α| · ‖v‖."""
        v = RNG.standard_normal(10)
        alpha = -3.7
        np.testing.assert_allclose(
            nrutils.vabs(alpha * v),
            abs(alpha) * nrutils.vabs(v),
            rtol=RTOL, atol=ATOL,
        )

    def test_triangle_inequality(self):
        """‖a + b‖ ≤ ‖a‖ + ‖b‖."""
        a = RNG.standard_normal(20)
        b = RNG.standard_normal(20)
        assert nrutils.vabs(a + b) <= nrutils.vabs(a) + nrutils.vabs(b) + ATOL

    def test_unit_vectors(self):
        """Standard basis vectors have norm 1."""
        for i in range(3):
            e = np.zeros(3)
            e[i] = 1.0
            np.testing.assert_allclose(nrutils.vabs(e), 1.0, rtol=RTOL, atol=ATOL)


# ======================================================================
#  dummy Jacobians — zero matrices of correct shape and dtype
# ======================================================================

class TestDummyJacobians:
    @pytest.mark.parametrize("n", [1, 5, 20])
    def test_dp_shapes_and_zeros(self, n):
        y = np.ones(n, dtype=np.float64)
        dfdx, dfdy = nrutils.dummy_jacobian_dp(0.0, y)
        assert dfdx.shape == (n,) and dfdy.shape == (n, n)
        assert dfdx.dtype == np.float64 and dfdy.dtype == np.float64
        np.testing.assert_array_equal(dfdx, 0.0)
        np.testing.assert_array_equal(dfdy, 0.0)

    @pytest.mark.parametrize("n", [1, 5, 20])
    def test_dpc_shapes_and_zeros(self, n):
        y = np.ones(n, dtype=np.complex128)
        dfdx, dfdy = nrutils.dummy_jacobian_dpc(0.0, y)
        assert dfdx.shape == (n,) and dfdy.shape == (n, n)
        assert dfdx.dtype == np.complex128 and dfdy.dtype == np.complex128
        np.testing.assert_array_equal(dfdx, 0.0)
        np.testing.assert_array_equal(dfdy, 0.0)


# ======================================================================
#  Backward-compatible aliases
# ======================================================================

class TestBackwardCompat:
    def test_impl_aliases_exist(self):
        """_dummy_jacobian_*_impl must be importable (used by integrator tests)."""
        from pulsesuite.libpulsesuite.nrutils import (
            _dummy_jacobian_dp_impl,
            _dummy_jacobian_dpc_impl,
        )
        assert _dummy_jacobian_dp_impl is nrutils.dummy_jacobian_dp
        assert _dummy_jacobian_dpc_impl is nrutils.dummy_jacobian_dpc


# ======================================================================
#  Lazy type aliases  (sp, dp)
# ======================================================================

class TestTypeAliases:
    def test_sp_is_float32(self):
        assert nrutils.sp is np.float32

    def test_dp_is_float64(self):
        assert nrutils.dp is np.float64
