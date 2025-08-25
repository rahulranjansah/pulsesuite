"""
Unit tests for nrutils.py utilities.

Covers array/matrix utilities, polynomial evaluation, and more. Uses pytest and numpy.
"""
import sys
import os
import numpy as np
import pytest

# Adjust sys.path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import nrutils

def test_arrayCopy():
    """
    Test arrayCopy copies elements correctly and returns correct counts.
    """
    src = np.arange(5)
    dest = np.zeros(5)
    nCopied, nNotCopied = nrutils.arrayCopy(src, dest)
    assert nCopied == 5
    assert nNotCopied == 0
    assert np.all(dest == src)
    # Test with dest smaller than src
    dest2 = np.zeros(3)
    nCopied, nNotCopied = nrutils.arrayCopy(src, dest2)
    assert nCopied == 3
    assert nNotCopied == 2
    assert np.all(dest2 == src[:3])

def test_swap():
    """
    Test swap swaps two values.
    """
    a, b = 1, 2
    b2, a2 = nrutils.swap(a, b)
    assert a2 == a and b2 == b
    arr1 = np.array([1, 2])
    arr2 = np.array([3, 4])
    arr2b, arr1b = nrutils.swap(arr1, arr2)
    assert np.all(arr1b == arr1) and np.all(arr2b == arr2)

def test_reallocate():
    """
    Test reallocate resizes arrays and copies data.
    """
    arr = np.arange(5)
    newArr = nrutils.reallocate(arr, 8)
    assert newArr.shape == (8,)
    assert np.all(newArr[:5] == arr)
    arr2 = np.arange(6).reshape(2, 3)
    newArr2 = nrutils.reallocate(arr2, (3, 4))
    assert newArr2.shape == (3, 4)
    assert np.all(newArr2[:2, :3] == arr2)

def test_imaxloc_iminloc():
    """
    Test imaxloc and iminloc return correct 1-based indices.
    """
    arr = np.array([1, 3, 2, 5, 4])
    assert nrutils.imaxloc(arr) == 4
    assert nrutils.iminloc(arr) == 1

def test_assertTrue_and_assertEq():
    """
    Test assertTrue and assertEq for correct assertion behavior.
    """
    # Should not raise
    nrutils.assertTrue(True, "Should not fail")
    assert nrutils.assertEq(1, 1, 1) == 1
    # Should raise SystemExit for assertTrue(False)
    with pytest.raises(SystemExit):
        nrutils.assertTrue(False, "Should fail")
    # Should raise SystemExit for assertEq with unequal args
    with pytest.raises(SystemExit):
        nrutils.assertEq(1, 2, 1)

def test_arth_geop():
    """
    Test arth and geop generate correct progressions.
    """
    arr = nrutils.arth(1, 2, 5)
    assert np.all(arr == np.array([1, 3, 5, 7, 9]))
    arr2 = nrutils.geop(2, 3, 4)
    assert np.all(arr2 == np.array([2, 6, 18, 54]))

def test_cumsum():
    """
    Test cumsum computes cumulative sum with and without seed.
    """
    arr = np.array([1, 2, 3])
    out = nrutils.cumsum(arr)
    assert np.all(out == np.array([1, 3, 6]))
    out2 = nrutils.cumsum(arr, seed=10)
    assert np.all(out2 == np.array([11, 13, 16]))

def test_poly_polyTerm():
    """
    Test poly and polyTerm for polynomial evaluation.
    """
    coeffs = np.array([1, 2, 3])  # 1 + 2x + 3x^2
    x = 2
    assert nrutils.poly(x, coeffs) == 1 + 2*2 + 3*4
    a = np.array([1, 2, 3])
    b = 2
    u = nrutils.polyTerm(a, b)
    assert np.allclose(u, [1, 2 + 2*1, 3 + 2*(2 + 2*1)])

def test_outerprod_outerdiff():
    """
    Test outerprod and outerdiff for correct array operations.
    """
    a = np.array([1, 2])
    b = np.array([3, 4])
    op = nrutils.outerprod(a, b)
    assert np.all(op == np.array([[3, 4], [6, 8]]))
    od = nrutils.outerdiff(a, b)
    assert np.all(od == np.array([[-2, -3], [-1, -2]]))

def test_scatterAdd_scatterMax():
    """
    Test scatterAdd and scatterMax for correct scatter operations.
    """
    dest = np.zeros(5)
    source = np.array([10, 20])
    destIndex = np.array([2, 4])
    nrutils.scatterAdd(dest, source, destIndex)
    assert np.all(dest == np.array([0, 10, 0, 20, 0]))
    dest2 = np.zeros(5)
    nrutils.scatterMax(dest2, source, destIndex)
    assert np.all(dest2 == np.array([0, 10, 0, 20, 0]))
    # Test max update
    nrutils.scatterMax(dest2, np.array([15, 5]), destIndex)
    assert np.all(dest2 == np.array([0, 15, 0, 20, 0]))

def test_diagAdd_diagMult_getDiag_putDiag():
    """
    Test diagAdd, diagMult, getDiag, putDiag for matrix diagonal operations.
    """
    mat = np.eye(3)
    nrutils.diagAdd(mat, 2)
    assert np.allclose(np.diag(mat), [3, 3, 3])
    nrutils.diagMult(mat, 2)
    assert np.allclose(np.diag(mat), [6, 6, 6])
    diag = nrutils.getDiag(mat)
    assert np.allclose(diag, [6, 6, 6])
    nrutils.putDiag(np.array([1, 2, 3]), mat)
    assert np.allclose(np.diag(mat), [1, 2, 3])

def test_unitMatrix_upper_lower_vabs():
    """
    Test unitMatrix, upperTriangle, lowerTriangle, vabs.
    """
    mat = np.zeros((3, 3))
    nrutils.unitMatrix(mat)
    assert np.allclose(mat, np.eye(3))
    upper = nrutils.upperTriangle(3, 3)
    lower = nrutils.lowerTriangle(3, 3)
    assert upper.shape == (3, 3)
    assert lower.shape == (3, 3)
    v = np.array([3, 4])
    assert np.isclose(nrutils.vabs(v), 5.0)


