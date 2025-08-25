"""
Unit tests for IntLengthCalculator using pytest.
Covers scalar, array, and edge cases for both integer and double length calculations.
"""
import numpy as np
import sys
import os
import pytest

# Ensure the src directory is in the path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from calcintlength import IntLengthCalculator

def testcalcIntLengthScalar():
    """
    Test CalcIntLength with scalar integer inputs.
    """
    assert IntLengthCalculator.calcIntLength(0) == 1
    assert IntLengthCalculator.calcIntLength(5) == 1
    assert IntLengthCalculator.calcIntLength(-5) == 2
    assert IntLengthCalculator.calcIntLength(12345) == 5
    assert IntLengthCalculator.calcIntLength(-12345) == 6
    assert IntLengthCalculator.calcIntLength(999999999) == 9
    assert IntLengthCalculator.calcIntLength(-999999999) == 10

def testcalcIntLengthArray():
    """
    Test CalcIntLength with array integer inputs.
    """
    arr = np.array([0, 1, -10, 123456, -999999])
    expected = np.array([1, 1, 3, 6, 7])
    np.testing.assert_array_equal(IntLengthCalculator.calcIntLength(arr), expected)

def testcalcDblLengthScalar():
    """
    Test CalcDblLength with scalar float inputs.
    """
    assert IntLengthCalculator.calcDblLength(3.14159, 6, 2, 'ES') == 12
    assert IntLengthCalculator.calcDblLength(-3.14159, 6, 2, 'ES') == 13
    assert IntLengthCalculator.calcDblLength(0.0, 6, 2, 'EN') >= 12

def testcalcDblLengthArray():
    """
    Test CalcDblLength with array float inputs.
    """
    darr = np.array([1.23, -456.789, 0.0])
    result = IntLengthCalculator.calcDblLength(darr, 6, 2, 'EN')
    assert isinstance(result, np.ndarray)
    assert result.shape == darr.shape
    assert np.all(result >= 12)

def testcalcDblLengthEdgeCases():
    """
    Test CalcDblLength with edge cases (very large, very small, negative floats).
    """
    darr = np.array([1e-100, -1e100, 0.0, 1.0, -1.0])
    result = IntLengthCalculator.calcDblLength(darr, 6, 2, 'E')
    assert np.all(result >= 12)