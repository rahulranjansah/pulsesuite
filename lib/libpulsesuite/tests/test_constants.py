"""
Unit tests for Constants using pytest.
Covers value and type checks for all constants.
"""
import numpy as np
import sys
import os
import pytest

# Ensure the src directory is in the path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from constants import Constants

def test_pi_value():
    """
    Test that pi is correct and of type float.
    """
    assert np.isclose(Constants.pi, np.pi, atol=1e-15)
    assert isinstance(Constants.pi, float) or isinstance(Constants.pi, np.floating)

def test_c0_value():
    """
    Test that c0 (speed of light) is correct and of type float.
    """
    assert np.isclose(Constants.c0, 299792458.0)
    assert isinstance(Constants.c0, float) or isinstance(Constants.c0, np.floating)

def test_e0_value():
    """
    Test that e0 (elementary charge) is correct and of type float.
    """
    assert np.isclose(Constants.e0, 1.60217733e-19)
    assert isinstance(Constants.e0, float) or isinstance(Constants.e0, np.floating)

def test_ii_type():
    """
    Test that ii is the imaginary unit and of type complex.
    """
    assert Constants.ii == 1j
    assert isinstance(Constants.ii, complex) or isinstance(Constants.ii, np.complexfloating)

def test_as_dict():
    """
    Test that as_dict returns a dictionary with all constants.
    """
    d = Constants.as_dict()
    assert isinstance(d, dict)
    assert 'pi' in d and np.isclose(d['pi'], np.pi, atol=1e-15)
    assert 'c0' in d and np.isclose(d['c0'], 299792458.0)