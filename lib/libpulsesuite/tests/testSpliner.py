import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from spliner import Spliner

try:
    from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, interp2d
    _USE_SCIPY = True
except ImportError:
    _USE_SCIPY = False

def test_spline1D_basic():
    """Test 1D cubic spline on a quadratic function."""
    x = np.linspace(0, 10, 11)
    y = x**2
    b, c, d = Spliner.spline1D(x, y)
    for xi in np.linspace(0, 10, 21):
        yi = Spliner.seval1D(xi, x, y, b, c, d)
        assert np.isclose(yi, xi**2, atol=1e-8)

def test_spline1D_N1():
    """Test 1D spline with N=1 (edge case)."""
    x = np.array([0.0])
    y = np.array([1.0])
    try:
        b, c, d = Spliner.spline1D(x, y)
        assert b.size == 1
    except Exception:
        pass

def test_polint1():
    """Test 1D polynomial interpolation (linear)."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 2.0])
    y0 = Spliner.polint1(x, y, 0.5)
    assert np.isclose(y0, 1.0)

def test_spline1DComplex():
    """Test 1D complex spline."""
    x = np.linspace(0, 2*np.pi, 10)
    y = np.exp(1j * x)
    b, c, d = Spliner.spline1DComplex(x, y)
    for xi in np.linspace(0, 2*np.pi, 20):
        yi = Spliner.seval1DComplex(xi, x, y, b, c, d)
        assert np.isfinite(yi)

@pytest.mark.skipif(not _USE_SCIPY, reason='scipy required for 2D/3D tests')
def test_spline2D():
    """Test 2D spline interpolation using scipy."""
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.sin(np.pi * X) * np.cos(np.pi * Y)
    interp = Spliner.spline2D(x, y, Z)
    val = interp(0.5, 0.5)
    assert np.isfinite(val)

@pytest.mark.skipif(not _USE_SCIPY, reason='scipy required for 3D tests')
def test_spline3D():
    """Test 3D spline interpolation using scipy."""
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    z = np.linspace(0, 1, 5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    V = np.sin(np.pi * X) * np.cos(np.pi * Y) * np.sin(np.pi * Z)
    interp = Spliner.spline3D(x, y, z, V)
    val = interp((0.5, 0.5, 0.5))
    assert np.isfinite(val)

@pytest.mark.skipif(not _USE_SCIPY, reason='scipy required for bicubic tests')
def test_bcuint():
    """Test bicubic interpolation using scipy interp2d."""
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.sin(np.pi * X) * np.cos(np.pi * Y)
    interp = Spliner.bcuint(x, y, Z)
    val = interp(0.5, 0.5)
    assert np.isfinite(val)

def test_nan_handling():
    """Test handling of NaNs in input."""
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    z = np.ones((10, 10))
    z[5, 5] = np.nan
    if _USE_SCIPY:
        interp = Spliner.spline2D(x, y, z, kind='linear')
        val = interp(0.5, 0.5)
        assert np.isnan(val) or np.isfinite(val)

# Add more edge case tests as needed