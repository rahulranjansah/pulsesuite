import sys
import os

import numpy as np
import pytest

# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pulsegenerator import (
    GaussPulse, Sech2Pulse, SquarePulse,
    FilePulse, pulsegen1, pulsegen2, multipulsegen
)


def test_GaussPulse_basic():
    x = np.linspace(-1, 1, 100)
    FWHM = 0.5
    y = GaussPulse(x, FWHM)
    assert y.shape == x.shape
    assert np.all(y >= 0)
    # Symmetry
    assert np.allclose(y, y[::-1])
    # Max at x=0
    idx0 = np.argmin(np.abs(x))
    assert np.isclose(y[idx0], y.max(), atol=1e-8)

def test_GaussPulse_edge():
    x = np.zeros(10)
    FWHM = 0.0
    y = GaussPulse(x, FWHM)
    assert np.all(np.isnan(y) | (y == 1))  # division by zero yields nan or 1

def test_Sech2Pulse_basic():
    x = np.linspace(-1, 1, 100)
    FWHM = 0.5
    y = Sech2Pulse(x, FWHM)
    assert y.shape == x.shape
    assert np.all(y > 0)
    assert np.allclose(y, y[::-1])
    idx0 = np.argmin(np.abs(x))
    assert np.isclose(y[idx0], y.max(), atol=1e-8)

def test_Sech2Pulse_edge():
    x = np.zeros(10)
    FWHM = 0.0
    y = Sech2Pulse(x, FWHM)
    # Should be nan or 0 due to division by zero
    assert np.all(np.isnan(y) | (y == 0))

def test_SquarePulse_basic():
    x = np.linspace(-1, 1, 100)
    FWHM = 1.0
    y = SquarePulse(x, FWHM)
    assert y.shape == x.shape
    assert set(np.unique(y)).issubset({0.0, 1.0})
    # Check correct region
    inside = np.abs(x) < 0.5 * FWHM
    assert np.all(y[inside] == 1.0)
    assert np.all(y[~inside] == 0.0)

def test_SquarePulse_edge():
    x = np.linspace(-1, 1, 10)
    FWHM = 0.0
    y = SquarePulse(x, FWHM)
    assert np.all(y == 0.0)

def test_FilePulse(tmp_path):
    # Create a file with 2 rows (real only)
    X1 = np.linspace(-1, 1, 10)
    Z1 = np.sin(X1)
    P = np.vstack([X1, Z1])
    fn = tmp_path / "pulse.txt"
    np.savetxt(fn, P)
    X = np.linspace(-1, 1, 20)
    Y = FilePulse(str(fn), X)
    assert Y.shape == X.shape
    assert np.issubdtype(Y.dtype, np.complexfloating)
    # Now with 3 rows (real + imag)
    Z1i = np.cos(X1)
    P2 = np.vstack([X1, Z1, Z1i])
    fn2 = tmp_path / "pulse2.txt"
    np.savetxt(fn2, P2)
    Y2 = FilePulse(str(fn2), X)
    assert Y2.shape == X.shape
    assert np.issubdtype(Y2.dtype, np.complexfloating)

def test_FilePulse_badfile(tmp_path):
    fn = tmp_path / "nofile.txt"
    X = np.linspace(-1, 1, 10)
    with pytest.raises(OSError):
        FilePulse(str(fn), X)

def test_pulsegen2_gauss():
    X = np.linspace(-1, 1, 100)
    FWHM = 0.5
    Y = pulsegen2("gauss", FWHM, X)
    assert Y.shape == X.shape
    assert np.issubdtype(Y.dtype, np.complexfloating)

def test_pulsegen2_sech2():
    X = np.linspace(-1, 1, 100)
    FWHM = 0.5
    Y = pulsegen2("sech2", FWHM, X)
    assert Y.shape == X.shape
    assert np.issubdtype(Y.dtype, np.complexfloating)

def test_pulsegen2_square():
    X = np.linspace(-1, 1, 100)
    FWHM = 0.5
    Y = pulsegen2("square", FWHM, X)
    assert Y.shape == X.shape
    assert np.issubdtype(Y.dtype, np.complexfloating)

def test_pulsegen2_file(tmp_path):
    X1 = np.linspace(-1, 1, 10)
    Z1 = np.sin(X1)
    P = np.vstack([X1, Z1])
    fn = tmp_path / "pulse.txt"
    np.savetxt(fn, P)
    X = np.linspace(-1, 1, 20)
    Y = pulsegen2(f"file:{fn}", 0, X)
    assert Y.shape == X.shape
    assert np.issubdtype(Y.dtype, np.complexfloating)

def test_pulsegen2_badshape():
    X = np.linspace(-1, 1, 10)
    FWHM = 0.5
    with pytest.raises(ValueError):
        pulsegen2("unknown", FWHM, X)

def test_pulsegen1_basic():
    N = 50
    dx = 0.1
    FWHM = 0.5
    Y, X = pulsegen1("gauss", FWHM, N, dx)
    assert Y.shape == X.shape == (N,)
    assert np.issubdtype(Y.dtype, np.complexfloating)

def test_pulsegen1_X_out():
    N = 20
    dx = 0.2
    FWHM = 0.5
    X_out = np.zeros(N)
    Y, X = pulsegen1("sech2", FWHM, N, dx, X_out)
    assert np.allclose(X, X_out)

def test_multipulsegen_2_gauss():
    t = np.linspace(-2, 2, 100)
    y = multipulsegen("gauss", 0.5, t, 1.0, 2)
    assert y.shape == t.shape
    assert np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.complexfloating)

def test_multipulsegen_3_gauss():
    t = np.linspace(-2, 2, 100)
    y = multipulsegen("gauss", 0.5, t, 1.0, 3)
    assert y.shape == t.shape
    assert np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.complexfloating)

def test_multipulsegen_2_sech():
    t = np.linspace(-2, 2, 100)
    y = multipulsegen("sech2", 0.5, t, 1.0, 2)
    assert y.shape == t.shape
    assert np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.complexfloating)

def test_multipulsegen_3_sech():
    t = np.linspace(-2, 2, 100)
    y = multipulsegen("sech2", 0.5, t, 1.0, 3)
    assert y.shape == t.shape
    assert np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.complexfloating)

def test_multipulsegen_uneven2a():
    t = np.linspace(-2, 2, 100)
    y = multipulsegen("uneven2a", 0.5, t, 1.0, 2)
    assert y.shape == t.shape
    assert np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.complexfloating)

def test_multipulsegen_uneven2b():
    t = np.linspace(-2, 2, 100)
    y = multipulsegen("uneven2b", 0.5, t, 1.0, 2)
    assert y.shape == t.shape
    assert np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.complexfloating)

def test_multipulsegen_badnum():
    t = np.linspace(-2, 2, 100)
    with pytest.raises(ValueError):
        multipulsegen("gauss", 0.5, t, 1.0, 4)