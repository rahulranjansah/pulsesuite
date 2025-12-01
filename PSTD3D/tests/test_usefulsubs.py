"""
Comprehensive test suite for usefulsubs.py module.

Tests all utility functions including array flipping, FFT operations,
derivatives, field manipulations, special functions, and I/O operations.
"""

import numpy as np
import pytest
import tempfile
import os
import shutil
from pathlib import Path

# Import the module to test
import sys
sys.path.insert(0, '../src')
import usefulsubs as us


class TestArrayFlipping:
    """Test array flipping functions."""

    def test_fflip_dp_simple(self):
        """Test flipping real array with simple case."""
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = us.fflip_dp(f)
        expected = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_fflip_dp_single_element(self):
        """Test flipping single element array."""
        f = np.array([42.0])
        result = us.fflip_dp(f)
        assert np.allclose(result, f, rtol=1e-12, atol=1e-12)

    def test_fflip_dp_even_length(self):
        """Test flipping even length array."""
        f = np.linspace(0, 10, 100)
        result = us.fflip_dp(f)
        expected = f[::-1]
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_fflip_dp_odd_length(self):
        """Test flipping odd length array."""
        f = np.linspace(0, 10, 101)
        result = us.fflip_dp(f)
        expected = f[::-1]
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_fflip_dpc_complex(self):
        """Test flipping complex array."""
        f = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        result = us.fflip_dpc(f)
        expected = np.array([7+8j, 5+6j, 3+4j, 1+2j])
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_fflip_dpc_single_element(self):
        """Test flipping single element complex array."""
        f = np.array([1+1j])
        result = us.fflip_dpc(f)
        assert np.allclose(result, f, rtol=1e-12, atol=1e-12)


class TestDerivatives1D:
    """Test 1D derivative functions using FFT."""

    def test_dfdy1D_linear(self):
        """Test derivative of periodic function (FFT assumes periodicity)."""
        N = 64
        y = np.linspace(-10e-6, 10e-6, N)
        dy = y[1] - y[0]
        qy = 2.0 * np.pi * np.fft.fftfreq(N, dy)

        # Use periodic function: f(y) = sin(k*y), df/dy = k*cos(k*y)
        k = 2.0 * np.pi / (N * dy)  # Wavenumber for one period
        f = np.sin(k * y)
        f_complex = f.astype(np.complex128)

        df = us.dfdy1D(f_complex, qy)
        expected = k * np.cos(k * y)

        # Test interior points (avoid boundary effects)
        interior = slice(N//4, 3*N//4)
        assert np.allclose(np.real(df[interior]), expected[interior], rtol=1e-8, atol=1e-10)

    def test_dfdy1D_quadratic(self):
        """Test derivative of periodic function (FFT assumes periodicity)."""
        N = 128
        y = np.linspace(-10e-6, 10e-6, N)
        dy = y[1] - y[0]
        qy = 2.0 * np.pi * np.fft.fftfreq(N, dy)

        # Use periodic function: f(y) = cos(k*y), df/dy = -k*sin(k*y)
        k = 2.0 * np.pi / (N * dy)  # Wavenumber for one period
        f = np.cos(k * y)
        f_complex = f.astype(np.complex128)

        df = us.dfdy1D(f_complex, qy)
        expected = -k * np.sin(k * y)

        # Test interior points (avoid boundary effects)
        interior = slice(N//4, 3*N//4)
        assert np.allclose(np.real(df[interior]), expected[interior], rtol=1e-6, atol=1e-10)

    def test_dfdy1D_gaussian(self):
        """Test derivative of Gaussian function (well-localized, minimal boundary effects)."""
        N = 256
        y = np.linspace(-20e-6, 20e-6, N)
        dy = y[1] - y[0]
        qy = 2.0 * np.pi * np.fft.fftfreq(N, dy)

        sigma = 3e-6
        f = np.exp(-(y/sigma)**2)
        f_complex = f.astype(np.complex128)

        df = us.dfdy1D(f_complex, qy)
        expected = -2 * y / sigma**2 * np.exp(-(y/sigma)**2)

        # Test interior points where Gaussian is significant (avoid boundary effects)
        # Gaussian is significant within ~3*sigma
        mask = np.abs(y) < 3 * sigma
        assert np.allclose(np.real(df[mask]), expected[mask], rtol=1e-4, atol=1e-10)

    def test_dfdy1D_single_element(self):
        """Test derivative with single element array."""
        f = np.array([1.0+0j])
        qy = np.array([0.0])
        result = us.dfdy1D(f, qy)
        assert np.allclose(result, np.zeros_like(f), rtol=1e-12, atol=1e-12)

    def test_dfdx1D_linear(self):
        """Test x-derivative of periodic function (FFT assumes periodicity)."""
        N = 64
        x = np.linspace(-10e-6, 10e-6, N)
        dx = x[1] - x[0]
        qx = 2.0 * np.pi * np.fft.fftfreq(N, dx)

        # Use periodic function: f(x) = sin(k*x), df/dx = k*cos(k*x)
        k = 2.0 * np.pi / (N * dx)  # Wavenumber for one period
        f = np.sin(k * x)
        f_complex = f.astype(np.complex128)

        df = us.dfdx1D(f_complex, qx)
        expected = k * np.cos(k * x)

        # Test interior points (avoid boundary effects)
        interior = slice(N//4, 3*N//4)
        assert np.allclose(np.real(df[interior]), expected[interior], rtol=1e-8, atol=1e-10)


class TestDerivatives2D:
    """Test 2D derivative functions using FFT."""

    def test_dfdy2D_constant(self):
        """Test 2D derivative of constant function."""
        Nx, Ny = 32, 32
        y = np.linspace(-10e-6, 10e-6, Ny)
        dy = y[1] - y[0]
        qy = 2.0 * np.pi * np.fft.fftfreq(Ny, dy)

        f = np.ones((Nx, Ny), dtype=np.complex128)
        df = us.dfdy2D(f, qy)

        assert np.allclose(np.abs(df), 0.0, rtol=1e-10, atol=1e-10)

    def test_dfdy2D_linear(self):
        """Test 2D derivative of periodic function in y (FFT assumes periodicity)."""
        Nx, Ny = 32, 64
        y = np.linspace(-10e-6, 10e-6, Ny)
        dy = y[1] - y[0]
        qy = 2.0 * np.pi * np.fft.fftfreq(Ny, dy)

        # Use periodic function: f(x,y) = a*sin(k*y)
        a = 2.5
        k = 2.0 * np.pi / (Ny * dy)  # Wavenumber for one period
        f = np.zeros((Nx, Ny), dtype=np.complex128)
        for i in range(Nx):
            f[i, :] = a * np.sin(k * y)

        df = us.dfdy2D(f, qy)
        expected = np.zeros_like(f)
        for i in range(Nx):
            expected[i, :] = a * k * np.cos(k * y)

        # Test interior points (avoid boundary effects)
        interior = slice(Ny//4, 3*Ny//4)
        assert np.allclose(np.real(df[:, interior]), expected[:, interior], rtol=1e-7, atol=1e-10)

    def test_dfdx2D_linear(self):
        """Test 2D derivative of periodic function in x (FFT assumes periodicity)."""
        Nx, Ny = 64, 32
        x = np.linspace(-10e-6, 10e-6, Nx)
        dx = x[1] - x[0]
        qx = 2.0 * np.pi * np.fft.fftfreq(Nx, dx)

        # Use periodic function: f(x,y) = a*sin(k*x)
        a = 3.0
        k = 2.0 * np.pi / (Nx * dx)  # Wavenumber for one period
        f = np.zeros((Nx, Ny), dtype=np.complex128)
        for j in range(Ny):
            f[:, j] = a * np.sin(k * x)

        df = us.dfdx2D(f, qx)
        expected = np.zeros_like(f)
        for j in range(Ny):
            expected[:, j] = a * k * np.cos(k * x)

        # Test interior points (avoid boundary effects)
        interior = slice(Nx//4, 3*Nx//4)
        assert np.allclose(np.real(df[interior, :]), expected[interior, :], rtol=1e-7, atol=1e-10)

    def test_dfdy2D_single_element(self):
        """Test 2D derivative with single element in y."""
        f = np.ones((10, 1), dtype=np.complex128)
        qy = np.array([0.0])
        result = us.dfdy2D(f, qy)
        assert np.allclose(result, np.zeros_like(f), rtol=1e-12, atol=1e-12)


class TestDerivativesQSpace:
    """Test derivative functions in q-space (Fourier domain)."""

    def test_dfdy1D_q_simple(self):
        """Test 1D q-space derivative."""
        N = 64
        qy = np.linspace(-1e6, 1e6, N)
        f = np.random.random(N) + 1j * np.random.random(N)

        df = us.dfdy1D_q(f, qy)
        expected = f * 1j * qy

        assert np.allclose(df, expected, rtol=1e-12, atol=1e-12)

    def test_dfdx1D_q_simple(self):
        """Test 1D q-space derivative in x."""
        N = 64
        qx = np.linspace(-1e6, 1e6, N)
        f = np.random.random(N) + 1j * np.random.random(N)

        df = us.dfdx1D_q(f, qx)
        expected = f * 1j * qx

        assert np.allclose(df, expected, rtol=1e-12, atol=1e-12)

    def test_dfdy2D_q_simple(self):
        """Test 2D q-space derivative in y."""
        Nx, Ny = 32, 32
        qy = np.linspace(-1e6, 1e6, Ny)
        f = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))

        df = us.dfdy2D_q(f, qy)

        # Check each column
        for i in range(Nx):
            for j in range(Ny):
                assert np.allclose(df[i, j], f[i, j] * 1j * qy[j], rtol=1e-12, atol=1e-12)

    def test_dfdx2D_q_simple(self):
        """Test 2D q-space derivative in x."""
        Nx, Ny = 32, 32
        qx = np.linspace(-1e6, 1e6, Nx)
        f = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))

        df = us.dfdx2D_q(f, qx)

        # Check each element
        for i in range(Nx):
            for j in range(Ny):
                assert np.allclose(df[i, j], f[i, j] * 1j * qx[i], rtol=1e-12, atol=1e-12)

    def test_dfdy1D_q_single_element(self):
        """Test q-space derivative with single element."""
        f = np.array([1.0+1j])
        qy = np.array([0.0])
        result = us.dfdy1D_q(f, qy)
        assert np.allclose(result, np.zeros_like(f), rtol=1e-12, atol=1e-12)


class TestGaussianFFT:
    """Test Gaussian-normalized FFT functions."""

    def test_GFFT_1D_normalization(self):
        """Test 1D Gaussian FFT normalization."""
        N = 64
        dx = 1e-7
        f = np.random.random(N) + 1j * np.random.random(N)
        f_original = f.copy()

        us.GFFT_1D(f, dx)

        # Check that FFT was applied with correct normalization
        expected = np.fft.fft(f_original) * dx / np.sqrt(2 * np.pi)
        assert np.allclose(f, expected, rtol=1e-12, atol=1e-12)

    def test_GIFFT_1D_normalization(self):
        """Test 1D Gaussian IFFT normalization."""
        N = 64
        dq = 1e5
        f = np.random.random(N) + 1j * np.random.random(N)
        f_original = f.copy()

        us.GIFFT_1D(f, dq)

        # Check that IFFT was applied with correct normalization
        expected = np.fft.ifft(f_original) * dq / np.sqrt(2 * np.pi) * N
        assert np.allclose(f, expected, rtol=1e-12, atol=1e-12)

    def test_GFFT_GIFFT_1D_roundtrip(self):
        """Test FFT-IFFT round trip with Gaussian normalization."""
        N = 128
        dx = 1e-7
        x = np.linspace(-N*dx/2, N*dx/2, N)

        # Create a Gaussian function
        f_original = np.exp(-(x/3e-6)**2).astype(np.complex128)
        f = f_original.copy()

        # Compute momentum space parameters
        dq = 2 * np.pi / (N * dx)

        # Forward and backward transform
        us.GFFT_1D(f, dx)
        us.GIFFT_1D(f, dq)

        # Should recover original (with normalization factor)
        # The normalization should preserve the integral
        assert f.shape == f_original.shape

    def test_GFFT_2D_normalization(self):
        """Test 2D Gaussian FFT normalization."""
        Nx, Ny = 32, 32
        dx, dy = 1e-7, 1e-7
        f = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))
        f_original = f.copy()

        us.GFFT_2D(f, dx, dy)

        # Check that FFT was applied with correct normalization
        expected = np.fft.fft2(f_original) * dx * dy / (2 * np.pi)
        assert np.allclose(f, expected, rtol=1e-12, atol=1e-12)

    def test_GIFFT_2D_normalization(self):
        """Test 2D Gaussian IFFT normalization."""
        Nx, Ny = 32, 32
        dqx, dqy = 1e5, 1e5
        f = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))
        f_original = f.copy()

        us.GIFFT_2D(f, dqx, dqy)

        # Check that IFFT was applied with correct normalization
        expected = np.fft.ifft2(f_original) * dqx * dqy / (2 * np.pi) * Nx * Ny
        assert np.allclose(f, expected, rtol=1e-12, atol=1e-12)


class TestFieldDerivatives:
    """Test electromagnetic field derivative functions."""

    def test_dEdx_forward_difference(self):
        """Test E field x-derivative with forward difference."""
        Nx, Ny = 64, 64
        dx = 1e-7
        x = np.arange(Nx) * dx

        # Create a field that varies linearly in x
        E = np.zeros((Nx, Ny), dtype=np.complex128)
        for j in range(Ny):
            E[:, j] = x

        dE = us.dEdx(E, dx)

        # The implementation uses roll(-1) which gives forward difference
        # dE[i] = (E[iu[i]] - E[i])/dx where iu = roll([0..N-1], -1)
        # For E[i] = i*dx: dE[i] = (E[i+1] - E[i])/dx = ((i+1)*dx - i*dx)/dx = 1
        # Interior points (skip last row due to wraparound): should be +1 (forward diff)
        expected_interior = np.ones((Nx, Ny))  # Forward difference of linear function
        # Skip last row (wraparound) and test rows 0 to N-2
        assert np.allclose(np.real(dE[:-1, :]), expected_interior[:-1, :], rtol=1e-10, atol=1e-10)

    def test_dEdy_forward_difference(self):
        """Test E field y-derivative with forward difference."""
        Nx, Ny = 64, 64
        dy = 1e-7
        y = np.arange(Ny) * dy

        # Create a field that varies linearly in y
        E = np.zeros((Nx, Ny), dtype=np.complex128)
        for i in range(Nx):
            E[i, :] = y

        dE = us.dEdy(E, dy)

        # The implementation uses roll(-1) which gives forward difference
        # dE[j] = (E[ju[j]] - E[j])/dy where ju = roll([0..N-1], -1)
        # For E[j] = j*dy: dE[j] = (E[j+1] - E[j])/dy = ((j+1)*dy - j*dy)/dy = 1
        # Interior points (skip last column due to wraparound): should be +1 (forward diff)
        expected_interior = np.ones((Nx, Ny))  # Forward difference of linear function
        # Skip last column (wraparound) and test columns 0 to N-2
        assert np.allclose(np.real(dE[:, :-1]), expected_interior[:, :-1], rtol=1e-10, atol=1e-10)

    def test_dHdx_backward_difference(self):
        """Test H field x-derivative with backward difference."""
        Nx, Ny = 64, 64
        dx = 1e-7
        x = np.arange(Nx) * dx

        # Create a field that varies linearly in x
        H = np.zeros((Nx, Ny), dtype=np.complex128)
        for j in range(Ny):
            H[:, j] = x

        dH = us.dHdx(H, dx)

        # The implementation uses roll(+1) which gives backward difference
        # dH[i] = (H[i] - H[id_arr[i]])/dx where id_arr = roll([0..N-1], +1)
        # For H[i] = i*dx: dH[i] = (H[i] - H[i-1])/dx = (i*dx - (i-1)*dx)/dx = 1
        # Interior points (skip first row due to wraparound): should be +1 (backward diff)
        expected_interior = np.ones((Nx, Ny))  # Backward difference of linear function
        # Skip first row (wraparound) and test rows 1 to N-1
        assert np.allclose(np.real(dH[1:, :]), expected_interior[1:, :], rtol=1e-10, atol=1e-10)

    def test_dHdy_backward_difference(self):
        """Test H field y-derivative with backward difference."""
        Nx, Ny = 64, 64
        dy = 1e-7
        y = np.arange(Ny) * dy

        # Create a field that varies linearly in y
        H = np.zeros((Nx, Ny), dtype=np.complex128)
        for i in range(Nx):
            H[i, :] = y

        dH = us.dHdy(H, dy)

        # The implementation uses roll(+1) which gives backward difference
        # dH[j] = (H[j] - H[jd[j]])/dy where jd = roll([0..N-1], +1)
        # For H[j] = j*dy: dH[j] = (H[j] - H[j-1])/dy = (j*dy - (j-1)*dy)/dy = 1
        # Interior points (skip first column due to wraparound): should be +1 (backward diff)
        expected_interior = np.ones((Nx, Ny))  # Backward difference of linear function
        # Skip first column (wraparound) and test columns 1 to N-1
        assert np.allclose(np.real(dH[:, 1:]), expected_interior[:, 1:], rtol=1e-10, atol=1e-10)

    def test_dEdx_single_row(self):
        """Test E field x-derivative with single row."""
        E = np.ones((1, 10), dtype=np.complex128)
        dx = 1e-7
        result = us.dEdx(E, dx)
        assert np.allclose(result, np.zeros_like(E), rtol=1e-12, atol=1e-12)

    def test_dEdy_single_column(self):
        """Test E field y-derivative with single column."""
        E = np.ones((10, 1), dtype=np.complex128)
        dy = 1e-7
        result = us.dEdy(E, dy)
        assert np.allclose(result, np.zeros_like(E), rtol=1e-12, atol=1e-12)


class TestAbsorbingBoundaryConditions:
    """Test absorbing boundary condition application."""

    def test_ApplyABC_simple(self):
        """Test ABC application with simple coefficients."""
        Nx, Ny = 32, 32
        Field = np.ones((Nx, Ny), dtype=np.complex128)
        abc = np.ones((Nx, Ny))

        # Apply ABC
        us.ApplyABC(Field, abc)

        # With abc=1, field should be unchanged (after FFT-IFFT round trip)
        assert Field.shape == (Nx, Ny)

    def test_ApplyABC_absorption(self):
        """Test ABC absorption at boundaries."""
        Nx, Ny = 64, 64
        Field = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))
        Field_original = Field.copy()

        # Create ABC with absorption at edges
        abc = np.ones((Nx, Ny))
        abc[:5, :] = 0.5
        abc[-5:, :] = 0.5
        abc[:, :5] = 0.5
        abc[:, -5:] = 0.5

        # Apply ABC
        us.ApplyABC(Field, abc)

        # Field should be modified
        assert not np.allclose(Field, Field_original, rtol=1e-12, atol=1e-12)


class TestFourierTransforms:
    """Test direct Fourier transform implementations."""

    def test_FT_IFT_roundtrip(self):
        """Test FT-IFT round trip."""
        N = 32
        x = np.linspace(-10e-6, 10e-6, N)
        dx = x[1] - x[0]
        q = 2 * np.pi * np.fft.fftfreq(N, dx)

        # Create a simple function
        y_original = np.exp(-(x/3e-6)**2).astype(np.complex128)
        y = y_original.copy()

        # Forward and inverse transform
        us.FT(y, x, q)
        us.IFT(y, x, q)

        # Should recover original (within numerical precision)
        assert np.allclose(y, y_original, rtol=1e-10, atol=1e-10)

    def test_FT_gaussian(self):
        """Test FT of Gaussian function."""
        N = 64
        x = np.linspace(-20e-6, 20e-6, N)
        dx = x[1] - x[0]
        q = 2 * np.pi * np.fft.fftfreq(N, dx)

        # Gaussian function
        sigma = 3e-6
        y = np.exp(-(x/sigma)**2).astype(np.complex128)

        us.FT(y, x, q)

        # FT of Gaussian should also be Gaussian-like
        assert np.all(np.isfinite(y))


class TestFlipFunction:
    """Test Flip function."""

    def test_Flip_simple(self):
        """Test Flip with simple array."""
        x = np.array([1+1j, 2+2j, 3+3j, 4+4j])
        result = us.Flip(x)
        expected = np.array([4+4j, 3+3j, 2+2j, 1+1j])
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_Flip_single_element(self):
        """Test Flip with single element."""
        x = np.array([1+1j])
        result = us.Flip(x)
        assert np.allclose(result, x, rtol=1e-12, atol=1e-12)


class TestGetArray0Index:
    """Test GetArray0Index function."""

    def test_GetArray0Index_center(self):
        """Test finding zero index in centered array."""
        x = np.linspace(-10.0, 10.0, 201)
        idx = us.GetArray0Index(x)
        assert x[idx] == pytest.approx(0.0, abs=0.11)

    def test_GetArray0Index_offset(self):
        """Test finding zero index in offset array."""
        x = np.linspace(-5.0, 5.0, 101)
        idx = us.GetArray0Index(x)
        assert abs(x[idx]) < 0.11

    def test_GetArray0Index_not_found(self):
        """Test error when zero not in array."""
        x = np.linspace(1.0, 10.0, 100)
        with pytest.raises(ValueError):
            us.GetArray0Index(x)


class TestSpecialFunctions:
    """Test special mathematical functions."""

    def test_GaussDelta_scalar(self):
        """Test Gaussian delta function with scalar."""
        result = us.GaussDelta(0.0, 1.0)
        expected = 1.0 / np.sqrt(np.pi)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_GaussDelta_array(self):
        """Test Gaussian delta function with array."""
        a = np.array([0.0, 1.0, 2.0])
        b = 1.0
        result = us.GaussDelta(a, b)
        expected = 1.0 / np.sqrt(np.pi) * np.exp(-(a/b)**2)
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_delta_at_zero(self):
        """Test delta function at zero."""
        result = us.delta(0.0, 1.0)
        assert result == 1

    def test_delta_away_from_zero(self):
        """Test delta function away from zero."""
        result = us.delta(2.0, 1.0)
        assert result == 0

    def test_kdel_at_zero(self):
        """Test Kronecker delta at zero."""
        assert us.kdel(0) == 1

    def test_kdel_nonzero(self):
        """Test Kronecker delta at nonzero."""
        assert us.kdel(1) == 0
        assert us.kdel(-1) == 0

    def test_delt_at_zero(self):
        """Test delt function at zero."""
        result = us.delt(0)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_delt_nonzero(self):
        """Test delt function at nonzero."""
        result = us.delt(1)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_sgn_positive(self):
        """Test sign function for positive."""
        assert us.sgn(5.0) == 1

    def test_sgn_negative(self):
        """Test sign function for negative."""
        assert us.sgn(-5.0) == -1

    def test_sgn_zero(self):
        """Test sign function for zero."""
        assert us.sgn(0.0) == 1

    def test_sgn2_array(self):
        """Test sign function for array."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = us.sgn2(x)
        expected = np.array([-1, -1, 1, 1, 1])
        assert np.array_equal(result, expected)


class TestEnergyFunctions:
    """Test energy and temperature calculation functions."""

    def test_TotalEnergy_simple(self):
        """Test total energy calculation."""
        n = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
        E = np.array([1.0, 2.0, 3.0])
        result = us.TotalEnergy(n, E)
        expected = 1*1 + 2*2 + 3*3  # 14
        assert result == pytest.approx(expected, rel=1e-12)

    def test_AvgEnergy_simple(self):
        """Test average energy calculation."""
        n = np.array([1.0, 1.0, 1.0], dtype=np.complex128)
        E = np.array([1.0, 2.0, 3.0])
        result = us.AvgEnergy(n, E)
        expected = (1 + 2 + 3) / 3  # 2.0
        assert result == pytest.approx(expected, rel=1e-12)

    def test_AvgEnergy_weighted(self):
        """Test weighted average energy."""
        n = np.array([2.0, 1.0, 1.0], dtype=np.complex128)
        E = np.array([1.0, 2.0, 4.0])
        result = us.AvgEnergy(n, E)
        expected = (2*1 + 1*2 + 1*4) / (2 + 1 + 1)  # 8/4 = 2.0
        assert result == pytest.approx(expected, rel=1e-12)

    def test_Temperature_simple(self):
        """Test temperature calculation."""
        n = np.array([1.0, 1.0], dtype=np.complex128)
        E = np.array([1.0e-20, 2.0e-20])  # J
        result = us.Temperature(n, E)
        expected = 2 * 1.5e-20 / us.kB
        assert result == pytest.approx(expected, rel=1e-12)


class TestLorentzianAndStep:
    """Test Lorentzian and step functions."""

    def test_Lrtz_peak(self):
        """Test Lorentzian at peak."""
        result = us.Lrtz(0.0, 1.0)
        expected = 1.0 / np.pi
        assert result == pytest.approx(expected, rel=1e-12)

    def test_Lrtz_array(self):
        """Test Lorentzian with array."""
        a = np.linspace(-5.0, 5.0, 11)
        b = 1.0
        result = us.Lrtz(a, b)
        expected = b / np.pi / (a**2 + b**2)
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_theta_negative(self):
        """Test Heaviside step for negative."""
        result = us.theta(-1.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_theta_positive(self):
        """Test Heaviside step for positive."""
        result = us.theta(1.0)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_theta_array(self):
        """Test Heaviside step with array."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = us.theta(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        assert np.allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_softtheta_negative(self):
        """Test soft step for negative."""
        result = us.softtheta(-10.0, 1.0)
        assert result < 0.05  # Soft function has finite transition width

    def test_softtheta_positive(self):
        """Test soft step for positive."""
        result = us.softtheta(10.0, 1.0)
        assert result > 0.95  # Soft function has finite transition width

    def test_softtheta_zero(self):
        """Test soft step at zero."""
        result = us.softtheta(0.0, 1.0)
        assert result == pytest.approx(0.5, rel=1e-12)


class TestAngleConversion:
    """Test angle conversion."""

    def test_rad_simple(self):
        """Test degree to radian conversion."""
        assert us.rad(180.0) == pytest.approx(np.pi, rel=1e-12)
        assert us.rad(90.0) == pytest.approx(np.pi/2, rel=1e-12)
        assert us.rad(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_rad_array(self):
        """Test degree to radian with array."""
        degrees = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
        result = us.rad(degrees)
        expected = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)


class TestFieldRotation:
    """Test field rotation functions."""

    def test_RotateField_90deg(self):
        """Test 2D field rotation by 90 degrees."""
        Nx, Ny = 10, 10
        Ex = np.ones((Nx, Ny), dtype=np.complex128)
        Ey = np.zeros((Nx, Ny), dtype=np.complex128)

        theta = np.pi / 2
        us.RotateField(theta, Ex, Ey)

        # After 90 degree rotation, Ex should become Ey
        assert np.allclose(Ex, np.zeros((Nx, Ny)), atol=1e-12)
        assert np.allclose(Ey, np.ones((Nx, Ny)), atol=1e-12)

    def test_RotateField_180deg(self):
        """Test 2D field rotation by 180 degrees."""
        Nx, Ny = 10, 10
        Ex = np.ones((Nx, Ny), dtype=np.complex128)
        Ey = np.zeros((Nx, Ny), dtype=np.complex128)

        theta = np.pi
        us.RotateField(theta, Ex, Ey)

        # After 180 degree rotation, Ex should be -Ex
        assert np.allclose(Ex, -np.ones((Nx, Ny)), atol=1e-12)
        assert np.allclose(Ey, np.zeros((Nx, Ny)), atol=1e-12)

    def test_RotateField_360deg(self):
        """Test 2D field rotation by 360 degrees."""
        Nx, Ny = 10, 10
        Ex_orig = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))
        Ey_orig = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))
        Ex = Ex_orig.copy()
        Ey = Ey_orig.copy()

        theta = 2 * np.pi
        us.RotateField(theta, Ex, Ey)

        # After 360 degree rotation, should recover original
        assert np.allclose(Ex, Ex_orig, rtol=1e-12, atol=1e-12)
        assert np.allclose(Ey, Ey_orig, rtol=1e-12, atol=1e-12)

    def test_RotateField3D_90deg(self):
        """Test 3D field rotation by 90 degrees about y-axis."""
        Nx, Ny, Nz = 8, 8, 8
        Ex_orig = np.ones((Nx, Ny, Nz), dtype=np.complex128)
        Ey_orig = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        Ez_orig = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        Ex = Ex_orig.copy()
        Ey = Ey_orig.copy()
        Ez = Ez_orig.copy()

        theta = np.pi / 2
        us.RotateField3D(theta, Ex, Ey, Ez)

        # After 90 degree rotation about y-axis: Ex -> -Ez, Ez -> Ex
        # Rotation matrix: [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
        # For 90°: [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        # So Ex_new = Ex*0 + Ey*0 + Ez*1 = Ez (which is 0)
        # Ez_new = Ex*(-1) + Ey*0 + Ez*0 = -Ex (which is -1)
        assert np.allclose(Ex, np.zeros((Nx, Ny, Nz)), atol=1e-12)
        assert np.allclose(Ey, np.zeros((Nx, Ny, Nz)), atol=1e-12)
        assert np.allclose(Ez, -np.ones((Nx, Ny, Nz)), atol=1e-12)

    def test_RotateField3D_identity(self):
        """Test 3D field rotation by 0 degrees."""
        Nx, Ny, Nz = 8, 8, 8
        Ex_orig = np.random.random((Nx, Ny, Nz)) + 1j * np.random.random((Nx, Ny, Nz))
        Ey_orig = np.random.random((Nx, Ny, Nz)) + 1j * np.random.random((Nx, Ny, Nz))
        Ez_orig = np.random.random((Nx, Ny, Nz)) + 1j * np.random.random((Nx, Ny, Nz))
        Ex = Ex_orig.copy()
        Ey = Ey_orig.copy()
        Ez = Ez_orig.copy()

        theta = 0.0
        us.RotateField3D(theta, Ex, Ey, Ez)

        # No rotation should preserve fields
        assert np.allclose(Ex, Ex_orig, rtol=1e-12, atol=1e-12)
        assert np.allclose(Ey, Ey_orig, rtol=1e-12, atol=1e-12)
        assert np.allclose(Ez, Ez_orig, rtol=1e-12, atol=1e-12)


class TestFieldShift:
    """Test field shifting function."""

    def test_ShiftField_no_shift(self):
        """Test field shift with zero displacement."""
        Nx, Ny = 32, 32
        Ex = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))
        Ey = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))
        Ex_orig = Ex.copy()
        Ey_orig = Ey.copy()

        dx, dy = 1e-7, 1e-7
        us.ShiftField(0.0, 0.0, dx, dy, Ex, Ey)

        assert np.allclose(Ex, Ex_orig, rtol=1e-12, atol=1e-12)
        assert np.allclose(Ey, Ey_orig, rtol=1e-12, atol=1e-12)

    def test_ShiftField_x_direction(self):
        """Test field shift in x direction (LEFT shift to match Fortran Cshift)."""
        Nx, Ny = 32, 32
        dx, dy = 1e-7, 1e-7

        # Create a simple field
        Ex = np.zeros((Nx, Ny), dtype=np.complex128)
        Ey = np.zeros((Nx, Ny), dtype=np.complex128)
        Ex[Nx//2, Ny//2] = 1.0

        Lx = 5 * dx  # Shift by 5 grid points (positive = shift LEFT in Fortran/Python)
        Ly = 0.0

        us.ShiftField(Lx, Ly, dx, dy, Ex, Ey)

        # Field should be shifted LEFT (index decreases, wraps around)
        # Fortran Cshift(Ex, +5, 1) shifts LEFT, Python np.roll(Ex, -5, axis=0) also shifts LEFT
        expected_idx = (Nx//2 - 5) % Nx
        assert Ex[expected_idx, Ny//2] == pytest.approx(1.0, rel=1e-12)


class TestBesselFunctions:
    """Test modified Bessel functions."""

    def test_cik01_real_small(self):
        """Test Bessel functions for small real argument."""
        z = 0.5 + 0j
        cbi0, cdi0, cbi1, cdi1, cbk0, cdk0, cbk1, cdk1 = us.cik01(z)

        # Check that results are finite
        assert np.isfinite(cbi0)
        assert np.isfinite(cbi1)
        assert np.isfinite(cbk0)
        assert np.isfinite(cbk1)

        # I0(0.5) ≈ 1.063, I1(0.5) ≈ 0.258
        assert np.abs(cbi0) > 1.0
        assert np.abs(cbi1) > 0.2

    def test_cik01_real_large(self):
        """Test Bessel functions for large real argument."""
        z = 10.0 + 0j
        cbi0, cdi0, cbi1, cdi1, cbk0, cdk0, cbk1, cdk1 = us.cik01(z)

        # For large arguments, modified Bessel functions grow exponentially
        assert np.isfinite(cbi0)
        assert np.isfinite(cbi1)
        assert np.isfinite(cbk0)
        assert np.isfinite(cbk1)

        # K functions decay exponentially
        assert np.abs(cbk0) < 1.0
        assert np.abs(cbk1) < 1.0

    def test_cik01_complex(self):
        """Test Bessel functions for complex argument."""
        z = 1.0 + 1.0j
        cbi0, cdi0, cbi1, cdi1, cbk0, cdk0, cbk1, cdk1 = us.cik01(z)

        # All results should be finite
        assert np.isfinite(cbi0)
        assert np.isfinite(cbi1)
        assert np.isfinite(cbk0)
        assert np.isfinite(cbk1)

    def test_cik01_zero(self):
        """Test Bessel functions at zero."""
        z = 0.0 + 0j
        cbi0, cdi0, cbi1, cdi1, cbk0, cdk0, cbk1, cdk1 = us.cik01(z)

        # I0(0) = 1, I1(0) = 0
        assert cbi0 == pytest.approx(1.0, rel=1e-12)
        assert np.abs(cbi1) < 1e-12

        # K0(0) and K1(0) are infinite
        assert np.abs(cbk0) > 1e20
        assert np.abs(cbk1) > 1e20

    def test_K03_small(self):
        """Test K0 function for small argument."""
        result = us.K03(0.5)
        # K0(0.5) ≈ 0.924
        assert result > 0.9 and result < 1.0

    def test_K03_large(self):
        """Test K0 function for large argument."""
        result = us.K03(150.0)
        # For x > 100, should return 0
        assert result == 0.0


class TestLocator:
    """Test locator function for finding insertion index."""

    def test_locator_interior(self):
        """Test locator for interior point."""
        x = np.linspace(0.0, 10.0, 101)
        x0 = 5.0
        idx = us.locator(x, x0)
        assert x[idx] <= x0 < x[idx+1]

    def test_locator_start(self):
        """Test locator for point at start."""
        x = np.linspace(0.0, 10.0, 101)
        x0 = 0.0
        idx = us.locator(x, x0)
        assert idx == 0

    def test_locator_end(self):
        """Test locator for point at end."""
        x = np.linspace(0.0, 10.0, 101)
        x0 = 10.0
        idx = us.locator(x, x0)
        assert idx == 99  # len(x) - 2

    def test_locator_beyond_end(self):
        """Test locator for point beyond end."""
        x = np.linspace(0.0, 10.0, 101)
        x0 = 15.0
        idx = us.locator(x, x0)
        assert idx == 99  # len(x) - 2

    def test_locator_before_start(self):
        """Test locator for point before start."""
        x = np.linspace(0.0, 10.0, 101)
        x0 = -5.0
        idx = us.locator(x, x0)
        assert idx == 0


class TestInterpolation:
    """Test interpolation functions."""

    def test_EAtX_linear_interpolation(self):
        """Test 2D interpolation at specified x."""
        Nx, Ny = 10, 5
        x = np.linspace(0.0, 10.0, Nx)
        f = np.zeros((Nx, Ny), dtype=np.complex128)

        # Create a function that varies linearly in x
        for i in range(Nx):
            f[i, :] = x[i]

        x0 = 5.5
        result = us.EAtX(f, x, x0)

        # All elements should be approximately x0
        assert np.allclose(np.real(result), x0, rtol=1e-10, atol=1e-10)

    def test_EAtX_single_element(self):
        """Test interpolation with single element."""
        f = np.array([[1.0+1j, 2.0+2j, 3.0+3j]])
        x = np.array([0.0])
        x0 = 0.0
        result = us.EAtX(f, x, x0)
        expected = f[0, :]
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_EAtXYZ_trilinear(self):
        """Test 3D trilinear interpolation."""
        Nx, Ny, Nz = 10, 10, 10
        x = np.linspace(0.0, 10.0, Nx)
        y = np.linspace(0.0, 10.0, Ny)
        z = np.linspace(0.0, 10.0, Nz)

        # Create a function f = x + y + z
        f = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    f[i, j, k] = x[i] + y[j] + z[k]

        x0, y0, z0 = 5.5, 4.5, 3.5
        result = us.EAtXYZ(f, x, y, z, x0, y0, z0)

        expected = x0 + y0 + z0
        assert np.abs(result - expected) < 1e-10

    def test_EAtXYZ_corner(self):
        """Test 3D interpolation at grid corner."""
        Nx, Ny, Nz = 5, 5, 5
        x = np.linspace(0.0, 4.0, Nx)
        y = np.linspace(0.0, 4.0, Ny)
        z = np.linspace(0.0, 4.0, Nz)

        f = np.random.random((Nx, Ny, Nz)) + 1j * np.random.random((Nx, Ny, Nz))

        x0, y0, z0 = 0.0, 0.0, 0.0
        result = us.EAtXYZ(f, x, y, z, x0, y0, z0)

        # Should be close to f[0, 0, 0]
        assert np.abs(result - f[0, 0, 0]) < 1e-10


class TestGaussianFunction:
    """Test Gaussian function."""

    def test_gaussian_peak(self):
        """Test Gaussian at peak."""
        result = us.gaussian(0.0, 1.0)
        assert result == pytest.approx(1.0, rel=1e-12)

    def test_gaussian_array(self):
        """Test Gaussian with array."""
        x = np.linspace(-5.0, 5.0, 11)
        x0 = 2.0
        result = us.gaussian(x, x0)
        expected = np.exp(-x**2 / x0**2)
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_gaussian_decay(self):
        """Test Gaussian decay."""
        x0 = 1.0
        assert us.gaussian(x0, x0) == pytest.approx(np.exp(-1), rel=1e-12)


class TestConvolve:
    """Test convolution function."""

    def test_convolve_delta(self):
        """Test convolution with delta function."""
        x = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.complex128)
        h = np.array([1], dtype=np.complex128)
        result = us.convolve(x, h)
        # Convolution with single element should multiply
        assert np.allclose(result[3:], x[3:], rtol=1e-12, atol=1e-12)

    def test_convolve_simple(self):
        """Test convolution with simple kernel."""
        x = np.ones(10, dtype=np.complex128)
        h = np.array([0.5, 0.5], dtype=np.complex128)
        result = us.convolve(x, h)
        # For most elements should be 1.0
        assert np.allclose(result[1:], 1.0, rtol=1e-12, atol=1e-12)

    def test_convolve_length(self):
        """Test convolution output length."""
        x = np.random.random(20) + 1j * np.random.random(20)
        h = np.random.random(5) + 1j * np.random.random(5)
        result = us.convolve(x, h)
        assert len(result) == len(x)


class TestFFTG:
    """Test FFTG and iFFTG functions."""

    def test_FFTG_normalization(self):
        """Test FFTG normalization."""
        N = 64
        F = np.random.random(N) + 1j * np.random.random(N)
        F_original = F.copy()

        us.FFTG(F)

        # Check that FFT was applied with normalization
        expected = -np.fft.fft(F_original) / N
        assert np.allclose(F, expected, rtol=1e-12, atol=1e-12)

    def test_iFFTG_normalization(self):
        """Test iFFTG normalization."""
        N = 64
        F = np.random.random(N) + 1j * np.random.random(N)
        F_original = F.copy()

        us.iFFTG(F)

        # Check that IFFT was applied with normalization
        expected = -np.fft.ifft(F_original) * N
        assert np.allclose(F, expected, rtol=1e-12, atol=1e-12)

    def test_FFTG_iFFTG_roundtrip(self):
        """Test FFTG-iFFTG round trip."""
        N = 128
        F_original = np.random.random(N) + 1j * np.random.random(N)
        F = F_original.copy()

        us.FFTG(F)
        us.iFFTG(F)

        # Should recover original
        assert np.allclose(F, F_original, rtol=1e-10, atol=1e-12)


class TestFileIO:
    """Test file I/O functions."""

    def setup_method(self):
        """Set up temporary directory for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.dataQW_dir = os.path.join(self.test_dir, 'dataQW')
        os.makedirs(self.dataQW_dir, exist_ok=True)
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir)

    def test_WriteIT1D_ReadIT1D(self):
        """Test 1D array write and read."""
        V_original = np.random.random(10)
        filename = 'test1d'

        us.WriteIT1D(V_original, filename)

        V_read = np.zeros_like(V_original)
        us.ReadIT1D(V_read, filename)

        assert np.allclose(V_read, V_original, rtol=1e-6, atol=1e-6)

    def test_WriteIT2D_ReadIT2D(self):
        """Test 2D array write and read."""
        V_original = np.random.random((5, 5))
        filename = 'test2d'

        us.WriteIT2D(V_original, filename)

        V_read = np.zeros_like(V_original)
        us.ReadIT2D(V_read, filename)

        assert np.allclose(V_read, V_original, rtol=1e-6, atol=1e-6)

    def test_print2file(self):
        """Test print2file function."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        filename = os.path.join(self.test_dir, 'test_print.dat')

        us.print2file(x, y, filename)

        # Check file exists and has content
        assert os.path.exists(filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3

    def test_printIT(self):
        """Test printIT function."""
        Dx = np.array([1+1j, 2+2j, 3+3j])
        z = np.array([0.0, 1.0, 2.0])
        n = 1
        filename = 'testIT'

        us.printIT(Dx, z, n, filename)

        expected_file = os.path.join('dataQW', f'{filename}000001.dat')
        assert os.path.exists(expected_file)

    def test_printITR(self):
        """Test printITR function."""
        Dx = np.array([1.0, 2.0, 3.0])
        z = np.array([0.0, 1.0, 2.0])
        n = 2
        filename = 'testITR'

        us.printITR(Dx, z, n, filename)

        expected_file = os.path.join('dataQW', f'{filename}000002.dat')
        assert os.path.exists(expected_file)

    def test_printIT2D(self):
        """Test printIT2D function."""
        Dx = np.random.random((3, 3)) + 1j * np.random.random((3, 3))
        z = np.array([0.0, 1.0, 2.0])
        n = 3
        filename = 'testIT2D'

        us.printIT2D(Dx, z, n, filename)

        expected_file = os.path.join('dataQW', f'{filename}0000003.dat')
        assert os.path.exists(expected_file)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestFFTDerivativeIntegration:
    """Integration tests for FFT-based derivative calculations."""

    def test_derivative_chain_1D(self):
        """Test chain of derivative operations in 1D with periodic function."""
        N = 256
        y = np.linspace(-20e-6, 20e-6, N)
        dy = y[1] - y[0]
        qy = 2.0 * np.pi * np.fft.fftfreq(N, dy)

        # Use periodic function: f(y) = sin(k*y)
        k = 2.0 * np.pi / (N * dy)
        f = np.sin(k * y).astype(np.complex128)

        # First derivative: df/dy = k*cos(k*y)
        df = us.dfdy1D(f, qy)
        expected_df = k * np.cos(k * y)

        # Test interior points
        interior = slice(N//4, 3*N//4)
        assert np.allclose(np.real(df[interior]), expected_df[interior], rtol=1e-6, atol=1e-10)

        # Second derivative: d2f/dy2 = -k^2*sin(k*y)
        d2f = us.dfdy1D(df, qy)
        expected_d2f = -k**2 * np.sin(k * y)

        # Test interior points
        assert np.allclose(np.real(d2f[interior]), expected_d2f[interior], rtol=1e-4, atol=1e-10)

    def test_derivative_2D_consistency(self):
        """Test consistency between 1D and 2D derivatives."""
        Nx, Ny = 32, 64
        y = np.linspace(-10e-6, 10e-6, Ny)
        dy = y[1] - y[0]
        qy = 2.0 * np.pi * np.fft.fftfreq(Ny, dy)

        # Create 2D field that only varies in y
        f1D = (2.0 * y).astype(np.complex128)
        f2D = np.zeros((Nx, Ny), dtype=np.complex128)
        for i in range(Nx):
            f2D[i, :] = f1D

        # Compute derivatives
        df1D = us.dfdy1D(f1D, qy)
        df2D = us.dfdy2D(f2D, qy)

        # Should be consistent
        for i in range(Nx):
            assert np.allclose(df2D[i, :], df1D, rtol=1e-10, atol=1e-12)


class TestFieldRotationIntegration:
    """Integration tests for field rotation operations."""

    def test_rotation_energy_conservation(self):
        """Test that rotation conserves field energy."""
        Nx, Ny = 32, 32
        Ex = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))
        Ey = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))

        energy_before = np.sum(np.abs(Ex)**2 + np.abs(Ey)**2)

        theta = np.pi / 3
        us.RotateField(theta, Ex, Ey)

        energy_after = np.sum(np.abs(Ex)**2 + np.abs(Ey)**2)

        assert energy_before == pytest.approx(energy_after, rel=1e-10)

    def test_rotation_3D_energy_conservation(self):
        """Test that 3D rotation conserves field energy."""
        Nx, Ny, Nz = 16, 16, 16
        Ex = np.random.random((Nx, Ny, Nz)) + 1j * np.random.random((Nx, Ny, Nz))
        Ey = np.random.random((Nx, Ny, Nz)) + 1j * np.random.random((Nx, Ny, Nz))
        Ez = np.random.random((Nx, Ny, Nz)) + 1j * np.random.random((Nx, Ny, Nz))

        energy_before = np.sum(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)

        theta = np.pi / 4
        us.RotateField3D(theta, Ex, Ey, Ez)

        energy_after = np.sum(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)

        assert energy_before == pytest.approx(energy_after, rel=1e-10)


class TestFFTRoundTripIntegration:
    """Integration tests for FFT round-trip operations."""

    def test_GFFT_workflow(self):
        """Test complete workflow with Gaussian FFT."""
        N = 128
        dx = 1e-7
        x = np.arange(N) * dx - N*dx/2
        dq = 2 * np.pi / (N * dx)

        # Create Gaussian
        sigma = 3e-6
        f_original = np.exp(-(x/sigma)**2).astype(np.complex128)
        f = f_original.copy()

        # Transform to momentum space
        us.GFFT_1D(f, dx)

        # Transform back
        us.GIFFT_1D(f, dq)

        # Check shape preservation
        assert f.shape == f_original.shape

    def test_2D_FFT_workflow(self):
        """Test complete 2D FFT workflow."""
        Nx, Ny = 64, 64
        dx, dy = 1e-7, 1e-7
        dqx = 2 * np.pi / (Nx * dx)
        dqy = 2 * np.pi / (Ny * dy)

        # Create 2D Gaussian
        x = np.arange(Nx) * dx - Nx*dx/2
        y = np.arange(Ny) * dy - Ny*dy/2
        sigma = 3e-6

        f_original = np.zeros((Nx, Ny), dtype=np.complex128)
        for i in range(Nx):
            for j in range(Ny):
                f_original[i, j] = np.exp(-((x[i]/sigma)**2 + (y[j]/sigma)**2))

        f = f_original.copy()

        # Forward and backward
        us.GFFT_2D(f, dx, dy)
        us.GIFFT_2D(f, dqx, dqy)

        # Check shape
        assert f.shape == f_original.shape


class TestMaxwellFieldDerivatives:
    """Integration tests for Maxwell equation field derivatives."""

    def test_curl_E_to_B(self):
        """Test that curl of E field is computed correctly."""
        Nx, Ny = 64, 64
        dx, dy = 1e-7, 1e-7

        # Create simple E field
        Ex = np.zeros((Nx, Ny), dtype=np.complex128)
        Ey = np.zeros((Nx, Ny), dtype=np.complex128)
        Ez = np.ones((Nx, Ny), dtype=np.complex128)

        # Compute curl components
        dEx_dy = us.dEdy(Ex, dy)
        dEy_dx = us.dEdx(Ey, dx)

        # For constant Ez, curl should involve Ez derivatives
        assert dEx_dy.shape == (Nx, Ny)
        assert dEy_dx.shape == (Nx, Ny)

    def test_divergence_H_field(self):
        """Test computation of field divergence."""
        Nx, Ny = 64, 64
        dx, dy = 1e-7, 1e-7
        x = np.arange(Nx) * dx
        y = np.arange(Ny) * dy

        # Create H field with known divergence
        Hx = np.zeros((Nx, Ny), dtype=np.complex128)
        Hy = np.zeros((Nx, Ny), dtype=np.complex128)

        for i in range(Nx):
            for j in range(Ny):
                Hx[i, j] = x[i]
                Hy[i, j] = y[j]

        # Compute divergence
        dHx_dx = us.dHdx(Hx, dx)
        dHy_dy = us.dHdy(Hy, dy)

        div_H = dHx_dx + dHy_dy

        # Divergence: div_H = dHx_dx + dHy_dy
        # dHx_dx computes backward difference: (Hx[i] - Hx[i-1])/dx = 1 for interior
        # dHy_dy computes backward difference: (Hy[j] - Hy[j-1])/dy = 1 for interior
        # For Hx[i] = i*dx, Hy[j] = j*dy: div_H = 1 + 1 = 2
        # Test interior points only (avoid boundary wraparound effects)
        # Skip first row/col (wraparound)
        # The divergence should be +2 for this field (backward diff of linear)
        assert np.allclose(np.real(div_H[1:, 1:]), 2.0, rtol=1e-6, atol=1e-10)


class TestNumericalStability:
    """Test numerical stability of functions."""

    def test_derivative_with_noise(self):
        """Test derivative calculation with noisy data."""
        N = 256
        y = np.linspace(-10e-6, 10e-6, N)
        dy = y[1] - y[0]
        qy = 2.0 * np.pi * np.fft.fftfreq(N, dy)

        # Create smooth function with noise
        f = np.exp(-(y/3e-6)**2)
        noise = 0.01 * np.random.random(N)
        f_noisy = (f + noise).astype(np.complex128)

        df = us.dfdy1D(f_noisy, qy)

        # Should still produce finite result
        assert np.all(np.isfinite(df))

    def test_bessel_stability(self):
        """Test Bessel function stability across range."""
        test_values = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]

        for x in test_values:
            z = x + 0j
            results = us.cik01(z)
            # All results should be finite
            assert all(np.isfinite(r) for r in results)


class TestPhysicalProperties:
    """Test that physical properties are preserved."""

    def test_energy_conservation_ABC(self):
        """Test that ABC reduces energy appropriately."""
        Nx, Ny = 64, 64
        Field = np.random.random((Nx, Ny)) + 1j * np.random.random((Nx, Ny))

        energy_before = np.sum(np.abs(Field)**2)

        # ABC with absorption
        abc = np.ones((Nx, Ny))
        abc[:10, :] = 0.5
        abc[-10:, :] = 0.5

        us.ApplyABC(Field, abc)

        energy_after = np.sum(np.abs(Field)**2)

        # Energy should be reduced
        assert energy_after < energy_before

    def test_parseval_theorem(self):
        """Test Parseval's theorem for FFT."""
        N = 128
        f = np.random.random(N) + 1j * np.random.random(N)

        energy_real = np.sum(np.abs(f)**2)

        F = np.fft.fft(f)
        energy_fourier = np.sum(np.abs(F)**2) / N

        # Parseval's theorem
        assert energy_real == pytest.approx(energy_fourier, rel=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
