"""
Comprehensive test suite for typespace.py module.

Tests all spatial grid structure functions including getters, setters,
array generation, I/O operations, and field manipulation functions.
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from io import StringIO, BytesIO

from pulsesuite.PSTD3D import typespace as ts


class TestSSDataclass:
    """Test the ss dataclass structure."""

    def test_ss_creation_3d(self):
        """Test creating 3D space structure."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        assert space.Dims == 3
        assert space.Nx == 100
        assert space.Ny == 50
        assert space.Nz == 25
        assert space.dx == 1e-9
        assert space.dy == 1e-9
        assert space.dz == 1e-9
        assert space.epsr == 3.0

    def test_ss_creation_1d(self):
        """Test creating 1D space structure."""
        space = ts.ss(Dims=1, Nx=100, Ny=1, Nz=1, dx=1e-9, dy=1.0, dz=1.0, epsr=3.0)
        assert space.Dims == 1
        assert space.Nx == 100
        assert space.Ny == 1
        assert space.Nz == 1

    def test_ss_creation_2d(self):
        """Test creating 2D space structure."""
        space = ts.ss(Dims=2, Nx=100, Ny=50, Nz=1, dx=1e-9, dy=1e-9, dz=1.0, epsr=3.0)
        assert space.Dims == 2
        assert space.Nx == 100
        assert space.Ny == 50
        assert space.Nz == 1


class TestGetFileParam:
    """Test GetFileParam function."""

    def test_getfileparam_simple(self):
        """Test reading simple parameter."""
        f = StringIO("42.5\n")
        result = ts.GetFileParam(f)
        assert result == 42.5

    def test_getfileparam_with_comment(self):
        """Test reading parameter with comment."""
        f = StringIO("3.14159  # pi\n")
        result = ts.GetFileParam(f)
        assert result == 3.14159

    def test_getfileparam_integer(self):
        """Test reading integer parameter."""
        f = StringIO("100\n")
        result = ts.GetFileParam(f)
        assert result == 100.0

    def test_getfileparam_scientific_notation(self):
        """Test reading scientific notation."""
        f = StringIO("1.23e-6\n")
        result = ts.GetFileParam(f)
        assert np.isclose(result, 1.23e-6, rtol=1e-12, atol=1e-12)

    def test_getfileparam_empty_line(self):
        """Test error on empty line."""
        f = StringIO("\n")
        with pytest.raises(ValueError, match="Empty line"):
            ts.GetFileParam(f)

    def test_getfileparam_eof(self):
        """Test error on end of file."""
        f = StringIO("")
        with pytest.raises(ValueError, match="Unexpected end of file"):
            ts.GetFileParam(f)


class TestGetterFunctions:
    """Test getter functions for space structure."""

    def test_getnx_getny_getnz(self):
        """Test getting grid dimensions."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        assert ts.GetNx(space) == 100
        assert ts.GetNy(space) == 50
        assert ts.GetNz(space) == 25

    def test_getdx_normal(self):
        """Test GetDx with normal case."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        assert ts.GetDx(space) == 1e-9

    def test_getdx_collapsed(self):
        """Test GetDx with collapsed dimension."""
        space = ts.ss(Dims=1, Nx=1, Ny=1, Nz=1, dx=1e-9, dy=1.0, dz=1.0, epsr=3.0)
        assert ts.GetDx(space) == 1.0

    def test_getdy_normal(self):
        """Test GetDy with normal case."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=2e-9, dz=1e-9, epsr=3.0)
        assert ts.GetDy(space) == 2e-9

    def test_getdy_collapsed(self):
        """Test GetDy with collapsed dimension."""
        space = ts.ss(Dims=1, Nx=100, Ny=1, Nz=1, dx=1e-9, dy=1.0, dz=1.0, epsr=3.0)
        assert ts.GetDy(space) == 1.0

    def test_getdz_normal(self):
        """Test GetDz with normal case."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=3e-9, epsr=3.0)
        assert ts.GetDz(space) == 3e-9

    def test_getdz_collapsed(self):
        """Test GetDz with collapsed dimension."""
        space = ts.ss(Dims=2, Nx=100, Ny=50, Nz=1, dx=1e-9, dy=1e-9, dz=1.0, epsr=3.0)
        assert ts.GetDz(space) == 1.0

    def test_getepsr_normal(self):
        """Test GetEpsr with normal case."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        assert ts.GetEpsr(space) == 3.0

    def test_getepsr_collapsed(self):
        """Test GetEpsr with collapsed z dimension."""
        space = ts.ss(Dims=2, Nx=100, Ny=50, Nz=1, dx=1e-9, dy=1e-9, dz=1.0, epsr=3.0)
        assert ts.GetEpsr(space) == 1.0


class TestSetterFunctions:
    """Test setter functions for space structure."""

    def test_setnx(self):
        """Test SetNx."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        ts.SetNx(space, 200)
        assert space.Nx == 200
        assert ts.GetNx(space) == 200

    def test_setny(self):
        """Test SetNy."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        ts.SetNy(space, 75)
        assert space.Ny == 75
        assert ts.GetNy(space) == 75

    def test_setnz(self):
        """Test SetNz."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        ts.SetNz(space, 30)
        assert space.Nz == 30
        assert ts.GetNz(space) == 30

    def test_setdx(self):
        """Test SetDx."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        ts.SetDx(space, 2e-9)
        assert space.dx == 2e-9
        assert ts.GetDx(space) == 2e-9

    def test_setdy(self):
        """Test SetDy."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        ts.SetDy(space, 3e-9)
        assert space.dy == 3e-9
        assert ts.GetDy(space) == 3e-9

    def test_setdz(self):
        """Test SetDz."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        ts.SetDz(space, 4e-9)
        assert space.dz == 4e-9
        assert ts.GetDz(space) == 4e-9


class TestWidthFunctions:
    """Test width calculation functions."""

    def test_getxwidth(self):
        """Test GetXWidth."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        expected = 1e-9 * (100 - 1)
        result = ts.GetXWidth(space)
        assert np.isclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_getywidth(self):
        """Test GetYWidth."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=2e-9, dz=1e-9, epsr=3.0)
        expected = 2e-9 * (50 - 1)
        result = ts.GetYWidth(space)
        assert np.isclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_getzwidth(self):
        """Test GetZWidth."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=3e-9, epsr=3.0)
        expected = 3e-9 * (25 - 1)
        result = ts.GetZWidth(space)
        assert np.isclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_getxwidth_single_point(self):
        """Test GetXWidth with single point."""
        space = ts.ss(Dims=1, Nx=1, Ny=1, Nz=1, dx=1e-9, dy=1.0, dz=1.0, epsr=3.0)
        result = ts.GetXWidth(space)
        assert result == 0.0


class TestArrayGeneration:
    """Test array generation functions."""

    def test_getspacearray_simple(self):
        """Test GetSpaceArray with simple case."""
        result = ts.GetSpaceArray(5, 1.0)
        expected = np.linspace(-0.5, 0.5, 5)
        assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_getspacearray_single_point(self):
        """Test GetSpaceArray with single point."""
        result = ts.GetSpaceArray(1, 1.0)
        assert len(result) == 1
        assert result[0] == 0.0

    def test_getspacearray_even_length(self):
        """Test GetSpaceArray with even length."""
        N = 100
        L = 1e-6
        result = ts.GetSpaceArray(N, L)
        assert len(result) == N
        assert np.isclose(result[0], -L/2.0, rtol=1e-12, atol=1e-12)
        assert np.isclose(result[-1], L/2.0, rtol=1e-12, atol=1e-12)

    def test_getspacearray_odd_length(self):
        """Test GetSpaceArray with odd length."""
        N = 101
        L = 1e-6
        result = ts.GetSpaceArray(N, L)
        assert len(result) == N
        assert np.isclose(result[N//2], 0.0, rtol=1e-12, atol=1e-12)

    def test_getkarray_simple(self):
        """Test GetKArray with simple case."""
        Nk = 5
        L = 1.0
        result = ts.GetKArray(Nk, L)
        assert len(result) == Nk
        # Check that it's centered at zero
        assert np.isclose(result[Nk//2], 0.0, rtol=1e-12, atol=1e-12)

    def test_getkarray_single_point(self):
        """Test GetKArray with single point."""
        result = ts.GetKArray(1, 1.0)
        assert len(result) == 1
        assert result[0] == 0.0

    def test_getkarray_spacing(self):
        """Test GetKArray spacing."""
        Nk = 100
        L = 1e-6
        result = ts.GetKArray(Nk, L)
        dk = 2.0 * np.pi / L
        # Check spacing (should be approximately dk)
        spacing = result[1] - result[0]
        assert np.isclose(spacing, dk, rtol=1e-10, atol=1e-12)

    def test_getxarray(self):
        """Test GetXArray."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        result = ts.GetXArray(space)
        assert len(result) == 100
        width = ts.GetXWidth(space)
        assert np.isclose(result[0], -width/2.0, rtol=1e-10, atol=1e-12)
        assert np.isclose(result[-1], width/2.0, rtol=1e-10, atol=1e-12)

    def test_getxarray_single_point(self):
        """Test GetXArray with single point."""
        space = ts.ss(Dims=1, Nx=1, Ny=1, Nz=1, dx=1e-9, dy=1.0, dz=1.0, epsr=3.0)
        result = ts.GetXArray(space)
        assert len(result) == 1
        assert result[0] == 0.0

    def test_getyarray(self):
        """Test GetYArray."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        result = ts.GetYArray(space)
        assert len(result) == 50

    def test_getzarray(self):
        """Test GetZArray."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        result = ts.GetZArray(space)
        assert len(result) == 25

    def test_getkxarray(self):
        """Test GetKxArray."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        result = ts.GetKxArray(space)
        assert len(result) == 100
        # Check centered at zero
        assert np.isclose(result[50], 0.0, rtol=1e-10, atol=1e-12)

    def test_getkyarray(self):
        """Test GetKyArray."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        result = ts.GetKyArray(space)
        assert len(result) == 50

    def test_getkzarray(self):
        """Test GetKzArray."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        result = ts.GetKzArray(space)
        assert len(result) == 25


class TestDifferentialFunctions:
    """Test differential functions for conjugate coordinate system."""

    def test_getdqx(self):
        """Test GetDQx."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        width = ts.GetXWidth(space)
        expected = 2.0 * np.pi / width
        result = ts.GetDQx(space)
        assert np.isclose(result, expected, rtol=1e-10, atol=1e-12)

    def test_getdqy(self):
        """Test GetDQy."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        width = ts.GetYWidth(space)
        expected = 2.0 * np.pi / width
        result = ts.GetDQy(space)
        assert np.isclose(result, expected, rtol=1e-10, atol=1e-12)

    def test_getdqz(self):
        """Test GetDQz."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        width = ts.GetZWidth(space)
        expected = 2.0 * np.pi / width
        result = ts.GetDQz(space)
        assert np.isclose(result, expected, rtol=1e-10, atol=1e-12)

    def test_getdqx_zero_width(self):
        """Test GetDQx with zero width."""
        space = ts.ss(Dims=1, Nx=1, Ny=1, Nz=1, dx=1e-9, dy=1.0, dz=1.0, epsr=3.0)
        result = ts.GetDQx(space)
        assert result == 0.0


class TestVolumeFunctions:
    """Test volume element functions."""

    def test_getdvol(self):
        """Test GetDVol."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=2e-9, dz=3e-9, epsr=3.0)
        expected = ts.GetDx(space) * ts.GetDy(space) * ts.GetDz(space)
        result = ts.GetDVol(space)
        assert np.isclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_getdqvol(self):
        """Test GetDQVol."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        expected = ts.GetDQx(space) * ts.GetDQy(space) * ts.GetDQz(space)
        result = ts.GetDQVol(space)
        assert np.isclose(result, expected, rtol=1e-10, atol=1e-12)


class TestFileIOParams:
    """Test file I/O functions for space parameters."""

    def test_readspaceparams_sub(self):
        """Test readspaceparams_sub."""
        space = ts.ss(Dims=0, Nx=0, Ny=0, Nz=0, dx=0.0, dy=0.0, dz=0.0, epsr=0.0)
        f = StringIO("3\n100\n50\n25\n1e-9\n2e-9\n3e-9\n3.0\n")
        ts.readspaceparams_sub(f, space)
        assert space.Dims == 3
        assert space.Nx == 100
        assert space.Ny == 50
        assert space.Nz == 25
        assert space.dx == 1e-9
        assert space.dy == 2e-9
        assert space.dz == 3e-9
        assert space.epsr == 3.0

    def test_writespaceparams_sub(self):
        """Test WriteSpaceParams_sub."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=2e-9, dz=3e-9, epsr=3.0)
        f = StringIO()
        ts.WriteSpaceParams_sub(f, space)
        output = f.getvalue()
        assert "3" in output
        assert "100" in output
        assert "50" in output
        assert "25" in output

    def test_readspaceparams_file(self):
        """Test ReadSpaceParams from file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.params') as f:
            f.write("3\n100\n50\n25\n1e-9\n2e-9\n3e-9\n3.0\n")
            fname = f.name

        try:
            space = ts.ss(Dims=0, Nx=0, Ny=0, Nz=0, dx=0.0, dy=0.0, dz=0.0, epsr=0.0)
            ts.ReadSpaceParams(fname, space)
            assert space.Dims == 3
            assert space.Nx == 100
        finally:
            os.unlink(fname)

    def test_writespaceparams_file(self):
        """Test writespaceparams to file."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=2e-9, dz=3e-9, epsr=3.0)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.params') as f:
            fname = f.name

        try:
            ts.writespaceparams(fname, space)
            # Read it back
            space2 = ts.ss(Dims=0, Nx=0, Ny=0, Nz=0, dx=0.0, dy=0.0, dz=0.0, epsr=0.0)
            ts.ReadSpaceParams(fname, space2)
            assert space2.Dims == space.Dims
            assert space2.Nx == space.Nx
            assert space2.Ny == space.Ny
            assert space2.Nz == space.Nz
        finally:
            os.unlink(fname)


class TestFieldInitialization:
    """Test field initialization functions."""

    def test_initialize_field(self):
        """Test initialize_field."""
        e = np.ones((10, 5, 3), dtype=complex) * (1.0 + 2.0j)
        ts.initialize_field(e)
        assert np.allclose(e, 0.0 + 0.0j, rtol=1e-12, atol=1e-12)

    def test_initialize_field_large(self):
        """Test initialize_field with large array."""
        e = np.ones((100, 50, 25), dtype=complex) * (1.0 + 2.0j)
        ts.initialize_field(e)
        assert np.allclose(e, 0.0 + 0.0j, rtol=1e-12, atol=1e-12)

    def test_initialize_field_small(self):
        """Test initialize_field with small array."""
        e = np.ones((1, 1, 1), dtype=complex) * (1.0 + 2.0j)
        ts.initialize_field(e)
        assert np.allclose(e, 0.0 + 0.0j, rtol=1e-12, atol=1e-12)


class TestBinaryIO:
    """Test binary I/O functions."""

    def test_unformatted_write_space(self):
        """Test unformatted_write_space."""
        space = ts.ss(Dims=3, Nx=100, Ny=50, Nz=25, dx=1e-9, dy=2e-9, dz=3e-9, epsr=3.0)
        f = BytesIO()
        ts.unformatted_write_space(f, space)
        f.seek(0)
        # Read it back
        space2 = ts.ss(Dims=0, Nx=0, Ny=0, Nz=0, dx=0.0, dy=0.0, dz=0.0, epsr=0.0)
        ts.unformatted_read_space(f, space2)
        assert space2.Dims == space.Dims
        assert space2.Nx == space.Nx
        assert space2.Ny == space.Ny
        assert space2.Nz == space.Nz
        assert np.isclose(space2.dx, space.dx, rtol=1e-12, atol=1e-12)
        assert np.isclose(space2.dy, space.dy, rtol=1e-12, atol=1e-12)
        assert np.isclose(space2.dz, space.dz, rtol=1e-12, atol=1e-12)
        assert np.isclose(space2.epsr, space.epsr, rtol=1e-12, atol=1e-12)

    def test_unformatted_write_e(self):
        """Test unformatted_write_e."""
        e = np.random.rand(10, 5, 3) + 1j * np.random.rand(10, 5, 3)
        e = e.astype(complex)
        f = BytesIO()
        ts.unformatted_write_e(f, e)
        f.seek(0)
        e2 = np.zeros_like(e)
        ts.unformatted_read_e(f, e2)
        assert np.allclose(e, e2, rtol=1e-12, atol=1e-12)

    def test_unformatted_read_e(self):
        """Test unformatted_read_e."""
        e = np.random.rand(10, 5, 3) + 1j * np.random.rand(10, 5, 3)
        e = e.astype(complex)
        f = BytesIO()
        ts.unformatted_write_e(f, e)
        f.seek(0)
        e2 = np.zeros_like(e)
        ts.unformatted_read_e(f, e2)
        assert np.allclose(e, e2, rtol=1e-12, atol=1e-12)


class TestTextIO:
    """Test text I/O functions."""

    def test_readfield_from_unit_text(self):
        """Test readfield_from_unit in text mode."""
        e = np.zeros((3, 2, 1), dtype=complex)
        # Create text data
        text_data = ""
        for k in range(1):
            for j in range(2):
                for i in range(3):
                    re = float(i + j + k)
                    im = float(i + j + k + 1)
                    text_data += f"{re} {im}\n"
        f = StringIO(text_data)
        ts.readfield_from_unit(f, e, binmode=False)
        # Check first element
        assert np.isclose(e[0, 0, 0], 0.0 + 1.0j, rtol=1e-12, atol=1e-12)
        # Check last element
        assert np.isclose(e[2, 1, 0], 3.0 + 4.0j, rtol=1e-12, atol=1e-12)

    def test_writefield_to_unit_text(self):
        """Test writefield_to_unit in text mode."""
        e = np.zeros((3, 2, 1), dtype=complex)
        e[0, 0, 0] = 1.0 + 2.0j
        e[1, 1, 0] = 3.0 + 4.0j
        f = StringIO()
        ts.writefield_to_unit(f, e, binmode=False)
        output = f.getvalue()
        lines = output.strip().split('\n')
        # Array has 3*2*1 = 6 elements
        assert len(lines) == 6
        # Check first line (element [0,0,0])
        parts = lines[0].split()
        assert len(parts) == 2
        assert np.isclose(float(parts[0]), 1.0, rtol=1e-10, atol=1e-12)
        assert np.isclose(float(parts[1]), 2.0, rtol=1e-10, atol=1e-12)


class TestIntegratedIO:
    """Test integrated I/O functions."""

    def test_writefield_binary(self):
        """Test writefield in binary mode."""
        space = ts.ss(Dims=3, Nx=10, Ny=5, Nz=3, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        e = np.random.rand(10, 5, 3) + 1j * np.random.rand(10, 5, 3)
        e = e.astype(complex)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            fname = f.name

        try:
            ts.writefield(fname, e, space, binmode=True, single=True)
            # Read it back
            space2 = ts.ss(Dims=0, Nx=0, Ny=0, Nz=0, dx=0.0, dy=0.0, dz=0.0, epsr=0.0)
            e2 = None
            e2 = ts.readfield(fname, e2, space2, binmode=True, single=True)
            assert space2.Dims == space.Dims
            assert space2.Nx == space.Nx
            assert np.allclose(e, e2, rtol=1e-12, atol=1e-12)
        finally:
            os.unlink(fname)

    def test_writefield_text(self):
        """Test writefield in text mode."""
        space = ts.ss(Dims=3, Nx=5, Ny=3, Nz=2, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        e = np.random.rand(5, 3, 2) + 1j * np.random.rand(5, 3, 2)
        e = e.astype(complex)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            fname = f.name

        try:
            ts.writefield(fname, e, space, binmode=False, single=True)
            # Read it back
            space2 = ts.ss(Dims=0, Nx=0, Ny=0, Nz=0, dx=0.0, dy=0.0, dz=0.0, epsr=0.0)
            e2 = None
            e2 = ts.readfield(fname, e2, space2, binmode=False, single=True)
            assert space2.Dims == space.Dims
            assert space2.Nx == space.Nx
            assert np.allclose(e, e2, rtol=1e-10, atol=1e-12)
        finally:
            os.unlink(fname)

    def test_readspace_only_binary(self):
        """Test readspace_only in binary mode."""
        space = ts.ss(Dims=3, Nx=10, Ny=5, Nz=3, dx=1e-9, dy=2e-9, dz=3e-9, epsr=3.0)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            fname = f.name

        try:
            # Write space
            f_bin = open(fname, 'wb')
            ts.unformatted_write_space(f_bin, space)
            f_bin.close()
            # Read it back
            space2 = ts.ss(Dims=0, Nx=0, Ny=0, Nz=0, dx=0.0, dy=0.0, dz=0.0, epsr=0.0)
            ts.readspace_only(fname, space2, binmode=True, single=True)
            assert space2.Dims == space.Dims
            assert space2.Nx == space.Nx
            assert np.isclose(space2.dx, space.dx, rtol=1e-12, atol=1e-12)
        finally:
            os.unlink(fname)

    def test_readfield_with_existing_array(self):
        """Test readfield with existing array."""
        space = ts.ss(Dims=3, Nx=5, Ny=3, Nz=2, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        e = np.random.rand(5, 3, 2) + 1j * np.random.rand(5, 3, 2)
        e = e.astype(complex)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            fname = f.name

        try:
            # Write field
            ts.writefield(fname, e, space, binmode=True, single=True)
            # Read with existing array
            space2 = ts.ss(Dims=0, Nx=0, Ny=0, Nz=0, dx=0.0, dy=0.0, dz=0.0, epsr=0.0)
            e2 = np.zeros((5, 3, 2), dtype=complex)
            e2 = ts.readfield(fname, e2, space2, binmode=True, single=True)
            assert np.allclose(e, e2, rtol=1e-12, atol=1e-12)
        finally:
            os.unlink(fname)

    def test_readfield_with_wrong_shape(self):
        """Test readfield with array of wrong shape."""
        space = ts.ss(Dims=3, Nx=5, Ny=3, Nz=2, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        e = np.random.rand(5, 3, 2) + 1j * np.random.rand(5, 3, 2)
        e = e.astype(complex)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            fname = f.name

        try:
            # Write field
            ts.writefield(fname, e, space, binmode=True, single=True)
            # Read with wrong shape array
            space2 = ts.ss(Dims=0, Nx=0, Ny=0, Nz=0, dx=0.0, dy=0.0, dz=0.0, epsr=0.0)
            e2 = np.zeros((10, 10, 10), dtype=complex)  # Wrong shape
            e2 = ts.readfield(fname, e2, space2, binmode=True, single=True)
            assert e2.shape == (5, 3, 2)
            assert np.allclose(e, e2, rtol=1e-12, atol=1e-12)
        finally:
            os.unlink(fname)


class TestEdgeCases:
    """Test edge cases and special values."""

    def test_getspacearray_zero_length(self):
        """Test GetSpaceArray with zero length."""
        result = ts.GetSpaceArray(1, 0.0)
        assert len(result) == 1
        assert result[0] == 0.0

    def test_getkarray_zero_length(self):
        """Test GetKArray with zero length."""
        result = ts.GetKArray(1, 0.0)
        assert len(result) == 1
        assert result[0] == 0.0

    def test_getdqx_zero_width(self):
        """Test GetDQx with zero width."""
        space = ts.ss(Dims=1, Nx=1, Ny=1, Nz=1, dx=1e-9, dy=1.0, dz=1.0, epsr=3.0)
        result = ts.GetDQx(space)
        assert result == 0.0

    def test_getdvol_collapsed_dimensions(self):
        """Test GetDVol with collapsed dimensions."""
        space = ts.ss(Dims=1, Nx=100, Ny=1, Nz=1, dx=1e-9, dy=1.0, dz=1.0, epsr=3.0)
        result = ts.GetDVol(space)
        expected = ts.GetDx(space) * ts.GetDy(space) * ts.GetDz(space)
        assert np.isclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_array_generation_1d(self):
        """Test array generation for 1D case."""
        space = ts.ss(Dims=1, Nx=100, Ny=1, Nz=1, dx=1e-9, dy=1.0, dz=1.0, epsr=3.0)
        x = ts.GetXArray(space)
        assert len(x) == 100
        y = ts.GetYArray(space)
        assert len(y) == 1
        assert y[0] == 0.0
        z = ts.GetZArray(space)
        assert len(z) == 1
        assert z[0] == 0.0

    def test_array_generation_2d(self):
        """Test array generation for 2D case."""
        space = ts.ss(Dims=2, Nx=100, Ny=50, Nz=1, dx=1e-9, dy=1e-9, dz=1.0, epsr=3.0)
        x = ts.GetXArray(space)
        assert len(x) == 100
        y = ts.GetYArray(space)
        assert len(y) == 50
        z = ts.GetZArray(space)
        assert len(z) == 1
        assert z[0] == 0.0


class TestParameterized:
    """Parameterized tests for various dimensions."""

    @pytest.mark.parametrize("dims,nx,ny,nz", [
        (1, 100, 1, 1),
        (2, 100, 50, 1),
        (3, 100, 50, 25),
        (1, 1, 1, 1),
        (2, 10, 10, 1),
        (3, 10, 10, 10),
    ])
    def test_getters_all_dimensions(self, dims, nx, ny, nz):
        """Test getters for all dimension types."""
        space = ts.ss(Dims=dims, Nx=nx, Ny=ny, Nz=nz, dx=1e-9, dy=1e-9, dz=1e-9, epsr=3.0)
        assert ts.GetNx(space) == nx
        assert ts.GetNy(space) == ny
        assert ts.GetNz(space) == nz

    @pytest.mark.parametrize("nx,dx", [
        (1, 1e-9),
        (2, 1e-9),
        (100, 1e-9),
        (101, 1e-9),
        (1000, 1e-9),
    ])
    def test_getxwidth_various_sizes(self, nx, dx):
        """Test GetXWidth with various sizes."""
        space = ts.ss(Dims=3, Nx=nx, Ny=50, Nz=25, dx=dx, dy=1e-9, dz=1e-9, epsr=3.0)
        expected = dx * (nx - 1)
        result = ts.GetXWidth(space)
        assert np.isclose(result, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("n,l", [
        (1, 1.0),
        (2, 1.0),
        (10, 1.0),
        (100, 1e-6),
        (101, 1e-6),
    ])
    def test_getspacearray_various_sizes(self, n, l):
        """Test GetSpaceArray with various sizes."""
        result = ts.GetSpaceArray(n, l)
        assert len(result) == n
        if n > 1:
            assert np.isclose(result[0], -l/2.0, rtol=1e-12, atol=1e-12)
            assert np.isclose(result[-1], l/2.0, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

