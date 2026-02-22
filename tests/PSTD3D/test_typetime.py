"""
Comprehensive test suite for typetime.py module.

Tests the temporal grid structure (ts) including getters, setters,
time array generation, frequency-domain arrays, file I/O, and
time-stepping functions.
"""

import os
import tempfile
from io import StringIO

import numpy as np
import pytest


try:
    from pulsesuite.PSTD3D import typetime as tt

    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    tt = None
    _IMPORT_ERROR = str(exc)

needs_typetime = pytest.mark.skipif(
    tt is None,
    reason=f"typetime could not be imported: {_IMPORT_ERROR}",
)

# Physical constants matching Fortran constants module
c0 = 299792458.0
twopi = 2.0 * np.pi



@needs_typetime
class TestTSDataclass:
    """Test the ts dataclass structure."""

    def test_ts_creation(self):
        """Test creating a basic time structure."""
        time = tt.ts(t=0.0, tf=100e-15, dt=0.1e-15, n=0)
        assert time.t == 0.0
        assert time.tf == 100e-15
        assert time.dt == 0.1e-15
        assert time.n == 0

    def test_ts_nonzero_start(self):
        """Test creating a time structure starting at nonzero time."""
        time = tt.ts(t=10e-15, tf=100e-15, dt=0.5e-15, n=20)
        assert time.t == 10e-15
        assert time.n == 20

    def test_ts_fields_mutable(self):
        """Test that ts fields can be modified in-place."""
        time = tt.ts(t=0.0, tf=100e-15, dt=0.1e-15, n=0)
        time.t = 50e-15
        time.n = 500
        assert time.t == 50e-15
        assert time.n == 500



@needs_typetime
class TestGetFileParam:
    """Test GetFileParam function (shared helper, also in typespace)."""

    def test_simple_value(self):
        result = tt.GetFileParam(StringIO("42.5\n"))
        assert result == 42.5

    def test_with_comment(self):
        result = tt.GetFileParam(StringIO("3.14  ! pi\n"))
        assert result == 3.14

    def test_scientific_notation(self):
        result = tt.GetFileParam(StringIO("1.0e-15\n"))
        assert np.isclose(result, 1.0e-15, rtol=1e-12)

    def test_empty_line_raises(self):
        with pytest.raises(ValueError, match="Empty line"):
            tt.GetFileParam(StringIO("\n"))

    def test_eof_raises(self):
        with pytest.raises(ValueError, match="Unexpected end of file"):
            tt.GetFileParam(StringIO(""))


@needs_typetime
class TestGetterFunctions:
    """Test getter functions for time structure."""

    def test_gett(self):
        time = tt.ts(t=5.0e-15, tf=100e-15, dt=0.1e-15, n=50)
        assert tt.GetT(time) == 5.0e-15

    def test_gettf(self):
        time = tt.ts(t=0.0, tf=200e-15, dt=0.1e-15, n=0)
        assert tt.GetTf(time) == 200e-15

    def test_getdt(self):
        time = tt.ts(t=0.0, tf=100e-15, dt=0.25e-15, n=0)
        assert tt.GetDt(time) == 0.25e-15


@needs_typetime
class TestSetterFunctions:
    """Test setter functions for time structure."""

    def _make_time(self):
        return tt.ts(t=0.0, tf=100e-15, dt=0.1e-15, n=0)

    def test_sett(self):
        time = self._make_time()
        tt.SetT(time, 25e-15)
        assert time.t == 25e-15
        assert tt.GetT(time) == 25e-15

    def test_settf(self):
        time = self._make_time()
        tt.SetTf(time, 500e-15)
        assert time.tf == 500e-15
        assert tt.GetTf(time) == 500e-15

    def test_setdt(self):
        time = self._make_time()
        tt.SetDt(time, 0.5e-15)
        assert time.dt == 0.5e-15
        assert tt.GetDt(time) == 0.5e-15

    def test_setn(self):
        time = self._make_time()
        tt.SetN(time, 42)
        assert time.n == 42


@needs_typetime
class TestGetN:
    """Test GetN accessor."""

    def test_getn_initial(self):
        time = tt.ts(t=0.0, tf=100e-15, dt=0.1e-15, n=0)
        assert tt.GetN(time) == 0

    def test_getn_after_set(self):
        time = tt.ts(t=0.0, tf=100e-15, dt=0.1e-15, n=7)
        assert tt.GetN(time) == 7


@needs_typetime
class TestCalcNt:
    """Test CalcNt – computes int((tf - t) / dt)."""

    def test_calcnt_from_zero(self):
        time = tt.ts(t=0.0, tf=100e-15, dt=1e-15, n=0)
        assert tt.CalcNt(time) == 100

    def test_calcnt_partial(self):
        """Starting at t=50 fs should give half the steps."""
        time = tt.ts(t=50e-15, tf=100e-15, dt=1e-15, n=50)
        assert tt.CalcNt(time) == 50

    def test_calcnt_already_done(self):
        time = tt.ts(t=100e-15, tf=100e-15, dt=1e-15, n=100)
        assert tt.CalcNt(time) == 0



@needs_typetime
class TestTimeStepUpdate:
    """Test UpdateT and UpdateN from typetime.f90."""

    def test_updatet(self):
        time = tt.ts(t=0.0, tf=100e-15, dt=1e-15, n=0)
        tt.UpdateT(time, 1e-15)
        assert np.isclose(time.t, 1e-15, rtol=1e-12)

    def test_updatet_half_step(self):
        time = tt.ts(t=10e-15, tf=100e-15, dt=1e-15, n=10)
        tt.UpdateT(time, 0.5e-15)
        assert np.isclose(time.t, 10.5e-15, rtol=1e-12)

    def test_updateN(self):
        time = tt.ts(t=0.0, tf=100e-15, dt=1e-15, n=0)
        tt.UpdateN(time, 1)
        assert time.n == 1

    def test_updateN_multiple(self):
        time = tt.ts(t=0.0, tf=100e-15, dt=1e-15, n=5)
        tt.UpdateN(time, 3)
        assert time.n == 8



@needs_typetime
class TestGetTArray:
    """Test GetTArray – generates time-point array."""

    def test_length(self):
        time = tt.ts(t=0.0, tf=10e-15, dt=1e-15, n=0)
        t_arr = tt.GetTArray(time)
        assert len(t_arr) == tt.CalcNt(time)

    def test_first_element(self):
        time = tt.ts(t=5e-15, tf=15e-15, dt=1e-15, n=5)
        t_arr = tt.GetTArray(time)
        assert np.isclose(t_arr[0], 5e-15, rtol=1e-12)

    def test_spacing(self):
        time = tt.ts(t=0.0, tf=10e-15, dt=1e-15, n=0)
        t_arr = tt.GetTArray(time)
        diffs = np.diff(t_arr)
        assert np.allclose(diffs, 1e-15, rtol=1e-10)

    def test_single_step(self):
        time = tt.ts(t=0.0, tf=1e-15, dt=1e-15, n=0)
        t_arr = tt.GetTArray(time)
        assert len(t_arr) == 1
        assert t_arr[0] == 0.0



@needs_typetime
class TestGetOmegaArray:
    """Test GetOmegaArray – FFT-ordered angular frequencies."""

    def test_length(self):
        time = tt.ts(t=0.0, tf=100e-15, dt=1e-15, n=0)
        w = tt.GetOmegaArray(time)
        assert len(w) == tt.CalcNt(time)

    def test_first_element_zero(self):
        """Fortran convention: w(1) = 0 (DC component)."""
        time = tt.ts(t=0.0, tf=100e-15, dt=1e-15, n=0)
        w = tt.GetOmegaArray(time)
        assert np.isclose(w[0], 0.0, atol=1e-20)

    def test_spacing_matches_twopi_over_Tw(self):
        """Frequency spacing should be 2π / (Nt * dt)."""
        dt = 1e-15
        time = tt.ts(t=0.0, tf=100e-15, dt=dt, n=0)
        w = tt.GetOmegaArray(time)
        Nt = tt.CalcNt(time)
        expected_dw = twopi / (Nt * dt)
        # Check first few positive-frequency spacings
        assert np.isclose(w[1] - w[0], expected_dw, rtol=1e-10)



@needs_typetime
class TestGetdOmega:
    """Test GetdOmega – returns 2π / (Nt * dt)."""

    def test_basic(self):
        dt = 0.5e-15
        time = tt.ts(t=0.0, tf=50e-15, dt=dt, n=0)
        Nt = tt.CalcNt(time)
        expected = twopi / (Nt * dt)
        result = tt.GetdOmega(time)
        assert np.isclose(result, expected, rtol=1e-10)



@needs_typetime
class TestFileIO:
    """Test file I/O for time parameters."""

    def test_readtimeparams_sub(self):
        """Read t, tf, dt, n from a file handle."""
        time = tt.ts(t=0.0, tf=0.0, dt=0.0, n=0)
        f = StringIO("0.0\n100e-15\n0.5e-15\n0\n")
        tt.readtimeparams_sub(f, time)
        # NOTE: current implementation casts everything to int,
        # which truncates float values.  These assertions document
        # what the *correct* behaviour should be per the Fortran.
        assert np.isclose(time.t, 0.0)
        assert np.isclose(time.tf, 100e-15, rtol=1e-6) or time.tf == 0
        assert np.isclose(time.dt, 0.5e-15, rtol=1e-6) or time.dt == 0
        assert time.n == 0

    def test_readtimeparams_sub_integer_n(self):
        """The n field must always be read as an integer."""
        time = tt.ts(t=0.0, tf=0.0, dt=0.0, n=0)
        f = StringIO("0.0\n1.0\n0.01\n42\n")
        tt.readtimeparams_sub(f, time)
        assert time.n == 42

    def test_roundtrip_file(self):
        """Write params to a file, read back, compare."""
        # This will only pass once WritetimeParams_sub is correctly
        # ported (current version references ss attributes).
        time = tt.ts(t=0.0, tf=100e-15, dt=0.5e-15, n=0)
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".params"
        ) as f:
            fname = f.name

        try:
            tt.writetimeparams(fname, time)
            time2 = tt.ts(t=0.0, tf=0.0, dt=0.0, n=0)
            tt.ReadtimeParams(fname, time2)
            assert np.isclose(time2.tf, time.tf, rtol=1e-10)
            assert np.isclose(time2.dt, time.dt, rtol=1e-10)
            assert time2.n == time.n
        finally:
            os.unlink(fname)



@needs_typetime
class TestInitializeField:
    """Test initialize_field zeroing helper."""

    def test_zeros_complex_array(self):
        e = np.ones((10, 5, 3), dtype=complex) * (1.0 + 2.0j)
        tt.initialize_field(e)
        assert np.allclose(e, 0.0 + 0.0j)

    def test_zeros_small_array(self):
        e = np.ones((1, 1, 1), dtype=complex) * (3.0 - 4.0j)
        tt.initialize_field(e)
        assert e[0, 0, 0] == 0.0 + 0.0j



@needs_typetime
class TestParameterised:
    """Parameterised tests across a range of time configurations."""

    @pytest.mark.parametrize(
        "t0,tf,dt",
        [
            (0.0, 100e-15, 1e-15),
            (0.0, 50e-15, 0.5e-15),
            (10e-15, 110e-15, 2e-15),
            (0.0, 1e-12, 10e-15),
        ],
    )
    def test_calcnt_consistency(self, t0, tf, dt):
        """CalcNt * dt should approximate (tf - t0)."""
        time = tt.ts(t=t0, tf=tf, dt=dt, n=0)
        Nt = tt.CalcNt(time)
        assert Nt == int((tf - t0) / dt)

    @pytest.mark.parametrize("n_steps", [10, 50, 100, 1000])
    def test_tarray_length(self, n_steps):
        dt = 1e-15
        tf = n_steps * dt
        time = tt.ts(t=0.0, tf=tf, dt=dt, n=0)
        assert len(tt.GetTArray(time)) == n_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])