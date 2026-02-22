"""
Comprehensive test suite for tfsf.py module (Total-Field / Scattered-Field).

Tests TFSF source array initialisation, field injection via UpdateTFSC,
and the expected properties of the source window (antisymmetry,
normalisation, spatial localisation).

Reference: PSTD3D.f90 subroutines InitializeTFSF, UpdateTFSC
"""

import numpy as np
import pytest



try:
    from pulsesuite.PSTD3D import tfsf

    _has_init = hasattr(tfsf, "InitializeTFSF")
    _IMPORT_ERROR = None if _has_init else "module loaded but 'InitializeTFSF' not found"
except Exception as exc:  # noqa: BLE001
    tfsf = None
    _has_init = False
    _IMPORT_ERROR = str(exc)

# Also try importing helper modules that TFSF depends on
try:
    from pulsesuite.PSTD3D import typespace as ts_mod
    _has_typespace = True
except Exception:
    _has_typespace = False

needs_tfsf = pytest.mark.skipif(
    not _has_init,
    reason=f"tfsf not available: {_IMPORT_ERROR or 'InitializeTFSF missing'}",
)

# Physical constants
c0 = 299792458.0
pi = np.pi
twopi = 2.0 * pi


def _make_space(Nx=256, Ny=8, Nz=8, dx=1e-9):
    """Create a typespace.ss object (or a simple namespace fallback)."""
    if _has_typespace:
        return ts_mod.ss(
            Dims=3, Nx=Nx, Ny=Ny, Nz=Nz,
            dx=dx, dy=dx, dz=dx, epsr=1.0,
        )
    # Fallback SimpleNamespace
    from types import SimpleNamespace
    return SimpleNamespace(
        Dims=3, Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dx, dz=dx, epsr=1.0,
    )


def _make_time(t=0.0, tf=100e-15, dt=0.1e-15, n=0):
    """Create a typetime.ts-like object."""
    from types import SimpleNamespace
    return SimpleNamespace(t=t, tf=tf, dt=dt, n=n)


def _make_pulse(lam=800e-9, Amp=1.25e8, Tw=5e-15, Tp=60e-15, chirp=0.0, pol=0):
    """Create a typepulse.ps-like object."""
    from types import SimpleNamespace
    return SimpleNamespace(
        lam=lam, Amp=Amp, Tw=Tw, Tp=Tp, chirp=chirp, pol=pol,
    )


@needs_tfsf
class TestInitializeTFSF:
    """Test TFSF array initialisation."""

    def test_returns_array(self):
        """InitializeTFSF should produce a 1-D complex array."""
        space = _make_space()
        time = _make_time()
        pulse = _make_pulse()
        result = tfsf.InitializeTFSF(space, time, pulse)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_length_matches_nx(self):
        """TFSF array length should equal Nx."""
        Nx = 128
        space = _make_space(Nx=Nx)
        time = _make_time()
        pulse = _make_pulse()
        result = tfsf.InitializeTFSF(space, time, pulse)
        assert len(result) == Nx

    def test_antisymmetric(self):
        """TFSF should be approximately antisymmetric: TFSF ≈ -flip(TFSF).

        The Fortran does: TFSF = TFSF - FFlip(TFSF)
        which makes the array equal to  f - flip(f), i.e. antisymmetric.
        So TFSF(i) + TFSF(N-1-i) ≈ 0 for all i.
        """
        Nx = 256
        space = _make_space(Nx=Nx)
        time = _make_time()
        pulse = _make_pulse()
        result = tfsf.InitializeTFSF(space, time, pulse)
        flipped = result[::-1]
        assert np.allclose(result + flipped, 0.0, atol=1e-20)

    def test_localised_near_quarter(self):
        """The TFSF source window should peak near 25 % of the x-array."""
        Nx = 256
        space = _make_space(Nx=Nx)
        time = _make_time()
        pulse = _make_pulse()
        result = tfsf.InitializeTFSF(space, time, pulse)
        peak_idx = np.argmax(np.abs(result))
        # Peak should be within ~10 % of the 25 % mark
        assert abs(peak_idx / Nx - 0.25) < 0.15

    def test_mostly_zero_in_center(self):
        """Interior of the array (between sources) should be near-zero."""
        Nx = 256
        space = _make_space(Nx=Nx)
        time = _make_time()
        pulse = _make_pulse()
        result = tfsf.InitializeTFSF(space, time, pulse)
        center = result[Nx // 3 : 2 * Nx // 3]
        assert np.max(np.abs(center)) < 0.01 * np.max(np.abs(result))


@needs_tfsf
class TestUpdateTFSC:
    """Test TFSF field injection into E or B arrays."""

    def _setup(self, Nx=64, Ny=4, Nz=4):
        space = _make_space(Nx=Nx, Ny=Ny, Nz=Nz)
        time = _make_time(t=60e-15)  # at pulse peak
        pulse = _make_pulse()
        TFSF = tfsf.InitializeTFSF(space, time, pulse)
        E = np.zeros((Nx, Ny, Nz), dtype=complex)
        return space, time, pulse, TFSF, E

    def test_modifies_field(self):
        """Injecting TFSF source should produce nonzero field."""
        space, time, pulse, TFSF, E = self._setup()
        tfsf.UpdateTFSC(space, time, pulse, pulse.Amp, E, TFSF)
        assert np.max(np.abs(E)) > 0

    def test_field_uniform_in_yz(self):
        """TFSF is x-dependent only; all y-z slices should be identical."""
        space, time, pulse, TFSF, E = self._setup()
        tfsf.UpdateTFSC(space, time, pulse, pulse.Amp, E, TFSF)
        for j in range(E.shape[1]):
            for k in range(E.shape[2]):
                assert np.allclose(E[:, j, k], E[:, 0, 0], rtol=1e-12)

    def test_zero_amplitude_no_change(self):
        """With Emax=0, field should not change."""
        space, time, pulse, TFSF, E = self._setup()
        E_before = E.copy()
        tfsf.UpdateTFSC(space, time, pulse, 0.0, E, TFSF)
        assert np.allclose(E, E_before)

    def test_linearity(self):
        """Doubling the amplitude should double the injected field."""
        space, time, pulse, TFSF, E1 = self._setup()
        E2 = np.zeros_like(E1)

        tfsf.UpdateTFSC(space, time, pulse, pulse.Amp, E1, TFSF)
        tfsf.UpdateTFSC(space, time, pulse, 2 * pulse.Amp, E2, TFSF)

        # E2 should be 2× E1 (both started from zero)
        assert np.allclose(E2, 2.0 * E1, rtol=1e-10)

    def test_far_from_peak_vanishes(self):
        """Long before the pulse arrives, injection should be negligible."""
        Nx, Ny, Nz = 64, 4, 4
        space = _make_space(Nx=Nx, Ny=Ny, Nz=Nz)
        time = _make_time(t=-1e-12)  # well before Tp
        pulse = _make_pulse()
        TFSF = tfsf.InitializeTFSF(space, time, pulse)
        E = np.zeros((Nx, Ny, Nz), dtype=complex)
        tfsf.UpdateTFSC(space, time, pulse, pulse.Amp, E, TFSF)
        assert np.max(np.abs(E)) < 1e-10 * pulse.Amp


@needs_tfsf
class TestFFlip:
    """Test the array-flip helper if exposed by the module."""

    @pytest.mark.skipif(
        tfsf is None or not hasattr(tfsf, "FFlip"),
        reason="FFlip not exposed",
    )
    def test_fflip_reverses(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tfsf.FFlip(a)
        expected = a[::-1]
        assert np.allclose(result, expected)

    @pytest.mark.skipif(
        tfsf is None or not hasattr(tfsf, "FFlip"),
        reason="FFlip not exposed",
    )
    def test_fflip_double_is_identity(self):
        a = np.array([1.0, 2.0, 3.0])
        assert np.allclose(tfsf.FFlip(tfsf.FFlip(a)), a)


@needs_tfsf
class TestParameterised:
    """Parameterised tests across grid sizes and wavelengths."""

    @pytest.mark.parametrize("Nx", [64, 128, 256, 512])
    def test_tfsf_length(self, Nx):
        space = _make_space(Nx=Nx)
        time = _make_time()
        pulse = _make_pulse()
        result = tfsf.InitializeTFSF(space, time, pulse)
        assert len(result) == Nx

    @pytest.mark.parametrize("lam", [400e-9, 800e-9, 1550e-9])
    def test_tfsf_different_wavelengths(self, lam):
        """InitializeTFSF should succeed for various wavelengths."""
        space = _make_space(Nx=256)
        time = _make_time()
        pulse = _make_pulse(lam=lam)
        result = tfsf.InitializeTFSF(space, time, pulse)
        assert result is not None
        assert len(result) == 256
        # Should still be antisymmetric
        assert np.allclose(result + result[::-1], 0.0, atol=1e-20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])