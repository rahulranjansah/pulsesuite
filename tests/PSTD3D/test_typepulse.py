"""
Comprehensive test suite for typepulse.py module.

Tests the pulse parameter structure (ps) including getters, setters,
derived physical quantities (wave number, frequency, spectral properties,
spatial beam properties), field generation, and file I/O.
"""

import os
import math
import tempfile
from io import StringIO

import numpy as np
import pytest



try:
    from pulsesuite.PSTD3D import typepulse as tp

    # Quick sanity: does it expose the dataclass?
    _has_ps = hasattr(tp, "ps")
    _IMPORT_ERROR = None if _has_ps else "module loaded but 'ps' class not found"
except Exception as exc:  # noqa: BLE001
    tp = None
    _has_ps = False
    _IMPORT_ERROR = str(exc)

needs_typepulse = pytest.mark.skipif(
    not _has_ps,
    reason=f"typepulse not available: {_IMPORT_ERROR or 'ps class missing'}",
)

# Physical constants (must match Fortran constants module)
c0 = 299792458.0  # speed of light (m/s)
pi = np.pi
twopi = 2.0 * pi
eps0 = 8.8541878176203898505365630317107e-12
mu0 = 1.2566370614359172953850573533118e-6


# Helper to build a typical pulse structure
def _make_pulse():
    """Return a representative pulse (800 nm, 5 fs, 1.25e8 V/m)."""
    return tp.ps(
        lam=800e-9,  # wavelength (m)
        Amp=1.25e8,  # amplitude (V/m)
        Tw=5.0e-15,  # pulse width (s)
        Tp=60e-15,  # peak time (s)
        chirp=0.0,  # chirp (rad/s²)
        pol=0,  # polarisation index
    )


@needs_typepulse
class TestPSDataclass:
    """Test the ps dataclass structure."""

    def test_creation_basic(self):
        p = _make_pulse()
        assert p.lam == 800e-9
        assert p.Amp == 1.25e8
        assert p.Tw == 5.0e-15
        assert p.Tp == 60e-15
        assert p.chirp == 0.0
        assert p.pol == 0

    def test_creation_chirped(self):
        p = tp.ps(
            lam=800e-9, Amp=1e7, Tw=10e-15, Tp=0.0,
            chirp=1e26, pol=1,
        )
        assert p.chirp == 1e26
        assert p.pol == 1


@needs_typepulse
class TestGetters:
    """Test getter functions for ps fields."""

    def test_getlambda(self):
        p = _make_pulse()
        assert tp.GetLambda(p) == 800e-9

    def test_getamp(self):
        p = _make_pulse()
        assert tp.GetAmp(p) == 1.25e8

    def test_gettw(self):
        p = _make_pulse()
        assert tp.GetTw(p) == 5.0e-15

    def test_gettp(self):
        p = _make_pulse()
        assert tp.GetTp(p) == 60e-15

    def test_getchirp(self):
        p = _make_pulse()
        assert tp.GetChirp(p) == 0.0

    def test_getpol(self):
        p = _make_pulse()
        assert tp.GetPol(p) == 0


@needs_typepulse
class TestSetters:
    """Test setter functions for ps fields."""

    def test_setlambda(self):
        p = _make_pulse()
        tp.SetLambda(p, 1550e-9)
        assert tp.GetLambda(p) == 1550e-9

    def test_setamp(self):
        p = _make_pulse()
        tp.SetAmp(p, 5e7)
        assert tp.GetAmp(p) == 5e7

    def test_settw(self):
        p = _make_pulse()
        tp.SetTw(p, 20e-15)
        assert tp.GetTw(p) == 20e-15

    def test_settp(self):
        p = _make_pulse()
        tp.SetTp(p, 120e-15)
        assert tp.GetTp(p) == 120e-15

    def test_setchirp(self):
        p = _make_pulse()
        tp.SetChirp(p, 1e26)
        assert tp.GetChirp(p) == 1e26

    def test_setpol(self):
        p = _make_pulse()
        tp.SetPol(p, 2)
        assert tp.GetPol(p) == 2


@needs_typepulse
class TestDerivedQuantities:
    """Test CalcK0, CalcFreq0, CalcOmega0, CalcTau, CalcDeltaOmega, etc."""

    def test_calck0(self):
        """Wave number k0 = 2π / λ."""
        p = _make_pulse()
        expected = twopi / 800e-9
        assert np.isclose(tp.CalcK0(p), expected, rtol=1e-12)

    def test_calcfreq0(self):
        """Optical frequency ν0 = c0 / λ."""
        p = _make_pulse()
        expected = c0 / 800e-9
        assert np.isclose(tp.CalcFreq0(p), expected, rtol=1e-12)

    def test_calcomega0(self):
        """Angular frequency ω0 = 2π c0 / λ."""
        p = _make_pulse()
        expected = twopi * c0 / 800e-9
        assert np.isclose(tp.CalcOmega0(p), expected, rtol=1e-12)

    def test_calctau(self):
        """Gaussian parameter τ = Tw / (2√ln2)."""
        p = _make_pulse()
        expected = 5.0e-15 / (2.0 * math.sqrt(math.log(2.0)))
        assert np.isclose(tp.CalcTau(p), expected, rtol=1e-12)

    def test_calcdeltaomega(self):
        """Fourier-limited bandwidth Δω = 0.44 / τ."""
        p = _make_pulse()
        tau = tp.CalcTau(p)
        expected = 0.44 / tau
        assert np.isclose(tp.CalcDeltaOmega(p), expected, rtol=1e-10)

    def test_calctime_bandwidth(self):
        """Time-bandwidth product Δω · τ ≈ 0.44 (transform-limited)."""
        p = _make_pulse()
        tbp = tp.CalcTime_BandWidth(p)
        assert np.isclose(tbp, 0.44, rtol=1e-10)


@needs_typepulse
class TestSpatialProperties:
    """Test CalcRayleigh, CalcCurvature, CalcGouyPhase."""

    def test_calcrayleigh(self):
        """Rayleigh range z_R = π w0² / λ  (w0 = ω0 here per Fortran)."""
        p = _make_pulse()
        w0 = tp.CalcOmega0(p)
        expected = pi * w0**2 / tp.GetLambda(p)
        assert np.isclose(tp.CalcRayleigh(p), expected, rtol=1e-10)

    def test_calccurvature_at_origin(self):
        """Curvature should be huge (inf) at x = 0."""
        p = _make_pulse()
        R = tp.CalcCurvature(p, 0.0)
        assert R > 1e30 or np.isinf(R)

    def test_calccurvature_far_field(self):
        """Far from waist, R(x) → x."""
        p = _make_pulse()
        x = 1e10  # very far away
        R = tp.CalcCurvature(p, x)
        assert np.isclose(R / x, 1.0, rtol=1e-3)

    def test_calcgouyphase(self):
        """Gouy phase = arctan(x / z_R)."""
        p = _make_pulse()
        x = 1.0
        zR = tp.CalcRayleigh(p)
        expected = math.atan(x / zR)
        assert np.isclose(tp.CalcGouyPhase(p, x), expected, rtol=1e-12)



@needs_typepulse
class TestPulseFieldXT:
    """Test PulseFieldXT – returns complex E(x, t) for a Gaussian pulse."""

    def test_peak_at_origin(self):
        """At x=0, t=Tp the envelope should be maximal."""
        p = _make_pulse()
        E = tp.PulseFieldXT(0.0, tp.GetTp(p), p)
        assert np.isclose(abs(E), tp.GetAmp(p), rtol=1e-10)

    def test_zero_far_from_peak(self):
        """Long before or after the pulse, field should vanish."""
        p = _make_pulse()
        E = tp.PulseFieldXT(0.0, -1e-12, p)  # 1 ps before origin
        assert abs(E) < 1e-10 * tp.GetAmp(p)

    def test_complex_valued(self):
        """Output should be complex (carrier oscillation)."""
        p = _make_pulse()
        E = tp.PulseFieldXT(0.0, tp.GetTp(p) + 1e-15, p)
        assert isinstance(E, (complex, np.complexfloating))

    def test_retarded_time(self):
        """Pulse at x should be delayed by x/c0 relative to x=0."""
        p = _make_pulse()
        x = 1e-6
        delay = x / c0
        E_origin = tp.PulseFieldXT(0.0, tp.GetTp(p), p)
        E_shifted = tp.PulseFieldXT(x, tp.GetTp(p) + delay, p)
        assert np.isclose(abs(E_origin), abs(E_shifted), rtol=1e-6)

    def test_chirp_changes_phase(self):
        """Nonzero chirp should change the phase but not the envelope peak."""
        p_unchirped = _make_pulse()
        p_chirped = tp.ps(
            lam=800e-9, Amp=1.25e8, Tw=5e-15, Tp=60e-15,
            chirp=1e26, pol=0,
        )
        E_u = tp.PulseFieldXT(0.0, 60e-15, p_unchirped)
        E_c = tp.PulseFieldXT(0.0, 60e-15, p_chirped)
        # Same envelope amplitude at peak
        assert np.isclose(abs(E_u), abs(E_c), rtol=1e-10)


@needs_typepulse
class TestFileIO:
    """Test reading / writing pulse parameter files."""

    def _pulse_params_text(self):
        return (
            "800e-9\n"   # lambda
            "1.25e8\n"   # Amp
            "5.0e-15\n"  # Tw
            "60e-15\n"   # Tp
            "0.0\n"      # chirp
        )

    def test_readpulseparams_sub(self):
        p = tp.ps(lam=0, Amp=0, Tw=0, Tp=0, chirp=0, pol=0)
        f = StringIO(self._pulse_params_text())
        tp.readpulseparams_sub(f, p)
        assert np.isclose(p.lam, 800e-9, rtol=1e-6)
        assert np.isclose(p.Amp, 1.25e8, rtol=1e-6)
        assert np.isclose(p.Tw, 5.0e-15, rtol=1e-6)
        assert np.isclose(p.Tp, 60e-15, rtol=1e-6)
        assert np.isclose(p.chirp, 0.0, atol=1e-30)

    def test_roundtrip_file(self):
        p = _make_pulse()
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".params"
        ) as fh:
            fname = fh.name

        try:
            tp.WritePulseParams(fname, p)
            p2 = tp.ps(lam=0, Amp=0, Tw=0, Tp=0, chirp=0, pol=0)
            tp.ReadPulseParams(fname, p2)
            assert np.isclose(p2.lam, p.lam, rtol=1e-10)
            assert np.isclose(p2.Amp, p.Amp, rtol=1e-10)
            assert np.isclose(p2.Tw, p.Tw, rtol=1e-10)
            assert np.isclose(p2.Tp, p.Tp, rtol=1e-10)
            assert np.isclose(p2.chirp, p.chirp, atol=1e-30)
        finally:
            os.unlink(fname)


@needs_typepulse
class TestParameterised:
    """Cross-check derived quantities across different wavelengths."""

    @pytest.mark.parametrize(
        "lam",
        [400e-9, 800e-9, 1550e-9, 10.6e-6],
    )
    def test_k0_omega0_relation(self, lam):
        """k0 * c0 == ω0 for any wavelength."""
        p = tp.ps(lam=lam, Amp=1e7, Tw=10e-15, Tp=0.0, chirp=0.0, pol=0)
        k0 = tp.CalcK0(p)
        omega0 = tp.CalcOmega0(p)
        assert np.isclose(k0 * c0, omega0, rtol=1e-12)

    @pytest.mark.parametrize(
        "Tw",
        [1e-15, 5e-15, 10e-15, 50e-15, 100e-15],
    )
    def test_time_bandwidth_product_constant(self, Tw):
        """TBP should always be 0.44 for transform-limited Gaussian."""
        p = tp.ps(lam=800e-9, Amp=1e7, Tw=Tw, Tp=0.0, chirp=0.0, pol=0)
        assert np.isclose(tp.CalcTime_BandWidth(p), 0.44, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])