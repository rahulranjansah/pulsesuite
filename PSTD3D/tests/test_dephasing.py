# tests/test_dephasing.py
import sys
import pathlib
import numpy as np
import pytest

sys.path.insert(0, "/mnt/hardisk/rahul_gulley/pulsesuiteXX_old_copy/src")
from constants import hbar, me0
# Make src importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from dephasing import Dephasing



@pytest.fixture(scope="module")
def small_grid():
    rng = np.random.default_rng(1234)

    N = 9  # odd size keeps ky centered at 0 more naturally
    kmax = 5.0e9
    ky = np.linspace(-kmax, kmax, N, dtype=np.float64)
    dk = ky[1] - ky[0]

    # simple parabolic bands (keep numbers realistic but small)
    me = 0.07 * me0
    mh = 0.45 * me0
    Ee = (hbar**2 * ky**2) / (2.0 * me)
    Eh = (hbar**2 * ky**2) / (2.0 * mh)

    # simple, positive Coulomb kernels (Veh, Vee, Vhh)
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Veh = 0.05 + 0.003 * (I + J)
    Vee = 0.07 + 0.003 * (I + J)
    Vhh = 0.09 + 0.003 * (I + J)
    VC = np.stack([Veh, Vee, Vhh], axis=-1).astype(np.float64)

    # occupations (complex, but rates use real parts)
    ne0 = (0.2 + 0.1 * np.tanh(ky / (0.6 * kmax))).astype(np.float64) + 0.0j
    nh0 = (0.25 + 0.1 * np.tanh(-ky / (0.5 * kmax))).astype(np.float64) + 0.0j

    # small dephasing rates (gee, ghh, geh)
    g = np.array([1.0e12, 1.2e12, 0.9e12], dtype=np.float64)

    # random complex polarization
    p = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * 1e-3

    return dict(
        N=N, ky=ky, dk=dk, me=me, mh=mh, Ee=Ee, Eh=Eh, VC=VC, ne0=ne0, nh0=nh0, g=g, p=p
    )


@pytest.fixture
def dph():
    return Dephasing()


def test_initialize_dephasing_maps(dph, small_grid):
    g = small_grid
    dph.InitializeDephasing(g["ky"], g["me"], g["mh"])

    # maps must exist with correct shapes
    N = g["N"]
    for name in ("k_p_q", "k_m_q", "k1_m_q", "k1p_m_q", "k1", "k1p"):
        arr = getattr(dph, name)
        assert arr is not None and arr.shape == (N, N)
        # index bounds
        assert np.all((arr >= 0) & (arr < N))

    # xe/xh present and finite; xe[center] ~ 0 because |ky| factor
    assert dph.xe is not None and dph.xh is not None
    assert dph.xe.shape == (N,) and dph.xh.shape == (N,)
    assert np.isfinite(dph.xe).all() and np.isfinite(dph.xh).all()

    mid = N // 2
    assert dph.xe[mid] == pytest.approx(0.0, abs=1e-14)
    assert dph.xh[mid] == pytest.approx(0.0, abs=1e-14)


def test_vxx2_consistency(dph, small_grid):
    g = small_grid
    ky = g["ky"]

    # Build a diagnostic V whose first column encodes row index (monotonic)
    N = g["N"]
    V = np.zeros((N, N), dtype=np.float64)
    V[:, 0] = np.arange(N, dtype=np.float64)

    out = dph.Vxx2(ky, V)
    # Expected: idx = clip(round(|q|/dq), 0, N-1); value = (V[idx,0])**2
    dq = ky[1] - ky[0]
    iq = np.rint(np.abs(ky / dq)).astype(int)
    idx = np.clip(iq, 0, N - 1)
    expected = (V[idx, 0]) ** 2

    assert np.array_equal(out, expected)


def test_calc_gammae_and_gammah_basic(dph, small_grid):
    g = small_grid
    dph.InitializeDephasing(g["ky"], g["me"], g["mh"])

    GammaE = np.zeros(g["N"], dtype=np.float64)
    GammaH = np.zeros(g["N"], dtype=np.float64)

    dph.CalcGammaE(g["ky"], g["ne0"], g["nh0"], g["VC"], GammaE)
    dph.CalcGammaH(g["ky"], g["ne0"], g["nh0"], g["VC"], GammaH)

    # Shapes and finiteness
    assert GammaE.shape == (g["N"],)
    assert GammaH.shape == (g["N"],)
    assert np.isfinite(GammaE).all()
    assert np.isfinite(GammaH).all()

    # Rates should be non-negative (allow tiny negative numerical noise)
    assert np.all(GammaE >= -1e-25)
    assert np.all(GammaH >= -1e-25)

    # With zero interactions, rates should be ~0
    VCzero = np.zeros_like(g["VC"])
    GEz = np.zeros_like(GammaE)
    GHz = np.zeros_like(GammaH)
    dph.CalcGammaE(g["ky"], g["ne0"], g["nh0"], VCzero, GEz)
    dph.CalcGammaH(g["ky"], g["ne0"], g["nh0"], VCzero, GHz)
    assert np.allclose(GEz, 0.0)
    assert np.allclose(GHz, 0.0)


def test_offdiagdephasing_shapes_and_finiteness(dph, small_grid):
    g = small_grid
    dph.InitializeDephasing(g["ky"], g["me"], g["mh"])

    x = np.zeros((g["N"], g["N"]), dtype=np.complex128)
    dph.OffDiagDephasing(g["ne0"], g["nh0"], g["p"], g["ky"], g["Ee"], g["Eh"], g["g"], g["VC"], x)

    assert x.shape == (g["N"], g["N"])
    assert np.isfinite(x.real).all() and np.isfinite(x.imag).all()

    # zero polarization should yield zero contribution
    x0 = np.zeros_like(x)
    dph.OffDiagDephasing(g["ne0"], g["nh0"], np.zeros_like(g["p"]), g["ky"], g["Ee"], g["Eh"], g["g"], g["VC"], x0)
    assert np.allclose(x0, 0.0)


def test_offdiagdephasing2_shapes_and_finiteness(dph, small_grid):
    g = small_grid
    dph.InitializeDephasing(g["ky"], g["me"], g["mh"])

    x = np.zeros((g["N"], g["N"]), dtype=np.complex128)
    dph.OffDiagDephasing2(
        g["ne0"], g["nh0"], g["p"], g["ky"], g["Ee"], g["Eh"], g["g"], g["VC"], t=0.0, x=x, write_extrema=False
    )
    assert x.shape == (g["N"], g["N"])
    assert np.isfinite(x.real).all() and np.isfinite(x.imag).all()

    # zero polarization should yield zero contribution here as well
    x0 = np.zeros_like(x)
    dph.OffDiagDephasing2(
        g["ne0"], g["nh0"], np.zeros_like(g["p"]), g["ky"], g["Ee"], g["Eh"], g["g"], g["VC"], t=0.0, x=x0
    )
    assert np.allclose(x0, 0.0)
