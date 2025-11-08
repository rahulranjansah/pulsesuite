# tests/test_dcfield.py
import sys
import pathlib

# Ensure we import from ../src
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np

# Try official constants; tests still work if not present thanks to module fallbacks
try:
    import constants as C  # expects: e0, hbar, pi, me0 maybe
    hbar = float(getattr(C, "hbar"))
    e0 = float(getattr(C, "e0"))
except Exception:
    hbar = 1.054_571_817e-34
    e0 = 1.602_176_634e-19

from dcfield import (
    DCField,
    InitializeDC,
    CalcDCE2,
    CalcDCH2,
    CalcDCE,
    CalcDCH,
    EkReNorm,
    DriftVt,
    FDrift2,
    CalcI0n,
    CalcI0,
    GetEDrift,
    GetHDrift,
    GetVEDrift,
    GetVHDrift,
    ShiftN1D,
    ShiftN2D,
    Transport,
)


# ----------------------- Fixtures -----------------------
import pytest


@pytest.fixture
def small_grid():
    """
    Minimal but meaningful grid and parameters for smoke + sanity tests.
    """
    N = 64  # small but enough to exercise FFT derivatives
    dk = 5.0e7  # 1/m spacing
    ky = (np.arange(N, dtype=np.float64) - N // 2) * dk
    ky.sort()

    # Parabolic dispersions Ee/Eh
    me = 0.07 * 9.1093837015e-31  # ~GaAs electron mass
    mh = 0.45 * 9.1093837015e-31  # heavy hole-ish
    Ee = hbar**2 * ky**2 / (2.0 * me)
    Eh = hbar**2 * ky**2 / (2.0 * mh)

    # Simple interaction matrices (Toeplitz-like, positive)
    idx = np.arange(N, dtype=np.float64)
    Vee = 1e-20 / (1.0 + np.add.outer(idx, idx))
    Vhh = 2e-20 / (1.0 + np.add.outer(idx, idx))
    Veh = 5e-21 / (1.0 + np.add.outer(idx, idx))
    VC = np.stack([Veh, Vee, Vhh], axis=-1)

    # Occupations: normalized Gaussian envelope (complex, but real-valued)
    ne = np.exp(-((ky / (5.0 * dk)) ** 2)).astype(np.complex128)
    nh = np.exp(-((ky / (6.0 * dk)) ** 2)).astype(np.complex128)

    # Phonon/scattering params
    Ephn = 36e-3 * e0  # 36 meV -> J
    N0 = 1.0
    ge = 1.0e12  # s^-1
    gh = 1.2e12  # s^-1
    Edc = 5e4  # V/m (moderate)
    Cq2 = np.full(N, 1.0e-40, dtype=np.float64)  # small positive constant

    return dict(
        N=N,
        ky=ky,
        dk=dk,
        me=me,
        mh=mh,
        Ee=Ee,
        Eh=Eh,
        VC=VC,
        Vee=Vee,
        Vhh=Vhh,
        Veh=Veh,
        ne=ne,
        nh=nh,
        Ephn=Ephn,
        N0=N0,
        ge=ge,
        gh=gh,
        Edc=Edc,
        Cq2=Cq2,
    )


@pytest.fixture
def dcobj(small_grid):
    g = small_grid
    dc = DCField()
    dc.InitializeDC(g["ky"], g["me"], g["mh"])
    return dc


# ----------------------- Tests -----------------------

def test_initialize_dc_basics(dcobj: DCField, small_grid):
    g = small_grid
    assert dcobj.Nk == g["N"]
    assert dcobj.Y is not None and dcobj.Y.shape == (g["N"],)
    assert dcobj.xe is not None and dcobj.xe.shape == (g["N"],)
    assert dcobj.xh is not None and dcobj.xh.shape == (g["N"],)
    assert dcobj.qinv is not None and dcobj.qinv.shape == (g["N"] + 2,)
    # Y is spectral wavenumbers: finite and not all zeros
    assert np.isfinite(dcobj.Y).all()
    assert not np.allclose(dcobj.Y, 0.0)


def test_ekrenorm_and_driftv(dcobj: DCField, small_grid):
    g = small_grid
    # Ec from e-e
    Ec = dcobj.EkReNorm(np.real(g["ne"]), g["Ee"], g["Vee"])
    assert Ec.shape == (g["N"],)
    assert np.isfinite(Ec).all()
    # symmetric n and parabolic energy -> drift should be small magnitude
    v = dcobj.DriftVt(np.real(g["ne"]), Ec)
    assert np.isfinite(v)
    # Not asserting exact ~0; just ensure it's a reasonable float
    assert abs(v) < 1e7  # m/s order cap (very loose bound)


def test_calc_dce2_and_dch2_paths(dcobj: DCField, small_grid):
    g = small_grid
    N = g["N"]
    # E-branch
    DC_e = np.zeros(N, dtype=np.float64)
    dcobj.CalcDCE2(
        True,
        g["ky"],
        g["Cq2"],
        g["Edc"],
        g["me"],
        g["ge"],
        g["Ephn"],
        g["N0"],
        g["ne"],
        g["Ee"],
        g["Vee"],
        n=10,
        j=25,
        DC=DC_e,
    )
    assert DC_e.shape == (N,)
    assert np.isfinite(DC_e).all()
    assert np.any(np.abs(DC_e) > 0)  # Edc != 0 and ne non-constant => nonzero derivative likely

    # H-branch
    DC_h = np.zeros(N, dtype=np.float64)
    dcobj.CalcDCH2(
        True,
        g["ky"],
        g["Cq2"],
        g["Edc"],
        g["mh"],
        g["gh"],
        g["Ephn"],
        g["N0"],
        g["nh"],
        g["Eh"],
        g["Vhh"],
        n=10,
        j=25,
        DC=DC_h,
    )
    assert DC_h.shape == (N,)
    assert np.isfinite(DC_h).all()
    assert np.any(np.abs(DC_h) > 0)

    # Drift getters updated
    assert np.isfinite(dcobj.GetEDrift())
    assert np.isfinite(dcobj.GetHDrift())
    assert np.isfinite(dcobj.GetVEDrift())
    assert np.isfinite(dcobj.GetVHDrift())


def test_calc_dce_and_dch_spectral(dcobj: DCField, small_grid):
    g = small_grid
    N = g["N"]
    # E spectral variant
    DC_e = np.zeros(N, dtype=np.float64)
    dcobj.CalcDCE(
        True,
        g["ky"],
        g["Cq2"],
        g["Edc"],
        g["me"],
        g["ge"],
        g["Ephn"],
        g["N0"],
        g["ne"],
        g["Ee"],
        g["Vee"],
        DC_e,
    )
    assert DC_e.shape == (N,)
    assert np.isfinite(DC_e).all()
    assert np.any(np.abs(DC_e) > 0)

    # H spectral variant
    DC_h = np.zeros(N, dtype=np.float64)
    dcobj.CalcDCH(
        True,
        g["ky"],
        g["Cq2"],
        g["Edc"],
        g["mh"],
        g["gh"],
        g["Ephn"],
        g["N0"],
        g["nh"],
        g["Eh"],
        g["Vhh"],
        DC_h,
    )
    assert DC_h.shape == (N,)
    assert np.isfinite(DC_h).all()
    assert np.any(np.abs(DC_h) > 0)


def test_fdrift2_smoke(dcobj: DCField, small_grid):
    g = small_grid
    v = dcobj.DriftVt(np.real(g["ne"]), g["Ee"])
    Fd = dcobj.FDrift2(
        g["Ephn"],
        g["me"],
        g["ge"],
        g["ky"],
        np.real(g["ne"]),
        g["Cq2"],
        v,
        g["N0"],
        dcobj.xe if dcobj.xe is not None else np.zeros_like(g["ky"]),
    )
    assert Fd.shape == (g["N"],)
    assert np.isfinite(Fd).all()


def test_currents_and_utils(dcobj: DCField, small_grid):
    g = small_grid
    # I0n
    Ie = dcobj.CalcI0n(g["ne"], g["me"], g["ky"])
    assert np.isfinite(Ie)
    # I0 requires VC with Veh,Vee,Vhh
    I0_out = np.zeros(1, dtype=np.float64)
    dcobj.CalcI0(g["ne"], g["nh"], g["Ee"], g["Eh"], g["VC"], g["dk"], g["ky"], I0_out)
    assert np.isfinite(I0_out[0])


def test_shift_and_transport(dcobj: DCField, small_grid):
    g = small_grid
    # ShiftN1D changes phase -> usually not identical for nonzero dk
    ne = g["ne"].copy()
    ne_orig = ne.copy()
    dk_shift = (g["ky"][1] - g["ky"][0]) * 0.3
    dcobj.ShiftN1D(ne, dk_shift)
    assert ne.shape == ne_orig.shape
    # change expected unless distribution is identically zero
    assert np.any(np.abs(ne - ne_orig) > 1e-14)

    # Transport on diagonal-only 2D correlation (k1==k2 path)
    C = np.diag(np.real(g["ne"])).astype(np.complex128)
    C_orig = C.copy()
    Transport(dcobj, C, Edc=1e5, Eac=0.0, dt=1e-13, DCTrans=True, k1nek2=False)
    assert C.shape == C_orig.shape
    # diagonal should move
    assert np.any(np.abs(np.diag(C) - np.diag(C_orig)) > 1e-14)
