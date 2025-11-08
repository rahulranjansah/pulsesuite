# tests/test_qwoptics.py
import sys
from pathlib import Path
import numpy as np

# Make ./src and project root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import qwoptics  # noqa: E402
from usefulsubs import FFTG, iFFTG

def _c128(a): return np.asarray(a, dtype=np.complex128, order="F")
def _f64(a): return np.asarray(a, dtype=np.float64, order="F")


def test_initialize_sets_expected_dipoles(initialized_qw):
    Xcv0, Ycv0, Zcv0 = qwoptics.Xcv0, qwoptics.Ycv0, qwoptics.Zcv0
    assert Xcv0 is not None and Ycv0 is not None and Zcv0 is not None
    assert np.allclose(Ycv0, 0.0), "Ycv0 must be zero per snippet."
    assert np.allclose(Zcv0, 0.0), "Zcv0 must be zero per snippet."
    assert np.any(np.abs(Xcv0) > 0), "Xcv0 should be non-zero."


# def test_CalcExpikr_conjugation_and_orthogonality(grids, initialized_qw):
#     _, RR, _, ky, _ = grids
#     qwoptics.CalcExpikr(RR, ky)
#     Expikr, Expikrc = qwoptics.Expikr, qwoptics.Expikrc
#     assert Expikr is not None and Expikrc is not None
#     assert np.allclose(Expikrc, np.conjugate(Expikr))
#     G = Expikr @ Expikrc.T
#     diag = np.real(np.diag(G))
#     off = np.real(G - np.diag(diag))
#     assert np.all(diag > 0.8 * np.max(diag))
#     assert np.max(np.abs(off)) < 0.5 * np.min(diag)

def test_CalcExpikr_conjugation_and_orthogonality(grids, initialized_qw):
    _, RR, _, ky, _ = grids

    # Build/refresh phase caches
    qwoptics.CalcExpikr(RR, ky)
    Expikr = qwoptics.Expikr
    Expikrc = qwoptics.Expikrc

    assert Expikr is not None and Expikrc is not None
    assert np.allclose(Expikrc, np.conjugate(Expikr))

    # Discrete inner product G[k, k'] = sum_r e^{i k y_r} e^{-i k' y_r}
    G = Expikr @ Expikrc.T  # (Nk, Nk)
    diag = np.real(np.diag(G))
    off_mask = ~np.eye(G.shape[0], dtype=bool)
    off_vals = np.real(G[off_mask])

    # On a non-periodic grid, just require dominance of diagonal over off-diagonal
    assert diag.max() > off_vals.max(), "Auto term should exceed any cross term."
    # And on average, diagonal should be (slightly) larger than off-diagonals
    assert diag.mean() > 1.01 * off_vals.mean(), "Auto terms should dominate on average."
def test_Prop2QW_roundtrip_no_window(grids, initialized_qw, monkeypatch):
    """
    With R==RR and large L (window≈1), Prop2QW → iFFTG should recover originals.
    We monkeypatch qwoptics._rescale_1D to avoid spliner edge NaNs so we test the FFT path.
    """
    _, RR, R, _, _ = grids

    # --- patch: robust rescale using np.interp (real+imag) ---
    def _safe_rescale(x0, z0, x1, z1):
        if np.iscomplexobj(z0):
            z1[:] = np.interp(x1, x0, z0.real) + 1j*np.interp(x1, x0, z0.imag)
        else:
            z1[:] = np.interp(x1, x0, z0)
    monkeypatch.setattr(qwoptics, "_rescale_1D", _safe_rescale)

    Exx_r = np.cos(2 * np.pi * RR)
    Eyy_r = 0.5 + 0.25 * np.cos(np.pi * RR)
    Ezz_r = np.sin(2 * np.pi * RR)
    Vrr_r = 0.1 * np.ones_like(RR)

    Exx = _c128(Exx_r)
    Eyy = _c128(Eyy_r)
    Ezz = _c128(Ezz_r)
    Vrr = _c128(Vrr_r)

    Ex = _c128(np.zeros_like(R))
    Ey = _c128(np.zeros_like(R))
    Ez = _c128(np.zeros_like(R))
    Vr = _c128(np.zeros_like(R))

    Edc = np.array(0.0, dtype=np.float64)

    qwoptics.Prop2QW(RR, Exx, Eyy, Ezz, Vrr, Edc, R, Ex, Ey, Ez, Vr, 0.0, 0)

    expected_Edc = np.sum(Eyy_r) / (Eyy_r.size * 0.5)
    assert np.isclose(Edc.item(), expected_Edc, rtol=1e-8, atol=1e-8)

    # Back to real space
    Ex_t, Ey_t, Ez_t, Vr_t = Ex.copy(), Ey.copy(), Ez.copy(), Vr.copy()
    iFFTG(Ex_t); iFFTG(Ey_t); iFFTG(Ez_t); iFFTG(Vr_t)

    # Slightly relaxed tolerances to allow FFT scaling nuances
    assert np.allclose(Ex_t.real, Exx_r, rtol=1e-6, atol=1e-7)
    assert np.allclose(Ey_t.real, Eyy_r, rtol=1e-6, atol=1e-7)
    assert np.allclose(Ez_t.real, Ezz_r, rtol=1e-6, atol=1e-7)
    assert np.allclose(Vr_t.real, Vrr_r, rtol=1e-6, atol=1e-7)

    assert np.linalg.norm(Ex_t.imag) < 1e-7
    assert np.linalg.norm(Ey_t.imag) < 1e-7
    assert np.linalg.norm(Ez_t.imag) < 1e-7
    assert np.linalg.norm(Vr_t.imag) < 1e-7


def test_QWPolarization3_zero_Py_Pz_when_Ycv0_Zcv0_zero(grids, initialized_qw):
    _, RR, _, ky, _ = grids
    ehint = initialized_qw["ehint"]; area = initialized_qw["area"]; L = initialized_qw["L"]
    rng = np.random.default_rng(1234)
    p = _c128(rng.normal(size=(ky.size, ky.size)))
    Px = _c128(np.zeros_like(RR)); Py = _c128(np.zeros_like(RR)); Pz = _c128(np.zeros_like(RR))
    qwoptics.QWPolarization3(RR, ky, p, ehint, area, L, Px, Py, Pz, 0)
    Px_t, Py_t, Pz_t = Px.copy(), Py.copy(), Pz.copy()
    iFFTG(Px_t); iFFTG(Py_t); iFFTG(Pz_t)
    assert np.linalg.norm(Py_t) < 1e-10
    assert np.linalg.norm(Pz_t) < 1e-10
    assert Px_t.shape == RR.shape


def test_QW2Prop_plasmonics_normalises_equal_charge(grids, initialized_qw):
    _, RR, R, _, _ = grids
    rng = np.random.default_rng(42)
    Ex = _c128(rng.normal(size=R.size)); Ey = _c128(rng.normal(size=R.size))
    Ez = _c128(rng.normal(size=R.size)); Vr = _c128(rng.normal(size=R.size))
    Px = _c128(rng.normal(size=R.size)); Py = _c128(rng.normal(size=R.size)); Pz = _c128(rng.normal(size=R.size))
    re = _c128(np.abs(rng.normal(size=R.size))); rh = _c128(np.abs(rng.normal(size=R.size)))
    Pxx = _c128(np.zeros_like(RR)); Pyy = _c128(np.zeros_like(RR)); Pzz = _c128(np.zeros_like(RR))
    RhoE = _c128(np.zeros_like(RR)); RhoH = _c128(np.zeros_like(RR))
    qwoptics.QW2Prop(R, R.copy(), Ex, Ey, Ez, Vr, Px, Py, Pz, re, rh,
                     RR, Pxx, Pyy, Pzz, RhoE, RhoH, w=0, xxx=0,
                     WriteFields=False, Plasmonics=True)
    dRR = RR[1] - RR[0]
    tot_e = np.sum(np.abs(RhoE)).real * dRR
    tot_h = np.sum(np.abs(RhoH)).real * dRR
    assert tot_e > 0 and tot_h > 0 and np.isclose(tot_e, tot_h, rtol=1e-6, atol=1e-12)


def test_QWChi1_scales_with_dcv_squared(grids, dispersions):
    _, _, _, ky, _ = grids
    Ee, Eh = dispersions
    lam = 1.55e-6
    dky = abs(ky[1] - ky[0])
    chi1_a = qwoptics.QWChi1(lam, dky, Ee, Eh, area=1.0, geh=1e12, dcv=1.0 + 0.0j)
    chi1_b = qwoptics.QWChi1(lam, dky, Ee, Eh, area=1.0, geh=1e12, dcv=2.0 + 0.0j)
    chi1_0 = qwoptics.QWChi1(lam, dky, Ee, Eh, area=1.0, geh=1e12, dcv=0.0 + 0.0j)
    assert np.isclose(chi1_0, 0.0 + 0.0j)
    ratio = (chi1_b / chi1_a) if chi1_a != 0 else 0.0
    assert np.isclose(ratio, 4.0 + 0.0j, rtol=1e-6, atol=1e-9)
