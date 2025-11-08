# tests/test_coulomb.py
import sys
import pathlib


import numpy as np
import pytest


sys.path.insert(0, "/mnt/hardisk/rahul_gulley/pulsesuiteXX_old_copy/src")
import constants as C

# Ensure we import from the project's src directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from coulomb import Coulomb, InitializeCoulomb, SetLorentzDelta

# ----------------------------- fixtures --------------------------------- #

@pytest.fixture(scope="module")
def small_grid():
    """
    Construct a tiny, reproducible quantum-wire grid and parameters.
    Keep sizes small so tests run fast while exercising code paths.
    """
    rng = np.random.default_rng(42)

    # Momentum grid (symmetric, evenly spaced)
    N = 8
    kmax = 4.0e8  # 1/m
    ky = np.linspace(-kmax, kmax, N, dtype=np.float64)

    # Real-space grid for integrals
    Ny = 64
    L = 1.0e-6  # 1 micron
    y = np.linspace(-L/2, L/2, Ny, dtype=np.float64)

    # Simple parabolic dispersions
    me = 0.067 * C.me0
    mh = 0.45  * C.me0
    Ee = (C.hbar**2) * ky**2 / (2.0 * me)
    Eh = (C.hbar**2) * ky**2 / (2.0 * mh)

    # Linewidths (Hz)
    ge = 1.0e12
    gh = 1.2e12

    # Confinement & background
    alphae = 1.5e9   # 1/m
    alphah = 1.1e9   # 1/m
    Delta0 = 3.0e-9  # m
    er = 12.5        # relative permittivity

    # Build Qy and the (k,q)->index mapping kkp
    # Qy as sorted unique |ky[k]-ky[q]| grid
    diffs = np.abs(ky[:, None] - ky[None, :]).ravel()
    Qy = np.unique(np.round(diffs, 12))  # de-duplicate with rounding
    # Map each (k,q) to nearest index in Qy
    def build_kkp(ky, Qy):
        N = ky.size
        kkp = np.full((N, N), -1, dtype=np.int32)
        for k in range(N):
            for q in range(N):
                val = abs(ky[k] - ky[q])
                idx = int(np.argmin(np.abs(Qy - val)))
                kkp[k, q] = idx
        return kkp
    kkp = build_kkp(ky, Qy)

    return {
        "N": N, "Ny": Ny,
        "ky": ky, "y": y, "L": L, "Delta0": Delta0,
        "me": me, "mh": mh, "Ee": Ee, "Eh": Eh,
        "ge": ge, "gh": gh,
        "alphae": alphae, "alphah": alphah,
        "er": er, "Qy": Qy, "kkp": kkp
    }


@pytest.fixture
def cobj():
    """Fresh instance per test to avoid cross-test state via the singleton."""
    return Coulomb()


# ----------------------------- tests ------------------------------------ #

def test_make_undel(cobj, small_grid):
    ky = small_grid["ky"]
    cobj.MakeUnDel(ky)
    UnDel = cobj.UnDel
    assert UnDel is not None
    N = ky.size
    assert UnDel.shape == (N+1, N+1)
    # Leading row/col must be zero
    assert np.allclose(UnDel[0, :], 0.0)
    assert np.allclose(UnDel[:, 0], 0.0)
    # Diagonal from 1..N must be zero
    diag = np.arange(1, N+1)
    assert np.allclose(UnDel[diag, diag], 0.0)
    # Off-diagonal should be ones
    off = UnDel.copy()
    off[0, :] = 0.0; off[:, 0] = 0.0
    off[diag, diag] = 0.0
    assert np.all((off == 1.0) | (off == 0.0))  # only 0 or 1


def test_makek3_and_makeqs(cobj, small_grid):
    ky = small_grid["ky"]
    N = small_grid["N"]
    cobj.MakeK3(ky)
    assert cobj.k3 is not None
    assert cobj.k3.shape == (N, N, N)

    cobj.MakeQs(ky, small_grid["alphae"], small_grid["alphah"])
    assert cobj.qe is not None and cobj.qh is not None
    assert cobj.qe.shape == (N, N)
    assert cobj.qh.shape == (N, N)
    # min constraints
    assert np.all(cobj.qe >= small_grid["alphae"]/2.0)
    assert np.all(cobj.qh >= small_grid["alphah"]/2.0)


def test_calc_coulomb_arrays_basic(cobj, small_grid):
    g = small_grid
    # ensure prerequisites
    cobj.MakeUnDel(g["ky"])
    cobj.MakeK3(g["ky"])
    cobj.MakeQs(g["ky"], g["alphae"], g["alphah"])

    # compute unscreened arrays
    cobj.CalcCoulombArrays(
        g["y"], g["ky"], g["er"],
        g["alphae"], g["alphah"],
        g["L"], g["Delta0"],
        g["Qy"], g["kkp"],
    )
    Veh0, Vee0, Vhh0 = cobj.Veh0, cobj.Vee0, cobj.Vhh0
    assert Veh0 is not None and Vee0 is not None and Vhh0 is not None
    N = g["N"]
    assert Veh0.shape == (N, N)
    assert Vee0.shape == (N, N)
    assert Vhh0.shape == (N, N)

    # Basic physical sanity: finite numbers and symmetry (depends on |Δk|)
    assert np.isfinite(Veh0).all()
    assert np.isfinite(Vee0).all()
    assert np.isfinite(Vhh0).all()
    assert np.allclose(Veh0, Veh0.T, atol=1e-10)
    assert np.allclose(Vee0, Vee0.T, atol=1e-10)
    assert np.allclose(Vhh0, Vhh0.T, atol=1e-10)


@pytest.mark.parametrize("lorentz", [False, True])
def test_calc_mbarrrays_and_zero_diagonal(cobj, small_grid, lorentz):
    g = small_grid
    cobj.MakeUnDel(g["ky"])
    cobj.MakeK3(g["ky"])
    SetLorentzDelta(lorentz)          # set on singleton…
    cobj.LorentzDelta = lorentz       # …and on instance used in tests
    cobj.CalcMBArrays(g["ky"], g["Ee"], g["Eh"], g["ge"], g["gh"])
    assert cobj.Ceh is not None and cobj.Cee is not None and cobj.Chh is not None
    N = g["N"]
    assert cobj.Ceh.shape == (N+1, N+1, N+1)
    # Zero whenever Undel(k,k)=0 on the appropriate indices (k==k4)
    UnDel = cobj.UnDel
    k = 3; k4 = k
    line = cobj.Ceh[:, :, k4].copy()
    assert np.allclose(line[k, :], 0.0, atol=0.0)


def test_screening_epsilon_and_arrays(cobj, small_grid):
    g = small_grid
    # Prereqs + veh/vee/vhh
    cobj.MakeUnDel(g["ky"])
    cobj.MakeK3(g["ky"])
    cobj.MakeQs(g["ky"], g["alphae"], g["alphah"])
    cobj.CalcCoulombArrays(
        g["y"], g["ky"], g["er"],
        g["alphae"], g["alphah"],
        g["L"], g["Delta0"],
        g["Qy"], g["kkp"],
    )
    cobj.CalcChi1D(g["ky"], g["alphae"], g["alphah"], g["Delta0"], g["L"], g["er"], g["me"], g["mh"])

    N = g["N"]
    VC = np.empty((N, N, 3), dtype=np.float64)
    E1D = np.empty((N, N), dtype=np.float64)
    ne = (0.05 * np.ones(N)).astype(np.complex128)
    nh = (0.05 * np.ones(N)).astype(np.complex128)

    cobj.CalcScreenedArrays(True, g["L"], ne, nh, VC, E1D)
    assert VC.shape == (N, N, 3)
    assert E1D.shape == (N, N)
    assert np.isfinite(E1D).all()
    # Screening should not produce zeros or NaNs
    assert np.all(E1D > 0.0)


def test_eps1dqw_and_chi1dqw(cobj, small_grid):
    g = small_grid
    # Dummy populations & linewidths per-k
    gamE = np.full_like(g["ky"], g["ge"])
    gamH = np.full_like(g["ky"], g["gh"])
    ne = np.zeros_like(g["ky"])
    nh = np.zeros_like(g["ky"])
    qq = abs(g["ky"][2] - g["ky"][1])
    w = 0.5e13  # rad/s

    chir, chii = cobj.GetChi1Dqw(
        g["alphae"], g["alphah"], g["Delta0"], g["L"], g["er"],
        gamE, gamH,
        g["ky"], g["Ee"], g["Eh"],
        ne, nh, qq, w
    )
    assert np.isfinite(chir)
    assert np.isfinite(chii)

    epr, epi = cobj.GetEps1Dqw(
        g["alphae"], g["alphah"], g["Delta0"], g["L"], g["er"],
        g["me"], g["mh"], 1e7,  # n1D
        qq, w
    )
    assert np.isfinite(epr)
    assert np.isfinite(epi)


def test_calcmveh_shapes_and_nontrivial(cobj, small_grid):
    g = small_grid
    N = g["N"]
    # Initialize full state once
    cobj.InitializeCoulomb(
        g["y"], g["ky"], g["L"], g["Delta0"], g["me"], g["mh"],
        g["Ee"], g["Eh"], g["ge"], g["gh"],
        g["alphae"], g["alphah"], g["er"], g["Qy"], g["kkp"], True
    )

    VC = np.empty((N, N, 3), dtype=np.float64)
    E1D = np.empty((N, N), dtype=np.float64)
    ne = (0.02 * np.ones(N)).astype(np.complex128)
    nh = (0.02 * np.ones(N)).astype(np.complex128)
    cobj.CalcScreenedArrays(True, g["L"], ne, nh, VC, E1D)

    # Polarization p with a single hot element
    F = 2
    p = np.zeros((N, N, F), dtype=np.complex128)
    p[N//2, N//2, 0] = 1.0 + 0.0j
    MVeh = np.zeros_like(p)
    cobj.CalcMVeh(p, VC, MVeh)
    assert MVeh.shape == p.shape
    # Expect some non-zero coupling into MVeh
    assert np.any(np.abs(MVeh) > 0.0)


def test_bgrenorm_and_renorms(cobj, small_grid):
    g = small_grid
    N = g["N"]
    # Minimal VC and UnDel prepared through initialization
    cobj.InitializeCoulomb(
        g["y"], g["ky"], g["L"], g["Delta0"], g["me"], g["mh"],
        g["Ee"], g["Eh"], g["ge"], g["gh"],
        g["alphae"], g["alphah"], g["er"], g["Qy"], g["kkp"], False
    )
    VC = np.zeros((N, N, 3), dtype=np.float64)
    VC[:, :, 0] = cobj.Veh0
    VC[:, :, 1] = cobj.Vee0
    VC[:, :, 2] = cobj.Vhh0

    # Diagonal density matrices (complex)
    Cmat = np.diag(0.01*np.ones(N)).astype(np.complex128)
    Dmat = np.diag(0.02*np.ones(N)).astype(np.complex128)
    BGR = np.zeros((N, N), dtype=np.complex128)

    cobj.BGRenorm(Cmat, Dmat, VC, BGR)
    assert BGR.shape == (N, N)
    assert np.isfinite(BGR.real).all()
    assert np.isfinite(BGR.imag).all()

    # Single-species renorms
    BGR2 = np.zeros_like(BGR)
    ne = np.diag(Cmat).copy()
    cobj.EeRenorm(ne, VC, BGR2)
    assert BGR2.shape == (N, N)

    BGR3 = np.zeros_like(BGR)
    nh = np.diag(Dmat).copy()
    cobj.EhRenorm(nh, VC, BGR3)
    assert BGR3.shape == (N, N)
