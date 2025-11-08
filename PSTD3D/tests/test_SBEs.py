# test_sbes.py — simple pytest coverage for SBEs.py
# Arrange sys.path so tests can import project modules when run from ./tests
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import constants as C
import SBEs as S
# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture(scope="module")
def grids():
    # small RR grid; Q is not directly used by InitializeSBE but keep parity
    RR = np.linspace(-2e-6, 2e-6, 192, dtype=np.float64)
    q  = np.linspace(-1.0, 1.0, 192, dtype=np.float64)
    return RR, q

@pytest.fixture(scope="module")
def inited(grids):
    RR, q = grids
    # Initialize with one wire, QW enabled; modest Emaxxx so gate can be overcome
    S.InitializeSBE(q=q, RR=RR, r0=0.0, Emaxxx=1.0, lam=1.55e-6, Nw=1, enable_QW=True)
    return True

# -----------------------------
# Import / basic API
# -----------------------------

def test_imports_ok():
    assert hasattr(S, "InitializeSBE")
    assert hasattr(S, "QWCalculator")
    assert hasattr(S, "SBECalculator")

# -----------------------------
# InitializeSBE effects
# -----------------------------

def test_initialize_allocations_and_layout(inited):
    # Shapes present
    assert S.Nk > 0 and S.Nr == 2 * S.Nk
    # Core arrays allocated
    for name in ("r","Qr","kr","Ee","Eh","YY1","YY2","YY3","CC1","CC2","CC3","DD1","DD2","DD3"):
        arr = getattr(S, name)
        assert arr is not None
        # Fortran‑contiguous where applicable
        if isinstance(arr, np.ndarray) and arr.ndim >= 1:
            assert arr.flags['F_CONTIGUOUS']


def test_indices_and_kkp(inited):
    assert S.kkp is not None
    # All indices must be within Qr bounds
    assert np.all(S.kkp >= 0)
    assert np.all(S.kkp < S.Nr)

# -----------------------------
# Small helpers
# -----------------------------

def test_QWArea_and_toggles(inited):
    A = S.QWArea()
    assert A > 0
    S.ShutOffOptics()
    assert S.Optics is False
    # restore for subsequent tests
    S.Optics = True

# -----------------------------
# Driver: QWCalculator gating and turn‑on
# -----------------------------

def _make_fields(n):
    # Complex128, Fortran‑order
    z = np.zeros(n, dtype=np.complex128, order='F')
    return z.copy(), z.copy(), z.copy(), z.copy()


def test_qwcalculator_gate_below_threshold(inited, grids):
    RR, _ = grids
    n = RR.size
    Exx, Eyy, Ezz, Vrr = _make_fields(n)
    Pxx, Pyy, Pzz, Rho = _make_fields(n)
    do_qwp, do_qwdl = S.QWCalculator(Exx, Eyy, Ezz, Vrr, RR, RR*0.0, 1e-15, 1,
                                     Pxx, Pyy, Pzz, Rho, np.array([False]), np.array([False]))
    # still gated off (wireoff True & |E| << 1e-3 Emax0)
    assert do_qwp is False and do_qwdl is False
    assert np.allclose(Pxx, 0) and np.allclose(Pyy, 0) and np.allclose(Pzz, 0) and np.allclose(Rho, 0)


def test_qwcalculator_turn_on(inited, grids):
    RR, _ = grids
    n = RR.size
    # Big enough field to exceed 1e-3 * Emax0 (Emax0=1.0 set in InitializeSBE)
    Exx = np.ones(n, dtype=np.complex128, order='F') * (2e-3 + 0.0j)
    Eyy = np.zeros_like(Exx)
    Ezz = np.zeros_like(Exx)
    Vrr = np.zeros_like(Exx)
    Pxx, Pyy, Pzz, Rho = _make_fields(n)
    do_qwp, do_qwdl = S.QWCalculator(Exx, Eyy, Ezz, Vrr, RR, RR*0.0, 1e-15, 1,
                                     Pxx, Pyy, Pzz, Rho, np.array([False]), np.array([False]))
    # Once turned on, flags reflect Optics/LF
    assert do_qwp is True and do_qwdl is True

# -----------------------------
# RHS kernels: dpdt/dCdt/dDdt
# -----------------------------

def test_dpdt_shapes_and_trivial_case(inited):
    Nk = S.Nk
    Z = np.zeros((Nk, Nk), dtype=np.complex128, order='F')
    Heh = np.eye(Nk, dtype=np.complex128, order='F')
    Hee = np.zeros_like(Heh)
    Hhh = np.zeros_like(Heh)
    GamE = np.zeros((Nk,), dtype=np.float64)
    GamH = np.zeros((Nk,), dtype=np.float64)
    OffP = np.zeros_like(Heh)
    out = S.dpdt(Z, Z, Z, Heh, Hee, Hhh, GamE, GamH, OffP)
    assert out.shape == (Nk, Nk)
    # With only Heh=I, C=D=P=0 => dp/dt = I/(i*hbar)
    expect = Heh / (1j * C.hbar)
    assert np.allclose(out, expect)


def test_dCdt_dDdt_shapes(inited):
    Nk = S.Nk
    Z = np.zeros((Nk, Nk), dtype=np.complex128, order='F')
    Heh = np.zeros_like(Z)
    Hee = np.zeros_like(Z)
    Hhh = np.zeros_like(Z)
    Off = np.zeros_like(Z)
    ge = np.zeros((Nk,), dtype=np.float64)
    gh = np.zeros((Nk,), dtype=np.float64)
    outC = S.dCdt(Z, Z, Z, Heh, Hee, Hhh, ge, gh, Off)
    outD = S.dDdt(Z, Z, Z, Heh, Hee, Hhh, ge, gh, Off)
    assert outC.shape == (Nk, Nk) and outD.shape == (Nk, Nk)
    assert np.allclose(outC, 0) and np.allclose(outD, 0)

# -----------------------------
# Relaxation / Preparation
# -----------------------------

def test_relaxation_no_nan(inited):
    Nk = S.Nk
    ne = np.zeros((Nk,), dtype=np.complex128)
    nh = np.zeros((Nk,), dtype=np.complex128)
    VC = np.zeros((Nk, Nk, 3), dtype=np.float64)
    E1D = np.zeros((Nk, Nk), dtype=np.float64)
    Rsp = np.zeros((Nk,), dtype=np.float64)
    S._Relaxation(ne, nh, VC, E1D, Rsp, 1e-15, 1, False)
    assert np.isfinite(ne.real).all() and np.isfinite(nh.real).all()


def test_preparation_runs(inited):
    Nk = S.Nk
    Ex = np.zeros((S.Nr,), dtype=np.complex128, order='F')
    Ey = np.zeros_like(Ex)
    Ez = np.zeros_like(Ex)
    Vr = np.zeros_like(Ex)

    P = np.zeros((Nk, Nk), dtype=np.complex128, order='F')
    Cee = np.zeros_like(P)
    Dhh = np.zeros_like(P)

    Heh = np.zeros_like(P)
    Hee = np.zeros_like(P)
    Hhh = np.zeros_like(P)
    VC  = np.zeros((Nk, Nk, 3), dtype=np.float64, order='F')
    E1D = np.zeros((Nk, Nk), dtype=np.float64, order='F')
    GamE = np.zeros((Nk,), dtype=np.float64)
    GamH = np.zeros((Nk,), dtype=np.float64)
    OffG = np.zeros((Nk, Nk, 3), dtype=np.complex128, order='F')
    Rsp  = np.zeros((Nk,), dtype=np.float64)

    S._Preparation(P, Cee, Dhh, Ex, Ey, Ez, Vr, Heh, Hee, Hhh, VC, E1D, GamE, GamH, OffG, Rsp)
    # Basic sanity
    assert Heh.shape == Hee.shape == Hhh.shape == (Nk, Nk)

# -----------------------------
# Energy accounting
# -----------------------------

def test_update_PdotE_accumulates(inited, grids):
    RR, _ = grids
    n = RR.size
    Exx = np.ones(n, dtype=np.complex128, order='F')
    Eyy = np.zeros_like(Exx)
    Ezz = np.zeros_like(Exx)
    Vrr = np.zeros_like(Exx)
    Pxx = np.ones_like(Exx) * 1e-6
    Pyy = np.zeros_like(Exx)
    Pzz = np.zeros_like(Exx)

    # local QW arrays (match sizes S.Nr)
    Ex = np.ones(S.Nr, dtype=np.complex128, order='F')
    Ey = np.zeros_like(Ex)
    Ez = np.zeros_like(Ex)
    Px = np.ones_like(Ex) * 1e-6
    Py = np.zeros_like(Ex)
    Pz = np.zeros_like(Ex)

    before = (S.EPEnergy, S.EPEnergyW)
    S._update_PdotE(RR, Exx, Eyy, Ezz, Pxx, Pyy, Pzz, S.r, Ex, Ey, Ez, Px, Py, Pz)
    after = (S.EPEnergy, S.EPEnergyW)
    assert after[0] != before[0] or after[1] != before[1]

# -----------------------------
# Checkout/Checkin
# -----------------------------

def test_checkout_checkin_roundtrip(inited):
    Nk = S.Nk
    w = 1
    P1 = np.zeros((Nk, Nk), dtype=np.complex128, order='F')
    P2 = np.zeros_like(P1)
    C1 = np.zeros_like(P1)
    C2 = np.zeros_like(P1)
    D1 = np.zeros_like(P1)
    D2 = np.zeros_like(P1)

    # Pull (no crash)
    S._Checkout(P1, P2, C1, C2, D1, D2, w)
    # Push back with a marker
    P3 = np.ones_like(P1) * (1+0j)
    C3 = np.ones_like(P1) * (2+0j)
    D3 = np.ones_like(P1) * (3+0j)
    S._Checkin(P1, P2, P3, C1, C2, C3, D1, D2, D3, w)
    assert np.allclose(S.YY3[:, :, w-1], 1+0j)
    assert np.allclose(S.CC3[:, :, w-1], 2+0j)
    assert np.allclose(S.DD3[:, :, w-1], 3+0j)

# -----------------------------
# chi/qc accessors
# -----------------------------

def test_accessors(inited):
    assert isinstance(S.chiqw(), complex)
    assert S.getqc() > 0.0
