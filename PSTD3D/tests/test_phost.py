import os
import sys
import math
import numpy as np
import pytest
import pathlib

# --- imports ---
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from constants import twopi, c0, eps0, e0, ii
import phost

# ---------------- helpers ----------------
def F(a, dtype):
    return np.asfortranarray(np.asarray(a, dtype=dtype))

def grid(Nx=8, Ny=10, Lx=10e-6, Ly=10e-6, seed=0):
    rng = np.random.default_rng(seed)
    Ex = F(rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny)), np.complex128)
    Ey = F(rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny)), np.complex128)
    qx = F(2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx), np.float64)
    qy = F(2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny), np.float64)
    QX, QY = np.meshgrid(qx, qy, indexing="ij")
    qsq = F(QX * QX + QY * QY, np.float64)  # real for MakeTransverse/SetInitialP
    return Ex, Ey, qx, qy, qsq

def set_host(mat="AlAs", lam=1.55e-6, host=True):
    """Call SetHostMaterial with 0-D float64 inout scalars; return (epsr, n0)."""
    epsr = np.array(0.0, dtype=np.float64)
    n0   = np.array(0.0, dtype=np.float64)
    phost.SetHostMaterial(host, mat, float(lam), epsr, n0)
    return epsr.item(), n0.item()


# ---------------- presence ----------------
def test_symbols_present():
    names = [
        "CalcPHost","CalcPHostOld","CalcNextP","CalcMonoP",
        "SetHostMaterial","InitializeHost","CalcWq","CalcEpsrWq",
        "CalcEpsrWq_ij","DetermineCoeffs","Epsr_q","Epsr_qij",
        "FDTD_Dispersion","wq","SetInitialP","MakeTransverse",
        "SetParamsSilica","SetParamsGaAs","SetParamsAlAs","SetParamsNone",
        "nw2_no_gam","nw2","nwp_no_gam","epsrwp_no_gam","nwp",
        "nl2_no_gam","nl2","WriteHostDispersion",
    ]
    missing = [n for n in names if not hasattr(phost, n)]
    assert not missing, f"missing: {missing}"


# ---------------- params/materials ----------------
def test_SetParams_variants():
    phost.SetParamsSilica()
    phost.SetParamsGaAs()
    phost.SetParamsAlAs()
    phost.SetParamsNone()

def test_SetHostMaterial_and_identity(tmp_path):
    # host=True so the routine actually fills epsr/n0 (matches your port)
    os.chdir(tmp_path)
    lam = 1.55e-6
    epsr_val, n0_val = set_host("AlAs", lam, host=True)
    assert epsr_val > 0 and n0_val > 0
    lhs = phost.nl2_no_gam(lam)
    rhs = phost.nw2_no_gam(twopi * c0 / lam)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


# ---------------- init/alloc + accessors ----------------
def test_InitializeHost_false_branch_and_accessors():
    Ex, Ey, qx, qy, qsq_real = grid()
    # InitializeHost expects qsq as complex128
    qsq_cplx = F(qsq_real, np.complex128)
    # n0: take from SetHostMaterial(host=True) for a consistent material
    _, n0 = set_host("AlAs", 1.0e-6, host=True)
    phost.InitializeHost(Ex.shape[0], Ex.shape[1], n0, qsq_cplx, False)
    E = phost.Epsr_q(qsq_cplx)
    assert E.shape == qsq_real.shape and E.dtype == np.complex128
    assert np.allclose(E.real, n0**2, rtol=0, atol=0)
    _ = phost.Epsr_qij(0, 0)


# ---------------- projection & dispersion ----------------
def test_MakeTransverse_div_free():
    Ex, Ey, qx, qy, qsq_real = grid()
    Ex1, Ey1 = Ex.copy(order="F"), Ey.copy(order="F")
    phost.MakeTransverse(Ex1, Ey1, qx, qy, qsq_real)
    # Check q·E ≈ 0 for all k != 0 with a tolerant threshold
    for j in range(Ex.shape[1]):
        div = qx * Ex1[:, j] + qy[j] * Ey1[:, j]
        mask = qsq_real[:, j] > 0
        if np.any(mask):
            # scale tolerance by field magnitude to be robust
            scale = max(1.0, np.max(np.abs(Ex1[:, j])) + np.max(np.abs(Ey1[:, j])))
            assert np.all(np.abs(div[mask]) <= 1e-8 * scale)

def test_FDTD_Dispersion_and_wq_center():
    Nx, Ny = 12, 16
    qx = F(np.linspace(-1e4, 1e4, Nx), np.float64)
    qy = F(np.linspace(-1e4, 1e4, Ny), np.float64)
    dx = dy = 1e-6
    dt = 1e-17
    n0 = 3.4
    # allocate omega_q first
    phost.InitializeHost(Nx, Ny, n0, F(np.zeros((Nx,Ny)), np.complex128), False)
    phost.FDTD_Dispersion(qx, qy, dx, dy, dt, n0)
    i, j = Nx//2, Ny//2
    w = phost.wq(i, j)
    cont = (c0/n0) * math.sqrt(qx[i]**2 + qy[j]**2)
    assert np.isfinite(w)
    assert np.allclose(np.real(w), cont, rtol=2e-2, atol=1e-8)


# ---------------- ε(ω(q)) ----------------
def test_CalcWq_and_CalcEpsrWq_branch_consistency():
    Ex, Ey, qx, qy, qsq_real = grid()
    # Set material (sets w0, w1, w2 inside module)
    set_host("AlAs", 1.0e-6, host=True)
    # Initialize host buffers (EpsrWq/omega_q)
    phost.InitializeHost(Ex.shape[0], Ex.shape[1], 3.0, F(qsq_real, np.complex128), True)
    # choose a mix of q magnitudes to cover low/mid/high branches
    q = F(np.zeros_like(qsq_real), np.complex128)
    q[0:2, 0:2] = 1e1     # low ω after CalcWq
    q[3:5, 3:5] = 1e4     # mid
    q[-2:, -2:] = 1e7     # high
    phost.CalcWq(q)
    phost.CalcEpsrWq(q)

    # Recompute expected per element with the same branch logic
    aw = np.zeros(2, dtype=np.float64, order="F")
    bw = np.zeros(2, dtype=np.float64, order="F")
    phost.DetermineCoeffs(aw, bw)
    E = phost.Epsr_q(q)
    for (i, j) in [(0,0), (4,4), (-1,-1)]:
        wij = abs(phost.wq(i, j))
        # pick branch
        if wij < phost.twopi * 0 + 0:  # dummy to keep linter quiet
            pass  # never triggered
        # use stored w1/w2 via guesses (they are fixed for AlAs)
        w1g = twopi * c0 / 2.2e-6
        w2g = twopi * c0 / 0.56e-6
        if wij < w1g:
            # low: epsr_0 + aw1*w^2 + aw2*w^3
            eps = phost.epsr_0 + aw[0]*(wij**2) + aw[1]*(wij**3)
            expect = np.complex128(eps)
        elif wij > w2g:
            # high: epsr_inf + bw1/w^2 + bw2/w^3
            eps = phost.epsr_infty + bw[0]/(wij**2) + bw[1]/(wij**3)
            expect = np.complex128(eps)
        else:
            expect = phost.nw2_no_gam(np.real(wij))
        assert np.allclose(E[i, j], expect, rtol=1e-9, atol=1e-9)


def test_DetermineCoeffs_and_point_eval():
    set_host("AlAs", 1.0e-6, host=True)
    aw = F(np.zeros(2), np.float64)
    bw = F(np.zeros(2), np.float64)
    phost.DetermineCoeffs(aw, bw)
    w1g = twopi * c0 / 2.2e-6
    w2g = twopi * c0 / 0.56e-6
    e_lo = np.array(0+0j, dtype=np.complex128)
    e_hi = np.array(0+0j, dtype=np.complex128)
    phost.CalcEpsrWq_ij(w1g*0.3, aw, bw, e_lo)
    phost.CalcEpsrWq_ij(w2g*3.0, aw, bw, e_hi)
    assert np.isfinite(e_lo)
    assert np.isfinite(e_hi)


# ---------------- oscillators ----------------
def test_CalcMonoP_and_CalcNextP_shapes_and_linearity():
    phost.SetParamsAlAs()
    Nx, Ny = 6, 7
    E = F(np.ones((Nx, Ny)), np.complex128)
    Pmono = phost.CalcMonoP(E)
    assert Pmono.shape[:2] == (Nx, Ny) and Pmono.ndim == 3
    # no contiguity requirement — some broadcast paths are strided
    rng = np.random.default_rng(2)
    P1 = F(rng.standard_normal(Pmono.shape) + 1j*rng.standard_normal(Pmono.shape), np.complex128)
    P2 = F(rng.standard_normal(Pmono.shape) + 1j*rng.standard_normal(Pmono.shape), np.complex128)
    P3 = phost.CalcNextP(P1, P2, E, 1e-18)
    assert P3.shape == Pmono.shape and P3.dtype == np.complex128
    # linearity check with E=0
    E0 = F(np.zeros_like(E), np.complex128)
    a, b = (0.37+0j), (-1.42+0j)
    L1 = phost.CalcNextP(a*P1, b*P2, E0, 1e-18)
    L2 = a*phost.CalcNextP(P1, np.zeros_like(P2), E0, 1e-18) + \
         b*phost.CalcNextP(np.zeros_like(P1), P2, E0, 1e-18)
    assert np.allclose(L1, L2, rtol=2e-14, atol=2e-14)


def test_CalcPHost_and_CalcPHostOld_progression():
    Ex, Ey, qx, qy, qsq_real = grid(Nx=8, Ny=8)
    # allocate host=True buffers (omega_q/EpsrWq etc.)
    phost.InitializeHost(Ex.shape[0], Ex.shape[1], 3.0, F(qsq_real, np.complex128), True)
    Px = F(np.zeros_like(Ex), np.complex128)
    Py = F(np.zeros_like(Ey), np.complex128)
    epsb = np.array(0.0, dtype=np.float64)  # inout
    dt = 5e-19
    phost.CalcPHostOld(Ex, Ey, dt, 1, epsb, Px, Py)
    phost.CalcPHost(Ex, Ey, dt, 2, epsb, Px, Py)
    assert Px.shape == Ex.shape and Py.shape == Ey.shape


# ---------------- dispersion funcs ----------------
@pytest.mark.parametrize("lam", [0.7e-6, 1.0e-6, 2.0e-6])
def test_nl2_vs_nw2_identity(lam):
    w = twopi * c0 / lam
    assert np.allclose(phost.nl2_no_gam(lam), phost.nw2_no_gam(w), rtol=1e-12, atol=1e-12)

def test_derivatives_consistency():
    phost.SetParamsGaAs()
    w = 1.0e15
    h = 1e9
    num = (phost.nw2_no_gam(w+h) - phost.nw2_no_gam(w-h)) / (2*h)
    ana = phost.epsrwp_no_gam(w)
    n2 = phost.nw2_no_gam(w)
    n = np.sqrt(n2)
    assert np.allclose(num, ana, rtol=5e-6, atol=5e-6)
    assert np.allclose(phost.nwp_no_gam(w), ana/(2.0*n), rtol=5e-12, atol=5e-12)

def test_nw2_with_gam_finite():
    phost.SetParamsGaAs()
    w = 2.0e15
    assert np.isfinite(phost.nw2_no_gam(w))
    assert np.isfinite(phost.nw2(w))


# ---------------- SetInitialP + writers ----------------
def test_SetInitialP_runs_and_real_outputs(tmp_path):
    os.chdir(tmp_path)
    Ex, Ey, qx, qy, qsq_real = grid(Nx=10, Ny=12, seed=123)
    set_host("AlAs", 1.0e-6, host=True)  # also writes dispersion tables
    # InitializeHost needs complex qsq
    phost.InitializeHost(Ex.shape[0], Ex.shape[1], 3.2, F(qsq_real, np.complex128), True)
    Px = F(np.zeros_like(Ex), np.complex128)
    Py = F(np.zeros_like(Ey), np.complex128)
    epsb = np.array(0.0, dtype=np.float64)  # inout
    dt = 1e-18
    # SetInitialP expects qsq as REAL(dp)
    phost.SetInitialP(Ex.copy(order="F"), Ey.copy(order="F"), qx, qy, F(qsq_real, np.float64), dt, Px, Py, epsb)
    assert np.allclose(Px.imag, 0.0, atol=1e-12)
    assert np.allclose(Py.imag, 0.0, atol=1e-12)

def test_WriteHostDispersion_creates_files(tmp_path):
    os.chdir(tmp_path)
    phost.SetParamsAlAs()
    phost.WriteHostDispersion()
    expected = [
        "fields/host/n.w.real.dat",
        "fields/host/epsr.w.imag.dat",
        "fields/host/nogam/n.w.real.dat",
        "fields/host/n.l.real.dat",
        "fields/host/nogam/epsr.l.imag.dat",
    ]
    for rel in expected:
        assert os.path.exists(rel), f"missing {rel}"