# tests/test_phonons.py
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, "/mnt/hardisk/rahul_gulley/pulsesuiteXX_old_copy/src")
import constants as C

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from phonons import Phonons, InitializePhonons, MBPE, MBPH, Cq2, FermiDistr, BoseDistr, N00

# ----------------------------- fixtures --------------------------------- #

@pytest.fixture(scope="module")
def small_grid():
    """
    Small, deterministic grid and parameters for phonon tests.
    """
    N = 8
    kmax = 4.0e8  # 1/m
    ky = np.linspace(-kmax, kmax, N, dtype=np.float64)

    # Parabolic bands
    me = 0.067 * C.me0
    mh = 0.45  * C.me0
    Ee = (C.hbar**2) * ky**2 / (2.0 * me)
    Eh = (C.hbar**2) * ky**2 / (2.0 * mh)

    # Environment
    L = 1.0e-6   # m
    epsr = 12.5  # relative permittivity

    # Phonon params
    Gph = 1.5e12       # Hz (damping)
    Oph = 3.0e13       # rad/s (LO phonon)

    return {
        "N": N, "ky": ky, "Ee": Ee, "Eh": Eh,
        "L": L, "epsr": epsr, "Gph": Gph, "Oph": Oph
    }


@pytest.fixture
def pobj():
    """Fresh instance per test to avoid state leakage."""
    return Phonons()


# ----------------------------- tests ------------------------------------ #
def test_initialize_phonons_basics(pobj, small_grid):
    g = small_grid
    pobj.InitializePhonons(g["ky"], g["Ee"], g["Eh"], g["L"], g["epsr"], g["Gph"], g["Oph"])

    assert pobj.EP is not None and pobj.HP is not None
    assert pobj.EPT is not None and pobj.HPT is not None
    assert pobj.idel is not None

    N = g["N"]
    # Shapes
    assert pobj.EP.shape == (N, N)
    assert pobj.HP.shape == (N, N)
    assert pobj.EPT.shape == (N, N)
    assert pobj.HPT.shape == (N, N)
    assert pobj.idel.shape == (N, N)

    # EPT/HPT are the stored transposes; kernels are finite and non-negative
    assert np.allclose(pobj.EPT, pobj.EP.T, rtol=0, atol=0)
    assert np.allclose(pobj.HPT, pobj.HP.T, rtol=0, atol=0)
    assert np.isfinite(pobj.EP).all() and (pobj.EP >= 0).all()
    assert np.isfinite(pobj.HP).all() and (pobj.HP >= 0).all()

    # Zero diagonal (idel applies)
    assert np.allclose(np.diag(pobj.EP), 0.0)
    assert np.allclose(np.diag(pobj.HP), 0.0)

    # idel mask: ones off-diagonal, zeros on diagonal
    diag = np.arange(N)
    assert np.allclose(np.diag(pobj.idel), 0.0)
    off = pobj.idel.copy()
    off[diag, diag] = 0.0
    assert np.all((off == 1.0) | (off == 0.0))



def test_mbpe_nontrivial_and_nonnegative(pobj, small_grid):
    g = small_grid
    N = g["N"]
    pobj.InitializePhonons(g["ky"], g["Ee"], g["Eh"], g["L"], g["epsr"], g["Gph"], g["Oph"])

    # Coulomb arrays placeholder: use positive values for Vee slice
    VC = np.zeros((N, N, 3), dtype=np.float64)
    VC[:, :, 1] = 1.0 + 0.1 * np.add.outer(np.linspace(0, 1, N), np.linspace(0, 1, N))  # Vee
    E1D = np.ones((N, N), dtype=np.float64)

    # Case 1: ne all zeros -> Win == 0, Wout > 0 (since EP>0 and Vep>0)
    ne = np.zeros(N, dtype=np.float64)
    Win = np.zeros(N, dtype=np.float64)
    Wout = np.zeros(N, dtype=np.float64)
    pobj.MBPE(ne, VC, E1D, Win, Wout)
    assert np.allclose(Win, 0.0)
    assert np.all(Wout >= 0.0)
    assert np.any(Wout > 0.0)

    # Case 2: ne mid occupancy -> both Win and Wout positive
    ne = np.full(N, 0.3, dtype=np.float64)
    Win2 = np.zeros(N, dtype=np.float64)
    Wout2 = np.zeros(N, dtype=np.float64)
    pobj.MBPE(ne, VC, E1D, Win2, Wout2)
    assert np.all(Win2 >= 0.0) and np.any(Win2 > 0.0)
    assert np.all(Wout2 >= 0.0) and np.any(Wout2 > 0.0)


def test_mbph_nontrivial_and_nonnegative(pobj, small_grid):
    g = small_grid
    N = g["N"]
    pobj.InitializePhonons(g["ky"], g["Ee"], g["Eh"], g["L"], g["epsr"], g["Gph"], g["Oph"])

    # Coulomb arrays placeholder: use positive values for Vhh slice
    VC = np.zeros((N, N, 3), dtype=np.float64)
    VC[:, :, 2] = 1.0 + 0.2 * np.add.outer(np.linspace(0, 1, N), np.linspace(0, 1, N))  # Vhh
    E1D = np.ones((N, N), dtype=np.float64)

    nh = np.zeros(N, dtype=np.float64)
    Win = np.zeros(N, dtype=np.float64)
    Wout = np.zeros(N, dtype=np.float64)
    pobj.MBPH(nh, VC, E1D, Win, Wout)
    assert np.allclose(Win, 0.0)
    assert np.any(Wout > 0.0)

    nh = np.full(N, 0.4, dtype=np.float64)
    Win2 = np.zeros(N, dtype=np.float64)
    Wout2 = np.zeros(N, dtype=np.float64)
    pobj.MBPH(nh, VC, E1D, Win2, Wout2)
    assert np.all(Win2 >= 0.0) and np.any(Win2 > 0.0)
    assert np.all(Wout2 >= 0.0) and np.any(Wout2 > 0.0)


def test_cq2_indexing_and_scaling(pobj, small_grid):
    g = small_grid
    N = g["N"]
    pobj.InitializePhonons(g["ky"], g["Ee"], g["Eh"], g["L"], g["epsr"], g["Gph"], g["Oph"])

    # Build V, E1D such that first column encodes index clearly
    V = np.zeros((N, N), dtype=np.float64)
    E1D = np.ones((N, N), dtype=np.float64)
    V[:, 0] = np.arange(N, dtype=np.float64)  # V[i,0] == i

    # q grid spaced by dq; choose values 0..(N-1)*dq
    dq = 1.23  # arbitrary spacing
    qvals = np.array([0, dq, 2*dq, 3*dq, 4*dq], dtype=np.float64)
    out = pobj.Cq2(qvals, V, E1D)

    # Expected: rint(|q/dq|) gives i; result = V[i,0]/E1D[i,0]*Vscale = i * Vscale
    iq = np.rint(np.abs(qvals / dq)).astype(int)
    iq = np.clip(iq, 0, N-1)
    expected = iq.astype(np.float64) * pobj.Vscale
    assert np.allclose(out, expected, rtol=0, atol=0)


def test_fermi_bose_and_n00(pobj, small_grid):
    g = small_grid
    pobj.InitializePhonons(g["ky"], g["Ee"], g["Eh"], g["L"], g["epsr"], g["Gph"], g["Oph"])

    # Fermi-Dirac: large positive -> ~0, large negative -> ~1
    f_pos = pobj.FermiDistr(50.0 * pobj.kB * pobj.Temp)
    f_neg = pobj.FermiDistr(-50.0 * pobj.kB * pobj.Temp)
    assert np.isclose(f_pos, 0.0 + 0.0j, atol=1e-12)
    assert np.isclose(f_neg, 1.0 + 0.0j, atol=1e-12)

    # Vectorized call returns complex128 array
    arr = np.array([-10, 0, 10], dtype=np.float64) * pobj.kB * pobj.Temp
    f_arr = pobj.FermiDistr(arr)
    assert f_arr.dtype == np.complex128
    assert f_arr.shape == (3,)

    # Bose: positive energy -> finite positive, grows as energy -> 0+
    b_mid = pobj.BoseDistr(5.0 * pobj.kB * pobj.Temp)
    assert b_mid > 0.0
    # Not testing near zero to avoid inf

    # N00 equals Bose at LO phonon energy
    NO_expected = 1.0 / (np.exp(C.hbar * g["Oph"] / (pobj.kB * pobj.Temp)) - 1.0)
    assert np.isclose(pobj.N00(), NO_expected, rtol=1e-12, atol=0.0)
