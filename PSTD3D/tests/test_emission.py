# tests/test_emission.py
import sys
import pathlib


sys.path.insert(0, "/mnt/hardisk/rahul_gulley/pulsesuiteXX_old_copy/src")
import constants as C

# Ensure we import from the project's src directory (same style as before)
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

# for monkeypatch mod
import emission as emission_mod
from emission import (
    Emission,
    InitializeEmission,
    SpontEmission,
    Ec,
    SpontIntegral,
    rho0,
    CalcHOmega,
    Calchw,
    PLSpectrum,
)



@pytest.fixture(scope="module")
def small_grid():
    """
    Small, deterministic grid and parameters for emission tests.
    """
    N = 8
    kmax = 4.0e8  # 1/m
    ky = np.linspace(-kmax, kmax, N, dtype=np.float64)

    # Parabolic bands
    me = 0.067 * C.me0
    mh = 0.45  * C.me0
    Ee = (C.hbar**2) * ky**2 / (2.0 * me)
    Eh = (C.hbar**2) * ky**2 / (2.0 * mh)

    # Environment / material params
    epsr = 12.5
    geh  = 1.5e12          # Hz
    gap  = 1.0e-19         # J (~0.62 eV)
    dcv  = 5.0e-29         # C·m (arbitrary but plausible)
    ehint = 0.8            # dimensionless overlap

    return {
        "N": N, "ky": ky, "Ee": Ee, "Eh": Eh,
        "epsr": epsr, "geh": geh, "gap": gap,
        "dcv": dcv, "ehint": ehint
    }


@pytest.fixture
def eobj():
    """Fresh instance per test to avoid state leakage."""
    return Emission()


# ----------------------------- tests ------------------------------------ #

def test_initialize_emission_basics(eobj, small_grid):
    g = small_grid
    eobj.InitializeEmission(g["ky"], g["Ee"], g["Eh"], g["dcv"], g["epsr"], g["geh"], g["ehint"])

    # Shapes and presence
    assert eobj.idel is not None and eobj.HOmega is not None and eobj.square is not None
    N = g["N"]
    assert eobj.idel.shape == (N, N)
    assert eobj.HOmega.ndim == 1 and eobj.HOmega.size > 10  # from CalcHOmega guard
    assert eobj.square.shape == eobj.HOmega.shape

    # idel: zeros on diagonal, ones off-diagonal
    diag = np.arange(N)
    assert np.allclose(np.diag(eobj.idel), 0.0)
    off = eobj.idel.copy()
    off[diag, diag] = 0.0
    assert np.all((off == 1.0) | (off == 0.0))

    # RScale and square non-negative, finite
    assert np.isfinite(eobj.RScale) and eobj.RScale >= 0.0
    from usefulsubs import Lrtz
    expected = (eobj.RScale / C.hbar) * Lrtz(eobj.HOmega, C.hbar * g["geh"]) * np.exp(-eobj.HOmega / (eobj.kB * eobj.Temp))
    assert np.allclose(eobj.square, expected, rtol=0, atol=0)


def test_ec_reduces_to_minus_Veh_diag(eobj, small_grid):
    g = small_grid
    eobj.InitializeEmission(g["ky"], g["Ee"], g["Eh"], g["dcv"], g["epsr"], g["geh"], g["ehint"])
    N = g["N"]
    # Simple diagonal VC
    Veh = np.eye(N, dtype=np.float64) * 2.0
    Vee = np.eye(N, dtype=np.float64) * 3.0
    Vhh = np.eye(N, dtype=np.float64) * 4.0
    VC  = np.stack([Veh, Vee, Vhh], axis=-1)

    ne = np.zeros(N, dtype=np.float64)
    nh = np.zeros(N, dtype=np.float64)

    Ec_vec = eobj.Ec(ne, nh, VC)
    # With ne==nh==0: Ec(k) = -Veh(k,k)
    assert np.allclose(Ec_vec, -np.diag(Veh))


def test_spontintegral_vectorization(eobj, small_grid):
    g = small_grid
    eobj.InitializeEmission(g["ky"], g["Ee"], g["Eh"], g["dcv"], g["epsr"], g["geh"], g["ehint"])
    # Two different Ek values
    Ek = np.array([0.0, 0.5 * C.hbar * g["geh"]], dtype=np.float64)
    vec = eobj.SpontIntegral(Ek)
    # Compare with scalar calls (squeeze to scalars)
    s0 = np.asarray(eobj.SpontIntegral(Ek[0])).squeeze()
    s1 = np.asarray(eobj.SpontIntegral(Ek[1])).squeeze()
    scalar = np.array([s0, s1], dtype=np.float64)
    assert np.allclose(vec, scalar)


def test_spontemission_shapes_and_nonneg(eobj, small_grid):
    g = small_grid
    eobj.InitializeEmission(g["ky"], g["Ee"], g["Eh"], g["dcv"], g["epsr"], g["geh"], g["ehint"])
    N = g["N"]

    # Use very small VC so Ec ≈ 0 and Ek > 0 (physically meaningful integrand)
    tiny = 1e-30
    Veh = np.full((N, N), tiny, dtype=np.float64)
    Vee = np.full((N, N), tiny, dtype=np.float64)
    Vhh = np.full((N, N), tiny, dtype=np.float64)
    VC  = np.stack([Veh, Vee, Vhh], axis=-1)

    ne = np.zeros(N, dtype=np.complex128)
    nh = np.zeros(N, dtype=np.complex128)

    RSP = np.zeros(N, dtype=np.float64)
    eobj.SpontEmission(ne, nh, g["Ee"], g["Eh"], g["gap"], g["geh"], VC, RSP)
    assert RSP.shape == (N,)
    assert np.isfinite(RSP).all()
    # With Ek>0 and positive kernel, RSP should be non-negative
    assert np.all(RSP >= 0.0)
    assert np.any(RSP > 0.0)


def test_calchw_and_plspectrum(eobj, small_grid, monkeypatch):
    g = small_grid
    eobj.InitializeEmission(g["ky"], g["Ee"], g["Eh"], g["dcv"], g["epsr"], g["geh"], g["ehint"])

    # Monkeypatch the temperature estimator used inside PLSpectrum to avoid ~0 K
    monkeypatch.setattr(emission_mod, "Temperature", lambda *args, **kwargs: 300.0)

    # Build hw around the gap so exp(-|hw-E|/kBT) and softtheta are not vanishing
    Nw = 64
    hw = np.linspace(0.6 * g["gap"], 1.4 * g["gap"], Nw, dtype=np.float64)
    PLS = np.zeros(Nw, dtype=np.float64)

    # Minimal VC (tiny) so Ec≈0
    N = g["N"]
    tiny = 1e-30
    Veh = np.full((N, N), tiny, dtype=np.float64)
    Vee = np.full((N, N), tiny, dtype=np.float64)
    Vhh = np.full((N, N), tiny, dtype=np.float64)
    VC  = np.stack([Veh, Vee, Vhh], axis=-1)

    # Light, non-zero populations
    ne = np.full(N, 0.1 + 0.0j, dtype=np.complex128)
    nh = np.full(N, 0.1 + 0.0j, dtype=np.complex128)

    eobj.PLSpectrum(ne, nh, g["Ee"], g["Eh"], g["gap"], g["geh"], VC, hw, t=1e-12, PLS=PLS)
    assert PLS.shape == (Nw,)
    assert np.isfinite(PLS).all()
    # Should be non-negative, with some power in window
    assert np.all(PLS >= 0.0)
    assert np.any(PLS > 0.0)
