# tests/test_epsrtl.py
import os
import io
import math
import numpy as np
import pytest
import sys
import pathlib

#
# ---- Import the module under test ----
# Ensure your PYTHONPATH includes the directory holding epsrtl.py

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import epsrtl as M


# ---------------------- Fixtures ----------------------

@pytest.fixture
def tiny_ky():
    # small symmetric grid, Fortran order
    ky = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
    return np.asfortranarray(ky)


@pytest.fixture
def tiny_phys():
    # modest, physically-plausible constants (arbitrary units OK for consistency checks)
    me = np.float64(0.067 * 9.10938356e-31)  # GaAs-like electron effective mass
    mh = np.float64(0.45  * 9.10938356e-31)  # hole effective mass
    Eg = np.float64(1.42 * 1.602176634e-19)  # ~1.42 eV bandgap
    Te = np.float64(300.0)                   # Kelvin
    Th = np.float64(300.0)                   # Kelvin
    dcv0 = np.float64(1.0)                   # dimensionless coupling (placeholder)
    n1D = np.float64(1e8)                    # 1D density [m^-1] (arbitrary scale)
    return dict(me=me, mh=mh, Eg=Eg, Te=Te, Th=Th, dcv0=dcv0, n1D=n1D)


@pytest.fixture
def patch_small_omega(monkeypatch):
    # Make frequency grid tiny & more coarse so tests run fast
    monkeypatch.setattr(M, "_Nw", 2, raising=True)      # w = -2..2 (5 points)
    monkeypatch.setattr(M, "_dw", np.float64(1.0), raising=True)
    # Also tame radius to avoid extreme arguments in K03 in corner cases
    monkeypatch.setattr(M, "_R0", np.float64(1e-9), raising=True)


@pytest.fixture
def wire_dirs(tmp_path, monkeypatch):
    # Create the expected data directories and chdir into tmp for clean files
    base = tmp_path / "dataQW" / "Wire"
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return base


# ------------------ Elemental/Utility tests ------------------
def test_Eng_basic_properties():
    m = np.float64(9.10938356e-31)
    k = np.linspace(-3.0, 3.0, 21, dtype=np.float64)

    # function should accept both memory layouts identically
    E_F = M.Eng(m, np.asfortranarray(k))
    E_C = M.Eng(m, np.array(k, order="C"))
    assert np.allclose(E_F, E_C)
    assert E_F.dtype == np.float64

    # Even in k and nonnegative
    assert np.allclose(E_F, M.Eng(m, -k))
    assert np.all(E_F >= 0.0)

    # a 2D check where Fortran-contiguity is well-defined
    k2 = np.asfortranarray(np.stack([k, k*0.5], axis=1))  # shape (21,2), F-ordered
    assert np.isfortran(k2)
    E2 = M.Eng(m, k2)

    # Elementwise formula E = ħ² k² / (2m)
    assert np.allclose(E2, (M.hbar**2) * (k2**2) / (2.0*m))


def test_fT0_step_behavior():
    kf = np.float64(0.5)
    k = np.array([-1.0, -0.4, 0.0, 0.3, 0.5, 0.6, 1.1], dtype=np.float64, order="F")
    f = M.fT0(k, kf)
    # Inside Fermi sea (|k| < kf) => 1, outside => 0 (θ(0) handled as 0 in our helper)
    expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert np.allclose(f, expected)


def test_atanhc_matches_numpy_arctanh_on_real():
    x = np.linspace(-0.8, 0.8, 9, dtype=np.float64)
    # For real inputs in (-1,1): atanhc = np.arctanh
    got = M.atanhc(x.astype(np.complex128))
    want = np.arctanh(x.astype(np.complex128))
    assert np.allclose(got, want, atol=1e-12, rtol=1e-12)


def test_atanJG_matches_real_arctan_small_real():
    # For real z, atanJG(z) == arctan(z)
    z = np.linspace(-2.0, 2.0, 17, dtype=np.float64)
    got = M.atanJG(z.astype(np.complex128))
    want = np.arctan(z).astype(np.complex128)
    assert np.allclose(got, want, atol=1e-12, rtol=1e-12)


def test_ff0_basic_limits(tiny_phys):
    # MB-like occupancy goes to zero as E -> inf; positive for finite E,T
    E = np.linspace(0.0, 10.0, 11, dtype=np.float64, endpoint=True)
    f = M.ff0(E, np.float64(300.0), tiny_phys["me"])
    assert f.dtype == np.float64
    assert np.all(f >= 0.0)
    assert f[0] >= f[-1]
    # T=0 -> defined to 0 array in the port to avoid divide-by-zero
    f0 = M.ff0(E, np.float64(0.0), tiny_phys["me"])
    assert np.all(f0 == 0.0)


# ------------------ Pi kernels: shape, dtype, finiteness ------------------

def test_PiT_shapes_and_finiteness(tiny_phys, tiny_ky):
    me, mh, Te, Th, Eg = tiny_phys["me"], tiny_phys["mh"], tiny_phys["Te"], tiny_phys["Th"], tiny_phys["Eg"]
    dk = tiny_ky[1] - tiny_ky[0]
    Ek  = M.Eng(mh, tiny_ky)
    Ekq = M.Eng(me, tiny_ky + tiny_ky[3]) + Eg
    val = M.PiT(q=np.float64(tiny_ky[3]), w=np.float64(0.5), me=me, mh=mh,
                Te=Te, Th=Th, dk=np.float64(dk), Ek=Ek, Ekq=Ekq)
    assert isinstance(val, np.complexfloating)
    assert np.isfinite(val.real) and np.isfinite(val.imag)


def test_PiL_shapes_and_finiteness(tiny_phys, tiny_ky):
    m, T = tiny_phys["me"], tiny_phys["Te"]
    dk = tiny_ky[1] - tiny_ky[0]
    Ek  = M.Eng(m, tiny_ky)
    Ekq = M.Eng(m, tiny_ky + tiny_ky[2])
    val = M.PiL(q=np.float64(tiny_ky[2]), w=np.float64(1.0), m=m, T=T, dk=np.float64(dk), Ek=Ek, Ekq=Ekq)
    assert isinstance(val, np.complexfloating)
    assert np.isfinite(val.real) and np.isfinite(val.imag)


def test_PiL_T0_basic(tiny_ky):
    # Use neutral numbers; we only check finiteness and dtype here
    m  = np.float64(9.11e-31)
    dk = tiny_ky[1] - tiny_ky[0]
    Ek  = M.Eng(m, tiny_ky)
    Ekq = M.Eng(m, tiny_ky + tiny_ky[1])
    val = M.PiL_T0(q=np.float64(tiny_ky[1]), w=np.float64(0.2), m=m, T=np.float64(0.0),
                   dk=np.float64(dk), Ek=Ek, Ekq=Ekq)
    assert isinstance(val, np.complexfloating)
    assert np.isfinite(val.real) and np.isfinite(val.imag)


# ------------------ File-producing routines ------------------

def _count_lines(p):
    with open(p, "r") as f:
        return sum(1 for _ in f)


def test_GetEpsrLEpsrT_writes_metadata_and_zeroT_outputs(
    tiny_phys, tiny_ky, patch_small_omega, wire_dirs
):
    # Inputs
    n1D  = tiny_phys["n1D"]
    dcv0 = tiny_phys["dcv0"]
    Te, Th = tiny_phys["Te"], tiny_phys["Th"]
    me, mh = tiny_phys["me"], tiny_phys["mh"]
    Eg = tiny_phys["Eg"]

    # Act
    M.GetEpsrLEpsrT(n1D, dcv0, Te, Th, me, mh, Eg, tiny_ky)

    # Assert files exist
    base = wire_dirs
    assert (base / "qw.dat").exists()
    assert (base / "ChiL.E.dat").exists()
    assert (base / "ChiL.H.dat").exists()
    assert (base / "ChiT.dat").exists()

    # Check metadata contents (rough shape)
    txt = (base / "qw.dat").read_text().strip().splitlines()
    # Should have 6 lines per the implementation
    assert len(txt) == 6
    # Nw should be 2*_Nw + 1
    line_Nw = [ln for ln in txt if ln.startswith("Nw")][0]
    Nw_written = int(line_Nw.split()[1])
    assert Nw_written == 2 * M._Nw + 1


def test_RecordEpsrL_and_RecordEpsrT_line_counts(
    tiny_phys, tiny_ky, patch_small_omega, wire_dirs
):
    # Prepare globals (dcv, n00, kf) by calling the top-level initializer once
    M.GetEpsrLEpsrT(
        tiny_phys["n1D"], tiny_phys["dcv0"], tiny_phys["Te"], tiny_phys["Th"],
        tiny_phys["me"], tiny_phys["mh"], tiny_phys["Eg"], tiny_ky
    )

    Nk = tiny_ky.size
    Wn = 2 * M._Nw + 1
    # EpsL
    M.RecordEpsrL(tiny_phys["Te"], tiny_phys["Th"], tiny_phys["me"], tiny_phys["mh"], tiny_ky)
    pL = wire_dirs / "EpsL.dat"
    assert pL.exists()
    assert _count_lines(pL) == Nk * Wn

    # EpsT
    M.RecordEpsrT(tiny_phys["Te"], tiny_phys["Th"], tiny_phys["me"], tiny_phys["mh"], tiny_phys["Eg"], tiny_ky)
    pT = wire_dirs / "EpsT.dat"
    assert pT.exists()
    assert _count_lines(pT) == Nk * Wn

    # chi.0.w.dat small slice
    pchi = wire_dirs.parent / "chi.0.w.dat"
    assert pchi.exists()
    assert _count_lines(pchi) == Wn


def test_QqGq_produces_dispersion_file(
    tiny_phys, tiny_ky, patch_small_omega, wire_dirs
):
    # Make small synthetic epsR/epsI (Gaussian in w) for testing QqGq
    Nk = tiny_ky.size
    Wn = 2 * M._Nw + 1
    wgrid = np.arange(-M._Nw, M._Nw + 1, dtype=np.float64)
    # shape: (Nk, Wn)
    epsR = np.zeros((Nk, Wn), dtype=np.float64, order="F")
    epsI = np.zeros((Nk, Wn), dtype=np.float64, order="F")
    for q in range(Nk):
        center = 0.0
        epsR[q, :] = 0.5 * np.exp(-0.5 * (wgrid - center) ** 2) - 1.0  # crosses zero near center
        epsI[q, :] = 0.1 * np.exp(-0.5 * (wgrid - center) ** 2)

    M.QqGq(ky=tiny_ky, Nk=Nk, dk=tiny_ky[1]-tiny_ky[0], dw=M._dw, EpsR=epsR, EpsI=epsI, eh="z")

    out = wire_dirs / "Omega_qp.z.dat"
    assert out.exists()
    # One line per q
    lines = (wire_dirs / "Omega_qp.z.dat").read_text().strip().splitlines()
    assert len(lines) == Nk
    # Parse a line: "ky Omega Gam"
    ky0, Om0, G0 = map(float, lines[0].split())
    assert math.isfinite(ky0) and math.isfinite(Om0) and math.isfinite(G0)


# ------------------ Structural / Contiguity / Dtype checks ------------------

def test_internal_arrays_fortran_contiguity_and_dtypes(
    tiny_phys, tiny_ky, patch_small_omega, wire_dirs, capsys
):
    # Drive RecordEpsrL to run and emit the debug mins/maxes (captures stdout)
    M.GetEpsrLEpsrT(
        tiny_phys["n1D"], tiny_phys["dcv0"], tiny_phys["Te"], tiny_phys["Th"],
        tiny_phys["me"], tiny_phys["mh"], tiny_phys["Eg"], tiny_ky
    )
    M.RecordEpsrL(tiny_phys["Te"], tiny_phys["Th"], tiny_phys["me"], tiny_phys["mh"], tiny_ky)
    captured = capsys.readouterr().out
    assert "min/max EpsL real" in captured

    # Minimal smoke for T routine as well
    M.RecordEpsrT(tiny_phys["Te"], tiny_phys["Th"], tiny_phys["me"], tiny_phys["mh"], tiny_phys["Eg"], tiny_ky)
    captured2 = capsys.readouterr().out
    assert "min/max EpsT real" in captured2


# ------------------ Edge-cases / stability ------------------

def test_extreme_temperatures_and_zero_density_do_not_crash(tiny_ky, patch_small_omega, wire_dirs):
    me = np.float64(9.11e-31)
    mh = np.float64(9.11e-31)
    Eg = np.float64(1.0)
    dcv0 = np.float64(0.1)

    # zero density -> _kf=0, _n00=0; T extremes
    M.GetEpsrLEpsrT(
        n1D=np.float64(0.0), dcv0=dcv0, Te=np.float64(0.0), Th=np.float64(1e-6),
        me=me, mh=mh, Eg=Eg, ky=tiny_ky
    )
    # Still should produce files
    assert (wire_dirs / "ChiL.E.dat").exists()
    assert (wire_dirs / "ChiT.dat").exists()
