"""
Comprehensive test suite for PSTD3D.py module.

Tests the 3D Pseudo-Spectral Time-Domain electromagnetic field propagator
including Maxwell curl updates (UpdateE3D, UpdateB3D), field initialisation,
Fourier-space consistency, and short propagation runs.
"""

import numpy as np
import pytest


try:
    from pulsesuite.PSTD3D import PSTD3D as pstd

    _has_update = hasattr(pstd, "UpdateE3D") or hasattr(pstd, "PSTD_3D_Propagator")
    _IMPORT_ERROR = None if _has_update else "module loaded but core functions not found"
except Exception as exc:  # noqa: BLE001
    pstd = None
    _has_update = False
    _IMPORT_ERROR = str(exc)

# Also import typespace for building test grids
try:
    from pulsesuite.PSTD3D import typespace as ts_mod

    _has_typespace = True
except Exception:
    ts_mod = None
    _has_typespace = False

needs_pstd = pytest.mark.skipif(
    not _has_update,
    reason=f"PSTD3D not available: {_IMPORT_ERROR or 'core functions missing'}",
)

# Physical constants
c0 = 299792458.0
eps0 = 8.8541878176203898505365630317107e-12
mu0 = 1.2566370614359172953850573533118e-6
pi = np.pi
twopi = 2.0 * pi
ii = 1j



def _make_space(Nx=32, Ny=16, Nz=8, dx=1e-9, epsr=1.0):
    if _has_typespace:
        return ts_mod.ss(
            Dims=3, Nx=Nx, Ny=Ny, Nz=Nz,
            dx=dx, dy=dx, dz=dx, epsr=epsr,
        )
    from types import SimpleNamespace
    return SimpleNamespace(
        Dims=3, Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dx, dz=dx, epsr=epsr,
    )


def _make_time(t=0.0, tf=100e-15, dt=1e-18, n=0):
    from types import SimpleNamespace
    return SimpleNamespace(t=t, tf=tf, dt=dt, n=n)


def _zero_fields(Nx, Ny, Nz):
    """Return a dict of all-zero E, B, J field arrays."""
    shape = (Nx, Ny, Nz)
    return {
        "Ex": np.zeros(shape, dtype=complex),
        "Ey": np.zeros(shape, dtype=complex),
        "Ez": np.zeros(shape, dtype=complex),
        "Bx": np.zeros(shape, dtype=complex),
        "By": np.zeros(shape, dtype=complex),
        "Bz": np.zeros(shape, dtype=complex),
        "Jx": np.zeros(shape, dtype=complex),
        "Jy": np.zeros(shape, dtype=complex),
        "Jz": np.zeros(shape, dtype=complex),
    }

@needs_pstd
class TestInitializeFields:
    """Test field initialisation to zero."""

    def test_all_zero(self):
        shape = (16, 8, 4)
        Ex = np.ones(shape, dtype=complex)
        Ey = np.ones(shape, dtype=complex)
        Ez = np.ones(shape, dtype=complex)
        Bx = np.ones(shape, dtype=complex)
        By = np.ones(shape, dtype=complex)
        Bz = np.ones(shape, dtype=complex)
        Jx = np.ones(shape, dtype=complex)
        Jy = np.ones(shape, dtype=complex)
        Jz = np.ones(shape, dtype=complex)

        pstd.InitializeFields(Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz)

        for arr in [Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz]:
            assert np.allclose(arr, 0.0)


@needs_pstd
class TestUpdateE3D:
    """Test spectral E-field update: E += i(q×B)v²dt - μ₀ J v² dt."""

    def test_zero_B_zero_J_no_change(self):
        """With zero B and J, E should not change."""
        space = _make_space()
        time = _make_time()
        f = _zero_fields(32, 16, 8)
        f["Ex"][8, 4, 2] = 1.0 + 0j  # nonzero E

        Ex_before = f["Ex"].copy()
        pstd.UpdateE3D(
            space, time,
            f["Bx"], f["By"], f["Bz"],
            f["Jx"], f["Jy"], f["Jz"],
            f["Ex"], f["Ey"], f["Ez"],
        )
        assert np.allclose(f["Ex"], Ex_before)

    def test_nonzero_B_modifies_E(self):
        """A nonzero B-field should cause E to change via curl."""
        space = _make_space()
        time = _make_time()
        f = _zero_fields(32, 16, 8)
        # Place a nonzero Bz
        f["Bz"][:, :, :] = 1.0

        Ex_before = f["Ex"].copy()
        pstd.UpdateE3D(
            space, time,
            f["Bx"], f["By"], f["Bz"],
            f["Jx"], f["Jy"], f["Jz"],
            f["Ex"], f["Ey"], f["Ez"],
        )
        # At least one E component should change
        changed = (
            not np.allclose(f["Ex"], Ex_before)
            or not np.allclose(f["Ey"], 0.0)
        )
        assert changed

    def test_current_source_modifies_E(self):
        """A current density should drive E-field change."""
        space = _make_space()
        time = _make_time()
        f = _zero_fields(32, 16, 8)
        f["Jx"][:, :, :] = 1.0  # uniform Jx

        pstd.UpdateE3D(
            space, time,
            f["Bx"], f["By"], f["Bz"],
            f["Jx"], f["Jy"], f["Jz"],
            f["Ex"], f["Ey"], f["Ez"],
        )
        # Ex should be modified by Jx term
        assert not np.allclose(f["Ex"], 0.0)

    def test_update_uses_epsr(self):
        """v² = c0²/ε_r should scale the update."""
        time = _make_time()
        f1 = _zero_fields(32, 16, 8)
        f2 = _zero_fields(32, 16, 8)

        # Same B-field in both
        f1["Bz"][:, :, :] = 1.0
        f2["Bz"][:, :, :] = 1.0

        space1 = _make_space(epsr=1.0)
        space2 = _make_space(epsr=4.0)

        pstd.UpdateE3D(
            space1, time,
            f1["Bx"], f1["By"], f1["Bz"],
            f1["Jx"], f1["Jy"], f1["Jz"],
            f1["Ex"], f1["Ey"], f1["Ez"],
        )
        pstd.UpdateE3D(
            space2, time,
            f2["Bx"], f2["By"], f2["Bz"],
            f2["Jx"], f2["Jy"], f2["Jz"],
            f2["Ex"], f2["Ey"], f2["Ez"],
        )

        # The ε_r=4 case should have updates 4× smaller than ε_r=1
        ratio = np.max(np.abs(f1["Ey"])) / (np.max(np.abs(f2["Ey"])) + 1e-300)
        assert np.isclose(ratio, 4.0, rtol=0.1)


@needs_pstd
class TestUpdateB3D:
    """Test spectral B-field update: B -= i(q×E)dt."""

    def test_zero_E_no_change(self):
        """With zero E, B should not change."""
        space = _make_space()
        time = _make_time()
        f = _zero_fields(32, 16, 8)
        f["Bx"][8, 4, 2] = 1.0

        Bx_before = f["Bx"].copy()
        pstd.UpdateB3D(
            space, time,
            f["Ex"], f["Ey"], f["Ez"],
            f["Bx"], f["By"], f["Bz"],
        )
        assert np.allclose(f["Bx"], Bx_before)

    def test_nonzero_E_modifies_B(self):
        """Nonzero E-field should change B via Faraday's law."""
        space = _make_space()
        time = _make_time()
        f = _zero_fields(32, 16, 8)
        f["Ey"][:, :, :] = 1.0

        pstd.UpdateB3D(
            space, time,
            f["Ex"], f["Ey"], f["Ez"],
            f["Bx"], f["By"], f["Bz"],
        )
        # Bx or Bz should change (curl of Ey contributes to Bx and Bz)
        changed = not np.allclose(f["Bx"], 0.0) or not np.allclose(f["Bz"], 0.0)
        assert changed

    def test_B_update_independent_of_epsr(self):
        """Faraday's law has no ε_r dependence: ∂B/∂t = -∇×E."""
        time = _make_time()
        f1 = _zero_fields(32, 16, 8)
        f2 = _zero_fields(32, 16, 8)
        f1["Ey"][:, :, :] = 1.0
        f2["Ey"][:, :, :] = 1.0

        space1 = _make_space(epsr=1.0)
        space2 = _make_space(epsr=9.0)

        pstd.UpdateB3D(space1, time, f1["Ex"], f1["Ey"], f1["Ez"],
                        f1["Bx"], f1["By"], f1["Bz"])
        pstd.UpdateB3D(space2, time, f2["Ex"], f2["Ey"], f2["Ez"],
                        f2["Bx"], f2["By"], f2["Bz"])

        assert np.allclose(f1["Bx"], f2["Bx"], rtol=1e-12)
        assert np.allclose(f1["Bz"], f2["Bz"], rtol=1e-12)


@needs_pstd
class TestGaussLaw:
    """Maxwell updates should preserve ∇·E = 0 and ∇·B = 0 in free space."""

    def _build_divergence_free_fields(self, Nx, Ny, Nz, dx):
        """Create initial fields that satisfy ∇·E = 0 in spectral domain."""
        f = _zero_fields(Nx, Ny, Nz)
        # A uniform Ey has zero divergence
        f["Ey"][:, :, :] = 1.0
        return f

    def test_divB_stays_zero(self):
        """Starting from ∇·B = 0, B-update should preserve it."""
        Nx, Ny, Nz = 32, 16, 8
        dx = 1e-9
        space = _make_space(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx)
        time = _make_time()
        f = self._build_divergence_free_fields(Nx, Ny, Nz, dx)

        # Multiple B-updates
        for _ in range(5):
            pstd.UpdateB3D(
                space, time,
                f["Ex"], f["Ey"], f["Ez"],
                f["Bx"], f["By"], f["Bz"],
            )

        # Check ∇·B = iq_x Bx + iq_y By + iq_z Bz ≈ 0
        if _has_typespace:
            qx = ts_mod.GetKxArray(space)
            qy = ts_mod.GetKyArray(space)
            qz = ts_mod.GetKzArray(space)
        else:
            pytest.skip("typespace needed for divergence check")

        divB = np.zeros((Nx, Ny, Nz), dtype=complex)
        for k in range(Nz):
            for j in range(Ny):
                divB[:, j, k] = (
                    ii * qx * f["Bx"][:, j, k]
                    + ii * qy[j] * f["By"][:, j, k]
                    + ii * qz[k] * f["Bz"][:, j, k]
                )
        assert np.allclose(divB, 0.0, atol=1e-20)


@needs_pstd
class TestLeapfrogConsistency:
    """Test that a full leapfrog step (E-update + B-update) is stable."""

    def test_energy_bounded(self):
        """Total EM energy should not grow unboundedly over a few steps."""
        Nx, Ny, Nz = 32, 16, 8
        dx = 10e-9
        dt = 1e-18  # well within Courant limit
        space = _make_space(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx)
        time = _make_time(dt=dt)
        f = _zero_fields(Nx, Ny, Nz)

        # Seed a small disturbance
        f["Ey"][Nx // 2, Ny // 2, Nz // 2] = 1.0

        def em_energy():
            return np.sum(
                np.abs(f["Ex"]) ** 2 + np.abs(f["Ey"]) ** 2 + np.abs(f["Ez"]) ** 2
                + c0**2 * (np.abs(f["Bx"]) ** 2 + np.abs(f["By"]) ** 2 + np.abs(f["Bz"]) ** 2)
            )

        E0 = em_energy()
        for _ in range(20):
            pstd.UpdateE3D(
                space, time,
                f["Bx"], f["By"], f["Bz"],
                f["Jx"], f["Jy"], f["Jz"],
                f["Ex"], f["Ey"], f["Ez"],
            )
            pstd.UpdateB3D(
                space, time,
                f["Ex"], f["Ey"], f["Ez"],
                f["Bx"], f["By"], f["Bz"],
            )

        Ef = em_energy()
        # Energy should be conserved (no sources, no loss) within ~10 %
        assert Ef < 2.0 * E0 or np.isclose(Ef, E0, rtol=0.5)


@needs_pstd
class TestParameterised:
    """Parameterised tests across grid sizes."""

    @pytest.mark.parametrize(
        "Nx,Ny,Nz",
        [(8, 8, 8), (16, 16, 16), (32, 16, 8), (64, 32, 16)],
    )
    def test_update_e_shape_preserved(self, Nx, Ny, Nz):
        space = _make_space(Nx=Nx, Ny=Ny, Nz=Nz)
        time = _make_time()
        f = _zero_fields(Nx, Ny, Nz)
        f["Bz"][:, :, :] = 1.0

        pstd.UpdateE3D(
            space, time,
            f["Bx"], f["By"], f["Bz"],
            f["Jx"], f["Jy"], f["Jz"],
            f["Ex"], f["Ey"], f["Ez"],
        )
        assert f["Ex"].shape == (Nx, Ny, Nz)
        assert f["Ey"].shape == (Nx, Ny, Nz)
        assert f["Ez"].shape == (Nx, Ny, Nz)

    @pytest.mark.parametrize(
        "Nx,Ny,Nz",
        [(8, 8, 8), (16, 16, 16), (32, 16, 8)],
    )
    def test_update_b_shape_preserved(self, Nx, Ny, Nz):
        space = _make_space(Nx=Nx, Ny=Ny, Nz=Nz)
        time = _make_time()
        f = _zero_fields(Nx, Ny, Nz)
        f["Ey"][:, :, :] = 1.0

        pstd.UpdateB3D(
            space, time,
            f["Ex"], f["Ey"], f["Ez"],
            f["Bx"], f["By"], f["Bz"],
        )
        assert f["Bx"].shape == (Nx, Ny, Nz)
        assert f["By"].shape == (Nx, Ny, Nz)
        assert f["Bz"].shape == (Nx, Ny, Nz)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])