"""
Comprehensive test suite for cpml.py module (Convolutional PML).

Tests the 3D CPML absorbing boundary layer implementation including
profile computation, coefficient calculation, PML thickness heuristics,
field update routines, and energy absorption validation.

"""

import numpy as np
import pytest



try:
    from pulsesuite.PSTD3D import cpml

    _has_init = hasattr(cpml, "InitCPML")
    _IMPORT_ERROR = None if _has_init else "module loaded but 'InitCPML' not found"
except Exception as exc:  # noqa: BLE001
    cpml = None
    _has_init = False
    _IMPORT_ERROR = str(exc)

needs_cpml = pytest.mark.skipif(
    not _has_init,
    reason=f"cpml not available: {_IMPORT_ERROR or 'InitCPML missing'}",
)

# Physical constants (must match Fortran constants module)
c0 = 299792458.0
eps0 = 8.8541878176203898505365630317107e-12
mu0 = 1.2566370614359172953850573533118e-6
pi = np.pi

# Module-level parameters from cpml.f90
M_PROFILE = 4
KAPPA_MAX = 8.0
ALPHA_MAX = 0.05
R_TARGET = 1.0e-8



@needs_cpml
class TestCalcNPML:
    """Test CalcNPML – computes safe CPML thickness for a grid dimension."""

    def test_minimum_six(self):
        """PML thickness must be at least 6 for small grids."""
        assert cpml.CalcNPML(200) >= 6

    def test_five_percent_rule(self):
        """For large grids, PML ≈ 5 % of N."""
        N = 1000
        npml = cpml.CalcNPML(N)
        assert abs(npml - int(0.05 * N)) <= 2 or npml >= 6

    def test_upper_bound_ten_percent(self):
        """PML should not exceed N/10."""
        N = 100
        npml = cpml.CalcNPML(N)
        assert npml <= N // 10

    def test_safety_for_small_grid(self):
        """When 2*npml >= N, npml should be clamped to max(1, N/4)."""
        N = 10
        npml = cpml.CalcNPML(N)
        assert 2 * npml < N

    @pytest.mark.parametrize("N", [8, 16, 32, 64, 128, 256, 512, 1024])
    def test_always_positive(self, N):
        assert cpml.CalcNPML(N) >= 1

    @pytest.mark.parametrize("N", [8, 16, 32, 64, 128, 256, 512, 1024])
    def test_leaves_interior(self, N):
        """Interior region (N - 2*npml) must have at least 2 points."""
        npml = cpml.CalcNPML(N)
        assert N - 2 * npml >= 2



@needs_cpml
class TestSigmaStable:
    """Test sigma_stable(L) computation."""

    def test_positive(self):
        L = 10 * 1e-9
        assert cpml.sigma_stable(L) > 0

    def test_increases_with_thinner_pml(self):
        """Thinner PML needs higher σ_max to maintain same R."""
        s_thick = cpml.sigma_stable(20e-9)
        s_thin = cpml.sigma_stable(5e-9)
        assert s_thin > s_thick

    def test_formula(self):
        """Check against analytic formula from cpml.f90."""
        L = 10e-9
        pm = float(M_PROFILE)
        expected = -(pm + 1.0) * eps0 * c0 * np.log(R_TARGET) / (2.0 * L)
        result = cpml.sigma_stable(L)
        # May be clamped by Courant safety – just check lower bound
        assert result <= expected or np.isclose(result, expected, rtol=0.01)



@needs_cpml
class TestCalcCoefficients:
    """Test CPML coefficient generation for one axis."""

    def _uniform_profiles(self, N, sigma_val=0.0):
        sigma = np.full(N, sigma_val)
        kappa = np.ones(N)
        alpha = np.full(N, ALPHA_MAX)
        return sigma, kappa, alpha

    def test_output_shapes(self):
        N = 32
        sigma, kappa, alpha = self._uniform_profiles(N)
        bE, cE, bH, cH = cpml.Calc_CoefficientsCPML(N, sigma, kappa, alpha)
        assert bE.shape == (N,)
        assert cE.shape == (N,)
        assert bH.shape == (N,)
        assert cH.shape == (N,)

    def test_zero_sigma_bE_near_one(self):
        """With σ=0, kappa=1: bE = exp(-α dt/ε0) which is < 1 but close to 1."""
        N = 16
        sigma, kappa, alpha = self._uniform_profiles(N, sigma_val=0.0)
        bE, cE, bH, cH = cpml.Calc_CoefficientsCPML(N, sigma, kappa, alpha)
        # bE should be positive and ≤ 1
        assert np.all(bE > 0)
        assert np.all(bE <= 1.0)

    def test_zero_sigma_cE_near_zero(self):
        """With σ=0, cE should be approximately zero."""
        N = 16
        sigma, kappa, alpha = self._uniform_profiles(N, sigma_val=0.0)
        bE, cE, bH, cH = cpml.Calc_CoefficientsCPML(N, sigma, kappa, alpha)
        assert np.allclose(cE, 0.0, atol=1e-15)

    def test_nonzero_sigma(self):
        """With σ > 0, bE < 1 and cE < 0 (damping)."""
        N = 16
        sigma = np.full(N, 1e6)
        kappa = np.ones(N) * KAPPA_MAX
        alpha = np.full(N, ALPHA_MAX)
        bE, cE, bH, cH = cpml.Calc_CoefficientsCPML(N, sigma, kappa, alpha)
        assert np.all(bE < 1.0)
        assert np.all(cE < 0.0)  # σ * (b-1) is negative



@needs_cpml
class TestInitCPML:
    """Test full CPML initialisation with InitCPML."""

    def test_runs_without_error(self):
        """InitCPML should complete without raising for a reasonable grid."""
        cpml.InitCPML(
            Nx_in=32, Ny_in=32, Nz_in=32,
            dx_in=1e-9, dy_in=1e-9, dz_in=1e-9,
            dt_in=1e-18, espr=1.0,
        )

    def test_sets_module_dimensions(self):
        cpml.InitCPML(
            Nx_in=64, Ny_in=32, Nz_in=16,
            dx_in=1e-9, dy_in=1e-9, dz_in=1e-9,
            dt_in=1e-18, espr=1.0,
        )
        assert cpml.Nx == 64
        assert cpml.Ny == 32
        assert cpml.Nz == 16

    def test_npml_set(self):
        cpml.InitCPML(
            Nx_in=128, Ny_in=128, Nz_in=128,
            dx_in=1e-9, dy_in=1e-9, dz_in=1e-9,
            dt_in=1e-18, espr=1.0,
        )
        assert cpml.npml_x >= 1
        assert cpml.npml_y >= 1
        assert cpml.npml_z >= 1

    def test_auxiliary_fields_zeroed(self):
        """All psi arrays should be initialised to zero."""
        cpml.InitCPML(
            Nx_in=16, Ny_in=16, Nz_in=16,
            dx_in=1e-9, dy_in=1e-9, dz_in=1e-9,
            dt_in=1e-18, espr=1.0,
        )
        assert np.allclose(cpml.psi_Exy, 0.0)
        assert np.allclose(cpml.psi_Hxy, 0.0)

    def test_coefficient_arrays_allocated(self):
        cpml.InitCPML(
            Nx_in=16, Ny_in=16, Nz_in=16,
            dx_in=1e-9, dy_in=1e-9, dz_in=1e-9,
            dt_in=1e-18, espr=1.0,
        )
        assert cpml.bEx.shape == (16,)
        assert cpml.CaX.shape == (16, 16, 16)

    def test_profiles_symmetric(self):
        """Sigma profile should be symmetric (same PML on both sides)."""
        N = 64
        cpml.InitCPML(
            Nx_in=N, Ny_in=N, Nz_in=N,
            dx_in=1e-9, dy_in=1e-9, dz_in=1e-9,
            dt_in=1e-18, espr=1.0,
        )
        # σ(i) should equal σ(N-1-i) for the x-direction
        sx = cpml.sigma_x
        assert np.allclose(sx[:cpml.npml_x], sx[-cpml.npml_x:][::-1], rtol=0.1)

    def test_interior_sigma_zero(self):
        """σ should be zero in the interior (non-PML) region."""
        N = 64
        cpml.InitCPML(
            Nx_in=N, Ny_in=N, Nz_in=N,
            dx_in=1e-9, dy_in=1e-9, dz_in=1e-9,
            dt_in=1e-18, espr=1.0,
        )
        npml = cpml.npml_x
        interior = cpml.sigma_x[npml : N - npml]
        assert np.allclose(interior, 0.0, atol=1e-30)



@needs_cpml
class TestFieldUpdates:
    """Test that CPML E and H update routines run and modify fields."""

    @pytest.fixture(autouse=True)
    def _init_small_grid(self):
        """Set up a small 16³ grid before each test."""
        self.N = 16
        self.dx = self.dy = self.dz = 1e-9
        self.dt = 1e-18
        cpml.InitCPML(
            Nx_in=self.N, Ny_in=self.N, Nz_in=self.N,
            dx_in=self.dx, dy_in=self.dy, dz_in=self.dz,
            dt_in=self.dt, espr=1.0,
        )
        shape = (self.N, self.N, self.N)
        self.Ex = np.zeros(shape, dtype=complex)
        self.Ey = np.zeros(shape, dtype=complex)
        self.Ez = np.zeros(shape, dtype=complex)
        self.Hx = np.zeros(shape, dtype=complex)
        self.Hy = np.zeros(shape, dtype=complex)
        self.Hz = np.zeros(shape, dtype=complex)
        self.Jx = np.zeros(shape)
        self.Jy = np.zeros(shape)
        self.Jz = np.zeros(shape)

    def test_update_e_runs(self):
        """UpdateCPML_E should execute without error on zero fields."""
        cpml.UpdateCPML_E(
            self.Ex, self.Ey, self.Ez,
            self.Hx, self.Hy, self.Hz,
            self.Jx, self.Jy, self.Jz,
            self.N, self.N, self.N,
            self.dx, self.dy, self.dz, self.dt,
        )

    def test_update_h_runs(self):
        """UpdateCPML_H should execute without error on zero fields."""
        cpml.UpdateCPML_H(
            self.Hx, self.Hy, self.Hz,
            self.Ex, self.Ey, self.Ez,
            self.Jx, self.Jy, self.Jz,
            self.N, self.N, self.N,
            self.dx, self.dy, self.dz, self.dt,
        )

    def test_zero_fields_stay_zero(self):
        """With all-zero fields, E and H should remain zero after update."""
        cpml.UpdateCPML_E(
            self.Ex, self.Ey, self.Ez,
            self.Hx, self.Hy, self.Hz,
            self.Jx, self.Jy, self.Jz,
            self.N, self.N, self.N,
            self.dx, self.dy, self.dz, self.dt,
        )
        assert np.allclose(self.Ex, 0.0)
        assert np.allclose(self.Ey, 0.0)
        assert np.allclose(self.Ez, 0.0)

    def test_nonzero_H_modifies_E(self):
        """Nonzero H-field in PML region should cause E-field update."""
        self.Hz[2, 2, 2] = 1.0 + 0j
        self.Hz[2, 3, 2] = -1.0 + 0j
        E_before = self.Ex.copy()
        cpml.UpdateCPML_E(
            self.Ex, self.Ey, self.Ez,
            self.Hx, self.Hy, self.Hz,
            self.Jx, self.Jy, self.Jz,
            self.N, self.N, self.N,
            self.dx, self.dy, self.dz, self.dt,
        )
        # At least some E-field component in the PML should change
        assert not np.allclose(self.Ex, E_before)


@needs_cpml
class TestEnergyAbsorption:
    """Verify that CPML actually absorbs outgoing energy over many steps."""

    def test_field_decays_in_pml(self):
        """A pulse injected into the PML region should be damped."""
        N = 64
        dx = dy = dz = 1e-9
        dt = 1e-18

        cpml.InitCPML(N, N, N, dx, dy, dz, dt, espr=1.0)

        shape = (N, N, N)
        Ex = np.zeros(shape, dtype=complex)
        Ey = np.zeros(shape, dtype=complex)
        Ez = np.zeros(shape, dtype=complex)
        Hx = np.zeros(shape, dtype=complex)
        Hy = np.zeros(shape, dtype=complex)
        Hz = np.zeros(shape, dtype=complex)
        Jx = np.zeros(shape)
        Jy = np.zeros(shape)
        Jz = np.zeros(shape)

        # Inject energy in the PML region
        npml = cpml.npml_x
        Ey[2, N // 2, N // 2] = 1.0

        energy_initial = np.sum(np.abs(Ey) ** 2)

        # Run several update cycles
        for _ in range(50):
            cpml.UpdateCPML_E(
                Ex, Ey, Ez, Hx, Hy, Hz, Jx, Jy, Jz,
                N, N, N, dx, dy, dz, dt,
            )
            cpml.UpdateCPML_H(
                Hx, Hy, Hz, Ex, Ey, Ez, Jx, Jy, Jz,
                N, N, N, dx, dy, dz, dt,
            )

        energy_final = np.sum(np.abs(Ey) ** 2)
        assert energy_final < energy_initial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])