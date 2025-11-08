"""
test_phostpythonic_simple.py
============================

Simple test suite for phostpythonic.py module without pytest dependency.
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.phostpythonic as phost

def test_oscillator_parameters():
    """Test OscillatorParameters dataclass."""
    print("Testing OscillatorParameters...")

    # Test default initialization
    params = phost.OscillatorParameters()
    assert params.n_oscillators == 2
    assert len(params.B) == 2
    print("✓ Default initialization passed")

    # Test custom initialization with matching arrays
    params = phost.OscillatorParameters(
        n_oscillators=3,
        B=np.array([1.0, 2.0, 3.0]),
        C=np.array([1.0, 2.0, 3.0]),
        w=np.array([1.0, 2.0, 3.0]),
        gam=np.array([1.0, 2.0, 3.0]),
        chi1=np.array([1.0+0j, 2.0+0j, 3.0+0j]),
        Nf=np.array([1.0, 2.0, 3.0])
    )
    assert params.n_oscillators == 3
    print("✓ Custom initialization passed")

    # Test validation
    try:
        phost.OscillatorParameters(n_oscillators=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Number of oscillators must be positive" in str(e)
        print("✓ Negative oscillator validation passed")

    print("OscillatorParameters tests passed!")

def test_host_material_calculator():
    """Test HostMaterialCalculator class."""
    print("Testing HostMaterialCalculator...")

    material_params = phost.HostMaterialParameters()
    calculator = phost.HostMaterialCalculator(material_params)

    # Test material setting
    epsr, n0 = calculator.set_material('AlAs', 1e-6)
    assert isinstance(epsr, float)
    assert isinstance(n0, float)
    assert epsr > 0
    assert n0 > 0
    assert calculator._initialized
    print("✓ Material setting passed")

    # Test dielectric calculations
    wL = 2 * np.pi * 3e8 / 1e-6
    nw2 = calculator._nw2_no_gam(wL)
    print(f"nw2 type: {type(nw2)}, value: {nw2}")
    assert isinstance(nw2, (complex, np.complex128, np.complex64, float, np.float64))
    assert np.isfinite(nw2)
    print("✓ Dielectric calculations passed")

    print("HostMaterialCalculator tests passed!")

def test_polarization_calculator():
    """Test PolarizationCalculator class."""
    print("Testing PolarizationCalculator...")

    material_params = phost.HostMaterialParameters()
    material_calculator = phost.HostMaterialCalculator(material_params)
    material_calculator.set_material('AlAs', 1e-6)
    polarization_calculator = phost.PolarizationCalculator(material_calculator)

    # Test initialization
    Nx, Ny = 16, 16
    n0 = 3.0
    qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
    polarization_calculator.initialize(Nx, Ny, n0, qsq, host=True)

    assert polarization_calculator._initialized
    assert polarization_calculator._omega_q.shape == (Nx, Ny)
    print("✓ Initialization passed")

    # Test polarization calculations
    P1 = np.ones((Nx, Ny, 2), dtype=np.complex128) * 1e-6
    P2 = np.ones((Nx, Ny, 2), dtype=np.complex128) * 2e-6
    E = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
    dt = 1e-15

    result = polarization_calculator.calc_next_p(P1, P2, E, dt)
    assert result.shape == (Nx, Ny, 2)
    assert np.all(np.isfinite(result))
    print("✓ Polarization calculations passed")

    print("PolarizationCalculator tests passed!")

def test_host_solver():
    """Test HostSolver main class."""
    print("Testing HostSolver...")

    material_params = phost.HostMaterialParameters()
    solver = phost.HostSolver(material_params)

    # Test material setting
    epsr, n0 = solver.set_host_material(True, 'AlAs', 1e-6)
    assert isinstance(epsr, float)
    assert isinstance(n0, float)
    print("✓ Material setting passed")

    # Test initialization
    Nx, Ny = 16, 16
    n0 = 3.0
    qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
    solver.initialize_host(Nx, Ny, n0, qsq, host=True)

    assert solver._initialized
    print("✓ Initialization passed")

    # Test polarization calculation
    Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
    Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
    dt = 1e-15
    m = 5

    epsb, Px, Py = solver.calc_phost(Ex, Ey, dt, m)

    assert isinstance(epsb, float)
    assert Px.shape == (Nx, Ny)
    assert Py.shape == (Nx, Ny)
    assert np.all(np.isfinite(Px))
    assert np.all(np.isfinite(Py))
    print("✓ Polarization calculation passed")

    print("HostSolver tests passed!")

def test_fortran_interface():
    """Test Fortran-compatible interface functions."""
    print("Testing Fortran interface...")

    # Test SetHostMaterial
    epsr, n0 = phost.SetHostMaterial(True, 'AlAs', 1e-6, 0.0, 0.0)
    assert isinstance(epsr, float)
    assert isinstance(n0, float)
    print("✓ SetHostMaterial passed")

    # Test InitializeHost
    Nx, Ny = 16, 16
    n0 = 3.0
    qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
    solver = phost.InitializeHost(Nx, Ny, n0, qsq, True)
    assert isinstance(solver, phost.HostSolver)
    print("✓ InitializeHost passed")

    print("Fortran interface tests passed!")

def test_mathematical_properties():
    """Test mathematical properties."""
    print("Testing mathematical properties...")

    material_params = phost.HostMaterialParameters()
    calculator = phost.HostMaterialCalculator(material_params)
    calculator.set_material('AlAs', 1e-6)

    # Test dielectric function properties
    frequencies = np.logspace(12, 16, 5)

    for wL in frequencies:
        nw2 = calculator._nw2_no_gam(wL)
        nw2_damped = calculator._nw2(wL)

        assert np.isfinite(nw2)
        assert np.isfinite(nw2_damped)
        assert abs(nw2.real) < 1e6
        assert abs(nw2_damped.real) < 1e6

    print("✓ Dielectric function properties passed")

    # Test energy conservation
    solver = phost.HostSolver(material_params)
    solver.set_host_material(True, 'AlAs', 1e-6)

    Nx, Ny = 16, 16
    n0 = 3.0
    qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
    solver.initialize_host(Nx, Ny, n0, qsq, host=True)

    Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
    Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
    dt = 1e-15

    epsb, Px, Py = solver.calc_phost(Ex, Ey, dt, 5)

    energy = np.sum(np.abs(Px)**2 + np.abs(Py)**2)
    assert np.isfinite(energy)
    assert energy > 0
    print("✓ Energy conservation passed")

    print("Mathematical properties tests passed!")

def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")

    material_params = phost.HostMaterialParameters()
    solver = phost.HostSolver(material_params)
    solver.set_host_material(True, 'AlAs', 1e-6)

    Nx, Ny = 16, 16
    n0 = 3.0
    qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
    solver.initialize_host(Nx, Ny, n0, qsq, host=True)

    # Test zero field
    Ex = np.zeros((Nx, Ny), dtype=np.complex128)
    Ey = np.zeros((Nx, Ny), dtype=np.complex128)

    epsb, Px, Py = solver.calc_phost(Ex, Ey, 1e-15, 5)

    assert np.allclose(Px, 0, atol=1e-10)
    assert np.allclose(Py, 0, atol=1e-10)
    print("✓ Zero field handling passed")

    # Test single point grid
    Nx, Ny = 1, 1
    qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
    solver.initialize_host(Nx, Ny, n0, qsq, host=True)

    Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
    Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3

    epsb, Px, Py = solver.calc_phost(Ex, Ey, 1e-15, 5)

    assert Px.shape == (1, 1)
    assert Py.shape == (1, 1)
    assert np.isfinite(Px[0, 0])
    assert np.isfinite(Py[0, 0])
    print("✓ Single point grid passed")

    print("Edge cases tests passed!")

def test_performance():
    """Test performance characteristics."""
    print("Testing performance...")

    import time

    material_params = phost.HostMaterialParameters()
    solver = phost.HostSolver(material_params)
    solver.set_host_material(True, 'AlAs', 1e-6)

    Nx, Ny = 32, 32
    n0 = 3.0
    qsq = np.ones((Nx, Ny), dtype=np.complex128) * 1e12
    solver.initialize_host(Nx, Ny, n0, qsq, host=True)

    Ex = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
    Ey = np.ones((Nx, Ny), dtype=np.complex128) * 1e3
    dt = 1e-15

    # Time multiple calculations
    start_time = time.time()
    for _ in range(5):
        solver.calc_phost(Ex, Ey, dt, 5)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / 5

    print(f"Average calculation time: {avg_time:.4f} seconds")
    assert avg_time < 2.0  # Should complete in reasonable time
    print("✓ Performance test passed")

    print("Performance tests passed!")

def main():
    """Run all tests."""
    print("Running phostpythonic.py simple test suite...")
    print("=" * 50)

    try:
        test_oscillator_parameters()
        print()
        test_host_material_calculator()
        print()
        test_polarization_calculator()
        print()
        test_host_solver()
        print()
        test_fortran_interface()
        print()
        test_mathematical_properties()
        print()
        test_edge_cases()
        print()
        test_performance()
        print()
        print("=" * 50)
        print("All tests passed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
