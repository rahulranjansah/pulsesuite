import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from helpers import Intensity, MathOps, Transforms, Grid, Interpolator, Stencils, Smoothers, Utils


# def testSech():
#     x = np.array([0.0, 1.0, -1.0])
#     expected = 1.0 / np.cosh(x)
#     assert np.allclose(Helpers.sech(x), expected)

# def testArg():
#     z = np.array([1+0j, 0+1j, -1+0j, 0-1j])
#     expected = np.angle(z)
#     assert np.allclose(Helpers.arg(z), expected)

# def testGauss():
#     x = np.array([0.0, 1.0, -1.0])
#     expected = np.exp(-x**2)
#     assert np.allclose(Helpers.gauss(x), expected)

# def testMagsq():
#     z = np.array([1+1j, 0+1j, 1+0j])
#     expected = np.real(z)**2 + np.imag(z)**2
#     assert np.allclose(Helpers.magsq(z), expected)

# def testConstrain():
#     x = np.array([-2, 0, 2, 4])
#     result = Helpers.constrain(x, 3, -1)
#     assert np.all(result >= -1) and np.all(result <= 3)

# def testAmpToInten():
#     e = np.array([1.0, 2.0])
#     n0 = 1.5
#     out = Helpers.ampToInten(e, n0)
#     assert out.shape == e.shape

# def testFldToInten():
#     e = np.array([1+1j, 2+0j])
#     n0 = 1.5
#     out = Helpers.fldToInten(e, n0)
#     assert out.shape == e.shape

# def testIntenToAmp():
#     inten = np.array([1.0, 4.0])
#     n0 = 1.5
#     out = Helpers.intenToAmp(inten, n0)
#     assert out.shape == inten.shape

# def testL2f():
#     lam = np.array([1.0, 2.0])
#     out = Helpers.l2f(lam)
#     assert out.shape == lam.shape

# def testL2w():
#     lam = np.array([1.0, 2.0])
#     out = Helpers.l2w(lam)
#     assert out.shape == lam.shape

# def testW2l():
#     w = np.array([1.0, 2.0])
#     out = Helpers.w2l(w)
#     assert out.shape == w.shape

# def testIsnan():
#     x = np.array([0.0, np.nan, 1.0])
#     out = Helpers.isnan(x)
#     assert out[1] == True and out[0] == False

# def testFactorial():
#     p = np.array([0, 1, 5])
#     expected = np.array([1, 1, math.factorial(5)])
#     assert np.allclose(Helpers.factorial(p), expected)

# def testUnwrap():
#     phase = np.array([0, np.pi, 2*np.pi, 3*np.pi])
#     out = Helpers.unwrap(phase)
#     assert out.shape == phase.shape

# def testLinearInterp():
#     x = np.array([0, 1, 2])
#     f = np.array([0, 10, 20])
#     x0 = 1.5
#     expected = 15.0
#     assert np.isclose(Helpers.linearInterp(f, x, x0), expected)

# def testGetSpaceArray():
#     arr = Helpers.getSpaceArray(5, 4.0)
#     assert arr.shape == (5,)
#     assert np.isclose(arr[0], -2.0)
#     assert np.isclose(arr[-1], 2.0)

# def testGetKArray():
#     arr = Helpers.getKArray(4, 2.0)
#     assert arr.shape == (4,)

# if __name__ == "__main__":
#     testSech()
#     testArg()
#     testGauss()
#     testMagsq()
#     testConstrain()
#     testAmpToInten()
#     testFldToInten()
#     testIntenToAmp()
#     testL2f()
#     testL2w()
#     testW2l()
#     testIsnan()
#     testFactorial()
#     testUnwrap()
#     testLinearInterp()
#     testGetSpaceArray()
#     testGetKArray()
#     print("All tests passed.")




EPS = 1e-9

def test_intensity_roundtrip_scalar():
    e = 2.5
    I = Intensity.amp_to_inten(e, n0=1.33)
    e_back = Intensity.inten_to_amp(I, n0=1.33)
    assert abs(e - e_back) < EPS

def test_intensity_field_array():
    arr = np.array([1+2j, -3+4j, 0-1j])
    I = Intensity.fld_to_inten(arr, n0=1.0)
    expected = (2.0 * 8.854187817e-12 * 299792458) * np.abs(arr)**2
    assert np.allclose(I, expected)

@pytest.mark.parametrize("x", [0.0, -1.5, 3.14])
def test_mathops_gauss_sech(x):
    # gauss
    g = MathOps.gauss(np.array([x]))[0]
    assert g == pytest.approx(np.exp(-x**2))
    # sech
    s = MathOps.sech(np.array([x]))[0]
    assert s == pytest.approx(1.0/np.cosh(x))

def test_mathops_magsq_and_constrain():
    data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    msq = MathOps.magsq(data + 1j*data)
    assert np.all(msq == (data**2 + data**2))
    clipped = MathOps.constrain(data, low=-1.0, high=1.0)
    assert clipped.min() >= -1.0 and clipped.max() <= 1.0

def test_transforms_inverse():
    lam = 0.75
    w = Transforms.l2w(lam)
    lam_back = Transforms.w2l(w)
    assert lam_back == pytest.approx(lam)

def test_grid_endpoints_and_length():
    arr3 = Grid.get_space_array(3, 10.0)
    assert np.allclose(arr3, np.array([-5.0, 0.0, 5.0]))
    k4 = Grid.get_k_array(4, length=2*np.pi)
    # should be [0,1, -2, -1] when scaled by dk=1
    assert np.allclose(k4, np.array([0,1,-2,-1]))

def test_interpolator_1d_and_complex():
    xp = np.array([0, 1, 2])
    fp_real = np.array([0.0, 2.0, 4.0])
    assert Interpolator.linear(1.5, xp, fp_real) == pytest.approx(3.0)
    fp_c = np.array([0+0j, 1+1j, 2+2j])
    out_c = Interpolator.linear_complex(0.5, xp, fp_c)
    assert out_c == pytest.approx(0.5+0.5j)

def test_interpolator_bilinear_and_trilinear():
    xp = yp = zp = np.linspace(0,1,2)
    fp = np.zeros((2,2,2))
    fp[1,1,1] = 8.0
    # at the corner (1,1,1) should return 8
    assert Interpolator.bilinear(1,1,xp,yp,fp[:,:,1]) == pytest.approx(8.0)
    assert Interpolator.trilinear(1,1,1,xp,yp,zp,fp) == pytest.approx(8.0)

def test_stencils_locator_and_gradient():
    x = np.linspace(0,10,11)
    assert Stencils.locator(x, 4.2) == 4
    f = np.array([0,1,4,9,16], dtype=float)
    grad = Stencils.gradient(f, 1.0)
    # simple derivative of x^2 is 2x, check interior
    assert grad[2] == pytest.approx(2*2, rel=1e-3)

def test_dfdt_index_and_full():
    f = np.arange(20.0)
    # central derivative ~ slope=1, central index
    assert Stencils.dfdt_index_real(f, 1.0, 10) == pytest.approx(1.0)
    full = Stencils.dfdt1d_real(f, 1.0)
    assert full.shape == f.shape

def test_smoothers_lax_and_no_lax():
    u = np.arange(27).reshape((3,3,3)).astype(complex)
    # at center
    val_lax = Smoothers.lax(u,1,1,1)
    neighbors = [u[0,1,1],u[2,1,1],u[1,0,1],u[1,2,1],u[1,1,0],u[1,1,2]]
    assert val_lax == pytest.approx(sum(neighbors)/6)
    assert Smoothers.no_lax(u,1,1,1) == u[1,1,1]

def test_utils_unwrap_and_factorial():
    ph = np.array([0,2*np.pi,4*np.pi])
    u = Utils.unwrap_phase(ph)
    assert u[-1] > u[0]
    assert Utils.factorial(6) == 720

def test_large_array_performance():
    big = np.random.rand(2_000_000)
    out = MathOps.gauss(big)
    assert out.size == big.size

if __name__ == "__main__":
    test_intensity_roundtrip_scalar()
    test_intensity_field_array()
    test_mathops_gauss_sech()
    test_mathops_magsq_and_constrain()
    test_transforms_inverse()
    test_grid_endpoints_and_length()
    test_interpolator_1d_and_complex()
    test_interpolator_bilinear_and_trilinear()
    test_stencils_locator_and_gradient()
    test_dfdt_index_and_full()
    test_smoothers_lax_and_no_lax()
    test_utils_unwrap_and_factorial()
    test_large_array_performance()
    print("All tests passed.")