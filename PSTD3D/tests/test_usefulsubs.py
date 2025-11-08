import sys
import os
import numpy as np
import pytest
from unittest.mock import patch

# Add PSTD3D to sys.path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import usefulsubs

def test_GaussDelta():
    a, b = 0.0, 1.0
    result = usefulsubs.GaussDelta(a, b)
    assert np.isclose(result, 1.0/np.sqrt(np.pi))

def test_delta():
    assert np.allclose(usefulsubs.delta(0.0), 1.0)
    assert np.allclose(usefulsubs.delta(np.pi), 1.0)
    assert np.allclose(usefulsubs.delta(2*np.pi), 0.0)

def test_kdel():
    assert usefulsubs.kdel(0) == 1
    assert usefulsubs.kdel(1) == 0

def test_delt():
    assert usefulsubs.delt(0) == 1.0
    assert usefulsubs.delt(1) == 0.0

def test_sgn():
    assert usefulsubs.sgn(-5.0) == -1
    assert usefulsubs.sgn(5.0) == 1

def test_sgn2():
    arr = np.array([-1.0, 0.0, 1.0])
    out = usefulsubs.sgn2(arr)
    assert np.all(out == np.array([-1, 1, 1]))

def test_TotalEnergy():
    n = np.array([1+0j, 2+0j])
    E = np.array([1.0, 2.0])
    assert usefulsubs.TotalEnergy(n, E) == 5.0

def test_AvgEnergy():
    n = np.array([1+0j, 1+0j])
    E = np.array([2.0, 4.0])
    assert np.isclose(usefulsubs.AvgEnergy(n, E), 3.0)

def test_Temperature():
    n = np.array([1+0j, 1+0j])
    E = np.array([2.0, 4.0])
    T = usefulsubs.Temperature(n, E)
    assert T > 0

def test_Lrtz():
    assert np.isclose(usefulsubs.Lrtz(0.0, 1.0), 1.0/np.pi)

def test_theta():
    assert usefulsubs.theta(1.0) == 1.0
    assert usefulsubs.theta(-1.0) == 0.0

def test_softtheta():
    assert np.isclose(usefulsubs.softtheta(0.0, 1.0), 0.5)

def test_rad():
    assert np.isclose(usefulsubs.rad(180), np.pi)

def test_gaussian():
    assert np.isclose(usefulsubs.gaussian(0.0, 1.0), 1.0)

def test_Flip():
    arr = np.array([1,2,3], dtype=np.float64)
    assert np.all(usefulsubs.Flip(arr) == np.array([3,2,1]))

def test_GetArray0Index():
    arr = np.array([-1.0, 0.0, 1.0])
    assert usefulsubs.GetArray0Index(arr) == 1

def test_EAtX():
    f = np.array([[1+0j, 2+0j], [3+0j, 4+0j]])
    x = np.array([0.0, 1.0])
    x0 = 0.0
    out = usefulsubs.EAtX(f, x, x0)
    assert np.allclose(out, np.zeros(f.shape[1], dtype=np.complex128))

def test_GFFT_1D_and_GIFFT_1D():
    arr = np.array([1+0j, 2+0j, 3+0j, 4+0j])
    arr_orig = arr.copy()
    usefulsubs.GFFT_1D(arr, 1.0)
    usefulsubs.GIFFT_1D(arr, 1.0)
    assert arr.shape == arr_orig.shape
    assert arr.dtype == arr_orig.dtype

def test_GFFT_2D_and_GIFFT_2D():
    arr = np.ones((2,2), dtype=np.complex128)
    arr_orig = arr.copy()
    usefulsubs.GFFT_2D(arr, 1.0, 1.0)
    usefulsubs.GIFFT_2D(arr, 1.0, 1.0)
    assert arr.shape == arr_orig.shape
    assert arr.dtype == arr_orig.dtype

def test_FFTG_and_iFFTG():
    arr = np.array([1+0j, 2+0j, 3+0j, 4+0j])
    arr2 = arr.copy()
    usefulsubs.FFTG(arr)
    usefulsubs.iFFTG(arr)
    assert np.allclose(arr, arr2)

def test_fflip_dp_and_fflip_dpc():
    arr = np.array([1.0, 2.0, 3.0])
    carr = np.array([1+0j, 2+0j, 3+0j])
    assert np.all(usefulsubs.fflip_dp(arr) == arr[::-1])
    assert np.all(usefulsubs.fflip_dpc(carr) == carr[::-1])

def test_dfdy1D_and_dfdx1D():
    arr = np.array([1+0j, 2+0j, 3+0j, 4+0j])
    q = np.array([0.0, 1.0, 2.0, 3.0])
    out1 = usefulsubs.dfdy1D(arr, q)
    out2 = usefulsubs.dfdx1D(arr, q)
    assert out1.shape == arr.shape
    assert out2.shape == arr.shape

def test_dfdy2D_and_dfdx2D():
    arr = np.ones((2,2), dtype=np.complex128)
    q = np.array([0.0, 1.0])
    out1 = usefulsubs.dfdy2D(arr, q)
    out2 = usefulsubs.dfdx2D(arr, q)
    assert out1.shape == arr.shape
    assert out2.shape == arr.shape

def test_dfdy1D_q_and_dfdx1D_q():
    arr = np.array([1+0j, 2+0j, 3+0j])
    q = np.array([0.0, 1.0, 2.0])
    out1 = usefulsubs.dfdy1D_q(arr, q)
    out2 = usefulsubs.dfdx1D_q(arr, q)
    assert out1.shape == arr.shape
    assert out2.shape == arr.shape

def test_dfdy2D_q_and_dfdx2D_q():
    arr = np.ones((2,2), dtype=np.complex128)
    q = np.array([0.0, 1.0])
    out1 = usefulsubs.dfdy2D_q(arr, q)
    out2 = usefulsubs.dfdx2D_q(arr, q)
    assert out1.shape == arr.shape
    assert out2.shape == arr.shape

def test_dEdx_and_dEdy():
    arr = np.ones((2,2), dtype=np.complex128)
    out1 = usefulsubs.dEdx(arr, 1.0)
    out2 = usefulsubs.dEdy(arr, 1.0)
    assert out1.shape == arr.shape
    assert out2.shape == arr.shape

def test_dHdx_and_dHdy():
    arr = np.ones((2,2), dtype=np.complex128)
    out1 = usefulsubs.dHdx(arr, 1.0)
    out2 = usefulsubs.dHdy(arr, 1.0)
    assert out1.shape == arr.shape
    assert out2.shape == arr.shape

def test_convolve():
    x = np.array([1+0j, 2+0j, 3+0j])
    h = np.array([0+0j, 1+0j, 0+0j])
    out = usefulsubs.convolve(x, h)
    assert out.shape == x.shape

def test_ApplyABC():
    arr = np.ones((2,2), dtype=np.complex128)
    abc = np.ones((2,2), dtype=np.float64)
    usefulsubs.ApplyABC(arr, abc)
    assert arr.shape == (2,2)

def test_RotateField():
    Ex = np.ones((2,2), dtype=np.complex128)
    Ey = np.ones((2,2), dtype=np.complex128)
    usefulsubs.RotateField(np.pi/2, Ex, Ey)
    assert Ex.shape == (2,2)
    assert Ey.shape == (2,2)

def test_ShiftField():
    Ex = np.ones((2,2), dtype=np.complex128)
    Ey = np.ones((2,2), dtype=np.complex128)
    usefulsubs.ShiftField(1.0, 1.0, 1.0, 1.0, Ex, Ey)
    assert Ex.shape == (2,2)
    assert Ey.shape == (2,2)

def test_RotateShiftEField():
    Ex = np.ones((2,2), dtype=np.complex128)
    Ey = np.ones((2,2), dtype=np.complex128)
    qx = np.array([0.0, 1.0])
    qy = np.array([0.0, 1.0])
    usefulsubs.RotateShiftEField(0.1, qx, qy, Ex, Ey)
    assert Ex.shape == (2,2)
    assert Ey.shape == (2,2)

def test_FT_and_IFT():
    y = np.ones(2, dtype=np.complex128)
    x = np.array([0.0, 1.0])
    q = np.array([0.0, 1.0])
    usefulsubs.FT(y, x, q)
    usefulsubs.IFT(y, x, q)
    assert y.shape == (2,)

def test_K03():
    assert usefulsubs.K03(1.0) > 0

# Mock test for WriteIT1D and ReadIT1D
def test_WriteIT1D_and_ReadIT1D(tmp_path):
    arr = np.array([1.0, 2.0, 3.0])
    file = tmp_path / "test1d"
    # Patch _as_path to just return the filename as is
    with patch("src.usefulsubs._as_path", lambda name: str(file)):
        usefulsubs.WriteIT1D(arr, str(file))
        arr2 = np.zeros(3)
        usefulsubs.ReadIT1D(arr2, str(file))
        assert np.allclose(arr, arr2)

# Mock test for WriteIT2D and ReadIT2D
def test_WriteIT2D_and_ReadIT2D(tmp_path):
    arr = np.ones((2,2))
    dataqw = tmp_path / "dataQW"
    dataqw.mkdir()
    file = dataqw / "test2d"
    with patch("src.usefulsubs._as_path", lambda name: str(file)):
        usefulsubs.WriteIT2D(arr, str(file))
        arr2 = np.zeros((2,2))
        usefulsubs.ReadIT2D(arr2, str(file))
        assert np.allclose(arr, arr2)

def test_print2file(tmp_path):
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    file = tmp_path / "testprint"
    usefulsubs.print2file(x, y, 0, str(file))
    with open(file) as f:
        lines = f.readlines()
    assert len(lines) == 2