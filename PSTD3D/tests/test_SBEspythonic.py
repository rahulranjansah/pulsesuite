"""
Test suite for SBEspythonic.py module.

Tests the Python port of the FORTRAN SBEs module for semiconductor Bloch equations.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the module under test
import sys
sys.path.insert(0, '/mnt/hardisk/rahul_gulley/PSTD3D/src')
from SBEspythonic import *


class TestSBEspythonic:
    """Test class for SBEspythonic module."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)

        # Create test parameter files
        self.create_test_params()

        # Set up test arrays
        self.Nk = 10
        self.Nr = 20
        self.Nw = 2

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_test_params(self):
        """Create test parameter files."""
        os.makedirs('params', exist_ok=True)

        # Create qw.params file
        with open('params/qw.params', 'w') as f:
            f.write("100e-9\n")  # L
            f.write("5e-9\n")    # Delta0
            f.write("1.5\n")     # gap (eV)
            f.write("0.07\n")    # me (me0)
            f.write("0.45\n")    # mh (me0)
            f.write("0.1\n")     # HO (eV)
            f.write("1e12\n")    # gam_e
            f.write("1e12\n")    # gam_h
            f.write("1e12\n")    # gam_eh
            f.write("9.1\n")     # epsr
            f.write("36e-3\n")   # Oph (eV)
            f.write("3e-3\n")    # Gph (eV)
            f.write("0.0\n")     # Edc
            f.write("1000\n")    # jmax
            f.write("100000\n")  # ntmax

        # Create mb.params file
        with open('params/mb.params', 'w') as f:
            f.write("1\n")  # Optics
            f.write("1\n")  # Excitons
            f.write("1\n")  # EHs
            f.write("1\n")  # Screened
            f.write("1\n")  # Phonon
            f.write("0\n")  # DCTrans
            f.write("1\n")  # LF
            f.write("0\n")  # FreePot
            f.write("1\n")  # DiagDph
            f.write("1\n")  # OffDiagDph
            f.write("0\n")  # Recomb
            f.write("0\n")  # PLSpec
            f.write("0\n")  # ignorewire
            f.write("0\n")  # Xqwparams
            f.write("0\n")  # LorentzDelta

    def test_module_initialization(self):
        """Test module-level variable initialization."""
        assert L == 100e-9
        assert Delta0 == 5e-9
        assert gap == 1.5 * eV
        assert me == 0.07 * me0
        assert mh == 0.45 * me0
        assert HO == 100e-3 * eV
        assert epsr == 9.1
        assert Optics is True
        assert Excitons is True
        assert EHs is True

    def test_parameter_reading(self):
        """Test parameter file reading."""
        ReadQWParams()
        ReadMBParams()

        # Check that parameters were read correctly
        assert L == 100e-9
        assert gap == 1.5 * eV
        assert me == 0.07 * me0
        assert mh == 0.45 * me0
        assert jmax == 1000
        assert ntmax == 100000

    def test_array_initialization(self):
        """Test array initialization functions."""
        # Test GetArrays
        x = np.zeros(20, dtype=np.float64)
        qx = np.zeros(20, dtype=np.float64)
        kx = np.zeros(10, dtype=np.float64)

        # Mock the global variables
        global Nr, Nk, L, dkr, NK0, NQ0
        Nr = 20
        Nk = 10
        L = 100e-9
        dkr = 1e6

        GetArrays(x, qx, kx)

        assert len(x) == 20
        assert len(qx) == 20
        assert len(kx) == 10
        assert np.allclose(x[0], -L)
        assert np.allclose(x[-1], L)

    def test_kkp_calculation(self):
        """Test MakeKKP function."""
        global Nk, kr, dkr, NQ0, kkp
        Nk = 5
        kr = np.array([-2, -1, 0, 1, 2], dtype=np.float64) * 1e6
        dkr = 1e6
        NQ0 = 2

        MakeKKP()

        assert kkp.shape == (5, 5)
        assert kkp[2, 2] == NQ0  # kr[2] - kr[2] = 0

    def test_checkout_checkin(self):
        """Test Checkout and Checkin functions."""
        global YY1, YY2, YY3, CC1, CC2, CC3, DD1, DD2, DD3
        Nk = 5
        Nw = 2

        # Initialize global arrays
        YY1 = np.zeros((Nk, Nk, Nw), dtype=np.complex128)
        YY2 = np.ones((Nk, Nk, Nw), dtype=np.complex128)
        YY3 = 2 * np.ones((Nk, Nk, Nw), dtype=np.complex128)
        CC1 = np.zeros((Nk, Nk, Nw), dtype=np.complex128)
        CC2 = np.ones((Nk, Nk, Nw), dtype=np.complex128)
        CC3 = 2 * np.ones((Nk, Nk, Nw), dtype=np.complex128)
        DD1 = np.zeros((Nk, Nk, Nw), dtype=np.complex128)
        DD2 = np.ones((Nk, Nk, Nw), dtype=np.complex128)
        DD3 = 2 * np.ones((Nk, Nk, Nw), dtype=np.complex128)

        # Test Checkout
        P1 = np.zeros((Nk, Nk), dtype=np.complex128)
        P2 = np.zeros((Nk, Nk), dtype=np.complex128)
        C1 = np.zeros((Nk, Nk), dtype=np.complex128)
        C2 = np.zeros((Nk, Nk), dtype=np.complex128)
        D1 = np.zeros((Nk, Nk), dtype=np.complex128)
        D2 = np.zeros((Nk, Nk), dtype=np.complex128)

        Checkout(P1, P2, C1, C2, D1, D2, 0)

        assert np.allclose(P1, YY2[:, :, 0])
        assert np.allclose(P2, YY3[:, :, 0])
        assert np.allclose(C1, CC2[:, :, 0])
        assert np.allclose(C2, CC3[:, :, 0])
        assert np.allclose(D1, DD2[:, :, 0])
        assert np.allclose(D2, DD3[:, :, 0])

        # Test Checkin
        P1[:] = 3.0
        P2[:] = 4.0
        P3 = 5.0 * np.ones((Nk, Nk), dtype=np.complex128)
        C1[:] = 3.0
        C2[:] = 4.0
        C3 = 5.0 * np.ones((Nk, Nk), dtype=np.complex128)
        D1[:] = 3.0
        D2[:] = 4.0
        D3 = 5.0 * np.ones((Nk, Nk), dtype=np.complex128)

        Checkin(P1, P2, P3, C1, C2, C3, D1, D2, D3, 0)

        assert np.allclose(YY1[:, :, 0], 3.0)
        assert np.allclose(YY2[:, :, 0], 4.0)
        assert np.allclose(YY3[:, :, 0], 5.0)
        assert np.allclose(CC1[:, :, 0], 3.0)
        assert np.allclose(CC2[:, :, 0], 4.0)
        assert np.allclose(CC3[:, :, 0], 5.0)
        assert np.allclose(DD1[:, :, 0], 3.0)
        assert np.allclose(DD2[:, :, 0], 4.0)
        assert np.allclose(DD3[:, :, 0], 5.0)

    def test_dpdt_function(self):
        """Test dpdt function."""
        global Nk, hbar
        Nk = 3
        hbar = 1.054571817e-34

        C = np.ones((Nk, Nk), dtype=np.complex128)
        D = np.ones((Nk, Nk), dtype=np.complex128)
        P = np.ones((Nk, Nk), dtype=np.complex128)
        Heh = np.ones((Nk, Nk), dtype=np.complex128)
        Hee = np.eye(Nk, dtype=np.complex128)
        Hhh = np.eye(Nk, dtype=np.complex128)
        GamE = np.ones(Nk, dtype=np.float64)
        GamH = np.ones(Nk, dtype=np.float64)
        OffP = np.zeros((Nk, Nk), dtype=np.complex128)

        result = dpdt(C, D, P, Heh, Hee, Hhh, GamE, GamH, OffP)

        assert result.shape == (Nk, Nk)
        assert np.all(np.isfinite(result))

    def test_dCdt_function(self):
        """Test dCdt function."""
        global Nk, hbar
        Nk = 3
        hbar = 1.054571817e-34

        Cee = np.ones((Nk, Nk), dtype=np.complex128)
        Dhh = np.ones((Nk, Nk), dtype=np.complex128)
        Phe = np.ones((Nk, Nk), dtype=np.complex128)
        Heh = np.ones((Nk, Nk), dtype=np.complex128)
        Hee = np.eye(Nk, dtype=np.complex128)
        Hhh = np.eye(Nk, dtype=np.complex128)
        GamE = np.ones(Nk, dtype=np.float64)
        GamH = np.ones(Nk, dtype=np.float64)
        OffE = np.zeros((Nk, Nk), dtype=np.complex128)

        result = dCdt(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffE)

        assert result.shape == (Nk, Nk)
        assert np.all(np.isfinite(result))

    def test_dDdt_function(self):
        """Test dDdt function."""
        global Nk, hbar
        Nk = 3
        hbar = 1.054571817e-34

        Cee = np.ones((Nk, Nk), dtype=np.complex128)
        Dhh = np.ones((Nk, Nk), dtype=np.complex128)
        Phe = np.ones((Nk, Nk), dtype=np.complex128)
        Heh = np.ones((Nk, Nk), dtype=np.complex128)
        Hee = np.eye(Nk, dtype=np.complex128)
        Hhh = np.eye(Nk, dtype=np.complex128)
        GamE = np.ones(Nk, dtype=np.float64)
        GamH = np.ones(Nk, dtype=np.float64)
        OffH = np.zeros((Nk, Nk), dtype=np.complex128)

        result = dDdt(Cee, Dhh, Phe, Heh, Hee, Hhh, GamE, GamH, OffH)

        assert result.shape == (Nk, Nk)
        assert np.all(np.isfinite(result))

    def test_CalcMeh_function(self):
        """Test CalcMeh function."""
        global Nk, ehint, kkp
        Nk = 3
        ehint = 1.0

        # Mock kkp array
        kkp = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.int32)

        Ex = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
        Ey = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
        Ez = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
        Meh = np.zeros((Nk, Nk), dtype=np.complex128)

        # Mock Xcv, Ycv, Zcv functions
        with patch('SBEspythonic.Xcv', return_value=1.0), \
             patch('SBEspythonic.Ycv', return_value=1.0), \
             patch('SBEspythonic.Zcv', return_value=1.0):
            CalcMeh(Ex, Ey, Ez, Meh)

        assert Meh.shape == (Nk, Nk)
        assert np.all(np.isfinite(Meh))

    def test_CalcWnn_function(self):
        """Test CalcWnn function."""
        global Nk, kkp
        Nk = 3

        # Mock kkp array
        kkp = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.int32)

        q0 = 1.0
        Vr = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
        Wnn = np.zeros((Nk, Nk), dtype=np.complex128)

        CalcWnn(q0, Vr, Wnn)

        assert Wnn.shape == (Nk, Nk)
        assert np.all(np.isfinite(Wnn))

    def test_utility_functions(self):
        """Test utility functions."""
        # Test QWArea
        result = QWArea()
        assert isinstance(result, float)

        # Test ShutOffOptics
        global Optics
        original_optics = Optics
        ShutOffOptics()
        assert Optics is False
        Optics = original_optics

        # Test chiqw
        result = chiqw()
        assert isinstance(result, complex)

        # Test getqc
        result = getqc()
        assert isinstance(result, float)

    def test_initialization_error_handling(self):
        """Test error handling in initialization."""
        # Test QWCalculator without initialization
        Exx = np.zeros(10, dtype=np.complex128)
        Eyy = np.zeros(10, dtype=np.complex128)
        Ezz = np.zeros(10, dtype=np.complex128)
        Vrr = np.zeros(10, dtype=np.complex128)
        rr = np.zeros(10, dtype=np.float64)
        q = np.zeros(10, dtype=np.float64)
        Pxx = np.zeros(10, dtype=np.complex128)
        Pyy = np.zeros(10, dtype=np.complex128)
        Pzz = np.zeros(10, dtype=np.complex128)
        Rho = np.zeros(10, dtype=np.complex128)
        DoQWP = False
        DoQWDl = False

        with pytest.raises(RuntimeError, match="Not initialized"):
            QWCalculator(Exx, Eyy, Ezz, Vrr, rr, q, 1e-15, 1, Pxx, Pyy, Pzz, Rho, DoQWP, DoQWDl)

    def test_numerical_stability(self):
        """Test numerical stability of calculations."""
        global Nk, hbar
        Nk = 5
        hbar = 1.054571817e-34

        # Test with small values
        C = np.ones((Nk, Nk), dtype=np.complex128) * 1e-10
        D = np.ones((Nk, Nk), dtype=np.complex128) * 1e-10
        P = np.ones((Nk, Nk), dtype=np.complex128) * 1e-10
        Heh = np.ones((Nk, Nk), dtype=np.complex128) * 1e-10
        Hee = np.eye(Nk, dtype=np.complex128) * 1e-10
        Hhh = np.eye(Nk, dtype=np.complex128) * 1e-10
        GamE = np.ones(Nk, dtype=np.float64) * 1e-10
        GamH = np.ones(Nk, dtype=np.float64) * 1e-10
        OffP = np.zeros((Nk, Nk), dtype=np.complex128)

        result = dpdt(C, D, P, Heh, Hee, Hhh, GamE, GamH, OffP)

        assert result.shape == (Nk, Nk)
        assert np.all(np.isfinite(result))

    def test_array_shapes(self):
        """Test that all functions handle array shapes correctly."""
        # Test with different array sizes
        sizes = [3, 5, 10]

        for size in sizes:
            global Nk
            Nk = size

            C = np.ones((size, size), dtype=np.complex128)
            D = np.ones((size, size), dtype=np.complex128)
            P = np.ones((size, size), dtype=np.complex128)
            Heh = np.ones((size, size), dtype=np.complex128)
            Hee = np.eye(size, dtype=np.complex128)
            Hhh = np.eye(size, dtype=np.complex128)
            GamE = np.ones(size, dtype=np.float64)
            GamH = np.ones(size, dtype=np.float64)
            OffP = np.zeros((size, size), dtype=np.complex128)

            result = dpdt(C, D, P, Heh, Hee, Hhh, GamE, GamH, OffP)
            assert result.shape == (size, size)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test with invalid parameters
        with pytest.raises((ValueError, TypeError)):
            dpdt(None, None, None, None, None, None, None, None, None)

        # Test with mismatched array sizes
        global Nk
        Nk = 3

        C = np.ones((2, 2), dtype=np.complex128)  # Wrong size
        D = np.ones((3, 3), dtype=np.complex128)
        P = np.ones((3, 3), dtype=np.complex128)
        Heh = np.ones((3, 3), dtype=np.complex128)
        Hee = np.eye(3, dtype=np.complex128)
        Hhh = np.eye(3, dtype=np.complex128)
        GamE = np.ones(3, dtype=np.float64)
        GamH = np.ones(3, dtype=np.float64)
        OffP = np.zeros((3, 3), dtype=np.complex128)

        # This should work but may produce unexpected results
        result = dpdt(C, D, P, Heh, Hee, Hhh, GamE, GamH, OffP)
        assert result.shape == (3, 3)


if __name__ == "__main__":
    pytest.main([__file__])
