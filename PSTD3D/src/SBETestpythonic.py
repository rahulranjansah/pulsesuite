"""
SBETestpythonic.py - Python translation of SBETest.f90

Minimal driver to initialize a 1D quantum wire test, generate a Gaussian-modulated
cosine electric field Exx in time, and (optionally) initialize the SBE module.

Outputs Exx and Px samples at the center point to files under fields/.
"""

from __future__ import annotations


import os, sys
import numpy as np
from numpy.typing import NDArray


# Local imports
sys.path.append(os.path.dirname(__file__))
from constants import twopi, c0
from helperspythonic import GetSpaceArray, GetKArray
from SBEspythonic import InitializeSBE  # QWCalculator is available but not required here

# Type aliases
_dp = np.float64
_dc = np.complex128


def main() -> None:
    # Geometry and time
    Nr: int = 100
    drr: _dp = _dp(10e-9)
    n0: _dp = _dp(3.1)

    Nt: int = 10000
    dt: _dp = _dp(10e-18)
    t: _dp = _dp(0.0)

    # Field parameters (X, Y, Z)
    E0x: _dp = _dp(1e5)
    twx: _dp = _dp(10e-15)
    tpx: _dp = _dp(50e-15)
    lamX: _dp = _dp(800e-9)

    E0y: _dp = _dp(1e5)
    twy: _dp = _dp(10e-15)
    tpy: _dp = _dp(100e-15)
    lamY: _dp = _dp(800e-9)

    E0z: _dp = _dp(0.0)
    twz: _dp = _dp(1.0)
    tpz: _dp = _dp(0.0)
    lamZ: _dp = _dp(0.0)

    # Allocate complex field arrays
    Exx: NDArray[_dc] = np.zeros(Nr, dtype=_dc, order='F')
    Eyy: NDArray[_dc] = np.zeros(Nr, dtype=_dc, order='F')
    Ezz: NDArray[_dc] = np.zeros(Nr, dtype=_dc, order='F')

    Pxx: NDArray[_dc] = np.zeros(Nr, dtype=_dc, order='F')
    Pyy: NDArray[_dc] = np.zeros(Nr, dtype=_dc, order='F')
    Pzz: NDArray[_dc] = np.zeros(Nr, dtype=_dc, order='F')

    Rho: NDArray[_dc] = np.zeros(Nr, dtype=_dc, order='F')
    Vrr: NDArray[_dc] = np.zeros(Nr, dtype=_dc, order='F')

    # Coordinate arrays
    rr: NDArray[_dp] = GetSpaceArray(Nr, (Nr - 1) * drr)
    qrr: NDArray[_dp] = GetKArray(Nr, Nr * drr)

    # Angular frequencies and optical cycles
    w0x: _dp = _dp(twopi * c0 / lamX)
    k0x: _dp = _dp(twopi / lamX * n0)
    Tcx: _dp = _dp(lamX / c0)

    w0y: _dp = _dp(twopi * c0 / lamY)
    k0y: _dp = _dp(twopi / lamY * n0)
    Tcy: _dp = _dp(lamY / c0)

    w0z: _dp = _dp(twopi * c0 / lamZ) if lamZ != 0.0 else _dp(0.0)
    k0z: _dp = _dp(twopi / lamZ * n0) if lamZ != 0.0 else _dp(0.0)
    Tcz: _dp = _dp(lamZ / c0) if lamZ != 0.0 else _dp(0.0)

    # Peak field for SBE initialization
    Emax0: _dp = _dp(np.sqrt(E0x**2 + E0y**2 + E0z**2))

    # Initialize SBEs (use QW=True; one frequency line Nw=1)
    # Matches Fortran: InitializeSBE(qrr, rr, r0=0, Emax0, lamX, Nw=1, QW=.true.)
    InitializeSBE(qrr, rr, _dp(0.0), Emax0, lamX, 1, True)

    # Prepare output files
    os.makedirs('fields', exist_ok=True)
    ex_path = os.path.join('fields', 'Ex.dat')
    px_path = os.path.join('fields', 'Px.dat')

    with open(ex_path, 'w') as ex_file, open(px_path, 'w') as px_file:
        for n in range(Nt):
            # Time-domain Exx(t) at all spatial points (1D plane wave envelope)
            tau = t - tpx
            env = np.exp(-(tau**2) / (twx**2)) * np.exp(-(tau**20) / ((2 * twx)**20))
            carrier = np.cos(w0x * (t - tpx))
            Exx[:] = (E0x * env * carrier).astype(_dp)

            # Center-sample records (mirrors Fortran behavior)
            mid = Nr // 2
            ex_file.write(f"{t:.8e} {Exx[mid].real:.8e}\n")
            px_file.write(f"{t:.8e} {Pxx[mid].real:.8e}\n")

            t = _dp(t + dt)


if __name__ == "__main__":
    main()
