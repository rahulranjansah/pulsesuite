# tests/conftest.py -- setup for qwoptics
import sys
from pathlib import Path
import numpy as np
import pytest

# Make ./src and project root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import qwoptics

if not hasattr(qwoptics, "_dp"):
    qwoptics._dp = np.float64
if not hasattr(qwoptics, "_dc"):
    qwoptics._dc = np.complex128


def _noop(*_a, **_k):  # no-op printer
    return None
for name in ("printIT", "printIT2D"):
    if hasattr(qwoptics, name):
        setattr(qwoptics, name, _noop)

@pytest.fixture(scope="module")
def grids():
    """Small uniform grids including y=0; float64/complex128; Fortran-order friendly."""
    N = 32
    RR = np.linspace(-1.0, 1.0, N, dtype=np.float64)
    R = RR.copy()
    ky = np.linspace(-5.0, 5.0, N, dtype=np.float64)
    Qr = ky.copy()
    return N, RR, R, ky, Qr

@pytest.fixture(scope="module")
def dispersions(grids):
    _, _, _, ky, _ = grids
    Ee = (ky**2) * 1e-22
    Eh = (ky**2) * 1e-22
    return Ee, Eh

@pytest.fixture(scope="module")
def initialized_qw(grids, dispersions):
    """Initialize with large L so window≈1 on [-1,1]."""
    _, RR, _, ky, Qr = grids
    Ee, Eh = dispersions
    L = 100.0
    dcv = 1.0 + 0.0j
    ehint = 1.0
    area = 1.0
    gap = 0.0
    qwoptics.InitializeQWOptics(RR, L, dcv, ky, Qr, Ee, Eh, ehint, area, gap)
    return {"L": L, "dcv": dcv, "ehint": ehint, "area": area, "gap": gap}
