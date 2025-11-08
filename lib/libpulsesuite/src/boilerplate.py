"""
Integrator module (Python version)
==================================
High-performance ODE integrators for scientific computing, ported from Fortran.

Dependencies: numpy, numba, numerictypes, logger, nrutils, constants, helpers

Implements: odeint, rkqs, rkck, rk4, idiot, and related routines.

Author: Auto-converted from Fortran by AI, with manual optimization for HPC.
"""
# Standard imports
import numpy as np
from numba import njit, prange

# Local imports
from logger import Logger  # Assuming logger.py provides a Logger class or similar
from nrutils import *
from guardrails.guardrails import with_guardrails
# from constants import * # Assuming constants are available
# from helpers import *   # Assuming helpers are available

# Constants for adaptive step-size control
SAFETY = 0.9
PGROW = -0.20
PSHRNK = -0.25
ERRCON = (5.0 / SAFETY) ** (-5)
TINY = 1.0e-30
MAXSTP = 100000000
KMAXX = 8

# Numba JIT decorator helper
try:
    import numba
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False

def _jit(fn):
    """Numba‑JIT `fn` (parallel, cache) if Numba is available."""
    return njit(parallel=True, cache=True)(fn) if _USE_NUMBA else fn