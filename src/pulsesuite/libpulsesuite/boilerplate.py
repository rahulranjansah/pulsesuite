"""
Integrator module (Python version)
==================================
Implements: odeint, rkqs, rkck, rk4, idiot, and related routines.

Author: Rahul R. Sah
"""
# Standard imports
import numpy as np
from numba import njit, prange

# Local imports
from .logger import Logger
from .nrutils import *
try:
    from guardrails.guardrails import with_guardrails
except ImportError:
    # Fallback if guardrails not available
    def with_guardrails(fn):
        return fn
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