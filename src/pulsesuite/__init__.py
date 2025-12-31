"""
PulseSuite: Simulation suite for ultrafast laser-matter interactions.

This package provides tools for simulating quantum wire and quantum well systems
under intense optical excitation using the Semiconductor Bloch Equations (SBEs)
coupled with Pseudo-Spectral Time Domain (PSTD) electromagnetic field propagation.
"""

# Import main sub-packages
from . import core
from . import PSTD3D
from . import libpulsesuite

__all__ = [
    "core",
    "PSTD3D",
    "libpulsesuite",
]

