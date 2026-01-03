"""Core utilities for PulseSuite."""
# Import the module on demand
import importlib

# Import modules themselves (allows: from pulsesuite.core import fftw)
# from . import constants
# from . import fftw
# from . import rungekutta
# from . import typelens
# from . import typemedium
# from . import type_plasma
# from . import typetpamedium

__all__ = [
    "constants",
    "fftw",
    "rungekutta",
    "typelens",
    "typemedium",
    "type_plasma",
    "typetpamedium",
]

# Lazy import - only load modules when they're actually accessed
def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


    module = importlib.import_module(f".{name}", __name__)
    # Cache it in globals for future access
    globals()[name] = module
    return module