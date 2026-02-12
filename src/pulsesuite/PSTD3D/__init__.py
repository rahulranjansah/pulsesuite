"""PSTD3D sub-package for quantum wire simulations."""

# Import modules themselves (allows: from pulsesuite.PSTD3D import coulomb)
# from . import helpers
# from . import usefulsubs
# from . import typespace


# from . import dcfield

# from . import phonons
# from . import emission
# from . import dephasing

# from . import epsrtl
# from . import phonons
# from . import phost
# from . import coulomb

# from . import qwoptics
# from . import SBEs


__all__ = [
    "coulomb",
    "dcfield",
    "SBEs",
    "qwoptics",
    "phonons",
    "emission",
    "dephasing",
    "helpers",
    "usefulsubs",
    "typespace",
    "epsrtl",
    "phonons",
    "phost",
]

# Lazy import - only load modules when they're actually accessed
def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # Import the module on demand
    import importlib
    module = importlib.import_module(f".{name}", __name__)
    # Cache it in globals for future access
    globals()[name] = module
    return module
