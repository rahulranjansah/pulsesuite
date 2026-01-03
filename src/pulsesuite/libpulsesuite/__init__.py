"""libpulsesuite sub-package for core utilities and integrators."""

# Import the module on demand
import importlib

# # Import modules themselves (allows: from pulsesuite.libpulsesuite import integrator)
# from . import boilerplate
# from . import calcintlength
# from . import constants
# from . import dumps
# from . import f2kcli
# from . import helpers
# from . import integrator
# from . import integrator_obj
# from . import logger
# from . import materialproperties
# from . import nrutils
# from . import numerictypes
# from . import pulsegenerator
# from . import spliner
# from . import strings
# from . import units

__all__ = [
    "boilerplate",
    "calcintlength",
    "constants",
    "dumps",
    "f2kcli",
    "helpers",
    "integrator",
    "integrator_obj",
    "logger",
    "materialproperties",
    "nrutils",
    "numerictypes",
    "pulsegenerator",
    "spliner",
    "strings",
    "units",
]

# Lazy import - only load modules when they're actually accessed
def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


    module = importlib.import_module(f".{name}", __name__)
    # Cache it in globals for future access
    globals()[name] = module
    return module