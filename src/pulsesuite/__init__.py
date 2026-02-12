"""
PulseSuite: Simulation suite for ultrafast laser-matter interactions.

This package provides tools for simulating quantum wire and quantum well systems
under intense optical excitation using the Semiconductor Bloch Equations (SBEs)
coupled with Pseudo-Spectral Time Domain (PSTD) electromagnetic field propagation.
"""

__all__ = [
    "core",
    "PSTD3D",
    "libpulsesuite",
]

# Lazy import — only load sub-packages when actually accessed
def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    import importlib
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module
