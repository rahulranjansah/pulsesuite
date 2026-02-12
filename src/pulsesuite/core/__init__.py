"""Core utilities for PulseSuite."""

__all__ = [
    "constants",
    "fftw",
    "rungekutta",
    "typelens",
    "typemedium",
    "type_plasma",
    "typetpamedium",
]

# Lazy import — only load modules when actually accessed
def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    import importlib
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module
