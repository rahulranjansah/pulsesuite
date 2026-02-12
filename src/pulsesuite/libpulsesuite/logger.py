"""
Thin wrapper around Python's ``logging`` module with pulsesuite-specific
log levels that mirror the Fortran ``logger.F90`` hierarchy.

Usage
-----
>>> from pulsesuite.core.logger import get_logger
>>> log = get_logger(__name__)
>>> log.info("standard message")        # LOGSTD equivalent
>>> log.debug2("inner-loop detail")     # custom level
"""

import logging
import sys

# ── Custom levels (below DEBUG=10) ──────────────────────────────────────
DEBUG2 = 9
DEBUG3 = 8

logging.addLevelName(DEBUG2, "DEBUG2")
logging.addLevelName(DEBUG3, "DEBUG3")


class _PulseLogger(logging.Logger):
    """Logger subclass that adds ``debug2`` and ``debug3`` convenience methods."""

    def debug2(self, msg, *args, **kwargs):
        if self.isEnabledFor(DEBUG2):
            self._log(DEBUG2, msg, args, **kwargs)

    def debug3(self, msg, *args, **kwargs):
        if self.isEnabledFor(DEBUG3):
            self._log(DEBUG3, msg, args, **kwargs)


logging.setLoggerClass(_PulseLogger)

# ── Mapping from Fortran integer levels to Python levels ────────────────
FORTRAN_LEVEL_MAP = {
    0: logging.ERROR,  # LOGERROR
    1: logging.WARNING,  # LOGWARN
    2: logging.INFO,  # LOGSTD
    3: logging.DEBUG,  # LOGVERBOSE
    4: logging.DEBUG,  # LOGDEBUG  (Python DEBUG = 10)
    5: DEBUG2,  # LOGDEBUG2
    6: DEBUG3,  # LOGDEBUG3
}


def get_logger(name: str | None = None) -> _PulseLogger:
    """Return a logger under the ``pulsesuite`` hierarchy.

    If *name* is a fully qualified module name (e.g.
    ``pulsesuite.core.integrator``), the logger inherits from the
    ``pulsesuite`` root logger so a single ``set_level()`` call
    controls everything.
    """
    return logging.getLogger(name or "pulsesuite")


def set_level(level: int | str = logging.INFO) -> None:
    """Set the log level for *all* pulsesuite loggers at once.

    Accepts Python level ints/names **or** legacy Fortran integer levels
    (0-6).
    """
    if isinstance(level, int) and level in FORTRAN_LEVEL_MAP:
        level = FORTRAN_LEVEL_MAP[level]
    root = logging.getLogger("pulsesuite")
    root.setLevel(level)


def setup(level: int | str = logging.INFO, stream=None) -> None:
    """One-time setup: attach a stderr handler with the pulsesuite format.

    Safe to call multiple times — extra calls are no-ops.
    """
    root = logging.getLogger("pulsesuite")
    if root.handlers:
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)-7s: %(message)s"))
    root.addHandler(handler)
    set_level(level)
