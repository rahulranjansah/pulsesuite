import sys
import threading
from contextlib import ContextDecorator

class Logger(ContextDecorator):
    """
    High-performance logger for scientific/HPC applications.

    Provides log levels, error/warning/info/debug, and assertion logic.
    Thread-safe and context-enabled. All methods use camelCase.

    Parameters
    ----------
    logLevel : int, optional
        Initial log level (default: 2 = LOGSTD)
    output : file-like, optional
        Output stream (default: sys.stdout)
    """
    # Log level constants
    LOGERROR   = 0
    LOGWARN    = 1
    LOGSTD     = 2
    LOGVERBOSE = 3
    LOGDEBUG   = 4
    LOGDEBUG2  = 5
    LOGDEBUG3  = 6
    LOGNAMES = [
        "ERROR  ", "WARNING", "STD    ", "VERBOSE", "DEBUG  ", "DEBUG2 ", "DEBUG3 "
    ]

    _instance = None
    _lock = threading.Lock()

    def __init__(self, logLevel=LOGSTD, output=sys.stdout):
        self.logLevel = logLevel
        self.output = output
        self._logLock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed, but method required for context manager
        return False

    @classmethod
    def getInstance(cls):
        """Singleton access to the logger."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = Logger()
            return cls._instance

    def setLogLevel(self, level):
        """Set the log level."""
        self.logLevel = int(level)

    def getLogLevel(self):
        """Get the current log level."""
        return self.logLevel

    def incrLogLevel(self):
        """Increase the log level by 1."""
        self.logLevel += 1

    def decrLogLevel(self):
        """Decrease the log level by 1."""
        self.logLevel -= 1

    def getLogLevelStr(self, level=None):
        """Get the string representation of a log level."""
        lvl = self.logLevel if level is None else int(level)
        idx = max(0, min(lvl, len(self.LOGNAMES)-1))
        return self.LOGNAMES[idx]

    def _log(self, msg, level):
        if self.logLevel < level:
            return
        with self._logLock:
            print(f"{self.getLogLevelStr(level)}: {msg}", file=self.output)

    def error(self, msg, exitCode=1, file=None, line=None):
        """
        Log an error message and exit.

        Parameters
        ----------
        msg : str
            The error message.
        exitCode : int, optional
            Exit code (default: 1)
        file : str, optional
            File name for error location.
        line : int, optional
            Line number for error location.
        """
        fullMsg = self._makeMsg(msg, file, line)
        self._log(fullMsg, self.LOGERROR)
        sys.exit(exitCode)

    def assertTrue(self, test, msg, file=None, line=None):
        """
        Assert that a condition is true, else log error and exit.

        Parameters
        ----------
        test : bool
            Condition to check.
        msg : str
            Message to log if assertion fails.
        file : str, optional
            File name for error location.
        line : int, optional
            Line number for error location.
        """
        if not test:
            self.error(msg, exitCode=16, file=file, line=line)

    def warning(self, msg, file=None, line=None):
        """
        Log a warning message.
        """
        if self.logLevel < self.LOGWARN:
            return
        fullMsg = self._makeMsg(msg, file, line)
        self._log(fullMsg, self.LOGWARN)

    def std(self, msg, file=None, line=None):
        """
        Log a standard message.
        """
        if self.logLevel < self.LOGSTD:
            return
        fullMsg = self._makeMsg(msg, file, line)
        self._log(fullMsg, self.LOGSTD)

    def verbose(self, msg, file=None, line=None):
        """
        Log a verbose message.
        """
        if self.logLevel < self.LOGVERBOSE:
            return
        fullMsg = self._makeMsg(msg, file, line)
        self._log(fullMsg, self.LOGVERBOSE)

    def debug(self, msg, file=None, line=None):
        """
        Log a debug message.
        """
        if self.logLevel < self.LOGDEBUG:
            return
        fullMsg = self._makeMsg(msg, file, line)
        self._log(fullMsg, self.LOGDEBUG)

    def debug2(self, msg, file=None, line=None):
        """
        Log a debug2 message.
        """
        if self.logLevel < self.LOGDEBUG2:
            return
        fullMsg = self._makeMsg(msg, file, line)
        self._log(fullMsg, self.LOGDEBUG2)

    def debug3(self, msg, file=None, line=None):
        """
        Log a debug3 message.
        """
        if self.logLevel < self.LOGDEBUG3:
            return
        fullMsg = self._makeMsg(msg, file, line)
        self._log(fullMsg, self.LOGDEBUG3)

    def _makeMsg(self, msg, file, line):
        if file is not None and line is not None:
            return f"{file}:{line} : {msg}"
        return msg