import sys
import os
import io
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from logger import Logger

def test_log_levels():
    buf = io.StringIO()
    logger = Logger(logLevel=Logger.LOGDEBUG3, output=buf)
    logger.std("std message")
    logger.warning("warn message")
    logger.verbose("verbose message")
    logger.debug("debug message")
    logger.debug2("debug2 message")
    logger.debug3("debug3 message")
    out = buf.getvalue()
    assert "STD" in out
    assert "WARNING" in out
    assert "VERBOSE" in out
    assert "DEBUG" in out
    assert "DEBUG2" in out
    assert "DEBUG3" in out

def test_assert_true_and_error():
    buf = io.StringIO()
    logger = Logger(logLevel=Logger.LOGERROR, output=buf)
    # error should exit
    with pytest.raises(SystemExit) as excinfo:
        logger.error("fail message", exitCode=99)
    assert excinfo.value.code == 99
    # assertTrue should exit if false
    with pytest.raises(SystemExit) as excinfo2:
        logger.assertTrue(False, "assert fail")
    assert excinfo2.value.code == 16
    # assertTrue should not exit if true
    logger.assertTrue(True, "should not fail")