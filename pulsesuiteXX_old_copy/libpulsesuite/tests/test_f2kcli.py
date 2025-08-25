import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from f2kcli import F2kcli

def testCommandArgumentCount(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'arg1', 'arg2'])
    assert F2kcli.commandArgumentCount() == 2

def testGetCommand(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'foo', 'bar'])
    cmd = ['']
    length = [0]
    status = [None]
    F2kcli.getCommand(cmd, length, status)
    assert cmd[0] == 'prog foo bar'
    assert length[0] == len('prog foo bar')
    assert status[0] == 0

def testGetCommandArgument(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', 'foo', 'bar'])
    value = ['']
    length = [0]
    status = [None]
    # Test program name
    F2kcli.getCommandArgument(0, value, length, status)
    assert value[0] == 'prog'
    assert length[0] == 4
    assert status[0] == 0
    # Test first argument
    F2kcli.getCommandArgument(1, value, length, status)
    assert value[0] == 'foo'
    assert length[0] == 3
    assert status[0] == 0
    # Test out-of-range
    F2kcli.getCommandArgument(10, value, length, status)
    assert value[0] == ''
    assert length[0] == 0
    assert status[0] == 1