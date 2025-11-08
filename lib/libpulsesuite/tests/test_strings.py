import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from strings import Strings

def test_int2str():
    assert Strings.int2str(42) == '42'
    assert Strings.int2str(42, 5) == '   42'

def test_bool2str():
    assert Strings.bool2str(True) == 'T'
    assert Strings.bool2str(False) == 'F'
    assert Strings.bool2str(True, 3) == '  T'

def test_dbl2str():
    s = Strings.dbl2str(1.23e4)
    assert 'E' in s
    s2 = Strings.dbl2str(1.23e4, frmt='D')
    assert 'D' in s2

def test_sgl2str():
    s = Strings.sgl2str(1.23e4)
    assert 'E' in s

def test_wordwrap():
    lines = Strings.wordwrap('a b c d e f g', 3)
    assert all(len(line) <= 3 for line in lines)

def test_trimb():
    assert Strings.trimb('  hello  ') == 'hello'

def test_toupper_tolower():
    assert Strings.toupper('abc') == 'ABC'
    assert Strings.tolower('ABC') == 'abc'

def test_cmplx2str():
    s = Strings.cmplx2str(1+2j)
    assert '(' in s and ',' in s and ')' in s