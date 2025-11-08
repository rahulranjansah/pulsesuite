import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from units import Units

def test_prefix_val():
    assert Units.prefixVal('k') == 1e3
    assert Units.prefixVal('M') == 1e6
    assert Units.prefixVal('') == 1.0
    assert Units.prefixVal('Ki') == 2.0**10

def test_split_unit():
    assert Units.splitUnit('km') == ('k', 'm')
    assert Units.splitUnit('MiB') == ('Mi', 'B')
    assert Units.splitUnit('m') == ('', 'm')
    assert Units.splitUnit('') == ('', '')

def test_unit_val():
    assert Units.unitVal('1.5 km', 'm') == 1500.0
    assert Units.unitVal('2 MiB', 'B') == 2 * 2**20
    assert Units.unitVal('42 m', 'm') == 42.0
    assert Units.unitVal('1000', 'm') == 1000.0

def test_write_with_unit():
    s = Units.writeWithUnit(1500, 'm')
    assert '1.50e+00 km' in s or '1.50 km' in s
    s2 = Units.writeWithUnit(2**20, 'B', binary=True)
    assert '1.00e+00 MiB' in s2 or '1.00e+00 MiB' in s2