import math
from typing import Optional, Tuple
from numerictypes import NumericTypes
from logger import Logger

class Units:
    """
    Scientific units and prefix handling for high-performance computing.

    Provides prefix scaling, unit parsing, and formatting for SI and binary units.
    All methods use camelCase. Thread-safe and context-enabled.

    Examples
    --------
    >>> Units.prefixVal('k')
    1000.0
    >>> Units.splitUnit('km')
    ('k', 'm')
    >>> Units.unitVal('1.5 km', 'm')
    1500.0
    """
    _si_prefixes = {
        'y': 1e-24, 'z': 1e-21, 'a': 1e-18, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9,
        'u': 1e-6, 'm': 1e-3, 'c': 1e-2, 'd': 1e-1, 'da': 1e1, 'h': 1e2, 'k': 1e3,
        'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15, 'E': 1e18, 'Z': 1e21, 'Y': 1e24
    }
    _binary_prefixes = {
        'Ki': 2.0**10, 'Mi': 2.0**20, 'Gi': 2.0**30, 'Ti': 2.0**40,
        'Pi': 2.0**50, 'Ei': 2.0**60
    }
    _all_prefixes = {**_si_prefixes, **_binary_prefixes}

    @classmethod
    def prefixVal(cls, pre: str) -> float:
        """
        Returns the scaling factor for a prefix.

        Parameters
        ----------
        pre : str
            The prefix string (e.g., 'k', 'M', 'u', 'Ki').

        Returns
        -------
        float
            The scaling factor for the prefix.
        """
        pre = pre.strip()
        if not pre:
            return 1.0
        if pre in cls._all_prefixes:
            return cls._all_prefixes[pre]
        Logger.getInstance().error(f"Unknown prefix: {pre}")
        return 1.0  # unreachable

    @classmethod
    def splitUnit(cls, unit: str) -> Tuple[str, str]:
        """
        Splits the prefix from the base unit.

        Parameters
        ----------
        unit : str
            The complete unit string (e.g., 'km').

        Returns
        -------
        tuple
            (prefix, base) where prefix may be '' and base is the unit base.
        """
        unit = unit.strip()
        if len(unit) <= 1:
            return '', unit
        # Try binary prefix first
        for pre in sorted(cls._binary_prefixes, key=len, reverse=True):
            if unit.startswith(pre):
                return pre, unit[len(pre):]
        # Then SI prefix
        for pre in sorted(cls._si_prefixes, key=len, reverse=True):
            if unit.startswith(pre):
                return pre, unit[len(pre):]
        return '', unit

    @classmethod
    def unitPrefix(cls, unit: str) -> str:
        """
        Returns the prefix part of a unit string.
        """
        return cls.splitUnit(unit)[0]

    @classmethod
    def unitForVal(cls, base: str, val: float, binary: bool = False) -> Tuple[float, str]:
        """
        Returns the closest prefix for a value and scales the value.

        Parameters
        ----------
        base : str
            The base unit (e.g., 'm').
        val : float
            The value to convert.
        binary : bool, optional
            Use binary prefixes (default: False).

        Returns
        -------
        tuple
            (scaled value, unit with prefix)
        """
        absval = abs(val)
        if absval == 0:
            return val, base
        if not binary:
            exp = int(math.floor(math.log10(absval))) if absval > 0 else 0
            for e, pre in [(-24, 'y'), (-21, 'z'), (-18, 'a'), (-15, 'f'), (-12, 'p'),
                           (-9, 'n'), (-6, 'u'), (-3, 'm'), (-2, 'c'), (0, ''), (3, 'k'),
                           (6, 'M'), (9, 'G'), (12, 'T'), (15, 'P'), (18, 'E'), (21, 'Z'), (24, 'Y')]:
                if exp <= e:
                    scale = cls._si_prefixes.get(pre, 1.0)
                    return val / scale, pre + base
            return val / 1e24, 'Y' + base
        else:
            exp2 = int(math.floor(math.log(absval) / math.log(2))) if absval > 0 else 0
            for e, pre in [(0, ''), (10, 'Ki'), (20, 'Mi'), (30, 'Gi'), (40, 'Ti'), (50, 'Pi'), (60, 'Ei')]:
                if exp2 <= e:
                    scale = cls._binary_prefixes.get(pre, 1.0)
                    return val / scale, pre + base
            return val / 2.0**60, 'Ei' + base

    @classmethod
    def unitVal(cls, n: str, unit: Optional[str] = None) -> float:
        """
        Attempts to cast a value and unit to a base.

        Parameters
        ----------
        n : str
            The input value and unit (e.g., '1.5 km').
        unit : str, optional
            The base unit to cast to (e.g., 'm').

        Returns
        -------
        float
            The value in the base unit.
        """
        # Parse value and unit
        parts = n.strip().split()
        if not parts:
            Logger.getInstance().error(f"No number in: {n}")
        val = float(parts[0])
        unitin = parts[1] if len(parts) > 1 else ''
        scaleout = 1.0
        baseout = None
        if unit:
            prefix, baseout = cls.splitUnit(unit)
            scaleout = cls.prefixVal(prefix)
        if unitin:
            prefix, basein = cls.splitUnit(unitin)
            scalein = cls.prefixVal(prefix)
            if unit and baseout is not None and basein != baseout:
                Logger.getInstance().error(f"Base units do not match ({basein} != {baseout}).")
            val = val * scalein / scaleout
        else:
            val = val / scaleout
        return val

    @classmethod
    def writeWithUnit(cls, val: float, base: str, frmt: Optional[str] = None, binary: bool = False) -> str:
        """
        Formats a value for printing with the closest prefix.

        Parameters
        ----------
        val : float
            The numeric value.
        base : str
            The base unit of val.
        frmt : str, optional
            Format string for the numeric value (default: '%.2e').
        binary : bool, optional
            Use binary prefixes (default: False).

        Returns
        -------
        str
            The formatted string.
        """
        val0, uni = cls.unitForVal(base, val, binary)
        if frmt is None:
            frmt = '%.2e'
        return f"{frmt % val0} {uni}"