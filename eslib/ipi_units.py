"""Contains atomic masses, fundamental constants, and unit conversions
to/from atomic units.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import re


class Constants(object):
    """Class whose members are fundamental constants.

    Attributes:
        kb: Boltzmann constant.
        hbar: Reduced Planck's constant.
        amu: Atomic mass unit.
    """

    kb = 1.0
    hbar = 1.0
    amu = 1822.8885
    e = 1.0  # elementary charge, electron charge = - |e|


# these are the conversion FROM the unit stated to internal (atomic) units
# "angstrom": 1.8897261 MEANS 1 angstrom = 1.8897261 atomic_unit
UnitMap = {
    "undefined": {"": 1.00, "automatic": 1.00, "atomic_unit": 1.00},
    "energy": {
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "electronvolt": 0.036749326,
        "j/mol": 0.00000038087989,
        "cal/mol": 0.0000015946679,
        "kelvin": 3.1668152e-06,
        "rydberg" : 0.5,
    },
    "charge" : {
        "" : 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "coulomb" : 1./1.602176634e-19 # 1 coulomb = 1/1.602176634e-19 atomic_unit
    },
    "temperature": {
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "kelvin": 3.1668152e-06,
    },
    "time": {"": 1.00, "automatic": 1.00, "atomic_unit": 1.00, "second": 4.1341373e16},
    "frequency": {  # NB Internally, ANGULAR frequencies are used.
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "inversecm": 4.5563353e-06,
        "hertz*rad": 2.4188843e-17,
        "hertz": 1.5198298e-16,
        "hz": 1.5198298e-16,
        "thz": 1.5198298e-4,
    },
    "electric-field": {  # Hartree/Bohr radius\
        # https://physics.nist.gov/cgi-bin/cuu/Value?auefld
        # 1Hartree = 27.2113862459 eV
        #    1Bohr = 0.5291772109  A
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "v/ang": 0.019446903811441516,  # 0.5291772109/27.2113862459
    },
    "electric-dipole": {  # electron charge * Bohr
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "eang": 1.8897261,  # electron charge * angstrom
    },
    "polarization": {  # electron charge * Bohr
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
    },
    "ms-momentum": {  # TODO fill up units here (mass-scaled momentum)
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
    },
    "length": {  # TODO move angles to separate entry;
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "angstrom": 1.8897261, # 1 angstrom = 1.8897261 atomic_unit
        "meter": 1.8897261e10,
        "radian": 1.00,
        "degree": 0.017453292519943295,
    },
    "volume": {
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "angstrom3": 6.748334231,
    },
    "velocity": {
        "": 1.00, 
        "automatic": 1.00, 
        "atomic_unit": 1.00, 
        "m/s": 4.5710289e-7,
    },
    "momentum": {"": 1.00, "automatic": 1.00, "atomic_unit": 1.00},
    "mass": {
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "dalton": 1.00 * Constants.amu,
        "electronmass": 1.00,
    },
    "pressure": {
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "bar": 3.398827377e-9,
        "atmosphere": 3.44386184e-9,
        "pascal": 3.398827377e-14,
        "ev/ang3": 0.0054456877,
    },
    "density": {"": 1.00, "automatic": 1.00, "atomic_unit": 1.00, "g/cm3": 162.67263},
    "force": {
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "newton": 12137805,
        "ev/ang": 0.019446904,
    },
    "hessian": {
        "": 1.00,
        "automatic": 1.00,
        "atomic_unit": 1.00,
        "ev/ang^2": 0.010290858,
    },
}

# a list of magnitude prefixes
UnitPrefix = {
    "": 1.0,
    "yotta": 1e24,
    "zetta": 1e21,
    "exa": 1e18,
    "peta": 1e15,
    "tera": 1e12,
    "giga": 1e9,
    "mega": 1e6,
    "kilo": 1e3,
    "deci" : 1e-1,
    "centi": 1e-2,
    "milli": 1e-3,
    "micro": 1e-6,
    "nano": 1e-9,
    "pico": 1e-12,
    "femto": 1e-15,
    "atto": 1e-18,
    "zepto": 1e-21,
    "yocto": 1e-24,
}


# builds a RE to match prefix and split out the base unit
UnitPrefixRE = ""
for key in UnitPrefix:
    UnitPrefixRE = UnitPrefixRE + key + "|"
UnitPrefixRE = " *(" + UnitPrefixRE[1:] + ")(.*) *"
UnitPrefixRE = re.compile(UnitPrefixRE)


#
# Atomic units are used EVERYWHERE internally. In order to quickly    #
# interface with any "outside" unit, we set up a simple conversion    #
# library.                                                            #
#


def unit_to_internal(family, unit, number):
    """Converts a number of given dimensions and units into internal units.

    Args:
        family: The dimensionality of the number.
        unit: The units 'number' is originally in.
        number: The value of the parameter in the units 'unit'.

    Returns:
        The number in internal units.

    Raises:
        ValueError: Raised if the user specified units aren't given in the
            UnitMap dictionary.
        IndexError: Raised if the programmer specified dimensionality for the
            parameter isn't in UnitMap. Shouldn't happen, for obvious reasons.
        TypeError: Raised if the prefix is correct, but the base unit is not, in
            the user specified unit string.
    """

    if not (family == "number" or family in UnitMap):
        raise IndexError(family + " is an undefined units kind.")
    if family == "number":
        return number

    if unit == "":
        prefix = ""
        base = ""
    else:
        m = UnitPrefixRE.match(unit)
        if m is None:
            raise ValueError(
                "Unit " + unit + " is not structured with a prefix+base syntax."
            )
        prefix = m.group(1)
        base = m.group(2)

    if prefix not in UnitPrefix:
        raise TypeError(prefix + " is not a valid unit prefix.")
    if not base.lower() in UnitMap[family]:
        raise TypeError(base + " is an undefined unit for kind " + family + ".")

    return number * UnitMap[family][base.lower()] * UnitPrefix[prefix]


def unit_to_user(family, unit, number):
    """Converts a number of given dimensions from internal to user units.

    Args:
        family: The dimensionality of the number.
        unit: The units 'number' should be changed to.
        number: The value of the parameter in internal units.

    Returns:
        The number in the user specified units
    """

    return number / unit_to_internal(family, unit, 1.0)
