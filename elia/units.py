import pint
import xarray as xr
from typing import Union
import numpy as np

__all__ = ["atomic_unit","ureg","remove_unit","set_unit","get_unit","check_dim"]

ureg = pint.UnitRegistry()

atomic_unit = {
    "energy"        : ureg.hartree , # if use_pint else 1,
    "length"        : ureg.bohr , # if use_pint else 1,
    "mass"          : ureg.electron_mass , # if use_pint else 1,
    "action"        : ureg.hbar , # if use_pint else 1,
    "dimensionless" : ureg.dimensionless , # if use_pint else 1
}
atomic_unit["time"]      = atomic_unit["action"] / atomic_unit["energy"]
atomic_unit["velocity"]  = atomic_unit["length"] / atomic_unit["time"]
atomic_unit["frequency"] = 1 / atomic_unit["time"]

dtype = Union[xr.DataArray,np.ndarray]
utype = pint.Unit

def remove_unit(array:dtype)->dtype:
    """Returns a tuple with the input array without the ```pint``` unit, and the removed ```pint``` unit."""
    out = array.copy()
    unit = get_unit(out)
    try:
        if isinstance(out,xr.DataArray):
            out.data = out.data.magnitude
        else:
            out = out.magnitude
    except:
        pass
    return out,unit

def get_unit(array:dtype)->utype:
    """Return the ```pint``` unit of the input array."""
    out = array.copy()
    if isinstance(out,pint.Quantity):
        return out.units
    elif isinstance(out,np.ndarray):
        return atomic_unit["dimensionless"]
    else:
        return get_unit(out.data)


def set_unit(array:dtype,unit:utype)->dtype:
    """Return the input array with the input ```pint``` unit."""
    out = array.copy()
    if isinstance(out,xr.DataArray):
        tmp = set_unit(out.data,unit)
        out.data = tmp
        return out
    else:
        out *= unit / get_unit(out)
        return out
    
def check_dim(array:dtype,dimension:str)->bool:
    out = array.copy()
    if isinstance(out,xr.DataArray):
        return check_dim(out.data,dimension)
    elif isinstance(out,pint.Quantity):
        return out.check(dimension)
    else:
        out *= atomic_unit["dimensionless"]
        return out.check(dimension)
