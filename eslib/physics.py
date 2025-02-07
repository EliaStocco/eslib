import re
from copy import copy, deepcopy
from typing import Any, List, Tuple, Union, Optional

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from eslib.classes.bec import bec as BEC
from eslib.tools import cart2frac
from eslib.tools import convert

#---------------------------------------#
def lorentzian(x, x0, y0, gamma):
    """
    Lorentzian function.
    
    Parameters:
    x : array-like
        The x-values where the function is evaluated.
    x0 : float
        The x-value at which the Lorentzian is centered.
    y0 : float
        The height of the Lorentzian peak.
    gamma : float
        The half-width at half-maximum (HWHM) of the Lorentzian peak.
        
    Returns:
    array-like
        The Lorentzian function evaluated at each x-value.
    """
    out = y0 * gamma**2 / ((x - x0)**2 + gamma**2)
    # assert np.allclose(np.max(out), y0), "y0 != max(out)"
    return out

#---------------------------------------#
def compute_density(atoms:Atoms)->float:
    """Return the density of an atomic structure in g/cm3"""
    tot_mass = atoms.get_masses().sum()
    volume = atoms.get_volume()
    tot_mass = convert(tot_mass,"mass","dalton","atomic_unit")
    volume = volume*convert(1,"length","angstrom","atomic_unit")**3
    density = convert(tot_mass/volume,"density","atomic_unit","g/cm3")
    return density

#---------------------------------------#
def debye_real(w: np.ndarray, chi0: float, tau: float) -> np.ndarray:
    """
    Compute the real part of the Debye function without using complex numbers.

    Parameters:
        w (np.ndarray): Angular frequency array.
        chi0 (float): Static susceptibility.
        tau (float): Relaxation time.

    Returns:
        np.ndarray: Real part of the Debye function.
    """
    denominator = 1 + (w * tau) ** 2
    return chi0 / denominator

def debye_imag(w: np.ndarray, chi0: float, tau: float) -> np.ndarray:
    """
    Compute the imaginary part of the Debye function without using complex numbers.

    Parameters:
        w (np.ndarray): Angular frequency array.
        chi0 (float): Static susceptibility.
        tau (float): Relaxation time.

    Returns:
        np.ndarray: Imaginary part of the Debye function.
    """
    numerator = -chi0 * w * tau
    denominator = 1 + (w * tau) ** 2
    return numerator / denominator

def Debye_unc(w:np.ndarray,chi0:float,tau:float):
    real = debye_real(w,chi0,tau)
    imag = debye_imag(w,chi0,tau)
    return np.abs(real),np.abs(imag)

def Debye(w:np.ndarray,chi0:float,tau:float)->np.ndarray:
    return chi0/(1-1j*w*tau)       
    
def Debye_fit(_w,chi0,tau):
    w = np.zeros_like(_w).reshape((-1,2))
    w[:,0] = _w[:int(len(_w)/2)]
    w[:,1] = _w[int(len(_w)/2):]
    assert np.allclose(w[:,0],w[:,1]), "w[:,0] != w[:,1]"
    w = w[:,0]
    res = Debye(w,chi0,tau)
    out = np.zeros((len(w),2),dtype=float)
    out[:,0] = np.real(res)
    out[:,1] = np.imag(res)
    return out.flatten()

def kin2temp(kin: float,Natoms:int,kin_unit:Optional[str]="atomic_unit",temp_unit:Optional[str]="atomic_unit") -> float:
    """
    Convert kinetic energy to temperature.
    
    Parameters:
    kin (float): Kinetic energy
    Natoms (int): Number of atoms
    kin_unit (str): Unit of the kinetic energy
    temp_unit (str): Unit of the temperature to be converted to
    
    Returns:
    float: Temperature in the specified unit
    
    Notes:
    The conversion is done using the formula T = 2K / (3N) where K is the kinetic energy and N is the number of atoms.
    The conversion is internally performed in atomic units where the Boltzmann constant is 1.
    """
    kin = convert(kin,"energy",kin_unit,"atomic_unit")
    temp = 2*kin / (3*Natoms) 
    temp = convert(temp,"temperature","atomic_unit",temp_unit)
    return temp

def get_element_mass(symbols: Union[str, List[str]]) -> Union[float, List[float]]:
    """
    Get the atomic mass of a chemical element or a list of elements by their symbol(s).
    
    Parameters:
    symbols (Union[str, List[str]]): A chemical symbol (e.g., 'O') or a list of chemical symbols (e.g., ['O', 'H', 'C']).
    
    Returns:
    Union[float, List[float]]: The atomic mass of the element if a single symbol is provided, 
                               or a list of atomic masses if a list of symbols is provided.

    Raises:
    TypeError: If the input is not a string or a list/tuple of strings.
    ValueError: If an invalid chemical symbol is provided.
    
    Example:
    >>> get_element_mass('O')
    15.999
    >>> get_element_mass(['O', 'H', 'C'])
    [15.999, 1.008, 12.011]
    """
    if isinstance(symbols, str):  # If input is a single string
        if symbols not in atomic_numbers:
            raise ValueError(f"Invalid chemical symbol: {symbols}")
        atomic_number = atomic_numbers[symbols]
        return atomic_masses[atomic_number]
    
    elif isinstance(symbols, (list, tuple,np.ndarray)):  # If input is a list or tuple
        masses = []
        for symbol in symbols:
            if symbol not in atomic_numbers:
                raise ValueError(f"Invalid chemical symbol: {symbol}")
            atomic_number = atomic_numbers[symbol]
            masses.append(atomic_masses[atomic_number])
        return masses
    
    else:
        raise TypeError("Input must be a string or a list/tuple of strings.")


def FWHM2sigma(FWHM: float) -> float:
    """
    Convert FWHM to sigma.

    Args:
        FWHM (float): FWHM.

    Returns:
        float: Sigma.
    """
    return FWHM / (2*np.sqrt(2*np.log(2)))

def sigma2FWHM(sigma: float) -> float:
    """
    Convert sigma to FWHM.

    Args:
        sigma (float): Sigma.

    Returns:
        float: FWHM.
    """
    return 2*np.sqrt(2*np.log(2)) * sigma

def oxidation_number(molecule: List[str], numbers: dict = None):
    default_oxidation_numbers = {
        ('H',): 1,
        ('O',): -2,
        ('F',): -1,
        ('Cl',): -1,
        ('Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'): 1,
        ('Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'): 2,
        ('Nb',) : 5,
    }

    if numbers is None:
        numbers = {}

    h = default_oxidation_numbers.copy()
    for key, value in numbers.items():
        if value is not None:
            for old_key in list(h.keys()):
                if key in old_key:
                    new_key = tuple(atom for atom in old_key if atom != key)
                    # h[new_key] = h.pop(old_key)
                    if new_key:
                        h[new_key] = h.pop(old_key)
                        # del h[old_key]
                        h[tuple([key])] = value
                        pass
                    else:
                        # h[new] = h.pop(old_key)
                        # del h[old_key]
                        h[tuple([key])] = value
                        pass
                        


    def find_oxidation(a):
        for key in h.keys():
            if a in key:
                return h[key]

    r = []
    for atom in molecule:
        match = re.match(r'([A-Z][a-z]*)(\d*)', atom)
        symbol, multiplier = match.groups()
        oxidation = find_oxidation(symbol)
        if oxidation is None:
            raise ValueError(f"Oxidation number for {symbol} not provided")
        if multiplier == '':
            multiplier = 1
        else:
            multiplier = int(multiplier)
        r.extend([oxidation] * multiplier)

    r_without_none = [x for x in r if x is not None]
    out = [x if x is not None else -sum(r_without_none) // r_without_none.count(None) for x in r]
    return np.asarray(out)

def bec_from_oxidation_number(atoms:Atoms,on:List[str])->BEC:
    """Construct diagonal Born Effective Charges from the oxidation numbers of the chemical species.
    The Acoustic Sum Rule is not guaranteed to be satisfied."""
    Natoms = atoms.get_global_number_of_atoms()
    bec = np.zeros((Natoms,3,3))
    for n in range(Natoms):
        bec[n,:,:] = on[n] * np.eye(3)
    bec = bec.reshape((1,-1,3))
    return BEC.from_numpy(bec)

def compute_dipole_quanta(trajectory:Union[List[Atoms],Any],in_keyword:str="dipole",out_keyword:str="quanta")->Tuple[Union[List[Atoms],Any],np.ndarray]:
    out:Union[List[Atoms],Any] = deepcopy(trajectory)
    quanta = np.full((len(trajectory),3),np.nan)
    for n,atoms in enumerate(trajectory):
        quanta[n,:] = cart2frac(cell=atoms.get_cell(),v=atoms.info[in_keyword])
        out[n].info[out_keyword] = quanta[n,:]
    return out, quanta

    
