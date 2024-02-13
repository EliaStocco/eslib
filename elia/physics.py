from typing import List
import re
import numpy as np
from ase import Atoms
from elia.classes.bec import bec as BEC

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
