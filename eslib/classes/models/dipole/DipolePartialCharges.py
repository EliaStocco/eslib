from eslib.classes.dipole.DipoleModel import DipoleModel
from dataclasses import dataclass
from ase import Atoms
from typing import List, Dict
import numpy as np

@dataclass
class DipolePartialCharges(DipoleModel):

    charges: Dict[str,float]

    def set_charges(self,charges:dict):
        if self.charges.keys() != charges.keys():
            raise ValueError("error: different chemical species specified")
        self.charges = charges

    def get_all_charges(self,structure:Atoms)->np.ndarray:
        charges = [ self.charges[s] for s in structure.get_chemical_symbols() ]
        return np.asarray(charges)

    def compute_total_charge(self,structure:Atoms)->float:
        charges = self.get_all_charges(structure)
        return np.sum(charges)

    def check_charge_neutrality(self,structure:Atoms)->bool:
        tot = self.compute_total_charge(structure)
        if np.abs(tot) > 1e-12:
            return False
        return True
    
    def impose_charge_neutrality(self,structure:Atoms,inplace:bool=True)->Dict[str,float]:
        tot_charge = self.compute_total_charge(structure)
        Natoms = structure.get_global_number_of_atoms()
        charges = self.get_all_charges(structure)
        charges -= tot_charge/Natoms

        symbols = structure.get_chemical_symbols()
        _, index = np.unique(symbols,return_index=True)

        neutral_charges = {}
        for s,n in zip(self.charges.keys(),index):
            neutral_charges[s] = charges[n]

        # shift = tot_charge#  / Natoms
        # occurrence = dict(Counter(structure.get_chemical_symbols())) 
        # neutral_charges = {}
        # for s in self.charges.keys():
        #     neutral_charges[s] = self.charges[s] - shift/occurrence[s]
            
        # test
        test = DipolePartialCharges(neutral_charges)
        # test.compute_total_charge(structure)
        if not test.check_charge_neutrality(structure):
            raise ValueError("coding error")
        if inplace:
            self.set_charges(neutral_charges)
        return neutral_charges
            
    def get(self,traj:List[Atoms]):
        return self.compute(traj)

    def compute(self,traj:List[Atoms],**argv):
        dipole = np.zeros((len(traj),3))
        for n,structure in enumerate(traj):
            if not self.check_charge_neutrality(structure):
                raise ValueError("structure {:d} is not charge neutral.".format(n))
            charges = [ self.charges[s] for s in structure.get_chemical_symbols() ]
            Natoms = structure.get_global_number_of_atoms()
            charges = np.asarray(charges).reshape((Natoms,1))
            positions:np.ndarray = structure.get_positions()
            atomic_dipoles = charges * positions
            dipole[n] = atomic_dipoles.sum(axis=0)
            # # little test
            # test  = ( charges * ( positions + np.random.rand(3) )).sum(axis=0)
            # if not np.allclose(test,dipole[n]):
            #     raise ValueError("coding error")
        return dipole

    def summary(self, string: str = "\t") -> None:
        """Print summary of the model."""       
        super().summary(string=string)
        for c in self.charges:
            print("\t{:s}{:s}: {:>f}".format(string, c, self.charges[c]))
        