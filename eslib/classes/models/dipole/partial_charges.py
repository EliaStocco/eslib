from ase.calculators.calculator import Calculator
from ase.calculators.calculator import all_changes
from classes.models.dipole.baseclass import DipoleModel
from eslib.tools import add_info_array
from dataclasses import dataclass, field
from ase import Atoms
from typing import List, Dict, Any
import numpy as np

@dataclass
class DipolePartialCharges(DipoleModel):

    charges: Dict[str,float]
    compute_BEC: bool                      = field(default=False,init=True)         
    implemented_properties: Dict[str, Any] = field(init=False) 

    def __post_init__(self) -> None:
        """Initialize DipolePartialCharges object."""
        Calculator.__init__(self)
        self.implemented_properties = {"dipole" : (float, 3),}
        if self.compute_BEC:
            new_prop = {
                "BEC" : (float, ("natoms", 9)),
                "BECx": (float, ("natoms", 3)),
                "BECy": (float, ("natoms", 3)),
                "BECz": (float, ("natoms", 3))
            }
            self.implemented_properties.update(new_prop)            

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
            
    def get(self,traj:List[Atoms])->np.ndarray:
        return self.compute(traj,raw=True)["dipole"]

    def compute(self,traj:List[Atoms],prefix:str="",raw:bool=False):
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
        data = {f"dipole": None}
        new_data = {f"{prefix}dipole": dipole}
        if self.compute_BEC:
            BEC = [None]*len(traj)
            for n,structure in enumerate(traj):
                BEC[n] = self._get_BEC(structure)
            data["BEC"] = None
            new_data[f"{prefix}BEC"] = np.asarray(BEC)
        shapes = {prefix + k: self.implemented_properties[k] for k in data.keys()}
        if raw:
            return new_data
        else:
            # shapes = {prefix + k: self.implemented_properties[k] for k in data.keys()}
            return add_info_array(traj,new_data,shapes)

        
    def calculate(self, atoms:Atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        dipole = self.get([atoms]).flatten()
        assert dipole.shape == (3,), f"Invalid shape for 'dipole'. Expected (3,), got {dipole.shape}"
        self.results = {"dipole": dipole}
        if self.compute_BEC:
            Z = self._get_BEC(atoms)
            assert Z.shape == (atoms.get_global_number_of_atoms(),9), f"Invalid shape for 'BEC'. Expected ({atoms.get_global_number_of_atoms()},9), got {Z.shape}"
            self.results["BEC"]  = Z
            Z = Z.reshape(atoms.get_global_number_of_atoms(),3,3)
            self.results["BECx"] = Z[:,0,:]
            self.results["BECy"] = Z[:,1,:]
            self.results["BECz"] = Z[:,2,:]


    def _get_BEC(self, atoms:Atoms):
        charges = self.get_all_charges(atoms)
        N = atoms.get_global_number_of_atoms()
        BEC = np.zeros((N,3,3))
        for n in range(N):
            for m in range(3):
                BEC[n,m,m] = charges[n]
        return BEC.reshape((N,9))


    def summary(self, string: str = "\t") -> None:
        """Print summary of the model."""       
        super().summary(string=string)
        args = {
            "compute_BEC": self.compute_BEC,
            "properties": list(self.implemented_properties.keys())
        }
        max_key_length = max(len(key) for key in args.keys())+1
        for k, v in args.items():
            print("\t{:s}{:<{width}}: {}".format(string, k, v, width=max_key_length))
        for c in self.charges:
            print("\t{:s}{:s}: {:>f}".format(string, c, self.charges[c]))
        