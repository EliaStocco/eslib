from ase.calculators.calculator import Calculator
from ase.calculators.calculator import all_changes
from eslib.classes.models.dipole.baseclass import DipoleModel
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
        
    def calculate(self, atoms: Atoms = None, properties: list = None,system_changes: str = all_changes) -> None:
        """
        Calculate the results for the given atoms.

        This method is necessary when using `ase`.

        Args:
            atoms (Atoms, optional): The atoms for which to calculate the results.
                Defaults to None.
            properties (list, optional): The properties to calculate.
                Defaults to None.
            system_changes (str, optional): The changes in the system.
                Defaults to all_changes.

        Returns:
            None
        """
        # Call the parent class's calculate method
        super().calculate(atoms, properties, system_changes)

        # Get the dipole for the given atoms
        self.results = self.compute([atoms])# .flatten()

    def compute(self,traj:List[Atoms],prefix:str="",raw:bool=False):
        """
        Compute the dipole moment and the Born-Electron-Cloud (BEC) for a given trajectory of atomic structures.

        Parameters:
            traj (List[Atoms]): The trajectory of atomic structures.
            prefix (str, optional): A prefix to be added to the keys in the returned dictionary. Defaults to "".
            raw (bool, optional): If True, return the raw data without any additional processing. Defaults to False.

        Returns:
            Union[Dict[str, np.ndarray], Dict[str, np.ndarray], List[np.ndarray]]: A dictionary containing the dipole moment and the BEC for each structure in the trajectory. If `raw` is True, return a dictionary with the raw data.

        Raises:
            ValueError: If a structure in the trajectory is not charge neutral.

        Notes:
            - The dipole moment is computed as the sum of the atomic dipoles.
            - The Born-Effective-Charges (BEC) are computed using the `self._get_BEC` method.
            - The BEC is stored as a 3D array with shape (N, 9), where N is the number of atoms in the structure.
            - The BEC is reshaped into three separate arrays: BECx, BECy, and BECz.
            - The prefix is added to the keys in the returned dictionary.

        """
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
            data["BECx"] = None
            data["BECy"] = None
            data["BECz"] = None
            BEC = np.asarray(BEC)
            new_data[f"{prefix}BEC"]  = BEC
            new_data[f"{prefix}BECx"] = [None]*len(traj)
            new_data[f"{prefix}BECy"] = [None]*len(traj)
            new_data[f"{prefix}BECz"] = [None]*len(traj)
            for n,atoms in enumerate(traj):
                Z = np.asarray(BEC[n]).reshape(atoms.get_global_number_of_atoms(),3,3)
                new_data[f"{prefix}BECx"][n] = Z[:,0,:]
                new_data[f"{prefix}BECy"][n] = Z[:,1,:]
                new_data[f"{prefix}BECz"][n] = Z[:,2,:]
            new_data[f"{prefix}BECx"] = np.asarray(new_data[f"{prefix}BECx"])
            new_data[f"{prefix}BECy"] = np.asarray(new_data[f"{prefix}BECy"])
            new_data[f"{prefix}BECz"] = np.asarray(new_data[f"{prefix}BECz"])
        shapes = {prefix + k: self.implemented_properties[k] for k in data.keys()}
        if raw:
            return new_data
        else:
            # shapes = {prefix + k: self.implemented_properties[k] for k in data.keys()}
            return add_info_array(traj,new_data,shapes)
        
    def get(self,traj:List[Atoms])->np.ndarray:
        """
        Calculate the dipole moment for a given trajectory of atomic structures.

        Args:
            traj (List[Atoms]): The trajectory of atomic structures.

        Returns:
            np.ndarray: The dipole moment for each structure in the trajectory.
        """
        return self.compute(traj,raw=True)["dipole"]

    def _get_BEC(self, atoms:Atoms):
        """
        Calculate the Born Effective Charges (BEC) tensor for a given Atoms object.

        Parameters:
            atoms (Atoms): The Atoms object for which to calculate the BEC tensor.

        Returns:
            np.ndarray: The BEC tensor with shape (N, 9), where N is the number of atoms.
        """
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
        