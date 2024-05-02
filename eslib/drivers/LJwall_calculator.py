from typing import Dict, List, Union
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
import json
import torch
from eslib.tools import convert
from dataclasses import dataclass
from ase.io import write

@dataclass
class LJPotential:
    """
    Class for computing Lennard-Jones potential and forces.
    """

    zplane: float
    sigma: float
    epsilon: float
    symbols: List[str]
    first_power: int
    second_power: int

    def potential(self, z: torch.Tensor):
        """
        Calculate the Lennard-Jones potential based on the z value of the atomic positions.

        Parameters:
            z (torch.Tensor): The z-coordinate values of the atoms.

        Returns:
            torch.Tensor: The Lennard-Jones potential energy.
        """
        r = torch.abs(z)  # Distance along z-axis
        lj_potential = 4 * self.epsilon * ((self.sigma / r) ** self.first_power - (self.sigma / r) ** self.second_power)
        return lj_potential

    def potential_and_forces(self, z: torch.Tensor):
        """
        Calculate the Lennard-Jones potential and forces based on the z value of the atomic positions.

        Parameters:
            z (torch.Tensor): The z-coordinate values of the atoms.

        Returns:
            tuple: A tuple containing the Lennard-Jones potential energy and forces.
        """
        lj_potential = self.potential(z)
        grad_outputs = [torch.ones((lj_potential.shape[0], 1))]
        lj_forces = -torch.autograd.grad([lj_potential.unsqueeze(-1)], [z],
                                         grad_outputs=grad_outputs)[0]
        return lj_potential.detach().cpu().numpy(), lj_forces.detach().cpu().numpy()

    def __call__(self, atoms: Atoms):
        """
        Evaluate the Lennard-Jones potential and forces for specified atoms in the atomic system.

        Parameters:
            atoms (ase.Atoms): Atomic system containing atoms.

        Returns:
            tuple: A tuple containing the Lennard-Jones potential energy and forces.
        """
        positions = torch.tensor(atoms.positions[:, 2] - self.zplane, dtype=torch.float64, requires_grad=True)
        indices = [atom.symbol in self.symbols for atom in atoms]
        positions = positions[indices]
        energies, forces = self.potential_and_forces(positions)

        lj_energy = np.zeros(atoms.get_global_number_of_atoms())
        lj_forces = np.zeros((atoms.get_global_number_of_atoms(), 3))

        lj_energy = np.sum(energies)
        lj_forces[indices, 2] = forces

        return lj_energy, lj_forces


class LennardJonesWall(Calculator):
    """
    Custom ASE Calculator for Lennard-Jones potential calculations.
    """
    implemented_properties = {
        "energy"      : None,
        "free_energy" : None,
        "forces"      : None,
        "stress"      : None
    }

    def __init__(
        self,
        instructions: Union[str, Dict],
        log_file:str=None,
        **kwargs,
    ):
        """
        Initialize the LennardJonesWall calculator.

        Parameters:
            instructions (str or dict): Either a JSON file path or a dictionary containing instructions for the LJ potential.
            **kwargs: Additional keyword arguments.
        """
        Calculator.__init__(self, **kwargs)
        self.results = {}

        # Read instructions from file or use provided dictionary
        if isinstance(instructions, str):
            with open(instructions, 'r') as f:
                instructions = json.load(f)
        elif isinstance(instructions, dict):
            pass
        else:
            raise ValueError("`instructions` can be `str` or `dict` only.")

        # Convert units if specified
        if "sigma_unit" in instructions and instructions["sigma_unit"] is not None:
            factor = convert(1, "length", _from=instructions["sigma_unit"], _to="atomic_unit")
            instructions["sigma"] *= factor
        if "epsilon_unit" in instructions and instructions["epsilon_unit"] is not None:
            factor = convert(1, "energy", _from=instructions["epsilon_unit"], _to="atomic_unit")
            instructions["epsilon"] *= factor

        # Remove unit keys from instructions
        if "sigma_unit" in instructions:
            del instructions["sigma_unit"]
        if "epsilon_unit" in instructions:
            del instructions["epsilon_unit"]

        # Initialize LJPotential engine
        self.engine = LJPotential(**instructions)
        self.logger = log_file
        self.save = None

    def calculate(self, atoms: Atoms = None, properties=None, system_changes=all_changes):
        """
        Calculate properties.

        Parameters:
            atoms (ase.Atoms): ASE Atoms object.
            properties ([str]): Properties to be computed (used internally by ASE).
            system_changes ([str]): System changes since last calculation (used internally by ASE).
        """
        # Call base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # Reset results
        self.results = {}

        # Compute LJ potential and forces
        e, f = self.engine(atoms)

        # Populate results dictionary
        self.results["energy"] = e
        self.results["free_energy"] = e
        self.results["forces"] = f
        self.results["stress"] = np.zeros((3, 3))

        # Log results
        if self.logger is not None:
            kwargs = {
                "symbols" : atoms.get_chemical_symbols(),
                "positions" : atoms.get_positions(),
                "cell" : atoms.get_cell(),
                "pbc" : atoms.get_pbc(),
            }
            self.save = Atoms(**kwargs)
            self.save.info = {
                "energy" : self.results["energy"],
                "free_energy" : self.results["energy"],
                "stress" : self.results["stress"]
            }
            self.save.arrays["forces"] = self.results["forces"]

            with open(self.logger,'a') as ff:
                write(ff,self.save)
        pass
        

def random_water_structure(num_molecules=1):
    symbols = ['H', 'H', 'O']  # Atomic symbols for water molecule
    water_structure = Atoms()  # Initialize ASE Atoms object

    # Generate random positions for each water molecule
    for _ in range(num_molecules):
        # Randomly generate positions for each atom in the water molecule
        positions = np.random.rand(3, 3)
        
        # Append the atoms of the water molecule to the overall structure
        water_structure.extend(Atoms(symbols=symbols, positions=positions))

    return water_structure

def main():
    # Generate a random structure with 10 water molecules
    bulk_water = random_water_structure(10)

    # Instructions for Lennard-Jones potential calculation
    instructions = {
        "zplane": 0,
        "sigma": 2.569,
        "epsilon": 2.754,
        "sigma_unit": "angstrom",
        "epsilon_unit": "kilocal/mol",
        "symbols": ['O'],
        "first_power": 9,
        "second_power": 3
    }

    # Initialize LennardJonesWall calculator
    LJW = LennardJonesWall(instructions)

    # Perform calculation
    LJW.calculate(bulk_water)

    # Print results
    print(LJW.results)


if __name__ == "__main__":
    main()