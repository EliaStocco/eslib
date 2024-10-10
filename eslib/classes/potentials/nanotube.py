from typing import Dict, List, Union
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
import json
import torch
from zmq import device
from eslib.tools import convert
from dataclasses import dataclass
from ase.io import write

def distance_from_line(point: torch.Tensor, direction: torch.Tensor, pos: torch.Tensor,is_normalized: bool = False) -> torch.Tensor:
    """
    Compute the distance of a set of points from a line defined by a point and a direction.

    Parameters:
    point (torch.Tensor): A point on the line.
    direction (torch.Tensor): The direction of the line.
    pos (torch.Tensor): The set of points for which to compute the distance.
    is_normalized (bool): Whether to normalize the distance by the length of the direction vector. Defaults to False.

    Returns:
    torch.Tensor: The distance of each point from the line.
    """
    v = pos - point
    cross = torch.cross(direction[None,:], v,dim=1)
    if is_normalized:
        return torch.linalg.norm(cross,dim=1)/torch.linalg.norm(direction)
    else:
        return torch.linalg.norm(cross,dim=1)

@dataclass
class NanoTube:
    """
    Class for computing confining potential inside a nanotube.
    """

    point       : Union[List[float],torch.Tensor]
    direction   : Union[List[float],torch.Tensor]
    degrees     : Union[List[int],torch.Tensor]
    coefficients: Union[List[float],torch.Tensor]
    symbols     : List[str]
    device      : torch.device
    
    def __post_init__(self):
        """
        Initializes the NanoTube object's attributes.

        Converts the point, direction, degrees, and coefficients into torch tensors with the specified data types.
        Normalizes the direction vector to have a length of 1.

        Parameters:
        None

        Returns:
        None
        """
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and str(self.device).lower() == "cuda" else 'cpu')
        
        self.point = torch.tensor(self.point, requires_grad=False, dtype=torch.float,device=self.device)
        self.direction = torch.tensor(self.direction, requires_grad=False, dtype=torch.float,device=self.device)
        self.direction = self.direction / torch.linalg.norm(self.direction)

        self.degrees = torch.as_tensor(self.degrees, dtype=torch.int32,device=self.device)
        self.coefficients = torch.as_tensor(self.coefficients, dtype=torch.float,device=self.device)
        
        assert self.point.shape == (3,), "Point must be a 3D vector."
        assert self.direction.shape == (3,), "Direction must be a 3D vector."

    def potential(self, pos: torch.Tensor):
        """
        Calculate the potential energy of a set of positions in a nanotube.

        Args:
            pos (torch.Tensor): A tensor of shape (N, 3) representing the positions of N particles in the nanotube.

        Returns:
            torch.Tensor: A scalar tensor representing the total potential energy of the system.

        Raises:
            AssertionError: If the input tensor `pos` is not a 2-dimensional tensor.

        Notes:
            - The potential energy is calculated by summing the contributions from each degree of the polynomial
              expansion.
            - The distance from the line defined by the point and direction is computed using the `distance_from_line`
              function.
        """
        assert pos.ndim == 2, "pos must be a 2D tensor"
        distance = distance_from_line(self.point, self.direction, pos,is_normalized=True)
        potential = torch.zeros_like(distance)
        
        # for deg,coeff in zip(self.degrees,self.coefficients):
        #     potential += coeff*torch.pow(distance,deg)
        # potential = torch.sum(potential)
        
        # Vectorized polynomial expansion using broadcasting
        potential = torch.sum(self.coefficients * torch.pow(distance[:, None], self.degrees), dim=1)
        
        assert potential.ndim == 0, "potential must be a scalar"
        return potential

    def potential_and_forces(self, pos: torch.Tensor):
        """
        Calculate the potential energy and forces of a set of positions in a nanotube.

        Parameters:
            pos (torch.Tensor): A tensor of shape (N, 3) representing the positions of N particles in the nanotube.

        Returns:
            tuple: A tuple containing the total potential energy of the system as a scalar numpy array and the forces as a numpy array of shape (N, 3).
        """
        assert pos.shape[1] == 3, "Position array must contain 3D coordinates."
        potential = self.potential(pos)
        grad_outputs = [torch.ones((1))]
        forces = -torch.autograd.grad([potential.unsqueeze(-1)], [pos],
                                         grad_outputs=grad_outputs)[0]
        return potential.detach().cpu().numpy(), forces.detach().cpu().numpy()

    def __call__(self, atoms: Atoms):
        """
        Evaluate the nanotube potential and forces for specified atoms in the atomic system.

        Parameters:
            atoms (ase.Atoms): Atomic system containing atoms.

        Returns:
            tuple: A tuple containing the nanotube potential energy and forces.
        """
        indices = [atom.symbol in self.symbols for atom in atoms]
        positions = torch.tensor(atoms.positions[indices,:], dtype=torch.float64, requires_grad=True,device=self.device)
        
        potential, forces_ii = self.potential_and_forces(positions)
        forces = np.zeros((atoms.get_global_number_of_atoms(), 3))
        forces[indices, :] = forces_ii

        return potential, forces


class NanoTubeCalculator(Calculator):
    """
    Custom ASE Calculator for confining potential inside a nanotube.
    
    Attentions:
        `coefficients_unit` is assumed to be an energy unit.
        Please provide the `coefficient` in `coefficients_unit`/Angstrom^`degree`.
        No unit conversion for the `lenght` part will be performed.
    """
    implemented_properties = {
        "energy"      : None,
        "free_energy" : None,
        "forces"      : None,
        "stress"      : None
    }

    dimensions = {
        "point"        : "length",
        "coefficients" : "energy",
    }

    units = {
        "length" : "angstrom",
        "energy" : "electronvolt"
    }


    def __init__(
        self,
        instructions: Union[str, Dict],
        log_file:str=None,
        **kwargs,
    ):
        """
        Initializes the NanoTubeCalculator object.

        Parameters:
            instructions (str or dict): Either a JSON file path or a dictionary containing instructions for the nanotube potential.
            log_file (str, optional): The file path to log the calculation. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            None
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

        to_delete = list()
        for k,dimension in self.dimensions.items():
            variable = f'{k}_unit'
            if variable in instructions:
                to_delete.append(variable)
                if instructions[variable] is not None:
                    factor = convert(1, dimension, _from=instructions[variable], _to=self.units[dimension])
                    if np.isscalar(instructions[k]):
                        instructions[k] *= factor
                    else:
                        instructions[k] = np.array(instructions[k]) * factor
        for k in to_delete:
            del instructions[k]
        self.engine = NanoTube(**instructions)
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
        "point": [0,0,0],
        "direction": [1,0,0],
        "degrees": [2,4,6,8],
        "coefficients": [0.2281,1.09,0.2341, 0.3254],
        "coefficients_unit": "kilocal/mol",
        "symbols": ['O'],
        "device" : "cpu"
    }

    # Initialize LennardJonesWall calculator
    NT = NanoTubeCalculator(instructions)

    # Perform calculation
    NT.calculate(bulk_water)

    # Print results
    print(NT.results)


if __name__ == "__main__":
    main()