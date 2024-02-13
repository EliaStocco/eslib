from ase.build import bulk
from ase.calculators.emt import EMT
from phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms
import numpy as np

# Create a primitive cell
primitive_atoms = bulk("Si", cubic=True)
primitive_atoms.set_calculator(EMT())
primitive_atoms.positions += [0.1, 0.2, 0.3]  # Offset for demonstration

# Create a supercell
supercell = primitive_atoms * (2, 2, 2)

# PhonopyAtoms object for the primitive cell
phonopy_primitive = PhonopyAtoms(symbols=primitive_atoms.get_chemical_symbols(),
                                 scaled_positions=primitive_atoms.get_scaled_positions(),
                                 cell=primitive_atoms.cell)

# Phonopy object
phonon = Phonopy(phonopy_primitive, np.asarray(supercell.cell))

# Get the mapping between atoms in the supercell and primitive cell
supercell_to_primitive_map = phonon.supercell_to_primitive_map()

# Print the mapping for each atom in the supercell
for i, atom in enumerate(supercell):
    primitive_index = supercell_to_primitive_map[i]
    print(f"Supercell atom {i+1} corresponds to primitive atom {primitive_index+1}")
