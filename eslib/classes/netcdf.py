import netCDF4 as nc
import numpy as np
from ase import Atoms
from typing import List


def read_netcdf(filename: str) -> List[Atoms]:
    """
    Read a list of Atoms objects from a NetCDF file, where positions and other attributes
    are stored collectively and reconstructed based on atom counts.
    """
    with nc.Dataset(filename, 'r') as f:
        num_atoms_list = np.array(f['num_atoms'][:])
        num_structures = len(num_atoms_list)

        positions = np.array(f['positions'][:])
        atomic_numbers = np.array(f['atomic_numbers'][:])
        cells = np.array(f['cells'][:])
        pbc = np.array(f['pbc'][:])

        atoms_list = []
        start_idx = 0

        for i in range(num_structures):
            num_atoms = num_atoms_list[i]
            end_idx = start_idx + num_atoms

            atom_positions = positions[start_idx:end_idx]
            atom_numbers = atomic_numbers[start_idx:end_idx]
            atom_cell = cells[i]
            atom_pbc = pbc[i]

            atoms = Atoms(positions=atom_positions, numbers=atom_numbers, cell=atom_cell, pbc=atom_pbc)

            for var_name in f.variables:
                if "_info_" in var_name:
                    info_key = var_name.replace("_info_", "")
                    atoms.info[info_key] = np.array(f[var_name][i])
                elif "_arrays_" in var_name:
                    array_key = var_name.replace("_arrays_", "")
                    atoms.arrays[array_key] = np.array(f[var_name][start_idx:end_idx])

            atoms_list.append(atoms)
            start_idx = end_idx

    return atoms_list


def write_netcdf(filename: str, atoms_list: List[Atoms]) -> None:
    """
    Write a list of Atoms objects to a NetCDF file, storing positions and other attributes
    collectively to save space.
    """
    # Collect data from atoms_list
    num_atoms_list = [len(atoms) for atoms in atoms_list]
    total_atoms = sum(num_atoms_list)

    positions = np.vstack([atoms.get_positions() for atoms in atoms_list])
    atomic_numbers = np.hstack([atoms.get_atomic_numbers() for atoms in atoms_list])
    cells = np.array([atoms.get_cell().array for atoms in atoms_list])
    pbc = np.array([atoms.get_pbc() for atoms in atoms_list])

    with nc.Dataset(filename, 'w', format='NETCDF4') as f:
        # Define dimensions
        f.createDimension('total_atoms', total_atoms)
        f.createDimension('structures', len(atoms_list))
        f.createDimension('cell_dim', 3)

        # Create variables with correct dimensions
        num_atoms_var = f.createVariable('num_atoms', 'i8', ('structures',))
        positions_var = f.createVariable('positions', 'f8', ('total_atoms', 'cell_dim'))
        atomic_numbers_var = f.createVariable('atomic_numbers', 'i8', ('total_atoms',))
        cells_var = f.createVariable('cells', 'f8', ('structures', 'cell_dim', 'cell_dim'))
        pbc_var = f.createVariable('pbc', 'i8', ('structures', 'cell_dim'))

        # Store the collective data
        num_atoms_var[:] = np.asarray(num_atoms_list).astype(np.int64)
        positions_var[:] = np.asarray(positions).astype(np.float64)
        atomic_numbers_var[:] = np.asarray(atomic_numbers).astype(np.int64)
        cells_var[:] = np.asarray(cells).astype(np.float64)
        pbc_var[:] = np.asarray(pbc).astype(np.int64)

        # Store info and arrays collectively
        for key in atoms_list[0].info.keys():
            info_var = f.createVariable(f'_info_{key}', 'f8', ('structures',))
            info_var[:] = [atoms.info[key] for atoms in atoms_list]

        for key in atoms_list[0].arrays.keys():
            if key in ["positions", "numbers"]:
                continue
            array_var = f.createVariable(f'_arrays_{key}', 'f8', ('total_atoms', atoms_list[0].arrays[key].shape[1]))
            array_var[:] = np.vstack([atoms.arrays[key] for atoms in atoms_list])
