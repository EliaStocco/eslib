# This file was copied from this repo: https://gitlab.com/gims-developers/gims
"""A Structure class based on ase.Atoms"""
import json
import os
import numpy as np
from ase import Atoms
from ase.io import read as ase_read
import spglib as spg

class StructureInfo:
    """Wrapper to get supportive information about the structure"""
    def __init__(self, system, sym_thresh):
        """Initializes the class from an ASE atoms object.

        Parameters:

        system: ASE atoms object

        sym_thresh: float
            Symmetry threshold for determining Bravais lattice (ASE feature) and
            Space group information (spglib feature).
        """
        self.info = {}
        self.atoms = system
        self.add_info("n_atoms", len(system), "Number of atoms")
        self.add_info("formula", system.get_chemical_formula(), "Chemical formula")

        if all(system.pbc):
            lattice = system.get_cell()[:]
            positions = system.get_scaled_positions()
            numbers = system.get_atomic_numbers()
            magmoms = system.get_initial_magnetic_moments()
            spg_cell = (lattice, positions, numbers, magmoms)
            self.add_info(
                "unit_cell_parameters",
                np.around(system.cell.cellpar(), 12),
                "Lattice parameters <br> (a, b, c, \u03B1, \u03B2, \u03B3)",
            )
            bravais = system.cell.get_bravais_lattice(eps=sym_thresh)
            self.add_info(
                "bravais", "{} {}".format(bravais.longname, bravais), "Bravais Lattice"
            )
            
            dataset = spg.get_symmetry_dataset(spg_cell, symprec=sym_thresh)
            if dataset:
                # self.dataset = dataset
                prim_lattice, prim_scaled_positions, prim_numbers = spg.find_primitive(
                    spg_cell, symprec=sym_thresh
                )
                self.primitive = Atoms(
                    cell=prim_lattice,
                    scaled_positions=prim_scaled_positions,
                    numbers=prim_numbers,
                    pbc=True,
                )
                conv_lattice, conv_scaled_positions, conv_numbers = spg.standardize_cell(spg_cell, symprec=sym_thresh)
                self.conventional = Atoms(
                    cell=conv_lattice,
                    scaled_positions=conv_scaled_positions,
                    numbers=conv_numbers,
                    pbc=True,
                )
                self.add_info("sym_thresh", sym_thresh, "Symmetry Threshold")
                self.add_info("spacegroup", dataset["number"], "Spacegroup number")
                self.add_info("hall_symbol", dataset["hall"], "Hall symbol")
                self.add_info(
                    "occupied_wyckoffs",
                    np.unique(dataset["wyckoffs"]),
                    "Occupied Wyckoff positions",
                )
                self.equivalent_atoms = dataset["equivalent_atoms"]
                self.unique_equivalent_atoms = np.unique(dataset["equivalent_atoms"]) + 1
                self.add_info(
                    "equivalent_atoms",
                    np.unique(dataset["equivalent_atoms"]) + 1,
                    "Unique equivalent atoms",
                )
                self.add_info(
                    "is_primitive", len(system) == len(prim_numbers), "Is primitive cell?"
                )

    def add_info(self, key, value, info_str=""):
        """Add new key to info dict"""

        v = value
        if isinstance(value, np.ndarray):
            v = value.tolist()
        self.info[key] = {"value": v, "info_str": info_str}

    def get_info(self):
        """Return the current info dictionary"""
        return self.info

    def __str__(self):
        str_str = "System Info\n" + "-" * 14 + "\n"
        for key, idict in self.info.items():
            # print(key,idict)
            if isinstance(idict["value"], (list, np.ndarray)):
                fmt_str = "{:30}: " + "{} " * len(idict["value"]) + "\n"
                str_str += fmt_str.format(idict["info_str"], *idict["value"])
            else:
                fmt_str = "{:30}: {}\n"
                str_str += fmt_str.format(idict["info_str"], idict["value"])
        return str_str



class Structure(Atoms):

    @classmethod
    def from_dict(cls, struct_dict):
        """Generates an ASE atoms object from the structure attached to the
        control-generator form.

        Parameters:
            struct_dict (dict):
                Dictionary containing the structure, which was sent from the client.
        """

        cell, pbc, positions, species, constraints, mag_moms, charges, c = (None,) * 8
        # find the keys in the dicts
        cell_name = 'cell' if 'cell' in struct_dict else 'latVectors'
        atoms_name = 'atoms' if 'atoms' in struct_dict else 'positions'

        if cell_name in struct_dict:
            cell = struct_dict[cell_name]
            pbc = True

        if atoms_name in struct_dict:
            positions = [p["position"] for p in struct_dict[atoms_name]]
            species = [p["species"] for p in struct_dict[atoms_name]]
            mag_moms = [p["initMoment"] for p in struct_dict[atoms_name]]
            constraints = [p["constraint"] for p in struct_dict[atoms_name]]
            charges = [p["charge"] for p in struct_dict[atoms_name]]

        if any(constraints):
            from ase.constraints import FixAtoms
            c = FixAtoms(mask=constraints)
        # print(positions,cell,species)

        return cls(
            symbols=species,
            positions=positions,
            magmoms=mag_moms,
            charges=charges,
            constraint=c,
            cell=cell,
            pbc=pbc,
        )

    @classmethod
    def from_form(cls, form_dict, is_periodic):
        """Generates a dummy ASE atoms object only from the control-generator form,
        if no structure was attached to it.

        Parameters:
            form_dict (dict):
                JSON Form data from the control generator. A list of species from the Form
                is needed.
            is_periodic (boolean):
                Whether the pseudo structure should be periodic or not.
        """
        cell, pbc = None, None
        if is_periodic:
            # Need dummy cell
            cell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            pbc = True
        return cls(symbols=form_dict["species"], cell=cell, pbc=pbc)

    @classmethod
    def from_atoms(cls, atoms):
        """Generates Structure instance from the parent Atoms instance

        Parameters:
            atoms (Atoms):
                an ase.Atoms instance
        """
        instance = cls()
        instance.__dict__ = atoms.__dict__
        return instance

    def get_info(self, sym_thresh):
        """Get additional supportive information about structure.

        Parameters:
            sym_thresh (float):
                Symmetry threshold for determining symmetry information for periodic
                structures.
        """
        s_info = StructureInfo(self, sym_thresh)
        return s_info.get_info()

    def get_primitive_cell(self, sym_thresh):
        """Get primitive cell

        Parameters:
            sym_thresh (float):
                Symmetry threshold for determining symmetry information for periodic
                structures.
        """
        s_info = StructureInfo(self, sym_thresh)
        return Structure.from_atoms(s_info.primitive)

    def get_conventional_cell(self, sym_thresh):
        """Get conventional cell

        Parameters:
            sym_thresh (float):
                Symmetry threshold
        """
        s_info = StructureInfo(self, sym_thresh)
        return Structure.from_atoms(s_info.conventional)

    def to_json(self, file_name, sym_thresh):
        """Converts ASE atoms object 'struct' to json object.

        Parameters:
            file_name (str):
                Name of the structure (inherited from the original file name)
            sym_thresh (float):
                Symmetry threshold for determining symmetry information for periodic
                structures.
        """

        json_s = {"lattice": [],
                  "atoms": [],
                  "fileName": os.path.basename(file_name),
                  "structureInfo": self.get_info(sym_thresh)}

        # print(json_s["structureInfo"])
        pbc = np.any(self.pbc)
        if pbc:
            for v in self.cell.array:
                json_s["lattice"].append(list(v))
        else:
            json_s["lattice"] = None
        atoms_data = zip(
            self.get_positions(),
            self.get_chemical_symbols(),
            self.get_initial_magnetic_moments(),
            self.get_initial_charges(),
        )
        # print(struct.constraints[0].index)
        for i, (pos, species, init_moment, charge) in enumerate(atoms_data):
            # Tried to bring it in the correct form for this function:
            # structure.addAtomData(atomPos, tokens[4], tokens[0] === ATOM_FRAC_KEYWORD)
            # in src/common/util.js. The last entry is always false, since we obtain
            # cartesian positions from ase.
            constraint = False
            try:
                if i in self.constraints[0].index:
                    constraint = True
                    # print([list(pos), species, "", constraint])
            except IndexError:
                pass
            json_s["atoms"].append([list(pos), species, "", init_moment, constraint, charge])
        return json.dumps(json_s)


def read(file_name, *args, **kwargs):
    """A revamped implementation of ase.io.read that returns gims.Structure"""
    return Structure.from_atoms(ase_read(file_name, *args, **kwargs))
