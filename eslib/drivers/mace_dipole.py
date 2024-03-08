import sys
import numpy as np
from ipi.utils.units import unit_to_internal, unit_to_user
from ipi._driver.pes.dummy import Dummy_driver
import torch

from ase.io import read
from mace.calculators import MACEliaCalculator


__DRIVER_NAME__ = "MACE_Dipole_driver"
__DRIVER_CLASS__ = "MACE_Dipole_driver"


class MACE_Dipole_driver(Dummy_driver):
    def __init__(self, args=None, verbose=False):
        self.error_msg = """MACE driver requires specification of a .json model,
                            and a template file that describes the chemical makeup of the structure.
                            Example: python driver.py -m mace -u -o model.json,template.xyz"""

        super().__init__(args, verbose)

    def check_arguments(self):
        """Check the arguments requuired to run the driver

        This loads the potential and atoms template in MACE
        """
        try:
            arglist = self.args.split(",")
        except ValueError:
            sys.exit(self.error_msg)

        self.model_atoms = read(arglist[0])
        self.driver_example_atoms = arglist[0]
        self.driver_model_path = arglist[1]
        self.mace_calculator = MACEliaCalculator(
            model_path=self.driver_model_path, device="cpu",model_type="AtomicDipolesMACE"
        )
        self.atoms = read(self.driver_example_atoms)
        self.atoms.set_calculator(self.mace_calculator)

    def __call__(self, cell, pos):
        """Get energies, forces, and stresses from the MACE model
        This routine assumes that the client will take positions
        in angstrom, and return energies in electronvolt, and forces
        in ev/ang.
        """
        pos_calc = unit_to_user("length", "angstrom", pos)
        cell_calc = unit_to_user("length", "angstrom", cell.T)

        # atoms = read(self.driver_example_atoms)
        self.atoms.set_pbc([True, True, True])
        self.atoms.set_cell(cell_calc, scale_atoms=True)
        self.atoms.set_positions(pos_calc)
        
        pot, force, vir, extras = super().__call__(cell, pos)

        dipole:torch.Tensor = self.atoms.get_dipole_moment()

        extras = {}
        extras["dipole"] = dipole.tolist()

        return pot, force, vir, extras
