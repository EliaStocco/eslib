#!/usr/bin/env python
import json
import numpy as np
from ase import Atoms
from ase.calculators.socketio import SocketIOCalculator
from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter
from ase.io import write
from ase.optimize import BFGS
from typing import Optional

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import str2bool
from eslib.show import show_dict

#---------------------------------------#
description = "Run an ASE optimizer with constrained symmetries."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str     , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str     , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-p" , "--port"        , **argv, required=False, type=int     , help="TCP/IP port number. Ignored when using UNIX domain sockets.")
    parser.add_argument("-a" , "--address"     , **argv, required=False, type=str     , help="Host name (for INET sockets) or name of the UNIX domain socket to connect to (default: %(default)s)" , default=None)
    parser.add_argument("-u" , "--unix"        , **argv, required=False, type=str2bool, help="Use a UNIX domain socket (default: %(default)s)", default=False)
    
    parser.add_argument("-f" , "--fmax"        , **argv, required=False, type=float   , help="max force (default: %(default)s)", default=0.05)
    parser.add_argument("-op" , "--opt_par"    , **argv, required=False, type=str     , help="JSON file with the optimizer parameters (default: %(default)s)", default=None)
    # parser.add_argument("-l" , "--logger"      , **argv, required=False, type=str     , help="logging file (default: %(default)s)", default=None)
    parser.add_argument("-t", "--trajectory"     , **argv, required=False, type=str   , help="minimization trajectory (default: %(default)s)", default='minimization-trajectory.extxyz')
    parser.add_argument("-r" , "--restart"     , **argv, required=False, type=str     , help="file to restart the optimization from (default: %(default)s)", default=None)
    parser.add_argument("-ms", "--maxstep"     , **argv, required=False, type=int     , help="maximum step size (default: %(default)s)", default=100)

    parser.add_argument("-cs" , "--constrain_symmetry"    , **argv, required=False, type=str2bool     , help="whether to constrain symmetry (default: %(default)s)", default=False)
    parser.add_argument("-sp" , "--symprec"    , **argv, required=False, type=float     , help="symmetry precicion (default: %(default)s)", default=0.01)
    parser.add_argument("-rc" , "--relax_cell"    , **argv, required=False, type=str2bool     , help="whether to relax the cell (default: %(default)s)", default=False)
    
    parser.add_argument(
        "-cc",
        "--cell_constraints",
        **argv,
        nargs="+",
        required=False,
        type=str,
        help="fixed cell components (e.g. az bz cz)",
        default=None
    )
    
    parser.add_argument("-o" , "--output"      , **argv, required=False , type=str     , help="file to save the relaxed atomic structure (default: %(default)s)", default="final.extxyz")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str    , help="output file format (default: %(default)s)", default=None)
    
    return parser

#---------------------------------------#
class ConstrainedUnitCellFilter(UnitCellFilter):
    """
    UnitCellFilter with selected lattice components fixed.

    cell_mask:
        3x3 array:
        1 -> relax this component
        0 -> keep this component fixed

        Rows correspond to lattice vectors a,b,c
        Columns correspond to x,y,z Cartesian components.

    Example:
        Fix z components of all lattice vectors:

        mask[:,2] = 0
    """
    def __init__(self, atoms:Atoms, cell_mask=None, **kwargs):
        super().__init__(atoms, **kwargs)
        if cell_mask is None:
            cell_mask = np.ones((3,3), dtype=bool)
        cell_mask = np.asarray(cell_mask, dtype=bool)
        if cell_mask.shape != (3, 3):
            raise ValueError(
                f"cell_mask must have shape (3,3), got {cell_mask.shape}"
            )
        self.cell_mask = cell_mask
        self.orig_cell = atoms.cell.array.copy()
        

    def _apply_constraints(self, positions: np.ndarray)-> np.ndarray:
        # UnitCellFilter stores the three lattice vectors as the final
        # three pseudo-atoms.
        cell = positions[-3:].reshape(3,3)
        fixed = ~self.cell_mask
        cell[fixed] = self.orig_cell[fixed]
        positions[-3:,:] = cell# .reshape(9)
        # return positions

    def get_positions(self)-> np.ndarray:
        positions = super().get_positions()
        self._apply_constraints(positions)
        return positions

    def set_positions(self, new: np.ndarray, **kwargs):    
        """
        new is an array with shape (natoms+3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor used to change the cell shape.

        the new cell is first set from original cell transformed by the new
        deformation gradient, then the positions are set with respect to the
        current cell by transforming them with the same deformation gradient
        """
        
        self._apply_constraints(new)

        natoms = len(self.atoms)
        new_atom_positions = new[:natoms]
        new_deform_grad = new[natoms:] / self.cell_factor
        deform = (new_deform_grad - np.eye(3)).T * self.mask
        deform[~self.cell_mask] = 0.
        # Set the new cell from the original cell and the new
        # deformation gradient.  Both current and final structures should
        # preserve symmetry, so if set_cell() calls FixSymmetry.adjust_cell(),
        # it should be OK
        newcell = self.orig_cell @ (np.eye(3) + deform)

        self.atoms.set_cell(newcell,
                            scale_atoms=True)
        # Set the positions from the ones passed in (which are without the
        # deformation gradient applied) and the new deformation gradient.
        # This should also preserve symmetry, so if set_positions() calls
        # FixSymmetyr.adjust_positions(), it should be OK
        self.atoms.set_positions(new_atom_positions @ (np.eye(3) + deform),
                                 **kwargs)


#---------------------------------------#
def parse_cell_constraints(
    constraints: Optional[list[str]],
) -> Optional[np.ndarray]:

    if constraints is None:
        return None

    # Start with everything free
    mask = np.ones((3,3), dtype=bool)

    index = {
        "a":0,
        "b":1,
        "c":2,
        "x":0,
        "y":1,
        "z":2
    }

    for item in constraints:
        item = item.lower()
        if len(item) != 2:
            raise ValueError(
                f"Invalid cell constraint '{item}'. "
                "Expected format: ax, by, cz, ..."
            )

        if item[0] not in index:
            raise ValueError(
                f"Unknown lattice vector '{item[0]}' in '{item}'. "
                "Allowed: a, b, c."
            )

        if item[1] not in index:
            raise ValueError(
                f"Unknown Cartesian component '{item[1]}' in '{item}'. "
                "Allowed: x, y, z."
            )
        vector = index[item[0]]
        component = index[item[1]]

        # False means fixed
        if not mask[vector, component]:
            raise ValueError(
                f"Cell component '{item}' specified more than once."
            )
        mask[vector, component] = False

    return mask


#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    print("\tn. of atoms: ",atoms.get_global_number_of_atoms())

    #------------------#
    if args.constrain_symmetry:
        print("\tSetting constraints for preserving symmetries ... ")
        atoms.set_constraint(
            FixSymmetry(
                atoms,
                symprec=args.symprec,
                verbose=True,
            )
        )
        print("done")

    #------------------#
    filter = None
    if args.relax_cell:
        print("\tPreparing filter to relax the cell ... ", end="")
        cell_mask = parse_cell_constraints(args.cell_constraints)
        if cell_mask is None:
            filter = UnitCellFilter(atoms)
        else:
            print("\n\tCell constraint mask:")
            print(cell_mask.astype(int))
            filter = ConstrainedUnitCellFilter(
            atoms,
            cell_mask=cell_mask
        )
        print("done")

    #------------------#
    opt_par = {}
    if args.opt_par is not None:
        print("\tReading optimizer parameters from file '{:s}' ... ".format(args.opt_par), end="")
        with open(args.opt_par, 'r') as f:
                opt_par = json.load(f)
        print("done")
        print("\tOptimizer parameters:")
        show_dict(opt_par,string="\t")
    
    #------------------#
    print("\tAllocating BFGS optimizer  ... ", end="")
    
    opt = BFGS(atoms if filter is None else filter,
               restart=args.restart,trajectory=args.trajectory,**opt_par)
    print("done")
    
    if filter is not None:
        def print_cell_and_stress():
            print("Cell:")
            print(np.round(atoms.cell.array,3).tolist())
            stress = atoms.get_stress(voigt=False)
            print("Stress tensor (eV/A^3):")
            print(np.round(stress,3).tolist())
            print()
        opt.attach(print_cell_and_stress, interval=1)

    #------------------#
    if args.port is not None:
        if not (1025 <= args.port <= 65535):
            raise ValueError("'port' should be between 1025 and 65535"   )
        
    unixsocket = args.address if args.unix else None
    port = None if args.unix else args.port
    
    print(f"\tunixsocket: {unixsocket}")
    print(f"\tport: {port}")

    print("\tRunning BFGS optimizer  ... ")
    with SocketIOCalculator(port=port,unixsocket=unixsocket) as calc:
        # Server is now running and waiting for connections.
        # If you want to launch the client process here directly,
        # instead of manually in the terminal, uncomment these lines:
        #
        # from subprocess import Popen
        # proc = Popen([sys.executable, 'example_client_gpaw.py'])

        atoms.calc = calc
        opt.run(fmax=args.fmax, 
                steps=args.maxstep)
        
    print("\tFinished running BFGS optimizer")    

    #------------------#
    # Save the relaxed structure and optimizer state to file
    print(f"\tSaving the relaxed atomic structure and optimizer state to file '{args.output}' ... ", end="")
    write(args.output, atoms, format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
