#!/usr/bin/env python
# External dependencies: install NumPy and ASE with
#     python -m pip install numpy ase
# FixSymmetry additionally requires spglib:
#     python -m pip install spglib
# All other imported modules below are included with Python.
import argparse
import functools
import json
import os
import sys
from datetime import datetime
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.calculators.socketio import SocketIOCalculator
from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter
from ase.io import read, write
from ase.optimize import BFGS
from typing import List, Optional

#---------------------------------------#
description = "Run an ASE optimizer with constrained symmetries."
documentation = """Optimize the first structure in an input file with ASE's BFGS optimizer.
Energies, forces, and stresses are obtained from an external calculator through
a TCP or UNIX socket. The script can preserve the initial symmetry, relax the
cell, fix selected deformation-gradient components, and write an ASE trajectory.

Examples:
  # Relax atomic positions using a TCP-connected calculator
  python optimize.py -i start.extxyz -p 6000 -o relaxed.extxyz

  # Relax atoms and cell through a UNIX socket, keeping all z cell components fixed
  python optimize.py -i start.extxyz -u driver.socket --relax-cell true \\
      --cell-constraints az bz cz --constrain-symmetry true -o relaxed.extxyz
"""


def str2bool(value):
    """Parse the boolean spellings accepted by the original eslib CLI."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def show_dict(values, prefix="", width=30):
    """Print dictionary entries using the layout of the original script."""
    for key, value in values.items():
        print(f"\t{prefix}{key:<{width}} : {value}")


def _separator():
    return "@" + "-" * 30


def _print_header(args, function, started):
    script_path = Path(__file__).resolve()
    print(_separator())
    print(f"{'script file':20s}: {script_path.name}")
    print(f"{'script global path':20s}: {script_path}")
    print(f"{'working directory':20s}: {Path.cwd()}")
    print(f"{'VScode debugging':20s}: \"args\" : {json.dumps(sys.argv[1:])}")
    print(f"{'running script as':20s}: {script_path.name} {' '.join(sys.argv[1:])}")
    print(f"{'python --version':20s}: {sys.version.replace(chr(10), ' ')}")
    print(f"{'which python':20s}: {sys.executable}")
    print(f"{'conda env':20s}: {os.environ.get('CONDA_DEFAULT_ENV', 'none')}")
    print(f"{'SLURM job ID':20s}: {os.environ.get('SLURM_JOB_ID', 'none')}")
    print(f"{'PID':20s}: {os.getpid()}")
    print(f"{'start date':20s}: {started:%Y-%m-%d}")
    print(f"{'start time':20s}: {started:%H:%M:%S}")
    print(_separator())
    print(f"\n\t {description}")
    print("\n\t Documentation:")
    print("\t " + documentation.strip().replace("\n", "\n\t "))
    print("\n\t Input arguments:")
    for key, value in vars(args).items():
        print(f"\t {key:>20s}: {value}")
    print()


def _print_end(started):
    ended = datetime.now()
    elapsed = int((ended - started).total_seconds())
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("\n\t Job done :)\n")
    print("-" * 30 + "@")
    print(f"end date: {ended:%Y-%m-%d}")
    print(f"end time: {ended:%H:%M:%S}s")
    print(f"elapsed seconds: {elapsed}s")
    print(f"elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print("-" * 30 + "@\n")


def script_format(prepare_parser):
    """Standalone replacement for eslib's command-line formatting decorator."""
    def decorator(function):
        @functools.wraps(function)
        def wrapper(args=None):
            if args is None:
                args = prepare_parser(description).parse_args()
            elif isinstance(args, dict):
                args = argparse.Namespace(**args)

            started = datetime.now()
            _print_header(args, function, started)
            result = function(args)
            if result is None or result == 0:
                _print_end(started)
            return result
        return wrapper
    return decorator

#---------------------------------------#
def prepare_args(description):
    parser = argparse.ArgumentParser(
        description=description,
        epilog=documentation,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str     , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format", "--input-format", **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    socket_group = parser.add_mutually_exclusive_group(required=True)
    socket_group.add_argument("-p", "--port", **argv, type=int,help="TCP port on which to listen; selects TCP mode")
    socket_group.add_argument("-u", "--unixsocket", **argv, type=str,help="UNIX-domain socket name/path; selects UNIX mode")
    parser.add_argument("-f" , "--fmax"        , **argv, required=False, type=float   , help="force convergence threshold in eV/A (default: %(default)s)", default=0.05)
    parser.add_argument("-op", "--opt_par", "--opt-par", **argv, required=False, type=str, help="JSON file containing BFGS constructor parameters (default: %(default)s)", default=None)
    parser.add_argument("-t", "--trajectory"     , **argv, required=False, type=str   , help="ASE binary trajectory file (default: %(default)s)", default='minimization-trajectory.traj')
    parser.add_argument("-r" , "--restart"     , **argv, required=False, type=str     , help="file to restart the optimization from (default: %(default)s)", default=None)
    parser.add_argument("-ms", "--max-steps", "--maxstep", dest="max_steps", **argv, required=False, type=int,
                        help="maximum number of optimizer steps (default: %(default)s)", default=100)

    parser.add_argument("-cs", "--constrain_symmetry", "--constrain-symmetry", **argv, required=False, type=str2bool, help="whether to preserve the initial symmetry (default: %(default)s)", default=False)
    parser.add_argument("-sp" , "--symprec"    , **argv, required=False, type=float     , help="symmetry precision (default: %(default)s)", default=0.01)
    parser.add_argument("-rc", "--relax_cell", "--relax-cell", **argv, required=False, type=str2bool, help="whether to relax the cell (default: %(default)s)", default=False)
    parser.add_argument("--print-cell", **argv, required=False, type=str2bool,
                        help="print cell and stress after each cell-relaxation step (default: %(default)s)", default=False)
    
    parser.add_argument(
        "-cc",
        "--cell_constraints",
        "--cell-constraints",
        **argv,
        nargs="+",
        required=False,
        type=str,
        help=("fixed deformation-gradient components: first letter is the "
              "lattice direction and second is Cartesian (e.g. az bz cz)"),
        default=None
    )
    
    parser.add_argument("-o" , "--output"      , **argv, required=False , type=str     , help="file to save the relaxed atomic structure (default: %(default)s)", default="final.extxyz")
    parser.add_argument("-of", "--output_format", "--output-format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    
    return parser

#---------------------------------------#
class ConstrainedUnitCellFilter(UnitCellFilter):
    """
    UnitCellFilter with selected deformation-gradient components fixed.

    cell_mask:
        3x3 array:
        1 -> relax this component
        0 -> keep this component fixed

        Rows correspond to lattice directions a,b,c
        Columns correspond to x,y,z Cartesian components.

    Example:
        Fix z components of all lattice vectors:

        mask[:,2] = 0
    """
    def __init__(self, atoms:Atoms, cell_mask=None, **kwargs):
        super().__init__(atoms, **kwargs)
        if cell_mask is None:
            cell_mask = np.ones((3,3), dtype=bool)
        cell_mask = np.array(cell_mask, dtype=bool, copy=True)
        if cell_mask.shape != (3, 3):
            raise ValueError(
                f"cell_mask must have shape (3,3), got {cell_mask.shape}"
            )
        self.cell_mask = cell_mask
        self.orig_cell = atoms.cell.array.copy()
        

    def _apply_constraints(self, positions: np.ndarray) -> None:
        """Reset fixed components of the filter's deformation gradient.

        ``UnitCellFilter`` stores ``cell_factor * deformation_gradient`` in
        its final three pseudo-atom positions, not the physical cell vectors.
        A fixed component must therefore be restored to the corresponding
        component of the identity deformation gradient.  The transpose is
        required because :meth:`set_positions` transposes the stored
        deformation gradient before applying ``cell_mask``.
        """
        deformation = positions[-3:]
        undeformed = self.cell_factor * np.eye(3)
        fixed = (~self.cell_mask).T
        deformation[fixed] = undeformed[fixed]

    def get_positions(self)-> np.ndarray:
        positions = super().get_positions()
        self._apply_constraints(positions)
        return positions

    def get_forces(self, **kwargs) -> np.ndarray:
        """Return forces with fixed deformation components projected out."""
        forces = super().get_forces(**kwargs)
        fixed = (~self.cell_mask).T
        forces[-3:][fixed] = 0.0
        return forces

    def set_positions(self, new: np.ndarray, **kwargs):    
        """
        new is an array with shape (natoms+3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor used to change the cell shape.

        the new cell is first set from original cell transformed by the new
        deformation gradient, then the positions are set with respect to the
        current cell by transforming them with the same deformation gradient
        """
        
        # Optimizers may reuse their positions array, so do not mutate it.
        new = np.array(new, copy=True)
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

        # FixSymmetry and the component mask are separate projections.  Fail
        # clearly if an atomic constraint changes a component fixed here.
        actual_deform = np.linalg.solve(self.orig_cell, self.atoms.cell.array) - np.eye(3)
        if not np.allclose(actual_deform[~self.cell_mask], 0.0, atol=1e-10, rtol=0.0):
            raise RuntimeError(
                "An atomic/cell constraint changed a fixed cell deformation "
                "component; the symmetry and cell constraints are incompatible."
            )


#---------------------------------------#
def parse_cell_constraints(
    constraints: Optional[List[str]],
) -> Optional[np.ndarray]:

    if constraints is None:
        return None

    # Start with everything free
    mask = np.ones((3,3), dtype=bool)

    vectors = {"a": 0, "b": 1, "c": 2}
    components = {"x": 0, "y": 1, "z": 2}

    for item in constraints:
        item = item.lower()
        if len(item) != 2:
            raise ValueError(
                f"Invalid cell constraint '{item}'. "
                "Expected format: ax, by, cz, ..."
            )

        if item[0] not in vectors:
            raise ValueError(
                f"Unknown lattice vector '{item[0]}' in '{item}'. "
                "Allowed: a, b, c."
            )

        if item[1] not in components:
            raise ValueError(
                f"Unknown Cartesian component '{item[1]}' in '{item}'. "
                "Allowed: x, y, z."
            )
        vector = vectors[item[0]]
        component = components[item[1]]

        # False means fixed
        if not mask[vector, component]:
            raise ValueError(
                f"Cell component '{item}' specified more than once."
            )
        mask[vector, component] = False

    return mask


def validate_args(args) -> None:
    """Validate combinations and values not expressible with argparse alone."""
    if args.fmax <= 0:
        raise ValueError("--fmax must be greater than zero")
    if args.symprec <= 0:
        raise ValueError("--symprec must be greater than zero")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be greater than zero")
    if args.cell_constraints and not args.relax_cell:
        raise ValueError("--cell-constraints requires --relax-cell true")
    if args.unixsocket is not None and not args.unixsocket.strip():
        raise ValueError("--unixsocket cannot be empty")
    if args.unixsocket is None:
        if not (1025 <= args.port <= 65535):
            raise ValueError("--port must be between 1025 and 65535")
    if Path(args.trajectory).suffix.lower() != ".traj":
        raise ValueError("--trajectory must use the .traj extension (ASE binary format)")


#---------------------------------------#
@script_format(prepare_args)
def main(args):

    validate_args(args)

    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms: Atoms = read(args.input, format=args.input_format, index=0)
    print("done")
    print("\tn. of atoms: ",atoms.get_global_number_of_atoms())

    #------------------#
    if args.constrain_symmetry:
        print("\tSetting constraints for preserving symmetries ... ")
        symmetry_constraint = FixSymmetry(
            atoms,
            symprec=args.symprec,
            verbose=True,
        )
        atoms.set_constraint([*atoms.constraints, symmetry_constraint])
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
                cell_mask=cell_mask,
            )
        print("done")

    #------------------#
    opt_par = {}
    if args.opt_par is not None:
        print("\tReading optimizer parameters from file '{:s}' ... ".format(args.opt_par), end="")
        with open(args.opt_par, 'r') as f:
            opt_par = json.load(f)
        if not isinstance(opt_par, dict):
            raise ValueError("--opt-par must contain a JSON object")
        reserved = {"restart", "trajectory"}.intersection(opt_par)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(
                f"Optimizer parameters {names} must be provided through "
                "their command-line options, not --opt-par"
            )
        print("done")
        print("\tOptimizer parameters:")
        show_dict(opt_par, prefix="\t")
    
    #------------------#
    print("\tAllocating BFGS optimizer  ... ", end="")
    
    opt = BFGS(atoms if filter is None else filter,
               restart=args.restart,trajectory=args.trajectory,**opt_par)
    print("done")
    
    if filter is not None and args.print_cell:
        def print_cell_and_stress():
            print("Cell:")
            print(np.round(atoms.cell.array,3).tolist())
            stress = atoms.get_stress(voigt=False)
            print("Stress tensor (eV/A^3):")
            print(np.round(stress,3).tolist())
            print()
        opt.attach(print_cell_and_stress, interval=1)

    #------------------#
    if args.unixsocket is not None:
        socket_parameters = {"unixsocket": args.unixsocket}
        print(f"\tUNIX socket: {args.unixsocket}")
    else:
        socket_parameters = {"port": args.port}
        print(f"\tTCP socket: listening on port {args.port}")

    print("\tRunning BFGS optimizer  ... ")
    with SocketIOCalculator(**socket_parameters) as calc:
        # Server is now running and waiting for connections.
        # If you want to launch the client process here directly,
        # instead of manually in the terminal, uncomment these lines:
        #
        # from subprocess import Popen
        # proc = Popen([sys.executable, 'example_client_gpaw.py'])

        atoms.calc = calc
        opt.run(fmax=args.fmax, 
                steps=args.max_steps)
        
    print("\tFinished running BFGS optimizer")    

    #------------------#
    # Save the relaxed structure and optimizer state to file
    print(f"\tSaving the relaxed atomic structure and optimizer state to file '{args.output}' ... ", end="")
    write(args.output, atoms, format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()
