#!/usr/bin/env python
import json

from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter as RelaxCell
from ase.io import write
from ase.optimize import BFGS

from classes.potentials.multisocket import MultiSocket
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import str2bool, slist, ilist, blist
from eslib.show import show_dict

#---------------------------------------#
description = "Run the BFGS geometry optimizer provided by ASE."
documentation = "This script allows to:\n\
 - constraint the symmetry of the system\n\
 - relax the cell (fully or partially)\n\
 - provide multiple drivers (the sum of the energy, forces and stresses will be provided to BFGS)"
# choices = ["BFGS"]

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str     , help="file with an atomic structure")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str     , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-p" , "--port"        , **argv, required=False, type=ilist   , help="TCP/IP port numbers. Ignored when using UNIX domain sockets.")
    parser.add_argument("-a" , "--address"     , **argv, required=True , type=slist   , help="Host names (for INET sockets) or names of the UNIX domain socket to connect to.")
    parser.add_argument("-u" , "--unix"        , **argv, required=False, type=str2bool, help="Use a UNIX domain socket (default: %(default)s)", default=False)
    
    parser.add_argument("-f" , "--fmax"        , **argv, required=False, type=float   , help="max force (default: %(default)s)", default=0.05)
    # parser.add_argument("-ot", "--opt_type"    , **argv, required=False, type=str     , help=f"optimizer type {choices}"+" (default: %(default)s)", default="BFGS")
    parser.add_argument("-op", "--opt_par"     , **argv, required=False, type=str     , help="JSON file with the optimizer parameters (default: %(default)s)", default=None)
    parser.add_argument("-l" , "--logger"      , **argv, required=False, type=str     , help="logging file (default: %(default)s)", default=None)
    # parser.add_argument("-t" , "--trajectory"  , **argv, required=False, type=str     , help="minimization trajectory (default: %(default)s)", default='minimization-trajectory.pickle')
    parser.add_argument("-r" , "--restart"     , **argv, required=False, type=str     , help="file to restart the optimization from (default: %(default)s)", default=None)
    parser.add_argument("-ms", "--maxstep"     , **argv, required=False, type=int     , help="maximum step size (default: %(default)s)", default=100)

    parser.add_argument("-cs" , "--constrain_symmetry", **argv, required=False, type=str2bool, help="whether to constrain symmetry (default: %(default)s)", default=False)
    parser.add_argument("-sp" , "--symprec"           , **argv, required=False, type=float   , help="symmetry precicion (default: %(default)s)", default=0.01)
    parser.add_argument("-rc" , "--relax_cell"        , **argv, required=False, type=str2bool, help="whether to relax the cell (default: %(default)s)", default=False)
    parser.add_argument("-cm" , "--cell_mask"         , **argv, required=False, type=blist   , help="mask for the cell in Voigt notation [xx, yy, zz, yz, xz, xy] (default: %(default)s)", default=[True, True, True, True, True, True])
    
    parser.add_argument("-o" , "--output"             , **argv, required=False, type=str     , help="file to save the relaxed atomic structure (default: %(default)s)", default="final.extxyz")
    parser.add_argument("-of", "--output_format"      , **argv, required=False, type=str     , help="output file format (default: %(default)s)", default=None)
    
    return parser

#---------------------------------------#
@esfmt(prepare_args,description,documentation)
def main(args):

    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    print("\tn. of atoms: ",atoms.get_global_number_of_atoms())

    #------------------#
    if args.constrain_symmetry:
        print("\tSetting constraints for preserving symmetries ... ")
        atoms_sym = atoms.copy()
        atoms_sym.set_constraint(FixSymmetry(atoms_sym,symprec=args.symprec,verbose=True))
        atoms = atoms_sym.copy()
        del atoms_sym

    #------------------#
    filter = None
    if args.relax_cell:
        print("\tPreparing filter to relax the cell ... ",end="")
        filter = RelaxCell(atoms)
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
               restart=args.restart,**opt_par)
    print("done")

    #------------------#
    ports = args.port
    unixsockets = args.address if args.unix else None
    log = args.logger
    
    if ports is None:
        ports = [None]*len(unixsockets)

    print("\tRunning BFGS optimizer  ... ")
    # try:
    with MultiSocket(log=log,
                    ports=ports,
                    unixsockets=unixsockets) as calc:
        # Server is now running and waiting for connections.
        # If you want to launch the client process here directly,
        # instead of manually in the terminal, uncomment these lines:
        #
        # from subprocess import Popen
        # proc = Popen([sys.executable, 'example_client_gpaw.py'])

        atoms.calc = calc
        opt.run(fmax=args.fmax, 
                steps=args.maxstep)
    # except Exception as e:
    #     print("\tError: {:s}".format(e))
    #     return -1
        
    print("\tFinished running BFGS optimizer")    

    #------------------#
    # Save the relaxed structure and optimizer state to file
    print(f"\tSaving the relaxed atomic structure and optimizer state to file '{args.output}' ... ", end="")
    write(args.output, atoms, format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()



from ase import Atoms
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.constraints import FrechetCellFilter

# Example atomic structure: initialize an Atoms object
atoms = Atoms('Cu4', positions=[[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0], [1.8, 1.8, 1.8]],
              cell=[[3.6, 0, 0], [0, 3.6, 0], [0, 0, 3.6]], pbc=True)

# Set calculator (e.g., EMT for this example)
atoms.set_calculator(EMT())

# Define mask for relaxing only the x-component of the first lattice vector
# 3x3 mask where the first row's x-component is True, and all others are False
mask = [[True, False, False],
        [False, False, False],
        [False, False, False]]

# Apply the FrechetCellFilter with the mask
cell_filter = FrechetCellFilter(atoms, mask=mask)

# Run optimization using BFGS (or another optimizer) on the cell filter
opt = BFGS(cell_filter)
opt.run(fmax=0.01)
