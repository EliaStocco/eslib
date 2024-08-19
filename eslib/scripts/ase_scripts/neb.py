#!/usr/bin/env python
from eslib.formatting import esfmt
from eslib.input import str2bool
from ase import Atoms
import json
from classes.atomic_structures import AtomicStructures
from eslib.show import show_dict
from ase.calculators.socketio import SocketIOCalculator
from ase.optimize import BFGS
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter as RelaxCell
from ase.neb import NEB
from ase.io import write

#---------------------------------------#
description = "Run an ASE Nudge Elastic Band calculation."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # Input
    parser.add_argument("-ii" , "--initial_input"       , **argv, required=True , type=str     , help="initial file with an atomic structure")
    parser.add_argument("-iif", "--initial_input_format", **argv, required=False, type=str     , help="initial input file format (default: %(default)s)" , default=None)
    parser.add_argument("-fi" , "--final_input"       , **argv, required=True , type=str     , help="final file with an atomic structure")
    parser.add_argument("-fif", "--final_input_format", **argv, required=False, type=str     , help="final input file format (default: %(default)s)" , default=None)
    # Parameters
    parser.add_argument("-N" , "--n_images"    , **argv, required=True , type=int     , help="number of images")
    # Socket 
    parser.add_argument("-p" , "--port"        , **argv, required=False, type=int     , help="TCP/IP port number. Ignored when using UNIX domain sockets.")
    parser.add_argument("-a" , "--address"     , **argv, required=True , type=str     , help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.")
    parser.add_argument("-u" , "--unix"        , **argv, required=False, type=str2bool, help="Use a UNIX domain socket (default: %(default)s)", default=False)
    # Output
    parser.add_argument("-o" , "--output"      , **argv, required=False , type=str     , help="output folder (default: %(default)s)", default="NEB")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str    , help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # initial structure
    print("\tReading the initial atomic structure from file '{:s}' ... ".format(args.initial_input), end="")
    initial:Atoms = AtomicStructures.from_file(file=args.initial_input,format=args.initial_input_format,index=0)[0]
    print("done")
    print("\tn. of atoms: ",initial.get_global_number_of_atoms())

    #------------------#
    # final structure
    print("\tReading the final atomic structure from file '{:s}' ... ".format(args.final_input), end="")
    final:Atoms = AtomicStructures.from_file(file=args.final_input,format=args.final_input_format,index=0)[0]
    print("done")
    print("\tn. of atoms: ",final.get_global_number_of_atoms())

    assert initial.get_global_number_of_atoms() == final.get_global_number_of_atoms(), "The initial and final structures must have the same number of atoms."

    #------------------#
    # Images
    print("\tGenerating '{:d}' images... ".format(args.n_images), end="")
    images = [initial]
    images += [initial.copy() for i in range(args.n_images)]
    images.append(final)
    print("done")

    #------------------#
    #  NEB
    print("\tAllocating NEB ... ", end="")
    neb = NEB(images,parallel=False)
    print("done")

    #------------------#
    if args.constrain_symmetry:
        print("\tSetting constraints for preserving symmetries ... ")
        atoms_sym = atoms.copy()
        atoms_sym.set_constraint(FixSymmetry(atoms_sym,symprec=args.symprec,verbose=True))
        atoms = atoms_sym.copy()
        del atoms_sym
        # atoms = UnitCellFilter(atoms_sym)
        # print("done")

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
               restart=args.restart,trajectory=args.trajectory,**opt_par)
    print("done")

    #------------------#
    port = args.port
    unixsocket = args.address if args.unix else None
    log = args.logger

    print("\tRunning BFGS optimizer  ... ")
    with SocketIOCalculator(log=log,
                            port=port,
                            unixsocket=unixsocket) as calc:
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
