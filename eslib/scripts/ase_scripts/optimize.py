#!/usr/bin/env python
from eslib.formatting import esfmt
from eslib.input import str2bool
from ase import Atoms
import json
from eslib.classes.trajectory import AtomicStructures
from eslib.show import show_dict
from ase.calculators.socketio import SocketIOCalculator
from ase.optimize import BFGS
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter as RelaxCell
from ase.io import write

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
    parser.add_argument("-a" , "--address"     , **argv, required=True , type=str     , help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.")
    parser.add_argument("-u" , "--unix"        , **argv, required=False, type=str2bool, help="Use a UNIX domain socket (default: %(default)s)", default=False)
    
    parser.add_argument("-f" , "--fmax"        , **argv, required=False, type=float   , help="max force (default: %(default)s)", default=0.05)
    parser.add_argument("-op" , "--opt_par"    , **argv, required=False, type=str     , help="JSON file with the optimizer parameters (default: %(default)s)", default=None)
    parser.add_argument("-l" , "--logger"      , **argv, required=False, type=str     , help="logging file (default: %(default)s)", default=None)
    parser.add_argument("-t", "--trajectory"     , **argv, required=False, type=str   , help="minimization trajectory (default: %(default)s)", default='minimization-trajectory.pickle')
    parser.add_argument("-r" , "--restart"     , **argv, required=False, type=str     , help="file to restart the optimization from (default: %(default)s)", default=None)
    parser.add_argument("-ms", "--maxstep"     , **argv, required=False, type=int     , help="maximum step size (default: %(default)s)", default=100)

    parser.add_argument("-cs" , "--constrain_symmetry"    , **argv, required=False, type=str2bool     , help="whether to constrain symmetry (default: %(default)s)", default=False)
    parser.add_argument("-sp" , "--symprec"    , **argv, required=False, type=float     , help="symmetry precicion (default: %(default)s)", default=0.01)
    parser.add_argument("-rc" , "--relax_cell"    , **argv, required=False, type=str2bool     , help="whether to relax the cell (default: %(default)s)", default=False)
    
    parser.add_argument("-o" , "--output"      , **argv, required=False , type=str     , help="file to save the relaxed atomic structure (default: %(default)s)", default="final.extxyz")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str    , help="output file format (default: %(default)s)", default=None)
    
    return parser

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
