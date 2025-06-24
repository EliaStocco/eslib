#!/usr/bin/env python
from ase.calculators.socketio import SocketClient
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures

from eslib.classes.potentials.LJwall_calculator import LennardJonesWall
from eslib.formatting import esfmt
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Torch-pme calculator."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str  , help="input file with the atomic structure")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-k" , "--keyword"     , **argv, required=True , type=str     , help="keyword for the charges (default: %(default)s)", default='Qs')
    parser.add_argument("-p" , "--port"        , **argv, required=False, type=int     , help="TCP/IP port number. Ignored when using UNIX domain sockets.")
    parser.add_argument("-a" , "--address"     , **argv, required=True , type=str     , help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.")
    parser.add_argument("-u" , "--unix"        , **argv, required=False, type=str2bool, help="Use a UNIX domain socket (default: %(default)s)", default=False)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    
    #------------------#
    # I don't know if I actually need this line
    import torch
    import torchpme
    import vesin.torch
    from torchpme.tuning import tune_pme

    dtype = torch.float64
    
    # Create the properties CsCl unit cell
    symbols = ("Cs", "Cl")
    types = torch.tensor([55, 17])
    charges = torch.tensor([[1.0], [-1.0]], dtype=dtype)
    positions = torch.tensor([(0, 0, 0), (0.5, 0.5, 0.5)], dtype=dtype)
    cell = torch.eye(3, dtype=dtype)
    pbc = torch.tensor([True, True, True])


    # %%
    #
    # Based on our system we will first *tune* the PME parameters for an accurate computation.
    # The ``sum_squared_charges`` is equal to ``2.0`` becaue each atom either has a charge
    # of 1 or -1 in units of elementary charges.

    cutoff = 4.4
    nl = vesin.torch.NeighborList(cutoff=cutoff, full_list=False)
    neighbor_indices, neighbor_distances = nl.compute(
        points=positions.to(dtype=torch.float64, device="cpu"),
        box=cell.to(dtype=torch.float64, device="cpu"),
        periodic=True,
        quantities="Pd",
    )
    smearing, pme_params, _ = tune_pme(
        charges=charges,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )

    calculator = torchpme.PMECalculator(
    torchpme.CoulombPotential(smearing=smearing), **pme_params
)
    calculator.to(dtype=dtype)
    
    print("\tLoading the LennardJonesWall calculator ... ", end="")
    calculator = LennardJonesWall(instructions=args.instructions,log_file=args.logger)         
    print("done")

    atoms.calc = calculator

    #------------------#
    print("\tPreparing the socket communication ... ", end="")
    client = SocketClient(host=args.address,\
                          port=args.port,\
                          unixsocket=args.address if args.unix else None)
    print("done")

    #------------------#
    print("\n\tRunning ... ", end="")
    client.run(atoms)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()