#!/usr/bin/env python
import numpy as np
from ase import Atoms
from ase.cell import Cell
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.mathematics import gaussian_cluster_indices

#---------------------------------------#
description = "Assembly a structure made of: graphene monolayer, water, graphene bilayer."


#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-m" , "--monolayer"    , **argv, type=str, required=True , help="graphene monolayer [extxyz]")
    parser.add_argument("-w" , "--water"        , **argv, type=str, required=True , help="water [extxyz]")
    parser.add_argument("-b" , "--bilayer"      , **argv, type=str, required=True , help="graphene bilayer [extxyz]")
    parser.add_argument("-o" , "--output"       , **argv, type=str, required=True , help="output file")
    parser.add_argument("-of", "--output_format", **argv, type=str, required=False, help="output file format (default: %(default)s)", default=None)
    return  parser

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    #-------------------#
    print("\tReading graphene monolayer from file '{:s}' ... ".format(args.monolayer), end="")
    monolayer:Atoms = AtomicStructures.from_file(file=args.monolayer,format="extxyz",index=0)[0]
    print("done")
    
    #-------------------#
    print("\tReading water from file '{:s}' ... ".format(args.water), end="")
    water:Atoms = AtomicStructures.from_file(file=args.water,format="extxyz",index=0)[0]
    print("done")
    
    #-------------------#
    print("\tReading bilayer monolayer from file '{:s}' ... ".format(args.bilayer), end="")
    bilayer:Atoms = AtomicStructures.from_file(file=args.bilayer,format="extxyz",index=0)[0]
    print("done")
    
    #-------------------#
    cellpar_m = monolayer.cell.cellpar()[[0,1,5]]
    cellpar_w = water.cell.cellpar()[[0,1,5]]
    cellpar_b = bilayer.cell.cellpar()[[0,1,5]]
    assert np.allclose(cellpar_m, cellpar_w) and np.allclose(cellpar_m, cellpar_b), "Cell parameters differ"
    cellpar =  monolayer.cell.cellpar()
    cellpar[2] = 100
    cellpar[3] = 90
    cellpar[4] = 90
    cell = Cell.fromcellpar(cellpar)
    
    #-------------------#
    zcell = water.cell.cellpar()[2]
    print()
    print(f"\tThe distance between the graphene layers will be set equal to the z-axis of the water box + 4Å: {zcell+4} Å")
    print(f"\tThe z-axis of the box will be set to  100 Å")
    print()
    
    #-------------------#
    print("\tPreparing monolayer ... ",end="")
    z = monolayer.positions[:,2]
    monolayer.positions[:,2] -= np.mean(z)
    print("done")
    
    #-------------------#
    print("\tPreparing water ... ",end="")
    water.positions[:,2] += 2
    print("done")
    
    #-------------------#
    print("\tPreparing bilayer ... ",end="")
    zb = bilayer.positions[:,2]
    z_dict = gaussian_cluster_indices(zb,2)
    z_shift = min(list(z_dict.keys()))
    bilayer.positions[:,2] -= z_shift 
    bilayer.positions[:,2] += zcell + 4
    print("done")
    
    #-------------------#
    print("\tPreparing final structure ... ",end="")
    Nm = monolayer.get_global_number_of_atoms()
    Nw = water.get_global_number_of_atoms()
    Nb = bilayer.get_global_number_of_atoms()
    Ntot = Nm+Nw+Nb
    positions = np.zeros((Ntot,3))
    positions[:Nm,:] = monolayer.get_positions()
    positions[Nm:Nm+Nw,:] = water.get_positions()
    positions[Nm+Nw:,:] = bilayer.get_positions()
    symbols = monolayer.get_chemical_symbols() + water.get_chemical_symbols() + bilayer.get_chemical_symbols()
    confined_water = Atoms(positions=positions,cell=cell,pbc=[True,True,True],symbols=symbols)
    print("done")
    
    #-------------------#
    print(f"\tSaving final structure to {args.output} ... ",end="")
    confined_water.write(args.output, format=args.output_format)
    print("done")

    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()
