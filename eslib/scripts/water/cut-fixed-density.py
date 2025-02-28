#!/usr/bin/env python
import numpy as np
from ase.cell import Cell
from ase import Atoms
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
from eslib.tools import convert
from eslib.physics import compute_density

#---------------------------------------#
# Description of the script's purpose
description = "Do somehting with the density"

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"        , **argv, required=True , type=str, help="input file [extxyz]")
    parser.add_argument("-if" , "--input_format" , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-n"  , "--n_molecules"  , **argv, required=True , type=int, help="n. of molecules")
    parser.add_argument("-o"  , "--output"       , **argv, required=True , type=str, help="output file (default: %(default)s)", default=None)
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading first atomic structure from input file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)
    # atoms.fold()
    atoms = atoms[0]
    print("done")
    print("\tn. of atoms: ", len(atoms))
    
    #------------------#
    # Calculate the density    
    density = compute_density(atoms)
    print("\tdensity [g/cm^3]: ",density)
    
    factor = 3*args.n_molecules/len(atoms)
    factor = np.power(factor,1./3)
    
    cellpar = atoms.get_cell().cellpar()
    cellpar[:3] *= factor
    cell = Cell.fromcellpar(cellpar)
    atoms.set_cell(cell)
    
    # cell = np.asarray(atoms.cell)
    Oatoms = Atoms([ atom for atom in atoms if atom.symbol == "O"],cell=atoms.cell,pbc=True)
    ii = np.asarray([ atoms.arrays["molecule"][n] for n,atom in enumerate(atoms) if atom.symbol == "O"])
    scaled = Oatoms.get_scaled_positions(wrap=False)
    inside = np.all(scaled < 1, axis=1)
    count = np.sum(inside)
    if count < args.n_molecules:
        raise ValueError("Number of atoms not correct: {} < {}".format(count,args.n_molecules))
    else:
        tmp = np.arange(len(inside))
        inside[tmp[inside][args.n_molecules:]]=False
        count = np.sum(inside)
    assert count == args.n_molecules, "Number of atoms not correct: {} != {}".format(count,args.n_molecules)
    
    ii = ii[inside]
    assert len(ii) == args.n_molecules, "Number of atoms not correct: {} != {}".format(len(ii),args.n_atoms)
    
    #------------------#
    new_atoms = Atoms([ atom for n,atom in enumerate(atoms) if atoms.arrays["molecule"][n] in ii],cell=atoms.cell,pbc=True)
    jj = [ n for n,atom in enumerate(atoms) if atoms.arrays["molecule"][n] in ii]
    new_atoms.arrays["molecule"] = atoms.arrays["molecule"][jj]
    assert np.allclose(new_atoms.positions,atoms.positions[jj,:]), "New positions not correct: {} != {}".format(new_atoms.positions,atoms.positions[jj,:])
    new_density = compute_density(new_atoms)
    assert np.allclose(new_density,density), "New density not correct: {} != {}".format(new_density,density)
    assert len(new_atoms) == 3*args.n_molecules, "Number of atoms not correct: {} != {}".format(len(new_atoms),3*args.n_molecules)
    
    # #------------------#
    # print("\n\tKeeping only first {} atoms ... ".format(args.n_atoms), end="")
    # atoms = atoms[:args.n_atoms]    
    # print("done")
    # print("\tn. of atoms: ", len(atoms))
    
    #------------------#
    print("\tWriting atomic structures to file '{:s}' ... ".format(args.output), end="")
    new_atoms = AtomicStructures([new_atoms])
    new_atoms.to_file(file=args.output,format=args.output_format)
    print("done")
    
    pass

#---------------------------------------#
if __name__ == "__main__":
    main()

