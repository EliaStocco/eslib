#!/usr/bin/env python
import numpy as np
from ase import Atoms
from ase.io import write
from eslib.classes.atomic_structures import AtomicStructures
from eslib.mathematics import gaussian_cluster_indices
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Displace a bilayer 2D ice structure."
documentatin = "Please run this script after have run 'divide-into-water-molecules.py'"

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"         , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"  , **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-m" , "--molecule"     , **argv, required=False, type=str     , help="molecule name (default: %(default)s)", default="molecule")
    parser.add_argument("-n" , "--number"        , **argv, required=False, type=int, help="number of output structures")
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str, help="output file")
    parser.add_argument("-of", "--output_format" , **argv, required=False, type=str, help="output file format (default: %(default)s)", default=None)
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structure A from input file '{:s}' ... ".format(args.input), end="")
    reference:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    
    #------------------#
    print("\tDividing into lower and upper layer ... ",end="")
    oxygen = Atoms([ a for a in reference if a.symbol == "O" ])
    iOx = np.asarray([ n for n,a in enumerate(reference) if a.symbol == "O" ])
    pos = oxygen.get_positions()[:,2]
    z = gaussian_cluster_indices(pos,2)
    assert len(z.keys()) == 2, "you are not provided a graphene double layer."
    
    index_lower = iOx[z[list(z.keys())[0]]]
    index_upper = iOx[z[list(z.keys())[1]]]
    
    oxygen_lower = reference[index_lower]
    oxygen_upper = reference[index_upper]
    
    molecules_lower = oxygen_lower.arrays[args.molecule]
    molecules_upper = oxygen_upper.arrays[args.molecule]
    
    all_molecules = reference.arrays[args.molecule]
    atoms_lower = []
    atoms_upper = []
    for n,m in enumerate(all_molecules):
        if m in molecules_lower:
            atoms_lower.append(n)
        elif m in molecules_upper:
            atoms_upper.append(n)
        else:
            raise ValueError("WTF?")
            
    atoms_lower = reference[np.asarray(atoms_lower)]
    atoms_upper = reference[np.asarray(atoms_upper)]
    
    write("lower.extxyz",atoms_lower)
    write("upper.extxyz",atoms_upper)
    
    print("done")
    
    #------------------#
    print("\tGenerating random numbers ... ",end="")
    frac_shits = np.random.rand(args.number, 3)
    frac_shits[:,2] = 0 
    
    shifts = reference.get_cell().cartesian_positions(frac_shits)
    assert np.allclose(shifts[:,2],0), "blah"
    
    print("done")   
    
    #------------------#
    print("\tShifting the upper layer ... ",end="")
    structures = [None]*args.number
    for n in range(args.number):
        tmp = atoms_upper.copy()
        tmp.set_positions(atoms_upper.get_positions() + shifts[n][None,:])
        structures[n] = tmp + atoms_lower
    print("done")
    structures = AtomicStructures(structures)
    
    #------------------#
    print(f"\tWriting atomic structure to '{args.output}' ... ", end="")
    structures.to_file(file=args.output, format=args.output_format)
    print("done")
        
    return 0
    
#---------------------------------------#
if __name__ == "__main__":
    main()
