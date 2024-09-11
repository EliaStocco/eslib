#!/usr/bin/env python
from eslib.show import matrix2str
import numpy as np
from eslib.formatting import esfmt
from classes.atomic_structures import AtomicStructures
from ase import Atoms
from eslib.input import ilist
from eslib.geometry import modular_norm
import yaml

PHONOPY_FILE = "phonopy.yaml"
    
#---------------------------------------#
# Description of the script's purpose
description = "Get the map between atoms of two structures."
documentation = "This script will compare fractional coordinates"

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str  , help="input file")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-p" , "--phonopy"       , **argv, required=True , type=str  , help="phonopy structure")
    parser.add_argument("-pf", "--phonopy_format", **argv, required=False, type=str  , help="phonopy structure file format (default: %(default)s)", default=None)
    parser.add_argument("-s" , "--supercell"     , **argv, required=True , type=ilist, help="supercell size")
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structure from input file '{:s}' ... ".format(args.input), end="")
    atoms_S:Atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    print("\tn. of atoms:",atoms_S.get_global_number_of_atoms())  
    symbols_S = atoms_S.get_chemical_symbols()
    species_S = set(symbols_S)
    print("\tspecies:",species_S)
    print("\tunit cell [ang]:")    
    line = matrix2str(atoms_S.cell.array.T,col_names=["1","2","3"],cols_align="^",width=10,digits=4)
    print(line)
    
    #------------------#
    print("\tReading atomic structure from phonopy file '{:s}' ... ".format(args.phonopy), end="")
    atoms_P:Atoms = AtomicStructures.from_file(file=args.phonopy,format=args.phonopy_format,index=0)[0]
    print("done")
    print("\tn. of atoms:",atoms_P.get_global_number_of_atoms())    
    symbols_P = atoms_P.get_chemical_symbols()
    species_P = set(symbols_P)
    print("\tspecies:",species_P)
    print("\tunit cell [ang]:")    
    line = matrix2str(atoms_P.cell.array.T,col_names=["1","2","3"],cols_align="^",width=10,digits=4)
    print(line)
    
    #---------------------------------------#
    # read input file ('phonopy.yaml')
    print("\n\tReading file '{:s}' ... ".format(PHONOPY_FILE), end="")
    with open(PHONOPY_FILE) as f:
        info = yaml.safe_load(f)
    print("done")
    
    Phonopy2Prim = np.asarray([a['reduced_to']  for a in info['supercell']['points']]) -1
    
    #------------------#
    Nfactor = int(np.prod(args.supercell))
    Natoms = atoms_S.get_global_number_of_atoms()
    NatomsPrim = int(Natoms/Nfactor)
    index = np.arange(Natoms)
    
    index = index.reshape((NatomsPrim,Nfactor)).T.flatten()
    
    test = [symbols_P[index[n]] for n,i in enumerate(index)] == symbols_S
    assert test, "the map is not the same"
    
    pos_S = atoms_S.get_scaled_positions()
    pos_P = atoms_P.get_scaled_positions()
           
    distance = pos_P[index,:] - pos_S
    
    distance = modular_norm(distance,modulus=1,threshold=1e-4)
    

        
    pass
    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
