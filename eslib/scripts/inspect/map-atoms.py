#!/usr/bin/env python
from eslib.classes.normal_modes import NormalModes
from eslib.show import matrix2str
from eslib.tools import convert
from eslib.output import output_folder
from eslib.input import size_type
from eslib.functions import phonopy2atoms
import numpy as np
import yaml
import pandas as pd
import os
from eslib.formatting import esfmt, warning
from classes.atomic_structures import AtomicStructures
from eslib.tools import is_sorted_ascending, w2_to_w
from phonopy.units import VaspToTHz
from ase import Atoms
from eslib.geometry import modular_norm

#---------------------------------------#
# Description of the script's purpose
description = "Get the map between atoms of two structures."
documentation = "This script will compare fractional coordinates"

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-a" , "--atoms_A"       , **argv, required=True , type=str  , help="file structure A")
    parser.add_argument("-af", "--atoms_A_format", **argv, required=False, type=str  , help="format file structure A (default: %(default)s)", default=None)
    parser.add_argument("-b" , "--atoms_B"       , **argv, required=True , type=str  , help="file structure B")
    parser.add_argument("-bf", "--atoms_B_format", **argv, required=False, type=str  , help="format file structure B (default: %(default)s)", default=None)
    parser.add_argument("-t" , "--tolerance"     , **argv, required=False, type=float, help="tolerance on the distance (default: %(default)s)",default=1e-4)
    parser.add_argument("-o" , "--output"        , **argv, required=False, type=str  , help="output file for the indices (default: %(default)s)",default="indices.txt")
    parser.add_argument("-io", "--inverse_output", **argv, required=False, type=str  , help="output file for the inverse indices (default: %(default)s)",default="inverse-indices.txt")
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structure A from input file '{:s}' ... ".format(args.atoms_A), end="")
    atoms_A:Atoms = AtomicStructures.from_file(file=args.atoms_A,format=args.atoms_A_format,index=0)[0]
    print("done")
    print("\tn. of atoms:",atoms_A.get_global_number_of_atoms())  
    symbols_A = atoms_A.get_chemical_symbols()
    species_A = set(symbols_A)
    print("\tspecies A   :",species_A)
    print("\tunit cell [ang] :")    
    line = matrix2str(atoms_A.cell.array.T,col_names=["1","2","3"],cols_align="^",width=10,digits=4)
    print(line)
    
    #------------------#
    print("\tReading atomic structure B from input file '{:s}' ... ".format(args.atoms_B), end="")
    atoms_B:Atoms = AtomicStructures.from_file(file=args.atoms_B,format=args.atoms_B_format,index=0)[0]
    print("done")
    print("\tn. of atoms:",atoms_B.get_global_number_of_atoms())    
    symbols_B = atoms_B.get_chemical_symbols()
    species_B = set(symbols_B)
    print("\tspecies B   :",species_B)
    print("\tunit cell [ang] :")    
    line = matrix2str(atoms_B.cell.array.T,col_names=["1","2","3"],cols_align="^",width=10,digits=4)
    print(line)
    
    #------------------#
    assert species_A == species_B, "different species"

    #------------------#
    symbols_A = np.asarray(symbols_A)
    symbols_B = np.asarray(symbols_B)
    
    pos_A = atoms_A.get_scaled_positions()
    pos_B = atoms_B.get_scaled_positions()
    
    mathces = np.full(len(symbols_A),np.nan)
    
    #------------------#
    k = 0
    for species in species_A:
        ii_a = np.where(symbols_A == species)[0]
        ii_b = np.where(symbols_B == species)[0]
        assert len(ii_a) == len(ii_b), "different number of atoms for species {:s}".format(species)
        print("\n\tAnalysizing species {:s}:".format(species))
        print("\t\tatoms A: ",ii_a)
        print("\t\tatoms B: ",ii_b)
        
        
        for i in ii_a:
            if not np.isnan(mathces[i]):
                raise ValueError("atom {:d} already matched".format(i))
            
            pos = pos_A[i,:]
            assert symbols_A[i] == species, "some coding error"
            # distances = np.zeros((len(ii_b),3))
            # for n,j in enumerate(ii_b):
            #     distances[n,:] = pos - pos_B[j,:]
            distances_vec = pos - pos_B[ii_b]
            distances_vec = modular_norm(distances_vec,1,args.tolerance)
            distances = np.linalg.norm(distances_vec,axis=1)
            
            value = ii_b[np.argmin(distances)]
            # assert np.allclose(pos_A[i,:],pos[value,:]), "some coding error"
            # assert np.allclose(pos_A[i,:],pos_B[value,:]), "some coding error"
            if value in mathces[ ~ np.isnan(mathces) ]:
                raise ValueError("coding error")
                # raise ValueError("atom {:d} already matched with {:d}. What to do with {:d}?".format(i,int(np.where(mathces == i)[0]),value))
            mathces[i] = value
            k += 1
            
    assert np.allclose(np.unique(mathces),np.arange(len(symbols_A))), "some coding error"
    
    mathces = mathces.astype(int)
    check = modular_norm(pos_B[mathces,:] - pos_A,args.tolerance)
    assert np.allclose(check,0,atol=args.tolerance), "some coding error"
    
    print("\n\tAtoms have been matched.\n")
    print("\tSaving the indices to file '{:s}' ... ".format(args.output),end="")
    np.savetxt(args.output,mathces,fmt="%d")
    print("done")
    
    print("\tHow to interpret the indices: posB[indices,:] == posA")
    
    print("\n\tCreating inverse indices ... ",end="")
    inv_mathces = np.empty_like(mathces)
    inv_mathces[mathces] = np.arange(len(mathces))
    inv_mathces = inv_mathces.astype(int)
    check = modular_norm(pos_A[inv_mathces,:] - pos_B,args.tolerance)
    assert np.allclose(check,0,atol=args.tolerance), "some coding error"
    print("done")
    print("\tSaving the inverse indices to file '{:s}' ... ".format(args.inverse_output),end="")
    np.savetxt(args.inverse_output,mathces,fmt="%d")
    print("done")
    
    print("\tHow to interpret the inverse indices: posA[inv_indices,:] == posB")
        
    return 0
    
    
#---------------------------------------#
if __name__ == "__main__":
    main()
