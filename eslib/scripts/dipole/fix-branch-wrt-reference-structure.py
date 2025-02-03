#!/usr/bin/env python
import json

import numpy as np

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.models.dipole import DipolePartialCharges
from eslib.formatting import esfmt, warning
from eslib.show import show_dict
from eslib.tools import is_integer
from eslib.tools import cart2frac, frac2cart
from eslib.io_tools import save2json

#---------------------------------------#
# Description of the script's purpose
description = "Fix the dipole branch of a structure w.r.t. a reference one."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"           , **argv, required=True , type=str, help="input file with the atomic structures")
    parser.add_argument("-if", "--input_format"    , **argv, required=False, type=str, help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-r" , "--reference"       , **argv, required=True , type=str, help="reference file with the atomic structures")
    parser.add_argument("-rf", "--reference_format", **argv, required=False, type=str, help="reference file format (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--name"            , **argv, required=False, type=str, help="name for the charges (default: %(default)s)", default='Qs')
    parser.add_argument("-c" , "--charges"         , **argv, required=True , type=str, help="JSON file with the charges")
    parser.add_argument("-o" , "--output"          , **argv, required=True , type=str, help="output file with the dipole difference")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # charges
    print("\tReading the charges from file '{:s}' ... ".format(args.charges), end="")
    with open(args.charges, 'r') as json_file:
        charges:dict = json.load(json_file)
    print("done")

    #------------------#
    print("\n\tCharges: ")
    show_dict(charges,"\t",2)

    for k,c in charges.items():
        if not is_integer(c):
            print("\t{:s}: '{:s}' charge is not an integer".format(warning,k))
        charges[k] = np.round(c,0)
        
    #------------------#
    print("\n\tCreating dipole model based on the charges ... ",end="")
    model = DipolePartialCharges(charges)
    print("done")

    #------------------#
    print("\n\tReading the atomic structure from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,format=args.input_format,index=0)[0]
    print("done")
    assert np.all(atoms.get_pbc()), "The atomic structure must be periodic."
    
    #------------------#
    print("\tReading the reference atomic structure from file '{:s}' ... ".format(args.input), end="")
    reference = AtomicStructures.from_file(file=args.reference,format=args.reference_format,index=0)[0]
    print("done")
    assert np.all(reference.get_pbc()), "The atomic structure must be periodic."
    
    assert atoms.get_global_number_of_atoms() == reference.get_global_number_of_atoms(), "Different number of atoms."
    cell_a = np.asarray(atoms.get_cell()).flatten()
    cell_b = np.asarray(reference.get_cell()).flatten()
    assert np.allclose(cell_a,cell_b), "Lattice vectors must be identical."
    
    #------------------#
    dipoles = model.compute([atoms,reference],raw=True)["dipole"]
    cell = atoms.get_cell()
    quanta = cart2frac(cell,dipoles)
    delta = np.asarray(quanta[1]-quanta[0]) 
    
    assert np.allclose(delta, delta.astype(int)), "The difference should contain only integers."
    
    #------------------#
    output = {
        "quanta" : delta.astype(int),
        "dipole" : frac2cart(cell,delta)
    }
    print("\n\tOutput data:")
    show_dict(output, "\t", 2)  # This will display the output dictionary before saving


    #------------------#
    print("\n\tWriting the dipole difference to file '{:s}' ... ".format(args.output), end="")
    save2json(args.output,output)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()