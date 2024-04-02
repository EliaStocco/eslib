#!/usr/bin/env python
from ase.io import read
import numpy as np
from ase import Atoms
from eslib.classes.dipole import DipoleLinearModel
from eslib.classes.trajectory import AtomicStructures
from eslib.input import flist
from eslib.tools import lattice2cart, frac2cart
from eslib.formatting import esfmt, warning, error

#---------------------------------------#
# Description of the script's purpose
description = "Create a linar model for the dipole of a system given the Born Effective Charges of a reference configuration."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # parser.add_argument("-i", "--input"    , **argv,type=str, help="file with the atomic configurations [a.u]")
    # parser.add_argument("-n", "--index"    , **argv,type=int, help="index of the reference configuration",default=None)
    parser.add_argument("-r", "--reference", **argv,type=str, help="file with the reference configuration [a.u.,xyz]")
    parser.add_argument("-d", "--dipole"   , **argv,type=flist, help="dipole (quanta) of the reference configuration (default: None)",default=None)
    parser.add_argument("-k", "--keyword"  , **argv,type=str, help="keyword for the dipole (default: 'dipole')", default='dipole')
    parser.add_argument("-z", "--bec"      , **argv,type=str, help="file with the BEC tensors of the reference configuration [txt] (default: None --> specify -n,--index)",default=None)
    parser.add_argument("-o", "--output"   , **argv,type=str, help="output file with the dipole linear model (default: 'DipoleModel.pickle')", default="DipoleModel.pickle")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    # #------------------#
    # # trajectory
    # print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    # trajectory = AtomicStructures.from_file(file=args.input)
    # print("done")

    #------------------#
    # index
    ref = None
    bec = None
    dipole = None

    if args.reference is not None:
        ref = read(args.reference)#.get_positions()
    if args.bec is not None:
        bec = np.loadtxt(args.bec)

    # if args.index is not None:
        # reference = trajectory[args.index]
        # if ref is None:
        #     try: ref = Atoms(reference)#.get_positions()
        #     except: pass
    if bec is None:
        try: bec = np.asarray(ref.arrays["bec"]) 
        except: pass
    if args.dipole is None:
        try: dipole = np.asarray(ref.info[args.keyword])
        except: pass
    # del reference

    #------------------#
    if dipole is None:
        if args.dipole is None:
            print("\t{:s}: you need to provide the dipole as an 'info' in the reference configuration or using -d,--dipole".format(error))
        if np.all(ref.get_pbc()):
            print("\tThe provided reference configuration is periodic: the input dipole will be interpreted as 'dipole quanta'")
            print("\tConverting the dipole quanta into cartesian coordinates:")
            print("\t{:10s}: ".format("quanta"),args.dipole)
            # dipole = lattice2cart(cell=ref.get_cell(),v=args.dipole)
            # 'args.dipole' is a 3-vectos with numbers in [0,1].
            # The cartesian components has to be stretched depending on the lattice vectors lenght. 
            dipole = frac2cart(cell=ref.get_cell(),v=args.dipole)
        else:
            print("\tThe provided reference configuration is not periodic: the input dipole will be interpreted as cartesian coordinates of the dipole.")
        
        print("\t{:10s}: ".format("dipole"),dipole)
    #------------------#
    print("\n\tCreating the linear model for the dipole ... ", end="")
    model = DipoleLinearModel(ref=ref,dipole=dipole.reshape((3,)),bec=bec)
    print("done")

    #------------------#
    print("\n\tSaving the model to file '{:s}' ... ".format(args.output), end="")
    model.to_pickle(args.output)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()