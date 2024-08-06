#!/usr/bin/env python
from ase.io import read
import numpy as np
from eslib.classes.models.dipole.baseclass import DipoleModel
from classes.atomic_structures import AtomicStructures
from classes.atomic_structures import info
from eslib.formatting import esfmt, float_format, warning
from eslib.physics import oxidation_number
from eslib.physics import bec_from_oxidation_number
from eslib.metrics import metrics
from scipy.optimize import minimize

#---------------------------------------#
# Description of the script's purpose
description = "Create a linar model for the dipole of a system given the Born Effective Charges of a reference configuration."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"    , **argv,type=str, help="file with the atomic configurations [a.u]")
    parser.add_argument("-k"  , "--keyword"     , **argv,type=str, help="keyword (default: %(default)s)" , default="dipole")
    parser.add_argument("-o", "--output"   , **argv,type=str, help="output file with the dipole linear model (default: %(default)s)", default="DipoleModel.pickle")
    parser.add_argument("-z", "--born_charges"   , **argv,type=str, help="output file with the Born Effective Charges (default: %(default)s)", default='bec.oxn.txt')
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\t{:s}: I should fix the dipole during the minimization!".format(warning))

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input)
    print("done")

    reference = trajectory[0].copy()
    dipole = reference.info[args.keyword].reshape((3,))
    yreal = info(trajectory,args.keyword)

    #------------------#
    symbols = reference.get_chemical_symbols()
    species = np.unique(symbols)
    input_oxn = np.random.rand(len(species))
    model = DipoleModel(ref=reference,dipole=dipole,bec=None)


    def bec_OxN(input_oxn):
        ox = dict(zip(species, input_oxn))
        on = oxidation_number(symbols,ox)
        bec = bec_from_oxidation_number(reference,on)
        bec.force_asr()
        return bec.isel(structure=0).to_numpy()

    def model_OxN(input_oxn):
        bec = bec_OxN(input_oxn)
        model.set_bec(bec)
        return model
    
    def loss(input_oxn):
        model = model_OxN(input_oxn)
        ypred = model.compute(trajectory)
        return metrics["rmse"](yreal,ypred)
    
    #------------------#
    print("\tMinimizing the loss function ... ",end="")
    result = minimize(loss, input_oxn)
    print("done")

    print("\tComputed oxidation numbers: ")
    ox = dict(zip(species, input_oxn)) 
    for key,value in ox.items():
        print("\t\t{:<2s}: {:.2f}".format(key,value))
    print()
    
    #------------------#
    print("\tSaving BECs to file '{:s}' ... ".format(args.born_charges), end="")
    bec = bec_OxN(result.x)
    np.savetxt(args.born_charges, bec,fmt=float_format)
    print("done")
   
    #------------------#
    print("\tSaving the model to file '{:s}' ... ".format(args.output), end="")
    model = model_OxN(result.x)
    model.to_pickle(args.output)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()