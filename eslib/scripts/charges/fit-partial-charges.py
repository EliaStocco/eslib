#!/usr/bin/env python
import numpy as np
from eslib.classes.dipole import DipolePartialCharges
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
from eslib.metrics import metrics
from scipy.optimize import minimize
from eslib.show import show_dict
import json

#---------------------------------------#
# Description of the script's purpose
description = "Compute the partial charges using a fit of the dipoles along a trajectory."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"    , **argv,type=str, required=True , help="file with the atomic configurations [a.u]")
    parser.add_argument("-n", "--name"     , **argv,type=str, required=False, help="keyword (default: %(default)s)" , default="dipole")
    parser.add_argument("-o", "--output"   , **argv,type=str, required=False, help="JSON output file with the partial charges (default: %(default)s)", default='partial-charges.json')
    # parser.add_argument("-z", "--born_charges"   , **argv,type=str, help="output file with the Born Effective Charges (default: %(default)s)", default='bec.oxn.txt')
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input)
    print("done")

    yreal = np.zeros((len(trajectory),3))
    for n,structure in enumerate(trajectory):
        yreal[n,:] = structure.info[args.name]

    #------------------#
    symbols = trajectory[0].get_chemical_symbols()
    species = np.unique(symbols)

    #------------------#
    def build_charges(charges):
        return {k: v for k, v in zip(species, charges)}

    #------------------#
    charges = np.unique(trajectory[0].get_atomic_numbers()) #np.random.rand(len(species))
    _charges = build_charges(charges)
    model = DipolePartialCharges(_charges)
    reference = trajectory[0]
    # model = DipolePartialCharges(ref=reference,dipole=dipole,bec=None)


    # def bec_OxN(charges):
    #     ox = dict(zip(species, input_oxn))
    #     on = oxidation_number(symbols,ox)
    #     bec = bec_from_oxidation_number(reference,on)
    #     bec.force_asr()
    #     return bec.isel(structure=0).to_numpy()

    def model_PC(charges):
        model.set_charges(build_charges(charges))
        return model
    
    def loss(charges):
        model = model_PC(charges)
        model.impose_charge_neutrality(reference)
        ypred = model.compute(trajectory)
        return metrics["rmse"](yreal,ypred)
    
    def constraint_func(charges):
        model = model_PC(charges)
        return model.compute_total_charge(reference)
    
    #------------------#
    print("\tMinimizing the loss function ... ",end="")
    result = minimize(loss, charges,constraints={'type': 'eq', 'fun': constraint_func})
    print("done")

    # print("\tComputed partial charges: ")
    # pc = build_charges(result.x)
    # for key,value in pc.items():
    #     print("\t\t{:<2s}: {:.2f}".format(key,value))
    # print()
    print("\n\tPartial charges: ")
    charges = build_charges(result.x)
    model.set_charges(charges)
    show_dict(model.charges,"\t\t",2)

    #------------------#
    
    # all_charges = [ charges[s] for s in reference.get_chemical_symbols() ]
    print("\n\tTotal charge: ",model.compute_total_charge(reference))
    model.impose_charge_neutrality(reference)
    # mean = np.mean(all_charges)
    # for k in charges.keys():
    #     charges[k] -= mean

    #------------------#
    print("\n\tPartial charges (corrected): ")
    show_dict(model.charges,"\t\t",2)
    
    #------------------#
    if args.output is not None:
        print("\n\tWriting partial charges to file '{:s}' ... ".format(args.output), end="")
        with open(args.output, 'w') as json_file:
            json.dump(charges, json_file, indent=4)
        print("done")   

#---------------------------------------#
if __name__ == "__main__":
    main()