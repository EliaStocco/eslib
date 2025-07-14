#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import json

from eslib.classes.atomic_structures import AtomicStructures
from eslib.classes.models.dipole import DipolePartialCharges
from eslib.formatting import esfmt, float_format
from eslib.output import output_folder
from eslib.physics import compute_dipole_quanta
from eslib.plot import correlation_plot
from eslib.tools import frac2cart
from eslib.show import show_dict


#---------------------------------------#
# Description of the script's purpose
description = "Fix the dipole jumps using a point-charges model."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"     , **argv, type=str, required=True , help="extxyz file with the atomic configurations [a.u]")
    parser.add_argument("-id", "--in_dipole" , **argv, type=str, required=True , help="name of the input dipoles")
    parser.add_argument("-od", "--out_dipole", **argv, type=str, required=True , help="name of the output dipoles")
    parser.add_argument("-c" , "--charges"   , **argv, type=str, required=True , help="JSON file with the partial charges")
    parser.add_argument("-o" , "--output"    , **argv, type=str, required=True , help="output file with the fixed trajectory")
    parser.add_argument("-f" , "--folder"    , **argv, type=str, required=False, help="output folder with additional output files (default: %(default)s)", default="fix-dipole")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input)
    print("done")
    print("\tn. of atomic structures: ",len(trajectory))
    
    #------------------#
    # charges
    print("\tReading the charges from file '{:s}' ... ".format(args.charges), end="")
    with open(args.charges, 'r') as json_file:
        charges:dict = json.load(json_file)
    print("done")

    #------------------#
    print("\n\tCharges: ")
    show_dict(charges,"\t\t",2)
        
    #------------------#
    print("\n\tCreating dipole model based on the charges ... ",end="")
    model = DipolePartialCharges(charges)
    print("done")
    
    model.summary()

    #------------------#
    # print("\n\tAdding charges as '{:s}' to the 'arrays' of the atomic structures ... ".format(args.out_dipole),end="")
    for n,structure in enumerate(trajectory):
        if not model.check_charge_neutrality(structure):
            raise ValueError("structure . {:d} is not charge neutral".format(n))
        # structure.arrays[args.out_dipole] = model.get_all_charges(structure)
    # print("done")

    #------------------#
    # dipole
    print("\n\tExtracting '{:s}' from the provided atomic structures ... ".format(args.in_dipole), end="")
    dft = trajectory.get(args.in_dipole)
    print("done")

    #------------------#
    # dipole
    print("\n\tComputing the dipoles using the linear model ... ", end="")
    linear = model.get(trajectory)
    print("done")

    #------------------#
    if args.folder is not None:
        output_folder(args.folder)
    
    if args.folder is not None:
        print()
        file = "{:s}/dipole.linear-model.txt".format(args.folder)
        print("\tSaving dipoles computed from the linear model to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,linear,fmt=float_format)
        print("done")

        file = "{:s}/dipole.dft.txt".format(args.folder)
        print("\tSaving dipoles computed from DFT to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,dft,fmt=float_format)
        print("done")

        file = "{:s}/dipole.correlation.pdf".format(args.folder)
        print("\tSaving correlation plot between DFT and LM dipole to file '{:s}' ... ".format(file), end="")        
        correlation_plot(dft,linear,"DFT","LM",file,bisectors=False)
        print("done")

    #------------------#
        
    # quanta
    print("\n\tComputing the dipole quanta ... ", end="")
    _,quanta_dft = compute_dipole_quanta(trajectory,in_keyword=args.in_dipole)
    model_traj = trajectory.copy()
    model_traj.set(args.in_dipole,linear,"info")
    _,quanta_model = compute_dipole_quanta(model_traj,in_keyword=args.in_dipole)
    quanta = {
        "DFT" : quanta_dft,
        "LM"  : quanta_model
    } 
    del model_traj
    print("done")

    #------------------#
    # jumps
    factor = np.asarray(quanta["DFT"] - quanta["LM"])
    intfactor = np.round(factor) #.astype(int)
    
    if args.folder is not None:
        print()

        file = "{:s}/quanta.dft.txt".format(args.folder)
        print("\tSaving quanta computed from DFT to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,quanta["DFT"],fmt=float_format)
        print("done")

        file = "{:s}/quanta.lm.txt".format(args.folder)
        print("\tSaving quanta computed from LM to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,quanta["LM"],fmt=float_format)
        print("done")

        file = "{:s}/factors.float.txt".format(args.folder)
        print("\tSaving differences between DFT and LM dipoles to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,factor,fmt=float_format)
        print("done")

        file = "{:s}/factors.int.txt".format(args.folder)
        print("\tSaving (rounded to integers) differences between DFT and LM dipoles to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,intfactor,fmt='%d')
        print("done")

        file = "{:s}/factors.diff.txt".format(args.folder)
        print("\tSaving differences between teh previous two saved quantities to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,factor-intfactor,fmt=float_format)
        print("done")

        file = "{:s}/quanta.correlation.pdf".format(args.folder)
        print("\tSaving correlation plot between DFT and LM quanta to file '{:s}' ... ".format(file), end="")        
        correlation_plot(quanta["DFT"],quanta["LM"],"DFT","LM",file,bisectors=True)
        print("done")

    #------------------#
    # correction
    print("\n\tCorrecting the DFT dipoles ... ", end="")
    fixed_dipole = np.full((len(trajectory),3),np.nan)
    for n,atoms in enumerate(trajectory):
        v = quanta["DFT"][n,:] -  intfactor[n,:]
        assert v.shape == (3,)
        fixed_dipole[n,:] = frac2cart(cell=atoms.get_cell(),v=v)
    print("done")

    print("\n\tn. of corrected values along:")
    print("\t\t1st lattice vector: ",np.count_nonzero(intfactor[:,0]))
    print("\t\t2nd lattice vector: ",np.count_nonzero(intfactor[:,1]))
    print("\t\t3rd lattice vector: ",np.count_nonzero(intfactor[:,2]))

    #------------------#
    # set info
    print("\n\tAdding fixed dipole as info '{:s}'  to atomic structures ... ".format(args.out_dipole), end="")
    trajectory.set(args.out_dipole,fixed_dipole,"info")
    assert np.allclose(trajectory.get(args.out_dipole),fixed_dipole)
    print("done")


    _,fixed_quanta =  compute_dipole_quanta(trajectory,in_keyword=args.out_dipole)
    factor = np.asarray(fixed_quanta - quanta["LM"])
    intfactor = np.round(factor)
    assert np.sum(np.abs(intfactor)) == 0
    assert trajectory.is_there(args.out_dipole)


    #------------------#
    # writing
    print("\n\tWriting output to file '{:s}' ... ".format(args.output), end="")
    trajectory.to_file(file=args.output,format="extxyz")
    print("done")

    if args.folder is not None:
        print()

        file = "{:s}/quanta.fixed.txt".format(args.folder)
        print("\tSaving quanta of the fixed dipoles to file '{:s}' ... ".format(file), end="")
        _,fixed_quanta = compute_dipole_quanta(trajectory,in_keyword=args.out_dipole)
        np.savetxt(file,fixed_quanta,fmt=float_format)
        print("done")

        file = "{:s}/quanta-fixed.correlation.pdf".format(args.folder)
        print("\tSaving correlation plot between fixed DFT and LM quanta to file '{:s}' ... ".format(file), end="")        
        correlation_plot(fixed_quanta,quanta["LM"],"DFT","LM",file,bisectors=True)
        print("done")

        file = "{:s}/dipole-fixed.correlation.pdf".format(args.folder)
        print("\tSaving correlation plot between fixed DFT and LM dipole to file '{:s}' ... ".format(file), end="")        
        _fixed_dipole = trajectory.get(args.out_dipole)
        correlation_plot(_fixed_dipole,linear,"DFT","LM",file,bisectors=False)
        print("done")
        
        assert np.allclose(_fixed_dipole,fixed_dipole)

        file = "{:s}/dipole.fixed.txt".format(args.folder)
        print("\tSaving the fixed dipoles to file '{:s}' ... ".format(args.output), end="")
        np.savetxt(file,fixed_dipole,fmt=float_format)
        print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()