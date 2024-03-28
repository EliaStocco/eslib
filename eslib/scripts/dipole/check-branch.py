#!/usr/bin/env python
from ase.io import read
import matplotlib.pyplot as plt
from eslib.input import slist
from eslib.formatting import esfmt
from eslib.classes.dipole import DipoleModel
from eslib.plot import plot_bisector
from eslib.physics import compute_dipole_quanta
from copy import copy
from eslib.plot import generate_colors

#---------------------------------------#
description = "Plot the branches of the dipoles for different datasets."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"  , **argv, type=slist, required=True , help="input 'extxyz' files")
    parser.add_argument("-m", "--model"  , **argv, type=str,   required=False, help="pickle file with the dipole linear model (default: 'DipoleModel.pickle')", default='DipoleModel.pickle')
    parser.add_argument("-o", "--output" , **argv, type=str,   required=False, help="file with the branches plot (default: 'branch.pdf')", default="branch.pdf")
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # linear model
    print("\tLoading the dipole linear model from file '{:s}' ... ".format(args.model), end="")
    model = DipoleModel.from_pickle(args.model)
    print("done")
    print("\tLinear model dipole: ",model.get_dipole())
    print("\tLinear model quanta: ",model.get_quanta())

    #------------------#
    # read
    print("\n\tReading atomic structures from file:")
    trajectories = [None]*len(args.input)
    for n,file in enumerate(args.input):
        print("\t\t{:d}: '{:s}' ... ".format(n,file), end="")
        trajectories[n] = read(file,format='extxyz',index=":")
        print("done --> (n. atomic structures: {:d})".format(len(trajectories[n])))

    #------------------#
    # DFT quanta
    quanta = [None]*len(args.input)
    print("\n\tComputing the DFT dipole quanta:")
    for n,atoms in enumerate(trajectories):
        print("\t\t{:d}: '{:s}' ... ".format(n,args.input[n]), end="")
        _,quanta[n] = compute_dipole_quanta(atoms)
        print("done")

    #------------------#
    # linear model dipoles
    model_dipoles = [None]*len(args.input)
    print("\n\tComputing the dipoles using the linear model:")
    for n,atoms in enumerate(trajectories):
        print("\t\t{:d}: '{:s}' ... ".format(n,args.input[n]), end="")
        model_dipoles[n] = model.get(atoms)
        print("done")

    #------------------#
    # linear model quanta
    model_quanta = [None]*len(args.input)
    print("\n\tComputing the linear model dipole quanta:")
    for n,atoms in enumerate(trajectories):
        print("\t\t{:d}: '{:s}' ... ".format(n,args.input[n]), end="")
        tmp = copy(atoms)
        for i,snapshot in enumerate(tmp):
            snapshot.info["dipole"] = model_dipoles[n][i]
        _,model_quanta[n] = compute_dipole_quanta(tmp)
        print("done")

    #------------------#
    # plot
    print("\n\tCreating the correlation plot ... ",end="")
    fig,axes = plt.subplots(ncols=3,figsize=(15,5))
    colors = generate_colors(len(args.input),"viridis")
    
    for n,atoms in enumerate(trajectories):
        print("\t\t{:d}: '{:s}' ... ".format(n,args.input[n]), end="")
        
        A = quanta[n]
        B = model_quanta[n]

        for k,ax in enumerate(axes):
            ax.scatter(A[:,k],B[:,k],label=str(n+1),color=colors[n])
            ax.grid()
        print("done")

    for ax in axes:
        ax.grid()
        ax.legend(loc="upper left")
        for i in range(-20,20):
            if i == 0 :
                plot_bisector(ax,i,argv={"color":"red"})
            else:
                plot_bisector(ax,i)

    plt.tight_layout()
    

    #------------------#
    # write
    print("\n\tSaving plot to file '{:s}' ... ".format(args.output),end="")
    plt.savefig(args.output)
    print("done")
    
#---------------------------------------#
if __name__ == "__main__":
    main()
