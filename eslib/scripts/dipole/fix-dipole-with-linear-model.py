#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.plot import plot_bisector
from eslib.classes.dipole import DipoleModel
from eslib.classes.trajectory import info
from eslib.classes.trajectory import trajectory as Trajectory
from eslib.tools import cart2lattice, cart2frac, frac2cart
from eslib.output import output_folder
from ase.io import write
from eslib.formatting import everythingok,warning

#---------------------------------------#
# Description of the script's purpose
description = "Fix the dipole jumps using a (previously created) linear model."
warning = "***Warning***"
error = "***Error***"
closure = "Job done :)"
information = "You should provide the positions as printed by i-PI."
input_arguments = "Input arguments"

#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN    + Style.BRIGHT + description             + Style.RESET_ALL
    warning         = Fore.MAGENTA  + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    error           = Fore.RED      + Style.BRIGHT + error.replace("*","")   + Style.RESET_ALL
    closure         = Fore.BLUE     + Style.BRIGHT + closure                 + Style.RESET_ALL
    information     = Fore.YELLOW   + Style.NORMAL + information             + Style.RESET_ALL
    input_arguments = Fore.GREEN    + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass

#---------------------------------------#
def correlation_plot(A,B,nameA,nameB,file):
    fig,axes = plt.subplots(ncols=3,figsize=(15,5))

    labels = ["x","y","z"]
    for n,ax in enumerate(axes):
        ax.scatter(A[:,n],B[:,n],label=labels[n])
        ax.grid()
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        for n in range(-20,20):
            plot_bisector(ax,n)
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        
    axes[1].set_xlabel(nameA)
    axes[0].set_ylabel(nameB)

    plt.tight_layout()
    plt.savefig(file)

#---------------------------------------#
def prepare_args():
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input"    , **argv, type=str, help="extxyz file with the atomic configurations [a.u]")
    parser.add_argument("-k", "--keyword"  , **argv, type=str, help="keyword related to dipole to be fixed in the extxyz file (default: 'dipole')", default='dipole')
    parser.add_argument("-m", "--model"    , **argv, type=str, help="pickle file with the dipole linear model (default: 'DipoleModel.pickle')", default='DipoleModel.pickle')
    parser.add_argument("-o", "--output"   , **argv, type=str, help="output file with the fixed trajectory (default: 'trajectory.fixed.extxyz')", default="trajectory.fixed.extxyz")
    parser.add_argument("-f", "--folder"   , **argv, type=str, help="output folder with additional output files (default: None)", default=None)
    return parser# .parse_args()

#---------------------------------------#
def main():

    #------------------#
    # Parse the command-line arguments
    args = prepare_args()

    # Print the script's description
    print("\n\t{:s}".format(description))

    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #------------------#
    # trajectory
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = Trajectory(args.input)
    print("done")
    print("\tn. of atomic structures: ",len(trajectory))

    #------------------#
    # dipole
    print("\tExtracting '{:s}' from the provided atomic structures ... ".format(args.keyword), end="")
    dft = info(trajectory,args.keyword)
    print("done")

    # #---------------------------------------#
    # # lattice vectors
    # print("\tExtracting the lattice vectors from the trajectory ... ", end="")
    # def get_cell(e:Atoms):
    #     return e.get_cell()
    # lattices = trajectory.call(get_cell)
    # print("done")

    #------------------#
    # linear model
    print("\tLoading the dipole linear model from file '{:s}' ... ".format(args.model), end="")
    model = DipoleModel.from_pickle(args.model)
    print("done")

    #------------------#
    print("\tCheck the distances w.r.t. the reference configuration ... ", end="")
    indices = model.control_periodicity(list(trajectory))
    print("done")
    if indices is None or len(indices) == 0:
        print("\t{:s}".format(everythingok))
    else:
        print("\t{:s}: found {:d} structures that could be wrong.".format(warning,len(indices)))
        # print("\tatomic structures indices: ", indices.tolist())

    #------------------#
    # dipole
    print("\n\tComputing the dipoles using the linear model ... ", end="")
    linear = model.get(list(trajectory))
    print("done")

    #------------------#
    if args.folder is not None:
        output_folder(args.folder)
    
    if args.folder is not None:
        print()
        file = "{:s}/dipole.linear-model.txt".format(args.folder)
        print("\tSaving dipoles computed from the linear model in file '{:s}' ... ".format(file), end="")
        np.savetxt(file,linear)
        print("done")

        file = "{:s}/dipole.dft.txt".format(args.folder)
        print("\tSaving dipoles computed from DFT in file '{:s}' ... ".format(file), end="")
        np.savetxt(file,dft)
        print("done")

        file = "{:s}/dipole.correlation.pdf".format(args.folder)
        print("\tSaving correlation plot between DFT and LM dipole to file '{:s}' ... ".format(file), end="")        
        correlation_plot(dft,linear,"DFT","LM",file)
        print("done")

    #------------------#
    # quanta
    print("\n\tComputing the dipole quanta ... ", end="")
    N = len(trajectory)
    quanta = {
        "DFT" : np.zeros((N,3)),
        "LM"  : np.zeros((N,3))
    } 
    cell = trajectory[0].get_cell()#lattices[n]
    lenght = cell.cellpar()[0:3]
    R = cart2lattice(cell)
    for n in range(N):       
        quanta["DFT"][n,:] = R @ dft[n]    / lenght
        quanta["LM"][n,:]  = R @ linear[n] / lenght
    print("done")

    #------------------#
    # jumps
    factor = np.asarray(quanta["DFT"] - quanta["LM"])
    intfactor = np.round(factor) #.astype(int)
    
    if args.folder is not None:
        print()

        file = "{:s}/quanta.dft.txt".format(args.folder)
        print("\tSaving quanta computed from DFT to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,quanta["DFT"])
        print("done")

        file = "{:s}/quanta.lm.txt".format(args.folder)
        print("\tSaving quanta computed from LM to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,quanta["LM"])
        print("done")

        file = "{:s}/factors.float.txt".format(args.folder)
        print("\tSaving differences between DFT and LM dipoles to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,factor)
        print("done")

        file = "{:s}/factors.int.txt".format(args.folder)
        print("\tSaving (rounded to integers) differences between DFT and LM dipoles to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,intfactor)
        print("done")

        file = "{:s}/factors.diff.txt".format(args.folder)
        print("\tSaving differences between teh previous two saved quantities to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,factor-intfactor)
        print("done")

        file = "{:s}/quanta.correlation.pdf".format(args.folder)
        print("\tSaving correlation plot between DFT and LM quanta to file '{:s}' ... ".format(file), end="")        
        correlation_plot(quanta["DFT"],quanta["LM"],"DFT","LM",file)
        print("done")

    #------------------#
    # correction
    
    print("\n\tCorrecting the DFT dipoles ... ", end="")
    N = len(trajectory)
    fixed_dipole = np.full((N,3),np.nan)
    cell = trajectory[0].get_cell()#lattices[n]
    lenght = cell.cellpar()[0:3]
    
    # R = lattice2cart(cell)
    # fixed_dipole = ( R @ (quanta["DFT"] -  intfactor).T ).T* lenght
    
    # The previous code lines should be the same as the following:
    fixed_dipole = frac2cart(cell=trajectory[0].get_cell(),v=(quanta["DFT"] -  intfactor))
    print("done")

    #------------------#
    # set info
    print("\tAdding fixed dipole as info 'dipole' and quanta as 'quanta' to atomic structures ... ", end="")
    for n in range(N):
        trajectory[n].info["dipole"] = fixed_dipole[n,:]
        _quanta = cart2frac(cell=trajectory[n].get_cell(),v=fixed_dipole[n,:])
        trajectory[n].info["quanta"] = _quanta.flatten()
    print("done")

    #------------------#
    print("\tLooking for outliers ... ", end="")
    for n in range(N):
        fd = trajectory[n].info["dipole"]
        fd_quanta  = cart2frac(cell=trajectory[n].get_cell(),v=fd)
        lm_quanta = cart2frac(cell=trajectory[n].get_cell(),v=linear[n])
        norm = np.sqrt(np.square(fd_quanta-lm_quanta).sum())
        # norm = np.linalg.norm(fd-linear[n])
        print(norm)
    print("done")

    #------------------#
    # writing
    print("\n\tWriting output to file '{:s}' ... ".format(args.output), end="")
    try:
        write(args.output, list(trajectory), format="extxyz") # fmt)
        print("done")
    except Exception as e:
        print(f"\n\t{error}: {e}")

    # #------------------#
    # # output
    # print("\tSaving the fixed dipoles to file '{:s}' ... ".format(args.output), end="")
    # np.savetxt(args.output,fixed_dipole,fmt='%24.18e')
    # print("done")

    if args.folder is not None:
        print()

        print("\tComputing the quanta of the fixed dipoles ... ", end="")
        R = cart2lattice(cell)
        fixed_quanta = (R @ fixed_dipole.T).T / lenght
        print("done")

        file = "{:s}/quanta.fixed.txt".format(args.folder)
        print("\tSaving quanta of the fixed dipoles to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,fixed_quanta)
        print("done")

        file = "{:s}/quanta-fixed.correlation.pdf".format(args.folder)
        print("\tSaving correlation plot between fixed DFT and LM quanta to file '{:s}' ... ".format(file), end="")        
        correlation_plot(fixed_quanta,quanta["LM"],"DFT","LM",file)
        print("done")

        # file = "{:s}/dipole-fixed.correlation.pdf".format(args.folder)
        # print("\tSaving correlation plot between fixed DFT and LM dipole to file '{:s}' ... ".format(file), end="")        
        # correlation_plot(fixed_dipole,linear,"DFT","LM",file)
        # print("done")

    #------------------#
    # Script completion message
    print("\n\t{:s}\n".format(closure))

#---------------------------------------#
if __name__ == "__main__":
    main()