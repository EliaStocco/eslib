#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.plot import plot_bisector, square_plot, legend_options
from eslib.input import str2bool
from eslib.mathematics import atleast_3d, atleast_2d

#---------------------------------------#
# Description of the script's purpose
description = "Compare two properties."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"               , **argv, type=str     , required=True , help="extxyz file with the atomic configurations [a.u]")
    parser.add_argument("-rn" , "--ref_name"            , **argv, type=str     , required=True , help="name of the reference quantity")
    parser.add_argument("-pn" , "--pred_name"           , **argv, type=str     , required=True , help="name of the predicted quantity")
    parser.add_argument("-f" , "--multiplicative_factor", **argv, type=float   , required=False, help="multiplicative factor (default: %(default)s)", default=1)
    parser.add_argument("-da" , "--divide_by_atoms"     , **argv, type=str2bool, required=False, help="divide the properties by the number of atoms (default: %(default)s)", default=False)
    parser.add_argument("-t"  , "--threshold"           , **argv, type=float   , required=False, help="RMSE threshold (default: %(default)s)", default=1e3)   
    parser.add_argument("-p"  , "--plot"                , **argv, type=str     , required=False, help="output file with the plot (default: %(default)s)", default='comparison.pdf')
    parser.add_argument("-o"  , "--output"              , **argv, type=str2bool, required=False, help="whether to output the good and outliers structures to file (default: %(default)s)", default=False)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input)
    print("done")
    print("\tn. of atomic structures: ",len(trajectory),end="\n\n")

    #------------------#
    assert trajectory.has(args.ref_name)
    assert trajectory.has(args.pred_name)
    
    #------------------#
    print("\tExtracting '{:s}' from the atomic structures... ".format(args.ref_name), end="")
    real = trajectory.get(args.ref_name)*args.multiplicative_factor
    # real = np.atleast_2d(real)
    print("done")
    print("\t'{:s}' shape: ".format(args.ref_name),real.shape,end="\n\n")

    #------------------#
    print("\tExtracting '{:s}' from the atomic structures... ".format(args.pred_name), end="")
    pred = trajectory.get(args.pred_name)*args.multiplicative_factor
    # pred = np.atleast_2d(pred)
    print("done")
    print("\t'{:s}' shape: ".format(args.pred_name),pred.shape,end="\n\n")
    
    #------------------#
    is_array = trajectory.search(args.ref_name) == "arrays"
    if is_array:
        real = atleast_3d(real)
        pred = atleast_3d(pred)
    else:
        real = atleast_2d(real)
        pred = atleast_2d(pred)
    
    components = real.shape[-1]
    
    #------------------#
    natoms = trajectory.num_atoms()
    if args.divide_by_atoms:
        print("\tDividing properties by the number of atoms ... ", end="")
        real = real / natoms
        pred = pred / natoms
        print("done",end="\n\n")
        
    #------------------#
    print("\tAnalysing results ... ", end="")
    if not is_array:
        rmse = np.sqrt( np.sum( (pred - real)**2 , axis=-1 ) ) # sum over components
    else:
        rmse = np.sqrt( np.sum( (pred - real)**2 , axis=(-1,-2) ) / natoms ) # sum over components, mean over atoms
    outliers = rmse > args.threshold
    n_outliers = np.sum(outliers)
    data4hist = rmse[~outliers]
    print("done")
    print("\tNumber of outliers (RMSE > {:g}): {:d} / {:d} ".format(args.threshold,n_outliers,len(rmse)),end="\n\n")
    RMSE = np.mean(data4hist)
    print("\tRMSE (without outliers): {:g} ".format(RMSE),end="\n\n")
    
    #------------------#
    print("\tPreparing correlation plot ... ", end="")
    fig, axes = plt.subplots(1, 3, figsize=(15,5)) 

    # ---------------- Correlation plot ----------------
    ax = axes[0]
    # Non-outliers (blue)
    ax.scatter(
        real[~outliers].flatten(),
        pred[~outliers].flatten(),
        s=12,
        color="blue",
        label="non-outliers"
    )

    # Outliers (red)
    ax.scatter(
        real[outliers].flatten(),
        pred[outliers].flatten(),
        s=20,
        color="red",
        label="outliers"
    )
    plot_bisector(ax)
    square_plot(ax)
    ax.grid()
    ax.legend(**legend_options)

    suffix = " / atoms" if args.divide_by_atoms else ""
    ax.set_xlabel(args.ref_name + suffix)
    ax.set_ylabel(args.pred_name + suffix)
    ax.set_title("correlation plot")

    # ---------------- RMSE histogram ----------------
    ax = axes[1]
    ax.hist(data4hist, bins=50, color='gray', edgecolor='black')
    ax.set_xlabel("RMSE" + suffix)
    ax.set_ylabel("counts")
    ax.set_title("RMSE distribution")
    ax.set_xlim(0, min(args.threshold, np.max(data4hist)))
    ax.grid()

    # ---------------- RMSE vs Threshold plot ----------------
    ax = axes[2]

    thr_list = (rmse[outliers]+1e-8).tolist() + (rmse[outliers]-1e-8).tolist() + np.linspace(args.threshold/20,args.threshold,100).tolist()
    thr_list = np.asarray(thr_list)
    thr_list.sort()
    all_rmse = np.zeros_like(thr_list)
    for i,thr in enumerate(thr_list):
        all_rmse[i] = np.mean(rmse[rmse < thr])

    ax.plot(thr_list, all_rmse, color='blue', label='RMSE')
    ax.set_xscale('log')
    ax.set_xlabel("threshold")
    ax.set_ylabel("RMSE"+suffix,color='blue')
    ax.tick_params(axis='y', colors='blue')  # make y-axis ticks blue
    ax.grid()
    ax.scatter(args.threshold, RMSE, color='purple', marker='x')
    # ax.legend(**legend_options)

    axcount = ax.twinx()
    count_below = np.array([np.sum(rmse < thr) for thr in thr_list])
    axcount.plot(thr_list, count_below, color='red', label='count')
    axcount.set_ylabel("# data",color='red')
    axcount.tick_params(axis='y', colors='red')  # make y-axis ticks blue

    ax.set_title("RMSE vs threshold")

    print("done")

    print("\tSaving plot to file '{:s}' ... ".format(args.plot), end="")
    plt.tight_layout()
    plt.savefig(args.plot)
    print("done")
    
    if args.output:
        
        indices = np.arange(len(trajectory))   
        ii_good = indices[~outliers]
        ii_outliers = indices[outliers]
        good = trajectory.subsample(ii_good)
        outliers = trajectory.subsample(ii_outliers)
        
        file = "outliers.extxyz"
        print("\tSaving outlier structures to file '{:s}' ... ".format(file), end="")
        outliers.to_file(file=file)
        print("done")
        
        file = "outliers.txt"
        print("\tSaving outlier indices to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,ii_outliers,fmt="%d")
        print("done")
        
        file = "good.extxyz"
        print("\tSaving good structures to file '{:s}' ... ".format(file), end="")
        good.to_file(file=file)
        print("done")
        
        file = "good.txt"
        print("\tSaving good indices to file '{:s}' ... ".format(file), end="")
        np.savetxt(file,ii_good,fmt="%d")
        print("done")

        
    return
    
#---------------------------------------#
if __name__ == "__main__":
    main()