#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
# Description of the script's purpose
description = "Evaluate a regression metric (using sklearn) between two datasets." 
# + """The possible metrics are: """ + str(list(metrics.keys()))

#---------------------------------------#
def prepare_args(description):
    # Define the command-line argument parser with a description
    import argparse
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=description,formatter_class=RawTextHelpFormatter)
    argv = {"metavar" : "\b"}
    parser.add_argument("-i" , "--input"         , **argv, type=str, required=True , help='input extxyz file')
    parser.add_argument("-e" , "--expected"      , **argv, type=str, required=True , help="keyword of the expected forces")
    parser.add_argument("-p" , "--predicted"     , **argv, type=str, required=True , help="keyword or the predicted forces")
    parser.add_argument("-po", "--plot"          , **argv, type=str, required=False, help="plot file (default: %(default)s)", default=None)
    parser.add_argument("-oi", "--output_indices", **argv, type=str, required=False, help="output file with indices (default: %(default)s)", default="indices.txt")
    parser.add_argument("-o" , "--output"        , **argv, type=str, required=True , help="output file")
    parser.add_argument("-of", "--output_format" , **argv, type=str, required=False, help="output file format (default: %(default)s)" , default=None)
    return parser

def squared_norm(arr:np.ndarray,axis:int=0):
    x = np.square(arr)
    x = np.mean(x,axis=axis)
    return np.sqrt(x)

@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    atoms = AtomicStructures.from_file(file=args.input,format="extxyz")
    print("done")

    #------------------#
    print("\n\tExtracting forces ... ",end="")
    predicted = atoms.get(args.predicted)
    expected = atoms.get(args.expected)
    N = len(atoms)
    print("done")
    
    #------------------#
    print("\tComputing atomic RMSE on forces ... ",end="")
    atomic_rmse = [None]*N
    diff = [None]*N
    for n in range(N):
        pred:np.ndarray = predicted[n]
        expe:np.ndarray = expected[n]
        assert pred.shape == expe.shape, "wrong shape"
        diff[n] = pred - expe
        atomic_rmse[n] = np.linalg.norm(diff[n],axis=1)
    print("done")
    
    #------------------#
    print("\tSetting delta forces as 'DELTA_forces' ... ",end="")
    atoms.set(f"DELTA_forces",diff,"arrays")
    print("done")
    
    print("\tSetting atomic RMSE on forces as 'RMSE_forces_atomic' ... ",end="")
    atoms.set(f"RMSE_forces_atomic",atomic_rmse,"arrays")
    print("done")
    
    #------------------#
    print("\tComputing global RMSE on forces ... ",end="")
    rmse = [None]*N
    for n in range(N):
        rmse[n] = squared_norm(atomic_rmse[n])
    print("done")
    
    print("\tSetting global RMSE on forces as 'RMSE_forces_atomic' ... ",end="")
    atoms.set(f"RMSE_forces",rmse,"info")
    print("done")
    
    #------------------#
    # Flatten arrays for statistics
    diff_flat = 1000*np.vstack(diff).ravel()                # all coordinates of all force diffs
    atomic_rmse_flat = 1000*np.hstack(atomic_rmse)           # all per-atom RMSE values
    rmse_array = 1000*np.array(rmse)                         # per-structure RMSEs

    # Helper to compute stats
    def stats(arr):
        return np.min(arr), np.max(arr), np.mean(arr)

    # Compute and print stats
    diff_min, diff_max, diff_mean = stats(diff_flat)
    atomic_min, atomic_max, atomic_mean = stats(atomic_rmse_flat)
    rmse_min, rmse_max, rmse_mean = stats(rmse_array)

    print("\n\tStatistics [meV/ang]:")
    print(f"\t    Diff (raw)        → min: {diff_min:.6f}, max: {diff_max:.6f}, mean: {diff_mean:.6f}")
    print(f"\t    Atomic RMSE       → min: {atomic_min:.6f}, max: {atomic_max:.6f}, mean: {atomic_mean:.6f}")
    print(f"\t    Global RMSE       → min: {rmse_min:.6f}, max: {rmse_max:.6f}, mean: {rmse_mean:.6f}")
    
    #-------------------#
    print(f"\n\tWriting structures to file '{args.output}' ... ", end="")
    atoms.to_file(file=args.output,format=args.output_format)
    print("done")
    
    # Find index of structure with worst atomic RMSE (max per-atom RMSE in each structure)
    max_atomic_per_structure = [np.max(rm) for rm in atomic_rmse]
    worst_atomic_idx = np.argmax(max_atomic_per_structure)

    # Find index of structure with worst global RMSE
    worst_global_idx = np.argmax(rmse)

    # Find index of structure with worst absolute diff (max abs diff among coordinates)
    max_abs_diff_per_structure = [np.max(np.abs(d)) for d in diff]
    worst_diff_idx = np.argmax(max_abs_diff_per_structure)

    print("\n\tWorst values per structure:")
    print(f"\t  Structure with worst atomic RMSE (max per-atom): {worst_atomic_idx} (value = {max_atomic_per_structure[worst_atomic_idx]:.6f})")
    print(f"\t  Structure with worst global RMSE: {worst_global_idx} (value = {rmse[worst_global_idx]:.6f})")
    print(f"\t  Structure with worst absolute force difference: {worst_diff_idx} (value = {max_abs_diff_per_structure[worst_diff_idx]:.6f})")
    
    #-------------------#
    # Number of structures
    # Compute arrays for sorting
    max_atomic_per_structure = np.array([np.max(rm) for rm in atomic_rmse])
    rmse_array_np = np.array(rmse)
    max_abs_diff_per_structure = np.array([np.max(np.abs(d)) for d in diff])

    # Get descending order indices by sorting negative values
    sorted_atomic_idx = np.argsort(-max_atomic_per_structure)
    sorted_global_idx = np.argsort(-rmse_array_np)
    sorted_diff_idx = np.argsort(-max_abs_diff_per_structure)

    # Stack indices into a (N,3) array: columns are [atomic, global, diff]
    sorted_indices = np.vstack((sorted_atomic_idx, sorted_global_idx, sorted_diff_idx)).T

    # Add global RMSE values (corresponding to sorted_global_idx) as 4th column
    sorted_global_rmse = rmse_array_np[sorted_global_idx]
    sorted_indices_with_rmse = np.hstack((sorted_indices, 1000*sorted_global_rmse[:, None]))

    # Save to file
    header = ("Index order sorting by descending values:\n"
            "Columns: Atomic_RMSE_idx  Global_RMSE_idx  Max_Abs_Diff_idx  Global_RMSE_value")
    np.savetxt(args.output_indices, sorted_indices_with_rmse, fmt=["%5d", "%5d", "%5d", "%12.6f"], header=header)

    print(f"\n\tSaved descending order indices with global RMSE values to '{args.output_indices}'")


    #-------------------#
    if args.plot is not None:
    
        print("\n\tPlotting histograms ... ", end="")

        # Flatten arrays for plotting
        atomic_rmse_flat = 1000 * np.hstack(atomic_rmse)  # all per-atom RMSEs
        rmse_array = 1000 * np.array(rmse)                # per-structure RMSEs
        diff_flat = 1000 * np.vstack(diff).ravel()        # raw diff values (no abs)

        # Bin widths
        diff_bin_width = 1
        atomic_rmse_bin_width = 1        # atomic_rmse bins width
        global_rmse_bin_width = diff_bin_width / 5  # 0.2 for global RMSE

        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        # --- First subplot: diff ---
        diff_min, diff_max = diff_flat.min(), diff_flat.max()
        diff_bins = np.arange(diff_min, diff_max + diff_bin_width, diff_bin_width)

        axes[0].hist(diff_flat, bins=diff_bins, alpha=0.7, color='tab:blue', density=False)
        axes[0].set_xlabel("Force difference [meV/Å]")
        axes[0].set_ylabel("Count")
        axes[0].set_xlim(diff_min, diff_max)

        # --- Second subplot: RMSE values with two y-axes ---
        rmse_min, rmse_max = atomic_rmse_flat.min(), atomic_rmse_flat.max()

        # Separate bins for atomic and global RMSE
        atomic_rmse_bins = np.arange(0, rmse_max + atomic_rmse_bin_width, atomic_rmse_bin_width)
        global_rmse_bins = np.arange(0, rmse_max + global_rmse_bin_width, global_rmse_bin_width)

        ax_left = axes[1]          # for Atomic RMSE
        ax_right = ax_left.twinx() # for Global RMSE

        # Atomic RMSE histogram with bin width = 1
        ax_left.hist(atomic_rmse_flat, bins=atomic_rmse_bins, alpha=0.5, color='tab:blue', label='Atomic RMSE')
        ax_left.set_ylabel("Count (atomic RMSE)", color='tab:blue')
        ax_left.tick_params(axis='y', labelcolor='tab:blue')

        # Global RMSE histogram with bin width = 0.2
        ax_right.hist(rmse_array, bins=global_rmse_bins, alpha=0.5, color='tab:orange', label='Global RMSE')
        ax_right.set_ylabel("Count (global RMSE)", color='tab:orange')
        ax_right.tick_params(axis='y', labelcolor='tab:orange')

        # Set integer y-ticks only for both y-axes
        ax_left.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_right.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Common settings
        ax_left.set_xlabel("RMSE value [meV/Å]")
        ax_left.set_xlim(0, rmse_max)

        plt.tight_layout()
        plt.savefig(args.plot, dpi=300)
        plt.close()
        print(f"done (saved as {args.plot})")



        
    pass

#---------------------------------------#
if __name__ == "__main__":
    main()
