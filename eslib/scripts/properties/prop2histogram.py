#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from eslib.classes.properties import Properties
from eslib.formatting import esfmt, warning
from eslib.functions import suppress_output
from eslib.input import str2bool, itype, slist
from eslib.tools import convert
from eslib.classes.aseio import integer_to_slice_string

#---------------------------------------#
# Description of the script's purpose
description = "Plot an histogram of some properties."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"          , type=str     , **argv, required=True , help='txt input file')
    parser.add_argument("-k" , "--keywords"       , type=slist   , **argv, required=True , help='keywords')
    parser.add_argument("-f" , "--family"         , type=str     , **argv, required=False, help="family (default: %(default)s)", default=None)
    parser.add_argument("-u" , "--unit"           , type=str     , **argv, required=False, help="output unit (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--index"          , type=itype   , **argv, required=False, help="index to be read from input file (default: %(default)s)", default=':')
    # parser.add_argument("-s" , "--subsample"      , type=str     , **argv, required=False, help="txt file with the indices to keep (default: %(default)s)", default=None)
    parser.add_argument("-fd", "--fix_data"       , type=str2bool, **argv, required=False, help='whether to remove replicas and fill missing values (default: %(default)s)', default=False)
    parser.add_argument("-d" , "--delimiter"      , type=str     , **argv, required=False, help='delimiter (default: %(default)s)', default=' ')
    parser.add_argument("-o" , "--output"         , type=str     , **argv, required=False, help='output file (default: %(default)s)', default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    # assert not args.remove_replicas, "-rr,--remove_replicas no longer supported."
    
    if args.index != ':' and args.subsample is not None:
        print(f"\n\t{warning}: pay attention when using both --index and --subsample.")

    #------------------#
    index = integer_to_slice_string(args.index)
    print("\tReading properties from file '{:s}' ... ".format(args.input), end="")
    with suppress_output():
        if str(args.input).endswith(".pickle"):
            assert index is None, "`index` must be None when reading from `*.pickle` files."                
            allproperties = Properties.from_pickle(file_path=args.input)
        else:
            allproperties = Properties.load(file=args.input,index=index)
    print("done")

    #---------------------------------------#
    # summary
    print("\n\tSummary of the properties: ")
    df = allproperties.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))

    #------------------#
    
    if args.fix_data:
        print("\n\tFixing data ... ", end="")
        allproperties, message = allproperties.fix()
        print("done")
        print(f"\tMessage : {message}")    
        
        print("\n\tSummary of the properties: ")
        df = allproperties.summary()
        tmp = "\n"+df.to_string(index=False)
        print(tmp.replace("\n", "\n\t"))

    #------------------#
    data = {}
    for k in args.keywords:
        print(f"\n\tExtracting {k} ... ",end="")
        data[k] = allproperties.get(k)
        print("done")
        print(f"\t - {k}.shape: {data[k].shape}")
        print(f"\t - {k} unit: {allproperties.units[k]}")
    
    # #------------------#
    # if args.subsample is not None:
    #     subsample = np.loadtxt(args.subsample).astype(int)
    #     assert subsample.ndim == 1, f"'{args.subsample}' should contain a 1D array."
    #     print("\n\tSubsampling data:")
    #     print("\tdata.shape : ",data.shape," (before subsampling)")
    #     data = data[subsample]
    #     print("\tdata.shape : ",data.shape," (after subsampling)")
    
    #------------------#
    if args.unit is not None:
        iu = allproperties.units[args.keywords[0]]
        ou = args.unit
        print(f"\n\tConverting data from {iu} to {ou} ... ",end="")
        for k in data.keys():
            data[k] = convert(data[k],args.family,_from=iu,_to=ou)
        print("done")        
        
        #------------------#
    # Plot all histograms on the same figure
    print("\n\tPlotting histograms ...")

        #------------------#
    # Plot histograms and violin plots side-by-side
    print("\n\tPlotting histograms and violin plots ...")

    # Strict dimensionality check
    for k, arr in data.items():
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError(
                f"Property '{k}' is {arr.ndim}D with shape {arr.shape}, "
                "but histogram plotting requires 1D data."
            )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- First subplot: histogram ---
    labels = []
    for k, arr in data.items():
        axes[0].hist(arr, bins=50, alpha=0.5, edgecolor='black', label=f"{k} [{allproperties.units[k]}]")
        labels.append(k)
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Histograms of selected properties")
    axes[0].legend()
    axes[0].grid(True)

    # --- Second subplot: violin plot with mean & quantiles ---
    import seaborn as sns
    combined_data = []
    combined_labels = []
    for k, arr in data.items():
        combined_data.extend(arr)
        combined_labels.extend([k] * len(arr))

    sns.violinplot(
        x=combined_labels,
        y=combined_data,
        ax=axes[1],
        inner=None,
        cut=0
    )

    # Add mean & quantile lines for each category
    unique_labels = list(data.keys())
    for i, k in enumerate(unique_labels):
        arr = np.array(data[k])
        mean_val = np.mean(arr)
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        axes[1].plot(i, mean_val, marker='o', color='red', label='Mean' if i == 0 else "")
        axes[1].hlines([q25, q50, q75], i - 0.2, i + 0.2, colors='black', linestyles='--', lw=1)

    axes[1].set_xlabel("Property")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Violin plot with mean & quantiles")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"\tSaved combined histogram & violin plot to '{args.output}'")
    else:
        plt.show()

    print("\tAll plots generated.")
    

#---------------------------------------#
if __name__ == "__main__":
    main()
