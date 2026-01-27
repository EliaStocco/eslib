#!/usr/bin/env python
import numpy as np
import pandas as pd
from eslib.formatting import esfmt
from eslib.input import slist

description = "Convert a txt to a csv file."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}

    parser.add_argument("-i", "--input", type=str, **argv, required=True, help="txt input file")
    parser.add_argument("-c", "--columns", type=slist, **argv, required=False,
                        help="columns (default: auto-generated)", default=None)
    parser.add_argument("-o", "--output", type=str, **argv, required=True, help="csv output file")
    return parser

@esfmt(prepare_args, description)
def main(args):

    # safer than loadtxt; handles missing values
    data = np.genfromtxt(args.input)

    # ensure 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    ncols = data.shape[1]

    # auto-generate columns if None
    if args.columns is None:
        args.columns = [f"col_{i+1}" for i in range(ncols)]

    # validate custom columns
    if len(args.columns) != ncols:
        raise ValueError(
            f"Number of provided column names ({len(args.columns)}) "
            f"does not match data columns ({ncols})"
        )

    df = pd.DataFrame(data=data, columns=args.columns)

    df.to_csv(args.output, index=True)

if __name__ == "__main__":
    main()
