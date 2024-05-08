#!/usr/bin/env python
from ase.io import write
from ase import Atoms
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt 

#---------------------------------------#
# Description of the script's purpose
description = "Compute the time autocorrelation function (TACF) of a quantity."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"        , **argv, required=True , type=str, help="txt/npy input file")
    # parser.add_argument("-if", "--input_format" , **argv, required=False, type=str, help="input file format (default: 'None')" , default=None)
    # parser.add_argument("-in", "--in_name"      , **argv, required=True , type=str, help="name of quantity whose TACF has to be computed")
    # parser.add_argument("-on", "--out_name"     , **argv, required=False, type=str, help="name used to save the TACF (default: '[in_name]-tacf')", default=None)
    parser.add_argument("-b" , "--blocks"       , **argv, required=False, type=int, help="number of blocks (default: 10)", default=10)
    parser.add_argument("-o" , "--output"       , **argv, required=True , type=str, help="txt/npy output file")
    # parser.add_argument("-of", "--output_format", **argv, required=False, type=str, help="output file format (default: 'None')", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    args.input = str(args.input)
    if args.input.endswith("npy"):
        data = np.load(args.input)
    else:
        data = np.load(args.input)
    print("done")
    print("\tdata shape: :",data.shape)

    #------------------#
    print("\tComputing the autocorrelation function ... ", end="")
    autocorr = correlate(data,data,mode="same")# [:len(data)]
    print("done")
    print("\tautocorr shape: :",autocorr.shape)

    #------------------#
    print("\tSaving TACF to file '{:s}' ... ".format(args.output), end="")
    if args.output.endswith("npy"):
        np.save(args.output,autocorr)
    else:
        np.savetxt(args.output,autocorr)
    print("done")

    lags = np.arange(len(autocorr))
    plt.plot(lags, autocorr)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function of Time Series')
    plt.show()

    plt.plot(data)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function of Time Series')
    plt.show()

    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()