#!/usr/bin/env python
import argparse
from ase.io import read
import numpy as np
from eslib.functions import str2bool
from eslib.input import size_type
from eslib.show import matrix2str

description="Return a vector in cartesian coordinates depending on its lattice vector coordinates."

#---------------------------------------#
def main():

    # Define the command-line argument parser with a description
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"     ,   **argv,type=str      , help="input file with the cell")
    parser.add_argument("-v" , "--vector"    ,   **argv,type=size_type, help="vector components in lattice coordinates")
    parser.add_argument("-n" , "--normalize" ,   **argv,type=str2bool , help="whether the vectors is expressed w.r.t. normalized lattice vectors (default: %(default)s)",default=True)
    parser.add_argument("-a" , "--amplitude" ,   **argv,type=float    , help="amplitude of the output vector (default: %(default)s)",default=1.)
    parser.add_argument("-d" , "--digit"     ,   **argv,type=int      , help="digit of the final result (default: %(default)s)",default=8)
    
    # Print the script's description
    print("\n{:s}".format(description))

    # Parse the command-line arguments
    print("\nReading input arguments ... ",end="")
    args = parser.parse_args()
    print("done")

    print("\n\tInput arguments:")
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    ###
    # read the MD trajectory from file
    print("\tReading lattice vectors from file '{:s}' ... ".format(args.input), end="")
    cell = np.asarray(read(args.input).cell).T
    print("done\n")

    print("\n\tLattice vectors:")
    line = matrix2str(cell,col_names=["1","2","3"],cols_align="^",width=6)
    print(line)
    # line = "{:2s}| {:^10s} | {:^10s} | {:^10s} |".format(" ","a1","a2","a3")
    # _end_line_ = "\t  |" + "-"*(len(line)-4) + "|"
    # print(_end_line_)
    # print("\t"+line)
    # print(_end_line_)
    # for n,d in enumerate(["x","y","z"]):
    #     line = "\t{:2s}| {:>10.6f} | {:>10.6f} | {:>10.6f} |".format(d,cell[n,0],cell[n,1],cell[n,2])
    #     print(line)
    # print(_end_line_)

    if args.normalize:
        cell /= np.linalg.norm(cell,axis=0)
        print("\n\tLattice vectors normalized:")
        line = matrix2str(cell,col_names=["1","2","3"],cols_align="^",width=6)
        print(line)

    out = cell[:,0]*args.vector[0] + cell[:,1]*args.vector[1] + cell[:,2]*args.vector[2]

    print()
    print("\t{:>20s}:".format("Required vector"),out)

    if args.amplitude != 1.:
        out *= args.amplitude
        print("\t{:>20s}:".format("Scaled vector"),out)

    out = np.round(out,args.digit)
    print("\t{:>20s}:".format("Rounded vector"),out)

    print()
    string = "{:>" + "{:d}".format(args.digit+8) + ".{:d}".format(args.digit) + "e}"
    string = "[{:s},{:s},{:s}]".format(string,string,string)
    print("\t{:>20s}:".format("Final vector")+string.format(out[0],out[1],out[2]))

    print("\n\tJob done :)\n")


if __name__ == "__main__":
    main()