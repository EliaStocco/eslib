#!/usr/bin/env python
import argparse
from ase.io import read
import numpy as np
from eslib.input import str2bool
from eslib.input import size_type
from eslib.show import matrix2str
from eslib.formatting import esfmt

#---------------------------------------#
description="Return a vector in cartesian coordinates depending on its lattice vector coordinates."
#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"           ,   **argv,type=str      , help="input file with the cell")
    parser.add_argument("-v" , "--vector"          ,   **argv,type=size_type, help="vector components in lattice coordinates")
    parser.add_argument("-nc" , "--normalize_cell"  ,   **argv,type=str2bool , help="whether the vectors is expressed w.r.t. normalized lattice vectors (default: %(default)s)",default=True)
    parser.add_argument("-nv" , "--normalize_vector",   **argv,type=str2bool , help="whether to normalize the input vector (default: %(default)s)",default=True)
    parser.add_argument("-a" , "--amplitude"       ,   **argv,type=float    , help="amplitude of the output vector (default: %(default)s)",default=1.)
    parser.add_argument("-d" , "--digit"           ,   **argv,type=int      , help="digit of the final result (default: %(default)s)",default=8)
    return parser# .parse_args(

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # read the MD trajectory from file
    print("\tReading lattice vectors from file '{:s}' ... ".format(args.input), end="")
    cell = np.asarray(read(args.input).cell).T
    print("done\n")

    #------------------#
    print("\tLattice vectors:")
    line = matrix2str(cell,col_names=["1","2","3"],cols_align="^",width=6)
    print(line)

    #------------------#
    if args.normalize_cell:
        cell /= np.linalg.norm(cell,axis=0)
        print("\tLattice vectors normalized:")
        line = matrix2str(cell,col_names=["1","2","3"],cols_align="^",width=6)
        print(line)

    out = cell[:,0]*args.vector[0] + cell[:,1]*args.vector[1] + cell[:,2]*args.vector[2]

    if args.normalize_vector:
        print("\tNormalizing the input vector:")
        norm = np.linalg.norm(out)
        print("\t - original norm: ",norm)
        out /= norm
        print("\t -  current norm: ",np.linalg.norm(out))
        assert not np.allclose(norm,1), "coding error"

    #------------------#
    print("\n\tResults:")
    print("\t{:>20s}:".format("(norm.) vector"),out)

    #------------------#
    if args.amplitude != 1.:
        out *= args.amplitude
        print("\t{:>20s}:".format("Amp * vector"),out)

    out = np.round(out,args.digit)
    print("\t{:>20s}:".format("Rounded vector"),out)

    #------------------#
    string = "{:>" + "{:d}".format(args.digit+8) + ".{:d}".format(args.digit) + "e}"
    string = "[{:s},{:s},{:s}]".format(string,string,string)
    print("\n\t{:>20s}: ".format("Final vector")+string.format(out[0],out[1],out[2]))

    print("\n\tThe vector has norm: ",np.linalg.norm(out))


#---------------------------------------#
if __name__ == "__main__":
    main()