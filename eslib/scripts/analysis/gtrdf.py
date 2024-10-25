#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   rdfs.py
@Time    :   2024/05/07 10:11:41
@Author  :   George Trenins
@Contact :   gstrenin@gmail.com
@Desc    :   None
'''


from __future__ import absolute_import, division, print_function

import sys

import numpy as np

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format
from eslib.fortran import fortran_rdfs
from eslib.input import itype, size_type
from eslib.physics import get_element_mass

description = "Calculate the radial distribution functions for salty water"

def index_in_slice(slice_obj: slice, index: int) -> bool:
    """
    Check if the given index is within the specified slice object.

    Parameters:
    slice_obj (slice): The slice object to check against.
    index (int): The index to check.

    Returns:
    bool: True if the index is within the slice, False otherwise.
    """
    # Create a range object from the slice object's start, stop, and step attributes
    range_obj = range(slice_obj.start if slice_obj.start else 0,
                      slice_obj.stop if slice_obj.stop else sys.maxsize,
                      slice_obj.step if slice_obj.step else 1)

    # Check if the index is within the range object
    return index in range_obj

def string_to_slice(s: str) -> slice:
    if s == ":":
        slc = slice(None)
    else:
        try:
            slc = int(s)
        except ValueError:
            slc = slice(*[int(s) if s else None for s in s.split(":")])
    return slc

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    el:callable = lambda s:size_type(s,str,2)
    parser.add_argument("-i" , "--input"       , **argv, required=True , type=str  , help="input file")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-in", "--index"       , **argv, required=False, type=itype, help="index to be read from input file (default: %(default)s)", default=':')
    parser.add_argument("-e" , "--elements"    , **argv, required=True , type=el   , help="elements list")
    parser.add_argument("-n" , "--nbins"       , **argv, required=False, type=int  , help="number of bins to divide RDF (default: %(default)s)", default=100)
    parser.add_argument("-m" , "--min"         , **argv, required=False, type=float, help="minimum distance in the plot (default: %(default)s)", default=1.)
    parser.add_argument("-r1", "--rmin"        , **argv, required=False, type=float, help="minimum distance of RDF (default: %(default)s)", default=1.)
    parser.add_argument("-r2", "--rmax"        , **argv, required=False, type=float, help="maximum distance of RDF (default: %(default)s)", default=5.)
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str  , help="output file (default: %(default)s)", default="rdf.csv")
    
    # parser.add_argument("--index", default=':', help="frames to include in the calculation")
    # parser.add_argument("--rmin", type=float, default=0, help="Minimum distance for RDF calculation in Angstrom")
    # parser.add_argument("--rmax", type=float, default=6, help="Maximum distance for RDF calculation in Angstrom")
    # parser.add_argument("--bins", type=int, default=100, help="Number of bins")
    # parser.add_argument("--stride", type=int, default=1, help="Stride with which to print the RDF data to file.")
    # parser.add_argument("a1", help="Chemical symbol for the first atom in the RDF pair")
    # parser.add_argument("a2", help="Chemical symbol for the second atom in the RDF pair")
    # parser.add_argument("traj", nargs='+', help="XYZ trajectory files produced by i-PI")
    return parser 
#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    #---------------------------------------#
    # atomic structures
    print("\tReading atomic structures from file '{:s}' ... ".format(args.input), end="")
    trajectory = AtomicStructures.from_file(file=args.input, format=args.input_format,index=args.index)
    print("done")
    N = len(trajectory)
    print("\tn. of atomic structures: {:d}".format(N))
    print("\tn. of atoms: {:d}".format(len(trajectory[0])))
    symbols = np.asarray(trajectory[0].get_chemical_symbols())
    species = np.unique(symbols)
    print("\tspecies: {}".format(species))
    
    #---------------------------------------#
    dr = (args.rmax - args.rmin) / args.nbins  # RDF step
    print("\n\tdr = {:.3f} angstrom".format(dr))
    # RDF array, first column contains the position grid, second column -- RDF proper
    rdf = np.array(
        [[args.rmin + (0.5 + i) * dr, 0] for i in range(args.nbins)], order="F"
    )
    shellVolumes = 4*np.pi/3 * ((rdf[:, 0] + 0.5 * dr) ** 3 - (rdf[:, 0] - 0.5 * dr) ** 3)
    
    mass = trajectory[0].get_masses()
    
    massA, massB = get_element_mass(args.elements) # Elements.mass(args.elements[0])
    
    species_A = [
            3 * i + j
            for i in np.where(symbols == args.elements[0])[0]
            for j in range(3)
        ]
    species_B = [
            3 * i + j
            for i in np.where(symbols == args.elements[1])[0]
            for j in range(3)
        ]
    natomsA = len(species_A)
    natomsB = len(species_B)
    print("\tindices for {:s}: {} (tot. {:d})".format(args.elements[0], species_A, natomsA))
    print("\tindices for {:s}: {} (tot. {:d})".format(args.elements[1], species_B, natomsB))
    
    posA = np.zeros(natomsA, order="F")
    posB = np.zeros(natomsB, order="F")
    
    #---------------------------------------#
    print()
    for n,atoms in enumerate(trajectory):
        print("\tCalculating RDF: {:d}/{:d}".format(n+1,N), end="\r")
        pos = atoms.get_positions().flatten()
        posA[:] = pos[species_A]
        posB[:] = pos[species_B]
        cell = np.asarray(atoms.get_cell()).T
        inverseCell = np.linalg.inv(cell)        
        fortran_rdfs(rdf, posA, posB, args.rmin, args.rmax, cell, inverseCell, massA, massB)
    print("\n\tdone")
        
    rdf = np.copy(rdf)
    rdf[:,1] /= len(trajectory)
    # Creating RDF from N(r)
    rdf[:, 1] *= 1/shellVolumes
    
    print("\n\tWriting RDF to file '{:s}' ... ".format(args.output), end="")
    header = "r [ang], RDF"
    np.savetxt(args.output, rdf,fmt=float_format,header=header)
    print("done")
    
    return 0 


if __name__ == "__main__":
    main()