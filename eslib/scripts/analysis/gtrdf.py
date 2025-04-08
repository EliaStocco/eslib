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

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, float_format, warning, eslog
from eslib.fortran import fortran_rdfs_fixed_cell, fortran_rdfs_variable_cell
from eslib.input import itype, size_type, str2bool
from eslib.physics import get_element_mass

description = "Calculate the Radial Distribution Function (RDF) using a Fortran routine."

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
    parser.add_argument("-r1", "--rmin"        , **argv, required=False, type=float, help="minimum distance of RDF (default: %(default)s)", default=1.)
    parser.add_argument("-r2", "--rmax"        , **argv, required=False, type=float, help="maximum distance of RDF (default: %(default)s)", default=5.)
    parser.add_argument("-sc", "--same_cell"   , **argv, required=False, type=str2bool, help="all the cells are the same (default: %(default)s)", default=True)
    parser.add_argument("-o" , "--output"      , **argv, required=False, type=str  , help="output file (default: %(default)s)", default="rdf.csv")
    parser.add_argument("-p" , "--plot"        , **argv, required=False, type=str  , help="plot file (default: %(default)s)", default=None)
    return parser 
#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):
    
    #---------------------------------------#
    # atomic structures
    with eslog(f"Reading atomic structures from file '{args.input}'"):
        trajectory = AtomicStructures.from_file(file=args.input, format=args.input_format,index=args.index)
    N = len(trajectory)
    Natoms = len(trajectory[0])
    print("\t n. of atomic structures: {:d}".format(N))
    print("\t n. of atoms: {:d}".format(Natoms))
    symbols = np.asarray(trajectory[0].get_chemical_symbols())
    species = np.unique(symbols)
    print("\t species: {}".format(species))
    
    #---------------------------------------#
    dr = (args.rmax - args.rmin) / args.nbins  # RDF step
    print("\n\t dr = {:.3f} angstrom".format(dr))
    # RDF array, first column contains the position grid, second column -- RDF proper
    rdf = np.array(
        [[args.rmin + (0.5 + i) * dr, 0] for i in range(args.nbins)], order="F"
    )
    shellVolumes = 4*np.pi/3 * ((rdf[:, 0] + 0.5 * dr) ** 3 - (rdf[:, 0] - 0.5 * dr) ** 3)
    
    # mass = trajectory[0].get_masses()
    
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
    print("\t n. of {:s} atoms: {:d}".format(args.elements[0], natomsA))
    print("\t n. of {:s} atoms: {:d}".format(args.elements[1], natomsB))
    
    #---------------------------------------#
    print()
    if args.same_cell:
        print("\t {:s}: the adopted algorithm will assume that all snapshots have the same cell".format(warning))
        with eslog("Calculating RDF"):
            posA = np.zeros((N,natomsA), order="F")
            posB = np.zeros((N,natomsB), order="F")
            pos = trajectory.get("positions").reshape((N,3*Natoms))
            posA[:,:] = pos[:,species_A].reshape((N,natomsA))
            posB[:,:] = pos[:,species_B].reshape((N,natomsB))
            cell = np.asarray(trajectory[0].get_cell()).T
            inverseCell = np.linalg.inv(cell)  
            fortran_rdfs_fixed_cell(rdf, posA, posB, args.rmin, args.rmax, cell, inverseCell, massA, massB)
    else:
        print("\t {:s}: the adopted algorithm will assume that all snapshots could have different cells".format(warning))
        posA = np.zeros(natomsA, order="F")
        posB = np.zeros(natomsB, order="F")
        with eslog("Calculating RDF"):
            for n,atoms in enumerate(trajectory):
                pos = atoms.get_positions().flatten()
                posA[:] = pos[species_A]
                posB[:] = pos[species_B]
                cell = np.asarray(atoms.get_cell()).T
                inverseCell = np.linalg.inv(cell)        
                fortran_rdfs_variable_cell(rdf, posA, posB, args.rmin, args.rmax, cell, inverseCell, massA, massB)
        
    rdf = np.copy(rdf)
    rdf[:,1] /= len(trajectory)
    # Creating RDF from N(r)
    rdf[:, 1] *= 1/shellVolumes
    
    print("\n\t Writing RDF to file '{:s}' ... ".format(args.output), end="")
    header = "r [ang], RDF"
    np.savetxt(args.output, rdf,fmt=float_format,header=header)
    print("done")
    
    if args.plot is not None:
        print("\t Preparing plot ... ".format(args.plot), end="")
        plt.figure(figsize=(6,4))
        plt.plot(rdf[:,0],rdf[:,1],color="blue")
        plt.xlabel("distance $[\\AA]$")
        plt.ylabel("RDF")
        plt.xlim(args.rmin,args.rmax)
        plt.ylim(0,None)
        plt.grid()
        # plt.legend()
        plt.tight_layout()        
        print("done\n")

        print("\t Saving plot to file '{:s}' ... ".format(args.plot), end="")
        plt.savefig(args.plot)
        plt.close()
        print("done\n")
    
    return 0 


if __name__ == "__main__":
    main()