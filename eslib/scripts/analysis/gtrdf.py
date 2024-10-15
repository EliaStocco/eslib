#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   rdfs.py
@Time    :   2024/05/07 10:11:41
@Author  :   George Trenins
@Contact :   gstrenin@gmail.com
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
from ase.geometry.analysis import Analysis
from ase import io
import argparse
import re
import numpy as np
from ipi.utils.io import read_file
from ipi.utils.units import unit_to_internal, unit_to_user, Constants, Elements
from eslib.fortran import fortran_rdfs
import sys
# from gtlib.utils.arrays import index_in_slice

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

def main() -> None:

    parser = argparse.ArgumentParser(description="Calculate the radial distribution functions for salty water")
    parser.add_argument("--index", default=':', help="frames to include in the calculation")
    parser.add_argument("--rmin", type=float, default=0, help="Minimum distance for RDF calculation in Angstrom")
    parser.add_argument("--rmax", type=float, default=6, help="Maximum distance for RDF calculation in Angstrom")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins")
    parser.add_argument("--stride", type=int, default=100, help="Stride with which to print the RDF data to file.")
    parser.add_argument("a1", help="Chemical symbol for the first atom in the RDF pair")
    parser.add_argument("a2", help="Chemical symbol for the second atom in the RDF pair")
    parser.add_argument("traj", nargs='+', help="XYZ trajectory files produced by i-PI")


    args = parser.parse_args()

    # Convert string representation to slice object
    slc = string_to_slice(args.index)
    nbeads = len(args.traj)
    pos_files = [open(fn, "r") for fn in args.traj]
    massA = Elements.mass(args.a1)
    massB = Elements.mass(args.a2)
    r_min = unit_to_internal("length", "angstrom", args.rmin)  # Minimal distance for RDF
    r_max = unit_to_internal("length", "angstrom", args.rmax)  # Maximal distance for RDF
    dr = (r_max - r_min) / args.bins  # RDF step
    # RDF array, first column contains the position grid, second column -- RDF proper
    rdf = np.array(
        [[r_min + (0.5 + i) * dr, 0] for i in range(args.bins)], order="F"
    )
    shellVolumes = 4*np.pi/3 * ((rdf[:, 0] + 0.5 * dr) ** 3 - (rdf[:, 0] - 0.5 * dr) ** 3)

    ifr = 0
    isample = 0
    natoms = None
    fn_out_rdf = f"rdf{args.a1}{args.a2}.csv"
    while True:
        if ifr % args.stride == 0:
            print("\rProcessing frame {:d}".format(ifr), end=" ")
            sys.stdout.flush()
        try:
            for i in range(nbeads):
                ret = read_file("xyz", pos_files[i], dimension="length")
                if not natoms:
                    mass, natoms = ret["atoms"].m, ret["atoms"].natoms
                    pos = np.zeros((nbeads, 3 * natoms), order="F")
                cell = ret["cell"].h
                inverseCell = ret["cell"].get_ih()
                cellVolume = ret["cell"].get_volume()
                pos[i, :] = ret["atoms"].q
        except EOFError:  # finished reading files
            break

        if index_in_slice(slc, ifr):
            # select the target atoms:
            species_A = [
                    3 * i + j
                    for i in np.where(mass == massA)[0]
                    for j in range(3)
                ]
            species_B = [
                    3 * i + j
                    for i in np.where(mass == massB)[0]
                    for j in range(3)
                ]
            natomsA = len(species_A)
            natomsB = len(species_B)    
            posA = np.zeros((nbeads, natomsA), order="F")
            posB = np.zeros((nbeads, natomsB), order="F")
            for bead in range(nbeads):
                posA[bead, :] = pos[bead, species_A]
                posB[bead, :] = pos[bead, species_B]
            fortran_rdfs.updateqrdf(
                rdf,
                posA,
                posB,
                r_min,
                r_max,
                cell,
                inverseCell,
                massA,
                massB)
            isample += 1
        ifr += 1
        if isample > 0 and ifr % args.stride == 0:
            # Normalization
            _rdf = np.copy(rdf)
            _rdf[:,1] /= isample * nbeads
            # Creating RDF from N(r)
            _rdf[:, 1] *= cellVolume/shellVolumes
            for bin in range(args.bins):
                _rdf[bin, 0] = unit_to_user("length", "angstrom", _rdf[bin, 0])
            np.savetxt(fn_out_rdf, _rdf)
    print()

    # # Get cell dimensions assuming i-PI comment-line format
    # with open(args.traj, 'r') as f:
    #     f.readline()
    #     cmnt = f.readline()

    # # Parse comment to extract cell 
    # pattern = re.compile(
    #     r".*CELL\(abcABC\):\s*(?P<a>[\d,\.]*)\s*(?P<b>[\d,\.]*)\s*(?P<c>[\d,\.]*)\s*(?P<alpha>[\d,\.]*)\s*(?P<beta>[\d,\.]*)\s*(?P<gamma>[\d,\.]*).*", re.VERBOSE)
    # res = pattern.match(cmnt)
    # if not res:
    #     raise RuntimeError(f"Could not parse the comment line to extract cell parameters, check that you are using the i-PI format:\n{cmnt}")
    # a, b, c = [res.group(s) for s in ['a', 'b', 'c']]
    # alpha, beta, gamma = [res.group(s) for s in 'alpha,beta,gamma'.split(',')]
    # frames = io.read(args.traj, index=':')
    # for frame in frames:
    #     frame.set_pbc(True)
    #     frame.set_cell([a, b, c, alpha, beta, gamma])
    # analyse = Analysis(frames)
    # rdf = analyse.get_rdf(args.rmax, args.bins, imageIdx=slc, elements=[args.a1, args.a2], return_dists=True)
    # rdf = np.mean(rdf,axis=0)
    # np.savetxt(f"rdf{args.a1}{args.a2}.csv", rdf[::-1,:].T, fmt="%14.7e", delimiter=" ")

if __name__ == "__main__":
    main()