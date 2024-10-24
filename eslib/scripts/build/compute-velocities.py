#!/usr/bin/env python
from ase.io import write
import numpy as np
from ase import Atoms
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, eslog
from eslib.tools import convert

#---------------------------------------#
description = "Compute the velocities from a trajectory."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"           , **argv, required=True , type=str, help="file with an atomic structure")
    parser.add_argument("-if", "--input_format"    , **argv, required=False, type=str, help="input file format (default: %(default)s)" , default=None)
    parser.add_argument("-up" , "--unit_positions" , **argv, required=False, type=str, help="positions unit (default: %(default)s)", default="angstrom")
    parser.add_argument("-dt" , "--time_step"      , **argv, required=True , type=float, help="time step")
    parser.add_argument("-udt" , "--unit_time_step", **argv, required=False, type=str  , help="time step unit (default: %(default)s)", default="femtosecond")
    parser.add_argument("-o"  , "--output"         , **argv, required=True , type=str, help="output file with the velocities")
    parser.add_argument("-of" , "--output_format"  , **argv, required=False, type=str, help="output file format (default: %(default)s)", default="xyz")
    parser.add_argument("-uv" , "--unit_velocity"  , **argv, required=False, type=str, help="velocity unit (default: %(default)s)", default="atomic_unit")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    # trajectory
    print("\tReading the first atomic structure from file '{:s}' ... ".format(args.input), end="")
    structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
    print("done")
    N = len(structures)
    print("\tn. of structures: ",len(structures),"\n")
    
    #------------------#
    if args.unit_positions != "atomic_unit":
        factor_pos = convert(what=1,family="length",_from=args.unit_positions,_to="atomic_unit")
        print(f"\tConverting positions from '{args.unit_positions}' to 'atomic_unit' using factor {factor_pos} ... ",end="")
        for n,atoms in enumerate(structures):
            atoms.positions *= factor_pos
        print("done")
        
    #------------------#
    if args.unit_time_step != "atomic_unit":
        factor_time = convert(what=1,family="time",_from=args.unit_time_step,_to="atomic_unit")
        print(f"\tConverting time step from '{args.unit_time_step}' to 'atomic_unit' using factor {factor_time} ... ",end="")
        args.time_step *= factor_time
        print("done")
        
    print("\tComputing velocities ... ", end="")
    pos = structures.get_array("positions")
    vel = np.gradient(pos,axis=0)/args.time_step
    velocities = [None]*N
    for n,atoms in enumerate(structures):
        velocities[n] = Atoms(symbols=atoms.get_chemical_symbols(),positions=vel[n],cell=None,pbc=False)
    velocities = AtomicStructures(velocities)
    print("done")
    
    print("\tComputing temperature ... ", end="")
    masses = structures[0].get_masses()
    masses = convert(masses,family="mass",_from="dalton",_to="atomic_unit")
    v2 = np.square(vel)
    kinetic = 0.5*(masses[np.newaxis,:,np.newaxis] * v2).sum(axis=(1,2))
    temperature = 2.*kinetic/(3.*structures[0].get_global_number_of_atoms())
    temperature = convert(temperature,"temperature",_from="atomic_unit",_to="kelvin")
    print("done")
    
    print("\n\tTemperature = ",temperature.mean(),"K")
    
    #------------------#
    factor_vel = 1
    if args.unit_velocity != "atomic_unit":
        factor_vel = convert(what=1,family="velocity",_from="atomic_unit",_to=args.unit_velocity)
        print(f"\tConverting time velocities from 'atomic_unit' to '{args.unit_velocity}' using factor {factor_vel} ... ",end="")
        for n,atoms in enumerate(velocities):
            atoms.positions *= factor_vel
        print("done")
    
    print("\n\tSaving velocities to file '{:s}' ... ".format(args.output), end="")
    velocities.to_file(file=args.output,format=args.output_format)
    print("done")    

#---------------------------------------#
if __name__ == "__main__":
    main()