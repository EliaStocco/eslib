#!/usr/bin/env python
import numpy as np
from ase.io import read
from ase import Atoms
from eslib.formatting import esfmt, float_format
from eslib.tools import convert
from eslib.io_tools import pattern2data, save2json
from eslib.mathematics import cumulative_mean, mean_std_err
from eslib.classes.trajectory import AtomicStructures
from eslib.input import itype
from eslib.classes.aseio import integer_to_slice_string

#---------------------------------------#
# Description of the script's purpose
description = "Compute the dielectric permittivity from dipole time series."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"          , **argv, required=True , type=str     , help="txt/npy input file with the dipoles [eang]")
    parser.add_argument("-n" , "--index"           , **argv, required=False, type=itype, help="index to be read from input file (default: %(default)s)", default=':')
    parser.add_argument("-at", "--axis_time"       , **argv, required=False, type=int     , help="axis along compute autocorrelation (default: %(default)s)", default=1)
    parser.add_argument("-ac", "--axis_component"  , **argv, required=False, type=int     , help="axis along compute the sum (default: %(default)s)", default=2)
    parser.add_argument("-dt" , "--time_step"      , **argv, required=True , type=float   , help="time step [fs]")
    parser.add_argument("-s"  , "--structure"      , **argv, required=True , type=str     , help="input file with the atomic structure")
    parser.add_argument("-f"  , "--format"         , **argv, required=False, type=str     , help="atomic structure format (default: %(default)s)", default=None)
    parser.add_argument("-T"  , "--temperature"    , **argv, required=True , type=float   , help="temperature [K]")
    parser.add_argument("-einf", "--epsilon_infinity", **argv, required=False, type=float   , help="epsilon infinity (default: %(default)s)", default=1.72)
    parser.add_argument("-o"  , "--output"         , **argv, required=False, type=str     , help="output file (default: %(default)s)", default='epsilon_r.txt')
    parser.add_argument("-oj"  , "--output_json"   , **argv, required=False, type=str     , help="JSON output file (default: %(default)s)", default='epsilon_r.json')
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    args.time_step = convert(args.time_step,"time","femtosecond","atomic_unit")
    index = integer_to_slice_string(args.index)

    #------------------#
    print("\n\tReading the input array from file '{:s}' ... ".format(args.input), end="")
    dipole:np.ndarray = pattern2data(args.input)
    ii = np.arange(dipole.shape[args.axis_time])
    ii =  ii[index]
    dipole = np.take(dipole,axis=args.axis_time,indices=ii)
    print("done")
    print("\tdata shape: ",dipole.shape)
    dipole = convert(dipole,"electric-dipole","eang","atomic_unit")
    
    #------------------#
    print(f"\n\tRemoving the mean along axis {args.axis_time} ... ",end="")
    dipole -= np.mean(dipole, axis=args.axis_time,keepdims=True)
    print("done")
    
    print(f"\n\tComputing the fluctuations along axis {args.axis_component} ... ",end="")
    fluctuations = np.sum(dipole**2,axis=args.axis_component)
    print("done")
    print("\tdata shape: ",fluctuations.shape)
    
    #------------------#
    print(f"\n\tCumulative mean along axis {args.axis_time} ... ",end="")
    cummean = cumulative_mean(fluctuations,axis=args.axis_time)
    print("done")
    print("\tdata shape: ",cummean.shape)
    
    #------------------#
    a = np.take(cummean,axis=args.axis_time,indices=0)
    b = np.take(fluctuations,axis=args.axis_time,indices=0)
    assert np.allclose(a,b), "cumulative mean and fluctuations must be equal"
    
    #------------------#
    print("\n\tReading the atomic structure from file '{:s}' ... ".format(args.structure), end="")
    structure:Atoms = AtomicStructures.from_file(file=args.structure,format=args.format)[0]
    print("done")
    volume = structure.get_volume()
    print("\tvolume [ang^3]: ",volume)
    volume = convert(volume,"volume","angstrom3","atomic_unit")
    
    #------------------#
    T = convert(args.temperature,"temperature","kelvin","atomic_unit")
    kB = 1
    epsilon_0 = 1/(4*np.pi)
    factor = 3*epsilon_0*kB*volume*T
    def fluctuations2epsilon_r(fluc)->np.ndarray:
        return fluc/factor + args.epsilon_infinity
    
    #------------------#
    print("\n\tComputing the dielectric constant ... ",end="")
    last = np.take(cummean,axis=args.axis_time,indices=-1)
    assert last.ndim == 1, "'last' mean must be 1D"
    epsilon = fluctuations2epsilon_r(last)
    print("done")
    print("\tdielectric constant: ",epsilon)
    
    #------------------#
    mean, std, err = mean_std_err(epsilon, axis=0)
    stats = {
        "values" : epsilon.tolist(),
        "mean" : mean,
        "std" : std,
        "err" : err
    }
    
    print("\n\tWriting the average dielectric constant to file '{:s}' ... ".format(args.output_json),end="")
    save2json(args.output_json,stats)
    print("done")
    
    #------------------#
    print("\n\tWriting the dielectric constant to file '{:s}' ... ".format(args.output),end="")
    epsilon_r = fluctuations2epsilon_r(cummean).T
    if str(args.output).endswith("npy"):
        np.save(args.output,epsilon_r)
    elif str(args.output).endswith("txt"):
        np.savetxt(args.output,epsilon_r,fmt=float_format)
    print("done")
    
#---------------------------------------#
if __name__ == "__main__":
    main()