#!/usr/bin/env python
from eslib.formatting import esfmt

#---------------------------------------#
description = "Compute minimum and maximum frequencies from simulation time and timestep."
#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-T","--simtime" , **argv, type=float, required=True, help="Total simulation time in picoseconds")
    parser.add_argument("-dt","--timestep", **argv, type=float, required=True, help="Time step in femtoseconds")
    return parser

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    # Convert to seconds
    simtime_s = args.simtime * 1e-12    # ps → s
    timestep_s = args.timestep * 1e-15  # fs → s

    # Frequency calculations
    f_min = 1.0 / simtime_s             # Hz
    f_max = 1.0 / (2.0 * timestep_s)    # Hz

    # Output
    print("\n\t=== Frequency Range Information ===")
    print(f"\tSimulation time  : {args.simtime:.4f} ps")
    print(f"\tTime step        : {args.timestep:.4f} fs")
    print(f"\tMinimum frequency: {f_min / 1e9:.3f} GHz")
    print(f"\tMaximum frequency: {f_max / 1e12:.3f} THz")

#---------------------------------------#
if __name__ == "__main__":
    main()
