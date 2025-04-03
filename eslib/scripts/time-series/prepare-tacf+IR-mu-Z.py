#!/usr/bin/env python
import numpy as np
from eslib.classes.physical_tensor import PhysicalTensor
from eslib.classes.tcf import TimeAutoCorrelation, compute_spectrum, get_freq
from eslib.classes.trajectory import AtomicStructures
from eslib.formatting import esfmt
from eslib.tools import convert
from eslib.io_tools import pattern2sorted_files

#---------------------------------------#
# Description of the script's purpose
description = "Compute the Infrared (IR) absorption spectrum from Born Charges and velocities time series in arbitrary units."
documentation = "This script computes the frequency dependent Beer-Lambert absorption coefficient of IR spectroscopy from the time derivative of the dipole."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"          , **argv, required=True , type=str  , help="pattern to the files with the atomic structures")
    parser.add_argument("-if" , "--input_format"   , **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-z"  , "--bec"            , **argv, required=False , type=str  , help="keyword of the BEC (default: %(default)s)", default="BEC")
    parser.add_argument("-v"  , "--velocities"       , **argv, required=False , type=str  , help="keyword of the velocities (default: %(default)s)", default="velocities")
    parser.add_argument("-vu" , "--velocities_unit"  , **argv, required=False , type=str  , help="unit of the velocities (default: %(default)s)", default="atomic_unit")
    parser.add_argument("-as" , "--axis_samples"   , **argv, required=False, type=int  , help="axis corresponding to independent trajectories/samples (default: %(default)s)", default=0)
    parser.add_argument("-at" , "--axis_time"      , **argv, required=False, type=int  , help="axis corresponding to time (default: %(default)s)", default=1)
    parser.add_argument("-ac" , "--axis_components", **argv, required=False, type=int  , help="axis corresponding to dipole components (default: %(default)s)", default=2)
    parser.add_argument("-o"  , "--output"         , **argv, required=False, type=str  , help="root of the npz output file (default: %(default)s)", default='data')
    parser.add_argument("-pad", "--padding"        , **argv, required=False, type=int  , help="padding length w.r.t. TACF length (default: %(default)s)", default=2)
    return parser
    
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #------------------#    
    print("\tAxes:")
    print("\t - samples:",args.axis_samples)
    print("\t - time:",args.axis_time)
    print("\t - components:",args.axis_components)    
    
    #------------------#
    print("\n\tData will be extracted from the following files:")
    files = pattern2sorted_files(args.input)
    for n,file in enumerate(files):
        print(f"\t - {n}) {file}")
    
    Zx = [None]*len(files)
    Zy = [None]*len(files)
    Zz = [None]*len(files)
    v  = [None]*len(files)
    V  = [None]*len(files)
    
    for n,file in enumerate(files):
        
        # print(f"\t - {n}) {file}")
        print(f"\n\t{n}) Reading the atomic structures from file '{args.input}' ... ", end="")
        structures = AtomicStructures.from_file(file=args.input,format=args.input_format)
        print("done")
        volumes = np.asarray([ atoms.get_volume() for atoms in structures ])
        print("\n\t\tvolume [ang^3]: ",volumes.mean()," +/- ",volumes.std())
        V[n] = convert(volumes.mean(),"volume","angstrom3","atomic_unit")

        #------------------#
        print(f"\n\t\tExtracting the Born Charges using the keyword '{args.bec}' ... ", end="")
        Z = structures.get(args.bec)
        print("done")
        print("\t\tZ.shape: ",Z.shape)
        
        Zx[n] = Z[:,:,0::3] # d mu_x / d R 
        Zy[n] = Z[:,:,1::3] # d mu_y / d R
        Zz[n] = Z[:,:,2::3] # d mu_z / d R
    
        #------------------#
        print(f"\n\t\tExtracting the velocities using the keyword '{args.velocities}' ... ", end="")
        v[n] = structures.get(args.velocities)
        print("done")
        print("\t\tv.shape: ",v[n].shape)
        
        if args.velocities_unit != 'atomic_unit':
            print(f"\n\t\tConvert velocities from '{args.velocities_unit}' to 'atomic_unit'")
            v[n] = convert(v[n],"velocity",args.velocities_unit,'atomic_unit')
            
    print("\tdone")
            
    Zx = np.asarray(Zx)
    Zy = np.asarray(Zy)
    Zz = np.asarray(Zz)
    v  = np.asarray(v)
    
    #------------------#
    print("\n\tComputing the dipole time derivative ... ",end="")
    d_mu_x_dt =  np.einsum("mijk,mijk-> mi",Zx,v)
    d_mu_y_dt =  np.einsum("mijk,mijk-> mi",Zy,v)
    d_mu_z_dt =  np.einsum("mijk,mijk-> mi",Zz,v)
    data = np.zeros((len(files),d_mu_x_dt.shape[1],3))
    data[:,:,0] = d_mu_x_dt
    data[:,:,1] = d_mu_y_dt
    data[:,:,2] = d_mu_z_dt
    print("done")
    print("\tdata shape: ",data.shape)   
    
    print("\n\tHere we have:")
    print(f"\t - {data.shape[args.axis_samples]} samples") 
    print(f"\t - {data.shape[args.axis_time]} time steps") 
    print(f"\t - {data.shape[args.axis_components]} components")   
    
    #------------------#
    V = np.asarray(V)
    print("\n\tSaving data to file '{:s}.npz' ... ".format(args.output), end="")
    np.savez(args.output, data=data, V=V)
    print("done")
    
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()