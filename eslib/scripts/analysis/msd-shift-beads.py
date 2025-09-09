#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List
from uncertainties import unumpy as unp
from uncertainties import ufloat
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt, eslog
from eslib.input import itype, slist
from eslib.plot import add_common_legend
from eslib.fortran import shift_msd_beads
from eslib.mathematics import std_err

#---------------------------------------#
description = "Compute the Mean Squared Displacement (MSD) and the diffusion coefficient from the beads positions."
documentation = "https://en.wikipedia.org/wiki/Mean_squared_displacement"

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i" , "--input"       , **argv, required=False, type=slist, help="input file")
    parser.add_argument("-if", "--input_format", **argv, required=False, type=str  , help="input file format (default: %(default)s)", default=None)
    parser.add_argument("-in", "--index"       , **argv, required=False, type=itype, help="index to be read from input file (default: %(default)s)", default=':')
    parser.add_argument("-dt", "--time_step"   , **argv, required=False, type=float, help="time step [fs] (default: %(default)s)", default=1)
    parser.add_argument("-e" , "--element"     , **argv, required=True , type=str  , help="element")
    parser.add_argument("-o" , "--output"      , **argv, type=str  , help="output file (default: %(default)s)", default="msd.csv")
    parser.add_argument("-p" , "--plot"        , **argv, type=str  , help="plot (default: %(default)s)", default=None)
    return parser 

#---------------------------------------#
@esfmt(prepare_args, description, documentation)
def main(args):
    
    #------------------#
    args.time_step /= 1000 # fs to ps

    #------------------#
    trajectories:List[AtomicStructures] = [None]*len(args.input)
    for n,file in enumerate(args.input):
        with eslog(f"\nReading atomic structures from file '{file}'"):
            trajectories[n] = AtomicStructures.from_file(file=file, format=args.input_format,index=args.index)
        N = len(trajectories[n])
        print(f"\t n. of atomic structures: {N}")
    Nall = [len(t) for t in trajectories]
    assert np.allclose(Nall,N), f"The trajectories have different number of snapshots: {Nall}"
    print(f"\t simulation time: {N*args.time_step}ps")
    
    #------------------#
    positions = np.asarray([ trajectory.get("positions") for trajectory in trajectories])
    print(f"\t positions.shape: {positions.shape}")
    
    #------------------#
    symbols = trajectories[0][0].get_chemical_symbols()
    indices = [ n for n,s in enumerate(symbols) if args.element == s ]
    
    with eslog(f"\nExtracting positions of '{args.element}' atoms"):
        positions = positions[:,:,indices,:]
    print(f"\t positions.shape: {positions.shape}") # (time, atoms, xyz)
    
    #------------------#
    # Mean Squared Displacement (MSD)
    with eslog(f"Computing the Mean Squared Displacement (MSD)"):
        
        M = int(positions.shape[1]/2)
        
        MSD = pd.DataFrame(columns=["time","MSD","MSD-std","MSD-err","D","D-err","MSD/time","MSD/time-std","MSD/time-err"])
        MSD["time"] = np.arange(M)*args.time_step
        
        Natoms = positions.shape[2]
        
        delta_squared = np.zeros((M,Natoms),order="F")
        pos = np.zeros(positions.shape,order="F")
        pos[:,:,:,:] = positions
        shift_msd_beads(delta_squared,pos,msg=False)
        
        # # This is what the Fortran routine called within shift_msd is doing
        # delta_squared_test = np.zeros((M,Natoms)) # (time, atoms)
        # k = 0
        # for i in range(M):
        #     ref = positions[i]
        #     # add extra dim
        #     pos = positions[i:i+M]
        #     delta_squared_test[:,:]  += np.sum(np.square(pos - ref),axis=2) # (time, atoms)
        #     k += 1
        # assert k == M,  "coding error"
        # delta_squared_test /= k # (time, atoms)
        # assert np.allclose(delta_squared,delta_squared_test), "Fortran and python routine do not match"
        
        MSD["MSD"]     = np.mean(delta_squared,axis=1) # (time)
        MSD["MSD-std"],MSD["MSD-err"] = std_err(delta_squared,axis=1)
        
        msd          = unp.uarray(MSD["MSD"], MSD["MSD-err"])
        D            = np.diff(msd)/args.time_step # np.gradient(msd)
        MSD["D"]     = np.concatenate(([0],unp.nominal_values(D)))
        MSD["D-err"] = np.concatenate(([0],unp.std_devs(D))) 
        
        with np.errstate(invalid='ignore'):
            msd_time = msd_time = delta_squared/np.asarray(MSD["time"])[:,np.newaxis]
        MSD["MSD/time"]     = np.mean(msd_time,axis=1)
        MSD["MSD/time-std"] = np.std(msd_time,axis=1)
        MSD["MSD/time-err"] = MSD["MSD/time-std"] / np.sqrt(msd_time.shape[1]-1)
    
    #------------------#
    minT, maxT = min(MSD["time"]), max(MSD["time"])
    print(f"\t MSD has been computed for the time span [{minT},{maxT}]ps")    

    #------------------#
    # Diffusion coefficient
    with eslog(f"\nComputing the diffusion coefficient"):        
        D = ufloat(MSD["MSD/time"].iloc[-1],MSD["MSD/time-err"].iloc[-1])/6.
    
    print( "\t D = {:.6f} +/- {:.6f} [angstrom^2 / n. of atoms / picosecond]".format(unp.nominal_values(D),unp.std_devs(D)))   
    print(f"\t D has been computed for the last snapshot at {maxT} ps")    
    
    with open(args.output, "w") as f:
        f.write(f"# Diffusion coefficient D (computed using the last snapshot at {maxT} ps):\n")
        f.write( "# D-mean = {:.8f} [angstrom^2 / n. of atoms / picosecond]\n".format(unp.nominal_values(D)))
        f.write( "#  D-err = {:.8f} [angstrom^2 / n. of atoms / picosecond]\n".format(unp.std_devs(D)))
        f.write( "# ATTENTION: only the D values above have been divided by 6.\n")
        
    #------------------#    
    # Measure units
    message = """
    \t The columns of the produced dataframe have the following units:
    \t  -          time : picosecond
    \t  -           MSD : angstrom^2 / n. of atoms               | (mean squared displacement)
    \t  -       MSD-std : angstrom^2 / n. of atoms               | (standard deviation of MSD)
    \t  -       MSD-err : angstrom^2 / n. of atoms               | (standard error of MSD)
    \t  -             D : angstrom^2 / n. of atoms / picosecond  | (MSD time derivative)
    \t  -         D-err : angstrom^2 / n. of atoms / picosecond  | (standard error of MSD time derivative)
    \t  -      MSD/time : angstrom^2 / n. of atoms / picosecond  | (MSD/time)
    \t  -  MSD/time-std : angstrom^2 / n. of atoms / picosecond  | (standard deviation of MSD/time)
    \t  -  MSD/time-err : angstrom^2 / n. of atoms / picosecond  | (standard error of MSD/time)"""
    print(message)
    
    #------------------#
    with eslog(f"\nWriting MSD to file '{args.output}'"):
        with open(args.output, "a") as f:
            message = message.replace("\n","\n#")
            f.write(message.strip() + "\n")  # Write the message as a comment
        MSD.to_csv(args.output,index=False,mode='a')
        
    #------------------#
    # plot
    if args.plot is not None:        
        with eslog(f"\nPreparing plot"):
            fig,ax = plt.subplots(figsize=(6,4))
            ax.plot(MSD["time"],MSD["MSD"],color="blue",label="MSD")
            ax.fill_between(MSD["time"],MSD["MSD"]-MSD["MSD-err"],MSD["MSD"]+MSD["MSD-err"],alpha=0.5,color="blue",lw=0)
            ax.set_ylabel(r"MSD [ $\AA^2$ / n. atoms ]")
            ax.set_ylim(0,None)
            
            ax.set_xlabel("time [ps]")
            ax.set_xlim(min(MSD["time"]),max(MSD["time"]))
            
            ax2 = ax.twinx()            
            ax2.plot(MSD["time"],MSD["MSD/time"],color="green",label=r"MSD/t")
            ax2.fill_between(MSD["time"],MSD["MSD/time"]-MSD["MSD/time-err"],MSD["MSD/time"]+MSD["MSD/time-err"],alpha=0.5,color="green",lw=0)
            ax2.plot(maxT,6*unp.nominal_values(D),"ro")
            
            ax2.set_ylabel(r"$\partial$MSD/$\partial$t [ $\AA^2$ / n. atoms / ps ]")
            ax2.set_ylim(0,None)
            
            add_common_legend(fig,ax,ax2,loc="upper left")
            
            # ax.grid()
            plt.tight_layout()    
            # plt.show()
                
        with eslog(f"\nSaving plot to file '{args.plot}'"):
            plt.savefig(args.plot,bbox_inches="tight",dpi=300)
    
    return 0

#---------------------------------------#
if __name__ == "__main__":
    main()

