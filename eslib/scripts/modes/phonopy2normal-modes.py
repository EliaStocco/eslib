#!/usr/bin/env python
from eslib.classes.normal_modes import NormalModes
from eslib.formatting import matrix2str
from eslib.tools import convert
from eslib.output import output_folder
from eslib.input import size_type
from eslib.functions import phonopy2atoms
import argparse
import numpy as np
import yaml
import pandas as pd
from icecream import ic
import os
# import warnings
# warnings.filterwarnings("error")
#---------------------------------------#
THRESHOLD = 1e-4
#---------------------------------------#
# Description of the script's purpose
description = "Prepare the necessary file to project a MD trajectory onto phonon modes: read results from phonopy."
warning = "***Warning***"
closure = "Job done :)"
keywords = "It's up to you to modify the required keywords."
input_arguments = "Input arguments"
#---------------------------------------#
# colors
try :
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    description     = Fore.GREEN  + Style.BRIGHT + description             + Style.RESET_ALL
    warning         = Fore.MAGENTA    + Style.BRIGHT + warning.replace("*","") + Style.RESET_ALL
    closure         = Fore.BLUE   + Style.BRIGHT + closure                 + Style.RESET_ALL
    keywords        = Fore.YELLOW + Style.NORMAL + keywords                + Style.RESET_ALL
    input_arguments = Fore.GREEN  + Style.NORMAL + input_arguments         + Style.RESET_ALL
except:
    pass
#---------------------------------------#
def prepare_args():
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-q",  "--qpoints",       type=str, **argv, 
                        help="qpoints file (default: 'qpoints.yaml')", default="qpoints.yaml")
    parser.add_argument("-i",  "--input",         type=str, **argv, 
                        help="general phonopy file (default: 'phonopy.yaml')", default="phonopy.yaml")
    parser.add_argument("-o",  "--output",        type=str, **argv, 
                        help="output file (default: 'phonon-modes.pickle')", default="phonon-modes.pickle")
    parser.add_argument("-of", "--output_folder", type=str, **argv, 
                        help="output folder for *.mode, *.eigvec and *.eigval files (default: None)", default=None)
    parser.add_argument("-f", "--factor", type=float, **argv, 
                        help="conversion factor to THz for the frequencies ω", default=None)
    parser.add_argument("-m",  "--matrices",      type=lambda x: size_type(x,str), **argv, 
                        help="matrices/vectors to print (default: ['eigval','eigvec','mode'])", default=['eigval','eigvec','mode'])
    return parser.parse_args()
#---------------------------------------#
def main():

    #---------------------------------------#
    # print the script's description
    print("\n\t{:s}".format(description))

    #---------------------------------------#
    # parse the user input
    args = prepare_args()
    print("\n\t{:s}:".format(input_arguments))
    for k in args.__dict__.keys():
        print("\t{:>20s}:".format(k),getattr(args,k))
    print()

    #---------------------------------------#
    # read input file ('phonopy.yaml')
    print("\tReading data from input file '{:s}' ... ".format(args.input), end="")
    with open(args.input) as f:
        info = yaml.safe_load(f)
    print("done")

    print("\t{:<10s} : ".format("dim"),info["phonopy"]["configuration"]["dim"])
    print("\t{:<10s} : ".format("qpoints"),info["phonopy"]["configuration"]["qpoints"].split(" "))
    print("\t{:<10s} : ".format("masses"),np.asarray([ a["mass"] for a in info["unit_cell"]["points"] ]).round(2))   

    size = np.asarray([ int(a) for a in info["phonopy"]["configuration"]["dim"].split(" ") ])
    factor = convert(1,"mass","dalton","atomic_unit")
    mass = factor * np.asarray([ [a["mass"]]*3 for a in info["unit_cell"]["points"] ]).flatten()

    #---------------------------------------#
    # read input file ('phonopy.yaml')
    print("\tExtracting reference atomic structure ... ", end="")
    reference = phonopy2atoms(info["supercell"])
    print("done")

    #---------------------------------------#
    # read qpoints file ('qpoints.yaml')
    print("\n\tReading qpoints from input file '{:s}' ... ".format(args.qpoints), end="")
    with open(args.qpoints) as f:
        qpoints = yaml.safe_load(f)
    print("done")
    
    print("\t{:<20s} : {:<10d}".format("# q points",qpoints["nqpoint"]))
    print("\t{:<20s} : {:<10d}".format("# atoms",qpoints["natom"]))
    print("\t{:<20s} :".format("reciprocal vectors"))
    tmp = np.asarray(qpoints["reciprocal_lattice"]).T
    line = matrix2str(tmp,digits=6,exp=False,width=12)
    print(line)

    #---------------------------------------#
    if args.factor is None:
        print("\n\tNo conversion factor to THz for the frequencies provided, using the dafault one.")
        args.factor = np.sqrt( convert(1,"energy","rydberg","atomic_unit") / convert(1,"mass","dalton","atomic_unit") ) * convert(1,"frequency","atomic_unit","terahertz")

    print("\tConversion factor to THz for the frequencies: {:<20.6e}".format(args.factor))
    print("\tAll the quantities will then be saved in Hartree atomic units.")
    THz2au  = convert(1,"frequency","terahertz","atomic_unit")
    THz2au2 = THz2au**2

    #---------------------------------------#
    # supercell variables
    Nmodes = None
    Ndof   = None   
    Ndofcell = None

    #---------------------------------------#
    # phonon modes
    print("\n\tReading phonon modes and related quantities ... ")
    index = [ tuple(a["q-position"]) for a in qpoints["phonon"] ]
    pm = pd.DataFrame(index=index,columns=["q","freq","eigval","cell","supercell"])
    factor = convert(1,"frequency","inversecm","atomic_unit")
    for n,phonon in enumerate(qpoints["phonon"]):
        q = tuple(phonon["q-position"])
        print("\t\tphonons {:d}: q point".format(n),q)
        N = len(phonon["band"])
        nm = NormalModes(N,N)
        nm.set_dynmat(phonon["dynamical_matrix"],mode="phonopy")
        nm.set_eigvec(phonon["band"],mode="phonopy")
        nm.masses = mass

        pm.at[q,"q"]     = tuple(q)
        pm.at[q,"freq"]  = [ a["frequency"] for a in phonon["band"] ]

        nm.set_eigval( ( factor*np.asarray(pm.at[q,"freq"]))**2 )
        
        snm = nm.build_supercell_displacement(size=size,q=q)
        snm.reference = reference
        
        w,f = np.linalg.eigh(nm.dynmat)
        freq = np.sqrt(np.abs(w)) * np.sign(w) * args.factor

        if np.square(freq - np.asarray(pm.at[q,"freq"]) ).sum() > THRESHOLD:
            raise ValueError("The frequencies computed by this script and the ones provided by phonopy do not match.")
        
        pm.at[q,"freq"]   = freq * THz2au
        pm.at[q,"eigval"] = w * THz2au2

        nm.set_eigval( w )
        nm.dynmat *= THz2au2

        pm.at[q,"cell"]      = nm
        pm.at[q,"supercell"] = snm

        if Ndofcell is None:
            Ndofcell = nm.Ndof
        if Nmodes is None:
            Nmodes = snm.Nmodes
        if Ndof is None:
            Ndof = snm.Ndof

    #---------------------------------------#
    print("\n\tCell summary:")
    print("\t\t{:>10}: {:d}".format("# modes",Nmodes))
    print("\t\t{:>10}: {:d}".format("# dof",Ndofcell))

    #---------------------------------------#
    print("\n\tSupercell summary:")
    print("\t{:>10}: {:d}".format("# modes",Nmodes))
    print("\t{:>10}: {:d}".format("# dof",Ndof))

    SHAPE = {
        "eigval" : (Nmodes,),
        "eigvec" : (Ndof,Nmodes),
        "mode"   : (Ndof,Nmodes),
        "dynmat" : (Ndof,Ndof),
    }

    #---------------------------------------#
    print("\n\tWriting phonon modes to file '{:s}' ... ".format(args.output), end="")
    pm.to_pickle(args.output)
    print("done")

    if args.output_folder is not None:
        print("\n\tWriting phonon modes to i-PI-like files in folder '{:s}':".format(args.output_folder))

        print("\tThe following quantities will be saved (and they should have the following shapes):")
        for name in args.matrices:
            print("\t\t- {:>8s} --> shape {:s}".format(name,str(SHAPE[name])))

        # create directory
        output_folder(args.output_folder)
        k = 0 
        for n,row in pm.iterrows():
            # ic(row)
            q = str(row["q"]).replace(" ","")
            print("\n\tphonons {:d}, with q point {:s}:".format(k,q))
            for name in args.matrices:
                matrix = getattr(row["supercell"],name)
                if np.any(np.isnan(matrix)):
                    print("\t\t{:s}: {:s} matrix/vector containes np.nan values: it will not be saved to file".format(warning,name))
                    continue
                file = os.path.normpath("{:s}/{:s}.{:s}.{:s}".format(args.output_folder,"phonopy",q,name))
                print("\t\t- saving {:>8s} to file '{:s}' ... ".format(name,file),end="")
                with open(file,"w") as f:
                    f.write("# q-point: {:s}, matrix: {:s}, shape: {:s}".format(q,name,str(matrix.shape)))
                    np.savetxt(file,matrix)
                # print(str(matrix.shape),end="")
                print("done")
                
            k += 1
    
    # Script completion message
    print("\n\t{:s}\n".format(closure))

if __name__ == "__main__":
    main()
