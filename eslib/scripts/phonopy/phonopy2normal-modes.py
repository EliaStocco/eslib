#!/usr/bin/env python
from eslib.classes.normal_modes import NormalModes
from eslib.show import matrix2str
from eslib.tools import convert
import numpy as np
import yaml
import pandas as pd
import os
from eslib.formatting import esfmt, warning
from phonopy.units import VaspToTHz
from ase import Atoms

#---------------------------------------#
THRESHOLD = 1e-4
CHECK = False
#---------------------------------------#
# Description of the script's purpose
description = "Prepare the necessary file to project a MD trajectory onto phonon modes: read results from phonopy."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-q" , "--qpoints"         , **argv, required=False, type=str  , help="qpoints file (default: %(default)s)", default="qpoints.yaml")
    parser.add_argument("-i" , "--input"           , **argv, required=False, type=str  , help="general phonopy file (default: %(default)s)", default="phonopy.yaml")
    parser.add_argument("-f" , "--factor"          , **argv, required=False, type=float, help="conversion factor to THz for the frequencies ω (default: %(default)s)", default=VaspToTHz)
    parser.add_argument("-o" , "--output"          , **argv, required=False, type=str  , help="output prefix file (default: %(default)s)", default="phonons")
    parser.add_argument("-of", "--output_folder"   , **argv, required=False, type=str  , help="output folder (default: %(default)s)", default="phonons")
    return parser
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    #---------------------------------------#
    print("\n\tReading data from input file '{:s}' ... ".format(args.input), end="")
    with open(args.input) as f:
        info = yaml.safe_load(f)
    print("done")

    print("\t{:<10s} : ".format("dim"),info["phonopy"]["configuration"]["dim"])
    print("\t{:<10s} : ".format("qpoints"),info["phonopy"]["configuration"]["qpoints"])
    print("\t{:<10s} : ".format("masses"),np.asarray([ a["mass"] for a in info["unit_cell"]["points"] ]).round(2))   

    size = np.asarray([ int(a) for a in info["phonopy"]["configuration"]["dim"].split(" ") ])
    
    cell = np.asarray(info['unit_cell']['lattice'])
    sc = np.asarray([a['coordinates']  for a in info['unit_cell']['points']])
    symbols = np.asarray([a['symbol']  for a in info['unit_cell']['points']])
    reference = Atoms(cell=cell,scaled_positions=sc,pbc=True,symbols=symbols)
    
    print("\n\tPrimitive unit cell (angstrom): ")
    print("\t - n. of atoms: ",reference.get_global_number_of_atoms())    
    print("\t - volume: {:f} ang^3".format(reference.get_volume()))
    print("\t - cell (ang):")    
    line = matrix2str(reference.cell.array.T,col_names=["1","2","3"],cols_align="^",width=10,digits=4)
    print(line)
    
    print("\n\tConverting reference atomic structure from angstrom to bohr... ", end="")
    reference_au = reference.copy()
    reference_au.positions *= convert(1,"length","angstrom","atomic_unit")
    reference_au.cell *= convert(1,"length","angstrom","atomic_unit")
    print("done")
    
    print("\n\tPrimitive unit cell (bohr): ")
    print("\t - n. of atoms: ",reference_au.get_global_number_of_atoms())    
    print("\t - volume: {:f} bohr^3".format(reference_au.get_volume()))
    print("\t - cell (bohr):")    
    line = matrix2str(reference_au.cell.array.T,col_names=["1","2","3"],cols_align="^",width=10,digits=4)
    print(line)
    del reference
    
    #---------------------------------------#
    print("\n\tReading qpoints from input file '{:s}' ... ".format(args.qpoints), end="")
    with open(args.qpoints) as f:
        qpoints:dict = yaml.safe_load(f)
    print("done")

    print("\t\t{:<20s} :".format("contained keys"),list(qpoints.keys()))
    print("\t\t{:<20s} : {:<10d}".format("n. of q points",qpoints["nqpoint"]))
    print("\t\t{:<20s} : {:<10d}".format("n. of atoms",qpoints["natom"]))
    # print("\t\t{:<20s} :".format("reciprocal vectors"))
    # tmp = np.asarray(qpoints["reciprocal_lattice"]).T
    # line = matrix2str(tmp,digits=6,exp=False,width=12)
    print(line)

    #---------------------------------------#
    if args.factor is None:
        print("\n\tNo conversion factor to THz for the frequencies provided, using the dafault one.")
        args.factor = np.sqrt( convert(1,"energy","rydberg","atomic_unit") / convert(1,"mass","dalton","atomic_unit") ) * convert(1,"frequency","atomic_unit","terahertz")

    print("\tConversion factor to THz for the frequencies: {:<20.6f}".format(args.factor))
    print("\tRatio between number of atoms in the supercell and the primitive cell: {:d}".format(size.prod()))
    print("\t{:s}: all quantities will be saved in Hartree atomic units.".format(warning))

    #---------------------------------------#
    os.makedirs(args.output_folder, exist_ok=True)

    #---------------------------------------#
    # phonon modes
    print("\n\tUnfolding the phonon modes into the supercell ... ")
    index = [ tuple(a["q-position"]) for a in qpoints["phonon"] ]
    pm = pd.DataFrame(index=index,columns=["q","freq","eigval","cell","supercell"])
    freq_factor =  convert(1,family="frequency",_from="terahertz",_to="atomic_unit")
    dynmat_factor = args.factor**2 * freq_factor**2
    tot = len(qpoints["phonon"])
    for n,phonon in enumerate(qpoints["phonon"]):
        q = tuple(phonon["q-position"])
        assert len(q) == 3, "q point must be a of lenght 3"
        print("\t - phonons {:d}/{:d}: q point".format(n+1,tot),q)
        N = len(phonon["band"])
        nm = NormalModes(Nmodes=N,Ndof=N,ref=reference_au)
        
        dynmat = np.asarray(phonon["dynamical_matrix"]) * dynmat_factor
        nm.set_dynmat(dynmat,mode="phonopy")
        nm.set_eigvec(phonon["band"],mode="phonopy")

        pm.at[q,"q"]     = tuple(q)
        pm.at[q,"freq"]  = [ a["frequency"]*freq_factor  for a in phonon["band"] ] # frequencies are in THz

        eigval = np.square(pm.at[q,"freq"]) * np.sign(pm.at[q,"freq"])
        nm.set_eigval( eigval )
        if CHECK:
            nm.check()
        nm.eigvec2modes()

        qstr = np.asarray(q)*size
        tmp = "{:d}-{:d}-{:d}".format(*[ int(i) for i in qstr])
        file = os.path.normpath("{:s}/{:s}-unitcell.{:s}.pickle".format(args.output_folder,args.output,tmp))

        print("\t    * saving normal modes (unit cell) for q point {:} to file {:s}".format(q,file))
        nm.to_file(file=file)
        
        file = os.path.normpath("{:s}/{:s}-supercell.{:s}.pickle".format(args.output_folder,args.output,tmp))
        print("\t    * saving normal modes (supercell) for q point {:} to file {:s}".format(q,file))
        snm = nm.build_supercell_displacement(size=size,q=q,info=info)
        snm.to_file(file=file)
    
    sc_au = snm.reference
    sc_ang = sc_au.copy()
    sc_ang.positions *= convert(1,"length","atomic_unit","angstrom")
    sc_ang.cell *= convert(1,"length","atomic_unit","angstrom")

    print("\n\tSupercell cell (angstrom): ")    
    print("\t - n. of atoms: ",sc_ang.get_global_number_of_atoms())    
    print("\t - volume: {:f} ang^3".format(sc_ang.get_volume()))
    print("\t - cell (ang):")    
    line = matrix2str(sc_ang.cell.array.T,col_names=["1","2","3"],cols_align="^",width=10,digits=4)
    print(line)
    
    
    print("\n\tSupercell cell (bohr): ")
    print("\t - n. of atoms: ",sc_au.get_global_number_of_atoms())    
    print("\t - volume: {:f} bohr^3".format(sc_au.get_volume()))
    print("\t - cell (bohr):")    
    line = matrix2str(sc_au.cell.array.T,col_names=["1","2","3"],cols_align="^",width=10,digits=4)
    print(line)
        
    
#---------------------------------------#
if __name__ == "__main__":
    main()
