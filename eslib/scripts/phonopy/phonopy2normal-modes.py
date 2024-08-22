#!/usr/bin/env python
from sqlite3 import converters
from eslib.classes.normal_modes import NormalModes
from eslib.show import matrix2str
from eslib.tools import convert
from eslib.output import output_folder
from eslib.input import size_type
from eslib.functions import phonopy2atoms
import numpy as np
import yaml
import pandas as pd
import os
from eslib.formatting import esfmt, warning
from classes.atomic_structures import AtomicStructures
from eslib.tools import is_sorted_ascending, w2_to_w
from phonopy.units import VaspToTHz
# import ruamel.yaml

# import h5py

# def inspect_hdf5(filename):
#     with h5py.File(filename, 'r') as f:
#         # Function to recursively print the structure of the HDF5 file
#         def print_structure(name, obj):
#             if isinstance(obj, h5py.Dataset):
#                 print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
#             elif isinstance(obj, h5py.Group):
#                 print(f"Group: {name}")
        
#         # Print the structure of the file
#         print("HDF5 File Structure:")
#         f.visititems(print_structure)
        
#         print("\nDetailed Dataset Information:")
#         # Iterate through all items and print details
#         for name, obj in f.items():
#             if isinstance(obj, h5py.Dataset):
#                 print(f"\nDataset: {name}")
#                 print(f"Shape: {obj.shape}")
#                 print(f"Data type: {obj.dtype}")
#                 if obj.attrs:
#                     print(f"Attributes:")
#                     for attr_name, attr_value in obj.attrs.items():
#                         print(f"    {attr_name}: {attr_value}")
#                 # Optional: Print the actual data (caution with large datasets)
#                 # print(f"Data:\n{obj[:]}")
#             elif isinstance(obj, h5py.Group):
#                 print(f"\nGroup: {name}")
#                 for sub_name, sub_obj in obj.items():
#                     if isinstance(sub_obj, h5py.Dataset):
#                         print(f"  Sub-Dataset: {sub_name}")
#                         print(f"  Shape: {sub_obj.shape}")
#                         print(f"  Data type: {sub_obj.dtype}")
#                         if sub_obj.attrs:
#                             print(f"  Attributes:")
#                             for attr_name, attr_value in sub_obj.attrs.items():
#                                 print(f"      {attr_name}: {attr_value}")
#                         # Optional: Print the actual data (caution with large datasets)
#                         # print(f"  Data:\n{sub_obj[:]}")
#                     elif isinstance(sub_obj, h5py.Group):
#                         print(f"  Sub-Group: {sub_name}")
#                         # You can add more detailed inspection here if needed


# AMU_to_electron_mass = 1822.888486
# angstrom_to_Bohr = 1.889726124565062
# eV_to_Hartree = 0.0367493081366
# force_constant_conversion = eV_to_Hartree / (angstrom_to_Bohr**2)


#---------------------------------------#
THRESHOLD = 1e-4
#---------------------------------------#
# Description of the script's purpose
description = "Prepare the necessary file to project a MD trajectory onto phonon modes: read results from phonopy."

def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-r",  "--reference",  required=True,     type=str, **argv, help="reference structure file [a.u.] (default: %(default)s)",default=None)
    parser.add_argument("-rf" , "--reference_format"  , **argv,required=False, type=str     , help="reference file format (default: %(default)s)" , default=None)
    parser.add_argument("-q",  "--qpoints",       type=str, **argv, 
                        help="qpoints file (default: %(default)s)", default="qpoints.yaml")
    parser.add_argument("-i",  "--input",         type=str, **argv, 
                        help="general phonopy file (default: %(default)s)", default="phonopy.yaml")
    parser.add_argument("-f", "--factor", type=float, **argv, 
                        help="conversion factor to THz for the frequencies ω", default=VaspToTHz)
    parser.add_argument("-o",  "--output",        type=str, **argv, 
                        help="output prefix file (default: %(default)s)", default="phonons")
    parser.add_argument("-of", "--output_folder", type=str, **argv, 
                        help="output folder (default: %(default)s)", default="phonons")
    
    # parser.add_argument("-m",  "--matrices",      type=lambda x: size_type(x,str), **argv, 
    #                     help="matrices/vectors to print (default: %(default)s)", default=['eigval','eigvec','mode'])
    return parser# .parse_args()
#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    print("\tReading reference atomic structure from input '{:s}' ... ".format(args.reference), end="")
    reference = AtomicStructures.from_file(file=args.reference,format=args.reference_format,index=0)[0]
    print("done")
    print("\tVolume: {:f} A^3".format(reference.get_volume()))

    
    #---------------------------------------#
    # read input file ('phonopy.yaml')
    print("\n\tReading data from input file '{:s}' ... ".format(args.input), end="")
    with open(args.input) as f:
        info = yaml.safe_load(f)
    # yaml = ruamel.yaml.YAML(typ='safe', pure=True)
    # with open(args.input) as f:
    #     # info = yaml.safe_load(f)
    #     info = yaml.load(f)
    # print("done")

    print("\t{:<10s} : ".format("dim"),info["phonopy"]["configuration"]["dim"])
    print("\t{:<10s} : ".format("qpoints"),info["phonopy"]["configuration"]["qpoints"].split(" "))
    print("\t{:<10s} : ".format("masses"),np.asarray([ a["mass"] for a in info["unit_cell"]["points"] ]).round(2))   

    size = np.asarray([ int(a) for a in info["phonopy"]["configuration"]["dim"].split(" ") ])
    factor = convert(1,"mass","dalton","atomic_unit")
    # mass = factor * np.asarray([ [a["mass"]]*3 for a in info["unit_cell"]["points"] ]).flatten()

    # #---------------------------------------#
    # # read input file ('phonopy.yaml')
    # print("\n\tExtracting reference atomic structure ... ", end="")
    # reference = phonopy2atoms(info["supercell"])
    # print("done")
    # print("\tVolume: {:f} A^3".format(reference.get_volume()))

    print("\n\tConverting reference atomic structure from angstrom to bohr... ", end="")
    reference.positions *= convert(1,"length","angstrom","atomic_unit")
    reference.cell *= convert(1,"length","angstrom","atomic_unit")
    print("done")
    print("\tVolume: {:f} bohr^3".format(reference.get_volume()))

    #---------------------------------------#
    # read qpoints file ('qpoints.yaml')
    print("\n\tReading qpoints from input file '{:s}' ... ".format(args.qpoints), end="")
    with open(args.qpoints) as f:
        qpoints = yaml.safe_load(f)
    print("done")

    # inspect_hdf5("qpoints.hdf5")

    print("\t\t{:<20s} :".format("contained keys"),list(qpoints.keys()))
    print("\t\t{:<20s} : {:<10d}".format("n. of q points",qpoints["nqpoint"]))
    print("\t\t{:<20s} : {:<10d}".format("n. of atoms",qpoints["natom"]))
    print("\t\t{:<20s} :".format("reciprocal vectors"))
    tmp = np.asarray(qpoints["reciprocal_lattice"]).T
    line = matrix2str(tmp,digits=6,exp=False,width=12)
    print(line)

    #---------------------------------------#
    if args.factor is None:
        print("\n\tNo conversion factor to THz for the frequencies provided, using the dafault one.")
        args.factor = np.sqrt( convert(1,"energy","rydberg","atomic_unit") / convert(1,"mass","dalton","atomic_unit") ) * convert(1,"frequency","atomic_unit","terahertz")

    print("\tConversion factor to THz for the frequencies: {:<20.6e}".format(args.factor))
    print("\tAll quantities will be saved in Hartree atomic units.")
    THz2au  = convert(1,"frequency","terahertz","atomic_unit")
    THz2au2 = THz2au**2

    # dynmat_factor = convert(1,"energy","electronvolt","atomic_unit")/ (convert(1,"length","angstrom","atomic_unit"))**2
    # dynmat_factor = force_constant_conversion * AMU_to_electron_mass

    #---------------------------------------#
    # supercell variables
    Nmodes = None
    Ndof   = None   
    Ndofcell = None

    os.makedirs(args.output_folder, exist_ok=True)

    #---------------------------------------#
    # phonon modes
    print("\n\tReading phonon modes and related quantities ... ")
    index = [ tuple(a["q-position"]) for a in qpoints["phonon"] ]
    pm = pd.DataFrame(index=index,columns=["q","freq","eigval","cell","supercell"])
    # factor = convert(1,"frequency","inversecm","atomic_unit")
    freq_factor =  convert(1,family="frequency",_from="terahertz",_to="atomic_unit")
    dynmat_factor = args.factor**2 * freq_factor**2
    for n,phonon in enumerate(qpoints["phonon"]):
        q = tuple(phonon["q-position"])
        assert len(q) == 3, "q point must be a of lenght 3"
        print("\t\tphonons {:d}: q point".format(n),q)
        N = len(phonon["band"])
        nm = NormalModes(Nmodes=N,Ndof=N,ref=reference)

        # test = convert(1,"energy","electronvolt","atomic_unit")/(convert(1,"length","angstrom","atomic_unit"))**2/convert(1,"mass","dalton","atomic_unit")
        
        dynmat = np.asarray(phonon["dynamical_matrix"]) * dynmat_factor
        nm.set_dynmat(dynmat,mode="phonopy")
        nm.set_eigvec(phonon["band"],mode="phonopy")
        # nm.masses = mass

        pm.at[q,"q"]     = tuple(q)
        pm.at[q,"freq"]  = [ a["frequency"]*freq_factor  for a in phonon["band"] ] # frequencies are in THz

        eigval = np.square(pm.at[q,"freq"]) * np.sign(pm.at[q,"freq"])
        nm.set_eigval( eigval )
        nm.check()
        nm.eigvec2modes()

        tmp = "{:d}-{:d}-{:d}".format(*[ int(i) for i in q])
        file = os.path.normpath("{:s}/{:s}.{:s}.pickle".format(args.output_folder,args.output,tmp))

        print("\t\tsaving normal modes for q point {:} to file {:s}".format(q,file))
        nm.to_file(file=file)

    #     try:        
    #         snm = nm.build_supercell_displacement(size=size,q=q)
    #         snm.reference = reference
            
    #         w,f = np.linalg.eigh(nm.dynmat)
    #         freq = np.sqrt(np.abs(w)) * np.sign(w) * args.factor

    #         if np.square(freq - np.asarray(pm.at[q,"freq"]) ).sum() > THRESHOLD:
    #             raise ValueError("The frequencies computed by this script and the ones provided by phonopy do not match.")
            
    #         pm.at[q,"freq"]   = freq * THz2au
    #         pm.at[q,"eigval"] = w * THz2au2

    #         nm.set_eigval( w )
    #         nm.dynmat *= THz2au2

    #         pm.at[q,"cell"]      = nm
    #         pm.at[q,"supercell"] = snm

    #         if Ndofcell is None:
    #             Ndofcell = nm.Ndof
    #         if Nmodes is None:
    #             Nmodes = snm.Nmodes
    #         if Ndof is None:
    #             Ndof = snm.Ndof
    #     except:
    #         pass

    # #---------------------------------------#
    # print("\n\tCell summary:")
    # print("\t\t{:>10}: {:d}".format("# modes",Nmodes))
    # print("\t\t{:>10}: {:d}".format("# dof",Ndofcell))

    # #---------------------------------------#
    # print("\n\tSupercell summary:")
    # print("\t{:>10}: {:d}".format("# modes",Nmodes))
    # print("\t{:>10}: {:d}".format("# dof",Ndof))

    # SHAPE = {
    #     "eigval" : (Nmodes,),
    #     "eigvec" : (Ndof,Nmodes),
    #     "mode"   : (Ndof,Nmodes),
    #     "dynmat" : (Ndof,Ndof),
    # }

    # #---------------------------------------#
    # print("\n\tWriting phonon modes to file '{:s}' ... ".format(args.output), end="")
    # pm.to_pickle(args.output)
    # print("done")

    # if args.output_folder is not None:
    #     print("\n\tWriting phonon modes to i-PI-like files in folder '{:s}':".format(args.output_folder))

    #     print("\tThe following quantities will be saved (and they should have the following shapes):")
    #     for name in args.matrices:
    #         print("\t\t- {:>8s} --> shape {:s}".format(name,str(SHAPE[name])))

    #     # create directory
    #     output_folder(args.output_folder)
    #     k = 0 
    #     for n,row in pm.iterrows():
    #         # ic(row)
    #         q = str(row["q"]).replace(" ","")
    #         print("\n\tphonons {:d}, with q point {:s}:".format(k,q))
    #         for name in args.matrices:
    #             matrix = getattr(row["supercell"],name)
    #             if np.any(np.isnan(matrix)):
    #                 print("\t\t{:s}: {:s} matrix/vector containes np.nan values: it will not be saved to file".format(warning,name))
    #                 continue
    #             file = os.path.normpath("{:s}/{:s}.{:s}.{:s}".format(args.output_folder,"phonopy",q,name))
    #             print("\t\t- saving {:>8s} to file '{:s}' ... ".format(name,file),end="")
    #             with open(file,"w") as f:
    #                 f.write("# q-point: {:s}, matrix: {:s}, shape: {:s}".format(q,name,str(matrix.shape)))
    #                 np.savetxt(file,matrix)
    #             # print(str(matrix.shape),end="")
    #             print("done")
                
    #         k += 1
    
#---------------------------------------#
if __name__ == "__main__":
    main()
