#!/usr/bin/env python
import numpy as np
import xml.etree.ElementTree as xmlet
from eslib.classes.properties import Properties
from eslib.formatting import esfmt
from eslib.input import flist
from eslib.tools import convert
from eslib.classes.efield import ElectricField

#---------------------------------------#
description = "Compute the actual conserved quantity of the system when an electric field is applied."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    # Input
    parser.add_argument("-i" , "--input"          , **argv, type=str  , required=True , help="input file")
    parser.add_argument("-xml" , "--xml"          , **argv, type=str  , required=True , help="xml i-PI input file (default: %(default)s)", default=None)
    # Keywords
    parser.add_argument("-c" , "--conserved"      , **argv, type=str  , required=False, help="`conserved` keyword (default: %(default)s)", default="conserved")
    parser.add_argument("-d" , "--dipole"         , **argv, type=str  , required=False, help="`dipole` keyword (default: %(default)s)", default="dipole")
    parser.add_argument("-e" , "--efield"         , **argv, type=str  , required=False, help="`Efield` keyword (default: %(default)s)", default="Efield")
    parser.add_argument("-ec", "--efield_constant", **argv, type=flist, required=False, help="Efield value when constant(default: %(default)s)", default=None)
    # Units
    parser.add_argument("-cu" , "--conserved_unit", **argv, type=str  , required=False, help="`conserved` unit (default: %(default)s)", default="atomic_unit")
    parser.add_argument("-du" , "--dipole_unit"   , **argv, type=str  , required=False, help="`dipole` unit (default: %(default)s)", default="atomic_unit")
    parser.add_argument("-eu" , "--efield_unit"   , **argv, type=str  , required=False, help="`Efield` unit (default: %(default)s)", default="atomic_unit")
    # Time step
    parser.add_argument("-dt" , "--time_step"     , **argv, type=float  , required=False, help="time step [fs](default: %(default)s)", default=1)
    # Output
    parser.add_argument("-on", "--output_name"    , **argv, type=str  , required=False, help="output `Econserved` keyword (default: %(default)s)", default="Econserved")
    parser.add_argument("-o" , "--output"         , **argv, type=str  , required=True , help="output file")
    return parser

def extract(keys,families,scope):

    data = {}
    for key,family in zip(keys,families):

        data[key] = None
        
        element = scope.find(key)

        if element is not None:
            #value = ast.literal_eval(element.text)
            text =  element.text
            try :
                value = text.split('[')[1].split(']')[0].split(',')
                value = [ float(i) for i in value ]
                if len(value) == 1:
                    value = float(value)
                else :
                    value = np.asarray(value)
            except :
                value = float(text)
            
            try :
                unit = element.attrib["units"]
                if unit is None :
                    unit = "atomic_unit"
            except:
                unit = "atomic_unit"

            # print(key,value,unit)

            value = convert(value,family,unit,"atomic_unit")
            data[key] = value

    return data

def get_Efield(file)->ElectricField:

    data = xmlet.parse(file).getroot()
    efield = None
    for element in data.iter():
        if element.tag == "efield":
            efield = element

    data     = {}
    keys     = [ "amp"           , "freq"     , "phase"    , "peak", "sigma" ]
    families = [ "electric-field", "frequency", "undefined", "time", "time"  ]
    
    data = extract(keys,families,efield)

    return ElectricField( amp=data["amp"],\
                        phase=data["phase"],\
                        freq=data["freq"],\
                        peak=data["peak"],\
                        sigma=data["sigma"])

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #------------------#
    # atomic structures
    print("\tReading properties from file '{:s}' ... ".format(args.input), end="")
    properties = Properties.from_file(file=args.input)
    print("done\n")
    print("\tn. of snapshots: {:d}".format(len(properties)))

    #------------------#
    # summary
    print("\n\tSummary of the properties: ")
    df = properties.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))
    print()

    #------------------#
    # Extraction/Construction
    print("\tExtracting 'conserved' using keyword '{:s}' ... ".format(args.conserved), end="")
    conserved = properties.get(args.conserved)
    print("done")

    print("\tExtracting 'dipole' using keyword '{:s}' ... ".format(args.dipole), end="")
    dipole = properties.get(args.dipole)
    print("done")

    if args.efield_constant is None:
        print("\tExtracting 'Efield' using keyword '{:s}' ... ".format(args.efield), end="")
        efield = properties.get(args.efield)
        print("done")
    else:
        print("\tConstructing 'Efield' ... ", end="")
        efield = np.asarray(args.efield_constant)
        assert efield.shape == (3,), "efield shape must be (3,)"
        efield = np.tile(efield, (len(properties), 1))
        print("done")

    print()
    print("\t'conserved'.shape: ",conserved.shape)
    print("\t   'dipole'.shape: ",dipole.shape)
    print("\t   'Efield'.shape: ",efield.shape)

    assert dipole.shape == efield.shape, "dipole and efield must have the same shape"
    assert len(conserved) == dipole.shape[0], "conserved and dipole must have the same length"
    assert conserved.ndim == 1, "conserved must be a 1D array"

    #------------------#
    # Conversion
    print()
    if args.conserved_unit not in ["","au","atomic_unit"]:
        print("\tConverting 'conserved' to 'atomic_unit' ... ", end="")
        conserved = convert(conserved,"energy",args.conserved_unit,"atomic_unit")
        print("done")

    if args.dipole_unit not in ["","au","atomic_unit"]:
        print("\tConverting 'dipole' to 'atomic_unit' ... ", end="")
        dipole = convert(dipole,"electric-dipole",args.dipole_unit,"atomic_unit")
        print("done")

    if args.efield_unit not in ["","au","atomic_unit"]:
        print("\tConverting 'efield' to 'atomic_unit' ... ", end="")
        efield = convert(efield,"electric-field",args.efield_unit,"atomic_unit")
        print("done")

    #------------------#
    # Computation
    print("\tComputing 'Econserved' ... ", end="")
    eda = (dipole * efield).sum(axis=1)
    Econserved = conserved - eda # + Tdep
    print("done")

    time = np.arange(len(conserved))*convert(args.time_step,"time","femtosecond","atomic_unit") 

    if args.xml is not None:
        print("\tAdding time-dependent part to 'Econserved':")
        print("\t - reading {:s} file ... ".format(args.xml),end="")
        Ef:ElectricField = get_Efield(args.xml)
        print("done")
        assert np.allclose(efield,Ef.Efield(time)), "efield and Ef.efield must be the same"

        print("\t - computing integrand ... ",end="")
        integrand = ( dipole * Ef.derivative(time) ).sum(axis=1)
        print("done")
        
        print("\t - integrating integrand ... ",end="")
        Tdep = np.cumsum(integrand)*args.time_step
        print("done")

        print("\t - adding new term to 'Econserved' ... ",end="")
        Econserved = Econserved + Tdep
        print("done")


    #------------------#
    # Drift
    # conserving `conserved` and `Econserved` to meV
    conserved = convert(conserved,"energy","atomic_unit","millielectronvolt")
    Econserved = convert(Econserved,"energy","atomic_unit","millielectronvolt")

    print("\n\tFitting 'conserved and 'Econserved' with a line ... ", end="")
    
    time = convert(time,"time","atomic_unit","picosecond")
    slope, _ = np.polyfit(time, conserved, 1)
    Eslope, _ = np.polyfit(time, Econserved, 1)
    print("done")
    print(f"\t conserved slope: {slope:.6f} meV/ps")
    print(f"\tEconserved slope: {Eslope:.6f} meV/ps")

    #------------------#
    # Setting
    print("\n\tAdding 'Econserved as '{:s}' to the properties ... ".format(args.output_name), end="")
    Econserved = convert(Econserved,"energy","millielectronvolt","atomic_unit")
    properties.set(args.output_name, Econserved)
    print("done")

    #------------------#
    # Saving
    print("\n\tSaving properties to '{:s}' ... ".format(args.output), end="")
    properties.to_file(file=args.output)
    print("done")



#---------------------------------------#
if __name__ == "__main__":
    main()
