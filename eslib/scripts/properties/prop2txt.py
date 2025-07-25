#!/usr/bin/env python
import numpy as np

from eslib.classes.properties import Properties
from eslib.formatting import esfmt, everythingok, warning, float_format
from eslib.functions import suppress_output
from eslib.input import str2bool, itype
from eslib.tools import convert
from eslib.classes.aseio import integer_to_slice_string

#---------------------------------------#
# Description of the script's purpose
description = "Export a property from an i-PI output file to a txt file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"          , type=str     , **argv, required=True , help='txt input file')
    parser.add_argument("-k" , "--keyword"        , type=str     , **argv, required=True , help='keyword')
    parser.add_argument("-f" , "--family"         , type=str     , **argv, required=False, help="family (default: %(default)s)", default=None)
    parser.add_argument("-u" , "--unit"           , type=str     , **argv, required=False, help="output unit (default: %(default)s)", default=None)
    parser.add_argument("-n" , "--index"          , type=itype   , **argv, required=False, help="index to be read from input file (default: %(default)s)", default=':')
    parser.add_argument("-s" , "--subsample"      , type=str     , **argv, required=False, help="txt file with the indices to keep (default: %(default)s)", default=None)
    parser.add_argument("-fd", "--fix_data"       , type=str2bool, **argv, required=False, help='whether to remove replicas and fill missing values (default: %(default)s)', default=False)
    parser.add_argument("-d" , "--delimiter"      , type=str     , **argv, required=False, help='delimiter (default: %(default)s)', default=' ')
    parser.add_argument("-o" , "--output"         , type=str     , **argv, required=False, help='output file (default: %(default)s)', default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):
    
    # assert not args.remove_replicas, "-rr,--remove_replicas no longer supported."
    
    if args.index != ':' and args.subsample is not None:
        print(f"\n\t{warning}: pay attention when using both --index and --subsample.")

    #------------------#
    index = integer_to_slice_string(args.index)
    print("\tReading properties from file '{:s}' ... ".format(args.input), end="")
    with suppress_output():
        if str(args.input).endswith(".pickle"):
            assert index is None, "`index` must be None when reading from `*.pickle` files."                
            allproperties = Properties.from_pickle(file_path=args.input)
        else:
            allproperties = Properties.load(file=args.input,index=index)
    print("done")

    #---------------------------------------#
    # summary
    print("\n\tSummary of the properties: ")
    df = allproperties.summary()
    tmp = "\n"+df.to_string(index=False)
    print(tmp.replace("\n", "\n\t"))

    #------------------#
    
    if args.fix_data:
        print("\n\tFixing data ... ", end="")
        allproperties, message = allproperties.fix()
        print("done")
        print(f"\tMessage : {message}")    
        
        print("\n\tSummary of the properties: ")
        df = allproperties.summary()
        tmp = "\n"+df.to_string(index=False)
        print(tmp.replace("\n", "\n\t"))

    #------------------#
    print(f"\n\tExtracting {args.keyword} ... ",end="")
    data = allproperties.get(args.keyword)
    print("done")
    print(f"\t{args.keyword}.shape: {data.shape}")
    
    print(f"\t{args.keyword} unit: {allproperties.units[args.keyword]}")
    
    #------------------#
    if args.subsample is not None:
        subsample = np.loadtxt(args.subsample).astype(int)
        assert subsample.ndim == 1, f"'{args.subsample}' should contain a 1D array."
        print("\n\tSubsampling data:")
        print("\tdata.shape : ",data.shape," (before subsampling)")
        data = data[subsample]
        print("\tdata.shape : ",data.shape," (after subsampling)")
    
    #------------------#
    if args.unit is not None:
        iu = allproperties.units[args.keyword]
        ou = args.unit
        print(f"\n\tConverting {args.keyword} from {iu} to {ou} ... ",end="")
        data = convert(data,args.family,_from=iu,_to=ou)
        print("done")        
    
    #------------------#
    # write 
    print("\n\tWriting data to file '{:s}' ... ".format(args.output), end="")
    if str(args.output).endswith("npy"):
        np.save(args.output,data)
    else:
        np.savetxt(args.output,data,fmt=float_format,delimiter=args.delimiter)
    print("done")
    

#---------------------------------------#
if __name__ == "__main__":
    main()
