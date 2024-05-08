#!/usr/bin/env python
import numpy as np
from eslib.formatting import esfmt, warning, everythingok
from eslib.classes.properties import properties as Properties
from eslib.functions import suppress_output
from eslib.input import str2bool

#---------------------------------------#
# Description of the script's purpose
description = "Check that a properties file is correctly formatted."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i" , "--input"          , type=str     , **argv, required=True , help='txt input file')
    parser.add_argument("-rr", "--remove_replicas", type=str2bool, **argv, required=False, help='whether to remove replicas (default: false)', default=False)
    parser.add_argument("-o" , "--output"         , type=str     , **argv, required=False, help='output pickle file (default: None)', default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_args,description)
def main(args):

    #------------------#
    print("\tReading properties from file '{:s}' ... ".format(args.input), end="")
    with suppress_output():
        if str(args.input).endswith(".pickle"):
            allproperties = Properties.from_pickle(file_path=args.input)
        else:
            allproperties = Properties.load(file=args.input)
    print("done")

    #------------------#
    print("\n\tSummary of the read properties:")
    df = allproperties.summary()
    print("\tNumber of snapshots : {:d}".format(len(allproperties)))
    print("\tNumber of properties: {:d}".format(len(df)))

    def line(): print("\t\t-----------------------------------------")
    
    line()
    print("\t\t|{:^15s}|{:^15s}|{:^7s}|".format("name","unit","shape"))
    line()
    for index, row in df.iterrows():
        print("\t\t|{:^15s}|{:^15s}|{:^7d}|".format(row["name"],row["unit"],row["shape"]))
    line()

    #------------------#
    # check steps
    steps = allproperties.properties['step'].astype(int)
    correct = np.arange(steps.shape[0]+1,dtype=int)

    print()
    if correct[-1] > steps[-1]:
        print("\t{:s}: there could be replicas.".format(warning))
        #------------------#
        # remove replicas
        if args.remove_replicas:
            print("\tRemoving replicas ... ", end="")
            allproperties = allproperties.remove_replicas(keyword="step",ofile="steps-without-replicas.properties.txt")
            print("done")
            print("\tNumber of snapshots : {:d}".format(len(allproperties)))

    elif correct[-1] < steps[-1]:
        print("\t{:s}: there could be missing steps.".format(warning))
    elif correct[-1] < steps[-1]:
        print("\t{:s}".format(everythingok))

    #------------------#
    # write 
    if args.output is not None:
        print("\n\tWriting properties to file '{:s}' ... ".format(args.output), end="")
        allproperties.to_pickle(args.output)
        print("done")
    else:
        print("\n\t{:s}: no ouput file provided (-o,--output)".format(warning))

#---------------------------------------#
if __name__ == "__main__":
    main()
