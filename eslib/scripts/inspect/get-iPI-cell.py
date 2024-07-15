#!/usr/bin/env python
# author: Elia Stocco
# email : elia.stocco@mpsd.mpg.de
from ase.io import read
import numpy as np
from eslib.show import matrix2str
from eslib.formatting import esfmt

#---------------------------------------#
description = "Save to file and/or print to screen the cell an atomic configuration in a i-PI compatible format."

#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar":"\b"}
    parser.add_argument("-i", "--input",  type=str,**argv,help="atomic structure input file")
    parser.add_argument("-d", "--digits",  type=int,**argv,help="number of digits (default: %(default)s)", default=6)
    parser.add_argument("-m", "--mode"  ,  type=int,**argv,help="mode (default: %(default)s)", default=None)
    parser.add_argument("-e", "--exponential",  type=bool,**argv,help="exponential notation (default: %(default)s)", default=False)
    parser.add_argument("-o", "--output",  type=str,**argv,help="output file (default: %(default)s)", default=None)
    return parser# .parse_args()

#---------------------------------------#
@esfmt(prepare_parser,description)
def main(args):

    otext = ""

    # structure A
    print("\tReading atomic structure from input file '{:s}' ... ".format(args.input), end="")
    atoms = read(args.input)
    print("done")

    pbc = np.any(atoms.get_pbc())

    if not pbc:
        print("\n\tThe system is not periodic: providing a huge cell")
        cell = np.eye(3)*1e8
    else:
        cell = np.asarray(atoms.cell).T

    print("\tCell:")
    line = matrix2str(cell,col_names=["1","2","3"],cols_align="^",width=args.digits+2)
    print(line)

    print("\tCell in i-PI compatible formats:")
    print("\tmanual:")
    exp = "e" if args.exponential else "f" 
    num_format = f'{{:>{args.digits+6}.{args.digits}{exp}}}'
    text_format =   "\t<cell mode='manual' units=' ... ' > \n" + \
                    ( "\t\t[" + (num_format+", ")*3 + "\n")*2  + "\t\t" + (num_format+", ")*2  + num_format + "]" + \
                    "\n\t</cell>"
    text =  text_format.format(*cell.flatten().tolist())
    print(text)
    if args.mode is None or args.mode == "manual":
        otext += text

    print("\tabcABC:")
    exp = "e" if args.exponential else "f" 
    num_format = f'{{:>{args.digits+6}.{args.digits}{exp}}}'
    text_format =   "\t<cell mode='abcABC' units=' ... ' > \n" + \
                    "\t\t[" + (num_format+", ")*3 + "\n"  + "\t\t" + (num_format+", ")*2  + num_format +  "]" + \
                    "\n\t</cell>"
    text =  text_format.format(*atoms.get_cell().cellpar().tolist())
    print(text)
    if args.mode is None or args.mode == "abcABC":
        otext += text
    
    if args.output is not None:
        print("\n\tWriting cell in the i-PI compatible format to file '{:s}' ... ".format(args.output), end="")
        # Open a file in write mode ('w')
        with open(args.output, 'w') as file:
            # Write a string to the file
            file.write(otext)
        print("done")

if __name__ == "__main__":
    main()