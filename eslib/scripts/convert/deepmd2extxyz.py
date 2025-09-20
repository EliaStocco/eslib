#!/usr/bin/env python
import os
import numpy as np
from ase import Atoms
from ase.io import write
from eslib.formatting import esfmt


#---------------------------------------#
# Description of the script's purpose
description = "Convert an DeepMD dataset to extxyz."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i"  , "--input"            , **argv,required=True , type=str     , help="folder with DeepMD files")
    parser.add_argument("-o"  , "--output"           , **argv,required=True , type=str     , help="output file")
    parser.add_argument("-of" , "--output_format"    , **argv,required=False, type=str     , help="output file format (default: %(default)s)", default=None)
    return parser

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    print(f"\tReading DeepMD dataset from {args.input} ... ", end="")
    set_dir = args.input
    base_dir = os.path.dirname(set_dir.rstrip("/"))

    # Load arrays
    coords  = np.load(os.path.join(set_dir, "coord.npy"))    # (nframes, natoms*3)
    box     = np.load(os.path.join(set_dir, "box.npy"))      # (nframes, 9)
    forces  = np.load(os.path.join(set_dir, "force.npy"))    # (nframes, natoms*3)
    energy  = np.load(os.path.join(set_dir, "energy.npy"))   # (nframes,)
    types   = np.loadtxt(os.path.join(base_dir, "type.raw"), dtype=int)
    print("done")

    print("\tcoords.shape: ", coords.shape)
    print("\tbox.shape:    ", box.shape)
    print("\tforces.shape: ", forces.shape)
    print("\tenergy.shape: ", energy.shape)
    print("\ttypes.shape:  ", types.shape)

    Nsnapshots = int(coords.shape[0])
    Natoms     = len(types)
    
    print(f"\n\tNumber of snapshots: {Nsnapshots}")
    print(f"\tNumber of atoms per snapshot: {Natoms}")

    print("\n\tReshaping data ... ", end="")
    coords = np.reshape(coords, (Nsnapshots, -1, 3))[:, :Natoms, :]
    box    = np.reshape(box,    (Nsnapshots, 3, 3))
    forces = np.reshape(forces, (Nsnapshots, -1, 3))[:, :Natoms, :]
    energy = np.reshape(energy, (Nsnapshots,))
    types  = np.reshape(types,  (Natoms,)).astype(int)
    print("done")
    print("\tcoords.shape: ", coords.shape)
    print("\tbox.shape:    ", box.shape)
    print("\tforces.shape: ", forces.shape)
    print("\tenergy.shape: ", energy.shape)
    print("\ttypes.shape:  ", types.shape)

    # Type mapping (DeepMD standard: 0 = O, 1 = H)
    type_map = {0: "O", 1: "H"}
    symbols = [type_map[t] for t in types]

    print(f"\n\tConverting {Nsnapshots} snapshots to ASE Atoms objects ... ", end="")
    frames = []
    for i in range(Nsnapshots):
        atoms = Atoms(
            symbols=symbols,
            positions=coords[i],
            cell=box[i],
            pbc=True
        )
        atoms.info["energy"] = float(energy[i])
        atoms.arrays["forces"] = forces[i]
        frames.append(atoms)
    print("done")

    # Output
    output_format = args.output_format if args.output_format else "extxyz"
    print(f"\n\tWriting {len(frames)} frames to {args.output} (format={output_format}) ... ", end="")
    write(args.output, frames, format=output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()