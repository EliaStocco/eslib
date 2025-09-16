#!/usr/bin/env python3
from ase.calculators.socketio import SocketIOCalculator
from ase.mep import NEB
from ase.optimize import BFGS
from ase.io import write

from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.input import str2bool
from eslib.show import show_dict

# --------------------------------------- #
description = "Run an ASE Nudge Elastic Band calculation (per-image socket calculators)."

# --------------------------------------- #
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b", }
    # Input
    parser.add_argument("-ii", "--initial_input", **argv, required=True, type=str,
                        help="initial file with an atomic structure")
    parser.add_argument("-iif", "--initial_input_format", **argv, required=False, type=str,
                        help="initial input file format (default: %(default)s)", default=None)
    parser.add_argument("-fi", "--final_input", **argv, required=True, type=str,
                        help="final file with an atomic structure")
    parser.add_argument("-fif", "--final_input_format", **argv, required=False, type=str,
                        help="final input file format (default: %(default)s)", default=None)
    # Parameters
    parser.add_argument("-N", "--n_images", **argv, required=True, type=int,
                        help="number of images (including endpoints)")
    # Socket
    parser.add_argument("-p", "--port", **argv, required=False, type=int,
                        help="TCP/IP base port number (used when --unix False). Default: %(default)s", default=6000)
    parser.add_argument("-a", "--address", **argv, required=True, type=str,
                        help="Host name (for INET sockets) or base name/path for UNIX domain socket")
    parser.add_argument("-u", "--unix", **argv, required=False, type=str2bool,
                        help="Use a UNIX domain socket (default: %(default)s)", default=False)
    # Optimizer / output
    parser.add_argument("-o", "--output", **argv, required=False, type=str,
                        help="output file (trajectory) (default: %(default)s)", default="NEB.traj")
    parser.add_argument("-of", "--output_format", **argv, required=False, type=str,
                        help="output file format (default: inferred from output filename)", default=None)
    parser.add_argument("--opt_par", **argv, required=False, type=str,
                        help="JSON file with optimizer params (passed to BFGS)", default=None)
    parser.add_argument("--restart", **argv, required=False, type=str,
                        help="BFGS restart file (optional)", default=None)
    parser.add_argument("--trajectory", **argv, required=False, type=str,
                        help="BFGS trajectory file (optional)", default=None)
    parser.add_argument("--fmax", **argv, required=False, type=float,
                        help="Force convergence criterion (eV/Å)", default=0.05)
    parser.add_argument("--maxstep", **argv, required=False, type=int,
                        help="Maximum optimizer steps", default=200)
    parser.add_argument("--logger", **argv, required=False, type=str,
                        help="optional logger file for SocketIOCalculator (not the optimizer)", default=None)
    return parser

# --------------------------------------- #
@esfmt(prepare_args, description)
def main(args):
    # --- Read initial/final structures ---
    print(f"Reading initial structure from {args.initial_input} ...")
    initial = AtomicStructures.from_file(file=args.initial_input, format=args.initial_input_format, index=0)[0]

    print(f"Reading final structure from {args.final_input} ...")
    final = AtomicStructures.from_file(file=args.final_input, format=args.final_input_format, index=0)[0]

    assert initial.get_global_number_of_atoms() == final.get_global_number_of_atoms(), \
        "Initial and final structures must have the same number of atoms."

    # --- Build NEB images ---
    if args.n_images < 2:
        raise SystemExit("n_images must be >=2 (endpoints included)")
    n_middle = args.n_images - 2
    images = [initial] + [initial.copy() for _ in range(n_middle)] + [final]
    print(f"Built {len(images)} images")

    # --- Single SocketIOCalculator ---
    print("Creating single SocketIOCalculator ...")
    if args.unix:
        calc = SocketIOCalculator(unixsocket=args.address, log=args.logger)
    else:
        calc = SocketIOCalculator(host=args.address, port=args.port, log=args.logger)

    # --- NEB with shared calculator (serial) ---
    neb = NEB(images, allow_shared_calculator=True)  # <- key change
    neb.interpolate()

    # Assign the single calculator to all images
    for img in images:
        img.calc = calc

    print("NEB interpolation done")

    # --- Run optimizer ---
    print(f"Running BFGS optimizer (fmax={args.fmax}, maxstep={args.maxstep}) ...")
    optimizer = BFGS(neb, trajectory=args.output)
    optimizer.run(fmax=args.fmax, steps=args.maxstep)
    print("NEB optimization finished")

    # --- Save final images ---
    print(f"Saving NEB images to {args.output} ...")
    write(args.output, images, format=args.output_format)
    print("Done")

# --------------------------------------- #
if __name__ == "__main__":
    main()