#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ase import Atoms
from ase.io import write
from typing import List
from dscribe.descriptors import SOAP
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt

#---------------------------------------#
description = "Generate and optimize atomic structures using SOAP descriptors and DScribe."

#---------------------------------------#
def prepare_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar": "\b"}
    parser.add_argument("-i" , "--input"       , type=str, required=True , **argv, help="input file [au]")
    parser.add_argument("-if", "--input_format", type=str, required=False, **argv, default=None, help="input file format (default: %(default)s)")
    parser.add_argument("-o" , "--output"      , type=str, required=False, **argv, default="final.extxyz", help="output file with optimized structures (default: %(default)s)")
    parser.add_argument("-n" , "--number"      , type=int, required=True , **argv, help="number of structures to generate/optimize")
    parser.add_argument("-t" , "--template"       , type=str, required=True , **argv, help="template file [au]")
    parser.add_argument("-tf", "--template_format", type=str, required=False, **argv, default=None, help="template file format (default: %(default)s)")
    parser.add_argument("--debug", action="store_true", help="if set, saves every structure during optimization to debug_structure_<N>.extxyz")
    return parser

#---------------------------------------#
@esfmt(prepare_parser, description)
def main(args):

    #-------------------#
    print(f"\n\tReading template structure from '{args.template}' ... ", end="")
    template: Atoms = AtomicStructures.from_file(file=args.template, format=args.template_format, index=0)[0]
    print("done")
    pbc = any(template.pbc)

    #-------------------#
    print(f"\n\tReading reference structures from '{args.input}' ... ", end="")
    structures: List[Atoms] = AtomicStructures.from_file(file=args.input, format=args.input_format)
    print("done")
    n_structures = len(structures)
    print(f"\tn. of reference structures: {n_structures}")
    Natoms = structures.call(lambda x: x.get_global_number_of_atoms())
    print(f"\tn. of atoms: {np.unique(Natoms)}")
    species = structures.get_chemical_symbols(unique=True)
    print(f"\tspecies: {species}")

    #-------------------#
    print("\n\tPreparing SOAP object ... ", end="")
    soap = SOAP(
        species=species,
        r_cut=2.0,
        n_max=8,
        l_max=6,
        sigma=0.3,
        periodic=pbc,
        sparse=False,
    )
    print("done")

    #-------------------#
    print("\tComputing SOAP descriptors for reference structures ... ", end="")
    X_list = [soap.create(structure) for structure in structures]
    print("done")
    # [a.shape for a in X_list] --> [(3, 952), (9, 952)]

    #-------------------#
    def x2atoms(x: np.ndarray) -> Atoms:
        """Convert flat x vector (scaled coordinates) back to ASE Atoms object using fixed template cell."""
        scaled_pos = x.reshape((-1, 3))  # scaled coordinates in [0,1]
        cart_pos = template.cell.cartesian_positions(scaled_pos)  # convert to Cartesian
        structure = Atoms(
            symbols=template.get_chemical_symbols(),
            positions=cart_pos,
            cell=template.cell,
            pbc=pbc
        )
        return structure

    #-------------------#
    def func(x: np.ndarray):
        x[:3] = 0
        structure = x2atoms(x)
        feat = soap.create(structure)
        symbols = np.array(structure.get_chemical_symbols())
        
        total_loss = 0.0
        
        for struct_feat, ref in zip(X_list, structures):
            ref_symbols = np.array(ref.get_chemical_symbols())
            loss_ref = 0.0
            
            # unique species in candidate and reference
            species_set = np.unique(np.concatenate([symbols, ref_symbols]))
            
            for s in species_set:
                idx_cand = np.where(symbols == s)[0]
                idx_ref  = np.where(ref_symbols == s)[0]
                
                if len(idx_cand) == 0 or len(idx_ref) == 0:
                    continue  # skip species not in one structure
                
                D_s = cdist(feat[idx_cand], struct_feat[idx_ref])
                row_ind, col_ind = linear_sum_assignment(D_s)
                loss_s = D_s[row_ind, col_ind].mean()
                loss_ref += loss_s
            
            total_loss += loss_ref
        
        return total_loss / len(X_list)



    #-------------------#
    np.random.seed(42)  # reproducibility

    initial_structures = []
    optimized_structures = []

    for i in range(args.number):
        print(f"\n\tGenerating structure {i+1}/{args.number} ...")

        # --- Initialize random positions only ---
        pos = template.positions
        n_pos = pos.size
        pos_min = 0.
        pos_max = 1.
        rand_pos = np.random.uniform(pos_min, pos_max, size=pos.shape)
        x = rand_pos.flatten()
        bounds = [(pos_min, pos_max) for j in range(n_pos)]  # bounds only for positions
        x[:3] = 0
        bounds[0] = (0,0)

        # Save initial random structure
        init_atoms = x2atoms(x)
        initial_structures.append(init_atoms)

        # --- Debug callback ---
        debug_traj = []
        def debug_callback(xk):
            if args.debug:
                step_atoms = x2atoms(xk)
                debug_traj.append(step_atoms)

        # --- Optimization ---
        result = minimize(func, x, bounds=bounds, callback=debug_callback)

        # Save debug trajectory if requested
        if args.debug and debug_traj:
            debug_filename = f"debug_structure_{i+1:03d}.extxyz"
            write(debug_filename, debug_traj)
            print(f"\t→ Saved debug trajectory to '{debug_filename}' ({len(debug_traj)} frames)")

        # Final structure
        final_atoms = x2atoms(result.x)
        optimized_structures.append(final_atoms)
        print(f"\t→ Structure {i+1} optimized (final loss = {result.fun:.6f})")

    #-------------------#
    print("\n\tWriting all structures to disk ... ", end="")
    write("initial.extxyz", initial_structures)
    write(args.output, optimized_structures)
    print("done")

    print(f"\n✅ Saved {len(initial_structures)} initial structures to 'initial.extxyz'")
    print(f"✅ Saved {len(optimized_structures)} optimized structures to '{args.output}'")
    if args.debug:
        print(f"✅ Saved {args.number} debug trajectories (debug_structure_<N>.extxyz)")
    print()

#---------------------------------------#
if __name__ == "__main__":
    main()
