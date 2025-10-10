#!/usr/bin/env python
import re, os
from ase.io import read
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from eslib.classes.atomic_structures import AtomicStructures
from eslib.formatting import esfmt
from eslib.io_tools import pattern2sorted_files
from eslib.functions import list2dict, dict_lists2np

#---------------------------------------#
description = "Extract Hirshfeld analysis restuls from a FHI-aims output file."

#---------------------------------------#
def prepare_args(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    argv = {"metavar" : "\b",}
    parser.add_argument("-i", "--input" ,**argv, type=str, required=True , help="FHI-aims file)")
    parser.add_argument("-o" , "--output"        , **argv, required=True , type=str, help="output file")
    parser.add_argument("-of" , "--output_format", **argv, required=False, type=str, help="output file format (default: %(default)s)", default="extxyz")
    return parser

# flexible float regex
_float_re = r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?'

def parse_hirshfeld_blocks_text(text: str) -> List[Dict]:
    """
    Parse Hirshfeld per-atom blocks from an FHI-aims output text and
    return a list of dicts (one per atom).

    Each dict contains:
      - atom_index (int)
      - element (str)
      - hirshfeld_charge (float or None)
      - free_atom_volume (float or None)
      - hirshfeld_volume (float or None)
      - hirshfeld_dipole_vector (list[float] of length 3 or None)
      - hirshfeld_dipole_moment (float or None)
      - hirshfeld_second_moments (3x3 nested list of floats or None)
    """
    # Header regex finds the start of each atom block (multi-line scan)
    header_re = re.compile(r'^\s*\|\s*Atom\s+(\d+):\s*([A-Za-z]{1,3})\b', re.M)
    results: List[Dict] = []

    # Helper to convert found single-number match to float or None
    def _find_scalar(block: str, label: str) -> Optional[float]:
        m = re.search(rf'{re.escape(label)}\s*:\s*({_float_re})', block)
        return float(m.group(1)) if m else None

    for hmatch in header_re.finditer(text):
        start = hmatch.end()
        # find next header to determine end-of-block
        next_h = header_re.search(text, start)
        end = next_h.start() if next_h else len(text)
        block = text[start:end]

        atom_index = int(hmatch.group(1))
        element = hmatch.group(2)

        data: Dict = {
            "atom_index": atom_index,
            "element": element,
            "hirshfeld_charge": _find_scalar(block, "Hirshfeld charge"),
            "free_atom_volume": _find_scalar(block, "Free atom volume"),
            "hirshfeld_volume": _find_scalar(block, "Hirshfeld volume"),
            "hirshfeld_dipole_vector": None,
            "hirshfeld_dipole_moment": _find_scalar(block, "Hirshfeld dipole moment"),
            "hirshfeld_second_moments": None,
        }

        # Dipole vector: capture numbers on the same line after the label
        m = re.search(r'Hirshfeld dipole vector\s*:\s*(.*)', block)
        if m:
            nums = re.findall(_float_re, m.group(1))
            if len(nums) >= 3:
                data["hirshfeld_dipole_vector"] = [float(x) for x in nums[:3]]

        # SECOND MOMENTS: find the line index and parse that line + next two lines
        lines = block.splitlines()
        idx = None
        for i, ln in enumerate(lines):
            if 'Hirshfeld second moments' in ln:
                idx = i
                break

        if idx is not None:
            sec = []
            for j in range(idx, min(idx + 3, len(lines))):
                ln = lines[j].lstrip()            # remove leading whitespace
                if ln.startswith('|'):
                    ln = ln.lstrip('|').strip()  # remove leading '|'
                # for the first line remove the label portion if still present
                if j == idx and ':' in ln:
                    ln = ln.split(':', 1)[1]
                nums = re.findall(_float_re, ln)
                if nums:
                    sec.append([float(x) for x in nums])
            if len(sec) == 3 and all(len(row) == 3 for row in sec):
                data["hirshfeld_second_moments"] = sec
            else:
                # fallback: try to flatten/find 9 numbers around the label
                around = ' '.join(lines[idx:idx+3])
                nums = re.findall(_float_re, around)
                if len(nums) >= 9:
                    sec = [ [ float(nums[0+i*3 + k]) for k in range(3) ] for i in range(3) ]
                    data["hirshfeld_second_moments"] = sec
                else:
                    data["hirshfeld_second_moments"] = None

        results.append(data)

    return results

def process_file(n,file):
    atoms = read(file,format="aims-output")
    with open(file, 'r') as fh:
        txt = fh.read()
    info = parse_hirshfeld_blocks_text(txt)
    Natoms = len(atoms)
    assert len(info) == Natoms, "different number of atoms"
    info = list2dict(info)
    info = dict_lists2np(info)
    for key in ['hirshfeld_charge', 'free_atom_volume', 'hirshfeld_volume', 
                 'hirshfeld_dipole_vector', 'hirshfeld_dipole_moment', 'hirshfeld_second_moments']:
        arr = info[key]
        if arr.ndim > 1:
            arr = arr.reshape((Natoms,-1))
        atoms.arrays[key] = arr
    return n,file,atoms

#---------------------------------------#
@esfmt(prepare_args, description)
def main(args):

    #------------------#
    files = pattern2sorted_files(args.input)
    N = len(files)
    print(f"\tFound {N} files matching the pattern '{args.input}'")
    print()
    results = [None]*N
   
    #------------------#
    # Use all available cores by default
    max_workers = min(N,int(os.cpu_count()/2))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, n, file): n for n, file in enumerate(files)}

        for future in as_completed(futures):
            n,file,atoms = future.result()
            results[n] = atoms
            print(f"\t{n+1}/{N}) Processing {file} ... done")

    #------------------#
    print("\tWriting the atomic structure to file '{:s}' ... ".format(args.output), end="")
    AtomicStructures(results).to_file(file=args.output,format=args.output_format)
    print("done")

#---------------------------------------#
if __name__ == "__main__":
    main()

