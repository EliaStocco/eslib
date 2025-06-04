#!/usr/bin/env python
import os
import subprocess
import sys
from eslib.scripts.drivers.macelia_gpu import main

# Check if SLURM_LOCALID is set
slurm_localid = os.environ.get("SLURM_LOCALID")
if slurm_localid is None:
    print("\tSLURM_LOCALID is not set. Exiting.")
    sys.exit(1)

slurm_localid = int(slurm_localid)

if slurm_localid % 2 == 0:
    print(f"\tSLURM_LOCALID is odd: {slurm_localid}")
    print("Preparing PES", file=open("log.out", "a"))
    os.environ["OUTPUT_SCRIPT_NAME"] = "pes"
    args = {
        "structure":"start.extxyz",
        "address": os.environ.get("HOST_PES", ""),
        "unix": True,
        "model_type": "MACE",
        "model": "../../../../64_bulk_watter_swa.model",
        "device":"cuda",
        "dtype": "float32",
        "socket_client": "ase"
    }

else:
    print(f"\tSLURM_LOCALID is odd: {slurm_localid}")
    print("Preparing DIPOLE", file=open("log.out", "a"))
    
    os.environ["OUTPUT_SCRIPT_NAME"] = "dipole"
    args = {
        "structure":"start.extxyz",
        "address": os.environ.get("HOST_DIP", ""),
        "unix": True,
        "model_type": "eslib",
        "model": "../../../../model.cuda.bec.pickle",
        "device":"cuda",
        "dtype": "float64",
        "socket_client": "eslib"
    }

main(args)