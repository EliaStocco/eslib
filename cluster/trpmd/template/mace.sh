#!/bin/bash

###################################################################
# Redefine the echo command to include a tab character
echo() {
  command echo -e "\t$@"
}

###################################################################
# Check if SLURM_LOCALID is set
if [ -z "$SLURM_LOCALID" ]; then
  echo "SLURM_LOCALID is not set. Exiting."
  exit 1
fi

# Determine if SLURM_LOCALID is even or odd
if [ $((SLURM_LOCALID % 2)) -eq 0 ]; then
    echo "SLURM_LOCALID is even: $SLURM_LOCALID"
    echo "Preparing PES " >> log.out
    export OUTPUT_SCRIPT_NAME="pes"
    script="macelia-gpu.py"
    model_type="MACE"
    model="../../../../64_bulk_watter_swa.model"
    device="cuda"
    args="-s start.extxyz"
    args="${args} -a ${HOST_PES}  -u true " 
    args="${args} -mt ${model_type} -m ${model}"
    args="${args} -d ${device} -dt float32 -sc ase"

    # Declare an array to hold PIDs
    NN_COMMAND="${script} ${args} "
    echo "${NN_COMMAND}" >> log.out
    eval "${NN_COMMAND}" &

    # Get the PID of the PES process
    PES_PID=$!
    echo "PES ID: ${PES_PID}" >> log.out
else
    echo "SLURM_LOCALID is odd: $SLURM_LOCALID"
    echo "Preparing DIPOLE " >> log.out
    export OUTPUT_SCRIPT_NAME="dipole"
    script="macelia-gpu.py"
    model_type="eslib"
    model="../../../../model.cuda.bec.pickle"
    device="cuda"
    args="-s start.extxyz"
    args="${args} -a ${HOST_DIP}  -u true "
    args="${args} -mt ${model_type} -m ${model}"
    args="${args} -d ${device} -dt float64 -sc eslib"

    # Declare an array to hold PIDs
    NN_COMMAND="${script} ${args} "
    echo "${NN_COMMAND}" >> log.out
    eval "${NN_COMMAND}" &

    # Get the PID of the PES process
    DIP_PID=$!
    echo "Dipole ID: ${DIP_PID}" >> log.out
fi