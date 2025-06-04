#!/bin/bash -l
#SBATCH -o slurm/output.txt
#SBATCH -e slurm/error.txt
#SBATCH -D ./
#SBATCH -J TRPMD-GPU

#SBATCH --nodes=1 
#SBATCH --partition=p.ada 
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=64
#SBATCH --mail-type=none 
#SBATCH --nvmps               

#SBATCH --mail-type=NONE
#SBATCH --time=08:00:00


###################################################################
source ~/scripts/elia.sh
source ~/scripts/import.sh
conda activate mace
module purge
module load anaconda/3/2021.11 cuda/11.6
module load cudnn/8.8.1 pytorch/gpu-cuda-11.6/2.0.0 gcc/11 openmpi/4 mpi4py/3.0.3
export PYTHONPATH="${PYTHONPATH}:/u/elsto/programs/mace"
export PYTHONPATH="${PYTHONPATH}:/u/elsto/programs/eslib"
source /u/elsto/programs/eslib/install.sh

echo "OMP_NUM_THREADS (default): ${OMP_NUM_THREADS}"
export OMP_NUM_THREADS=${SLURM_NTASKS_PER_NODE}
echo "OMP_NUM_THREADS (user): ${OMP_NUM_THREADS}"
# rm slurm/*
 
# Initialize an array to store PIDs
pids=()

# export NBEADS=32
# export MACE_PREFIX="srun --mem=64000 --gres=gpu:a100:1"
export MACE_PREFIX="srun"
nohup ./nvt.sh $@ & 
pids+=($!)  # Add PID to array 
sleep 1

# Start nvidia-smi logging every 10 seconds in the background
(while true; do nvidia-smi > gpu.out; sleep 10; done) &
nvidia_smi_pid=$!  # Store the PID of the nvidia-smi logging process
pids+=($nvidia_smi_pid)  # Add PID to the array to track this background job

# Wait for all background jobs to complete
for pid in "${pids[@]}"; do
    echo "wait $pid" >> slurm/my_output.txt
    wait $pid
done

echo "All jobs completed."

