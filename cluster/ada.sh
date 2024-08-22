#!/bin/bash -l
#SBATCH -o slurm/output.txt
#SBATCH -e slurm/error.txt
#SBATCH -D ./
#SBATCH -J TEMP

#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --partition=p.ada    # in the GPU partition
#SBATCH --gres=gpu:a100:4    # Request 4 GPUs per node.
## SBATCH --ntasks-per-node=4  # Run one task per GPU
## SBATCH --cpus-per-task=18   # using 18 cores each.
#SBATCH --ntasks-per-node=72
## SBATCH --mem-per-cpu=6000
#SBATCH --threads-per-core=1

#SBATCH --mail-type=NONE
#SBATCH --time=01:20:00


###################################################################
conda activate mace
module purge
module load anaconda/3/2021.11 cuda/11.6
module load cudnn/8.8.1 pytorch/gpu-cuda-11.6/2.0.0 gcc/11 openmpi/4 mpi4py/3.0.3
export PYTHONPATH="${PYTHONPATH}:/u/elsto/programs/mace"
export PYTHONPATH="${PYTHONPATH}:/u/elsto/programs/eslib"
source /u/elsto/programs/eslib/install.sh

python ~/programs/eslib/clusters/slurm-info.py > slurm/slurm-info.txt

rm slurm/*
rm files/pid.txt
exec >> slurm/my_output.txt 2>&1   # Redirect all subsequent output to output.txt
source nve.sh $@

source ~/programs/eslib/clusters/report.sh > slurm/report.txt
source ~/programs/eslib/clusters/save.sh