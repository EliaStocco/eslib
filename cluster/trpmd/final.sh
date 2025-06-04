#!/bin/bash -l
#SBATCH -o slurm/output.txt
#SBATCH -e slurm/error.txt
#SBATCH -D ./
#SBATCH -J TRPMD

#SBATCH --nodes=1 
#SBATCH --partition=p.ada 
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00

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

mkdir -p final
efield="0.00"
mkdir -p final/E=${efield}
for n in {0..7..1}; do 
    xfile="mace/E=${efield}/xc.run-${n}.extxyz"
    vfile="pos/E=${efield}/vc.run-${n}.extxyz"
    ofile="final/E=${efield}/xc.run-${n}.extxyz"

    # compress and remove 'ipi_comment'
    ofile="final/E=0.00/xc.run-${n}.npz"
    remove-field.py -i  ${xfile} -o ${ofile} -n "ipi_comment"

    # add the velocities
    extxyz2array.py -i ${vfile} -k positions -o vel.npy
    add2extxyz.py -i ${ofile} -n velocities -d vel.npy -w a -o ${ofile}
    
done
rm vel.npy



