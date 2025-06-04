#!/bin/bash -l
#SBATCH -o slurm/output.txt
#SBATCH -e slurm/error.txt
#SBATCH -D ./
#SBATCH -J TRPMD

#SBATCH --nodes=1 
## SBATCH --partition=p.ada 
## SBATCH --gres=gpu:a100:4
## SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00

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

mkdir -p spectrum
# for n in {0..6..1}; do 
#     ifile="final/E=0.00/xc.run-${n}.npz"

#     tacf+IR-mu-Z.py -i ${ifile} -z MACE_BEC -v velocities -o spectrum/IR.n=${n}.txt -dt 0.25 -s 4 -T 300
#     plot.py spectrum/IR.n=${n}.txt spectrum/IR.n=${n}.png
    
# done

# prepare-tacf+IR-mu-Z.py -i "final/E=0.00/xc.run-*.npz" -z MACE_BEC -v velocities -o spectrum/data
if [ -e spectrum/data.npz ]; then
    for wt in 500 800 1000 1500 2000 ; do
        compute-tacf+IR-mu-Z.py -i spectrum/data.npz -o spectrum/IR.wt=${wt}.txt -t0 20 -dt 0.25 -s 4 -T 300 -wt ${wt} > spectrum/IR.summary.wt=${wt}.txt
        python plot.py spectrum/IR.wt=${wt}.txt spectrum/IR.wt=${wt}.png
    done
fi


