#!/bin/bash -l
#SBATCH -o slurm/output.txt
#SBATCH -e slurm/error.txt
#SBATCH -D ./
#SBATCH -J TRPMD

#SBATCH --nodes=1 
#SBATCH --partition=p.ada 
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
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

mkdir -p mace

rm log.*.out

efield="0.00"
for n in 1 5 6 ; do
    ifile="pos/E=${efield}/xc.run-${n}.extxyz"
    ofile="mace/E=${efield}/xc.run-${n}.extxyz"
    srun -n 1 --gres=gpu:a100:1 --exclusive eval-model.py -i ${ifile} -o ${ofile} -m ../../model.cuda.bec.pickle -c Qs -cf charges.json > "log.n=${n}.out" 2> "log.n=${n}.err" &
done
wait

# for n in {4..7..1}; do
#     ifile="pos/E=${efield}/xc.run-${n}.extxyz"
#     ofile="mace/E=${efield}/xc.run-${n}.extxyz"
#     srun -n 1 --gres=gpu:a100:1 --exclusive eval-model.py -i ${ifile} -o ${ofile} -m ../../model.cuda.bec.pickle -c Qs -cf charges.json > "log.n=${n}.out" 2> "log.n=${n}.err" &
# done
# wait



