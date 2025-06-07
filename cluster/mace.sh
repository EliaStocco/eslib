# this is in ADA
module purge
module load anaconda/3/2023.03 gcc/11 openmpi/4
module load cuda/11.6 cudnn/8.8.1
module load pytorch/gpu-cuda-11.6/2.0.0
module load mpi4py/3.1.4

# Currently Loaded Modulefiles:
#  1) anaconda/3/2023.03   2) gcc/11   3) openmpi/4   4) cuda/11.6   5) cudnn/8.8.1   6) pytorch/gpu-cuda-11.6/2.0.0   7) mpi4py/3.1.4

python -m venv venv/mace-main

module purge
module load anaconda/3/2023.03 
module load gcc/15

pip install -e .

echo $(which python)
echo $(python --version)