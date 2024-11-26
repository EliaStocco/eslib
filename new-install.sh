# module purge
# module load anaconda
# module load pytorch

conda create -n eslib python==3.9 -y
conda activate eslib
pip install -r requirements.txt
cd ..
git clone git@github.com:lab-cosmo/librascal.git
cd librascal
mkdir build
cd build
cmake ..
make
cd ../../eslib