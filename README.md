# `eslib`
Elia Stocco's personal repository.

# Installation
Using `conda`:
```bash
# create a conda environment
conda create -n eslib python==3.9 -y
conda activate eslib
# install packages
conda install pytorch torchvision torchaudio -c pytorch -y
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
conda install anaconda::pandas -y
pip install -r requirements.txt
# optional
```

Install the package in `edit` mode with:
```bash
pip install -e .
```
and make all the scripts executable:
```bash
source install.sh
```
In case add `source eslib_path/install.sh` to `~/.bashrc`.

# Testing
Run the automatic tests using `pytest` by typing
```bash
./run-test.sh
```

## Help
You don't find a script? Just type:
```bash
eslib-help.py 
```
and you'll see all the script contained in this repository with a short description for each of them.
If you know in which folder the script that you are looking for is (e.g. `inspect`), you can type:
```bash
eslib-help.py -f inspect
```