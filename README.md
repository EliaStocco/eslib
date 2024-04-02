# `eslib`
Elia Stocco's personal repository.

# Installation
Add this line in your `.bashrc` file.
```bash
pip install black
pip install nbconvert
pip install sphinx
pip install matplotlib
conda install pytorch torchvision torchaudio -c pytorch
CUDA=cpu
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+${CUDA}.html
pip install torch-geometric
pip install e3nn
conda install anaconda::pandas
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