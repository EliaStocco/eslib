# ESLib
Elia Stocco's personal repository.

# Installation

Using `conda`:
```bash
conda create -n eslib python==3.9 -y
conda activate eslib
```
or `python` virtual environment:
```bash
python -m venv ~/venvs/eslib
source ~/venvs/eslib/bin/activate
```

Install the package in editable mode with:
```bash
pip install --upgrade pip
pip install -e .
```
and make all the scripts executable:
```bash
source install.sh
```
In case add `source <eslib_path>/install.sh` to `~/.bashrc` to have the scripts always available.

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