# `eslib`
My personal repository.

# Installation, or kind of
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
conda install anaconda::pandas
```

Make the scripts executable:
```bash
ESLIB="/home/stoccoel/google-personal/eslib"
export PATH="$PATH:${ESLIB}/eslib/"
export PYTHONPATH="$PYTHONPATH:${ESLIB}/eslib/"
for dir in "${ESLIB}/eslib/scripts/*"; do
    if [ -d "$dir" ]; then
        export PATH="$PATH:$dir"

        # Make Python files executable in each subdirectory
        find "$dir" -name "*.py" -type f -exec chmod +x {} \;
    fi
done
```
and install the package in `edit` mode with:
```bash
pip install -e .
```