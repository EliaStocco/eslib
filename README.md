# `eslib`
My personal repository.

# Installation, or kind of
Add this line in your `.bashrc` file.
```bash
ELIAMISC="/home/stoccoel/google-personal/eslib/eslib"
export PATH="$PATH:${ELIAMISC}"
export PYTHONPATH="$PYTHONPATH:${ELIAMISC}"
for dir in "${ELIAMISC}"/scripts/*; do
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