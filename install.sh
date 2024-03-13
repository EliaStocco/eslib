pip install -e .

ESLIB="/home/stoccoel/google-personal/codes/eslib"
export PATH="$PATH:${ESLIB}/eslib"
export PYTHONPATH="$PYTHONPATH:${ESLIB}/eslib"
export PATH="$PATH:${ESLIB}"
export PYTHONPATH="$PYTHONPATH:${ESLIB}"
for dir in "${ESLIB}"/eslib/scripts/*; do
    if [ -d "$dir" ]; then
        export PATH="$PATH:$dir"
        # echo ${dir}
        # Make Python files executable in each subdirectory
        find "$dir" -name "*.py" -type f -exec chmod +x {} \;
    fi
done