#!/bin/bash -l
#SBATCH -o slurm/output.txt
#SBATCH -e slurm/error.txt
#SBATCH -D ./
#SBATCH -J MSD

#SBATCH --nodes=1 
#SBATCH --partition=p.ada 
#SBATCH --ntasks=40
#SBATCH --mail-type=NONE
#SBATCH --time=01:00:00

# --- Environment setup ---
module purge
module load anaconda/3/2023.03
source ~/scripts/elia.sh
source ~/scripts/import.sh
source /u/elsto/venv/eslib/bin/activate
source /u/elsto/programs/eslib/install.sh

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/u/elsto/programs/eslib"

# --- Prepare directories ---
mkdir -p msd pos plot log

# --- Collect commands ---
commands=()
logs=()
for EFIELD in 0.00 0.02 0.05 0.10 0.15; do
    mkdir -p pos/E=${EFIELD}
    for n in {0..7}; do
        ifile="pos/E=${EFIELD}/xc.n=${n}.npz"
        logfile="log/E=${EFIELD}_n=${n}.log"

        if [ ! -e "${ifile}" ]; then
            cmd="convert-file.py -i E=${EFIELD}/run-${n}/i-pi.xc.extxyz \
                                 -pk x_centroid -ra true -rp true \
                                 -o ${ifile}"
            commands+=("$cmd")
            logs+=("$logfile")
        fi
    done
done

# --- Run in parallel ---
pids=()
for i in "${!commands[@]}"; do
    srun --exclusive -N1 -n1 bash -c "${commands[$i]}" \
        > "${logs[$i]}" 2>&1 &
    pids+=($!)
done

# --- Wait & check exit codes ---
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    logfile=${logs[$i]}
    if ! wait $pid; then
        echo "Job failed: see $logfile" >&2
    fi
done

echo "All jobs completed."
