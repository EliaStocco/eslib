###################################################################
# Redefine the echo command to include a tab character
echo() {
  command echo -e "\t$@"
}

###################################################################
check_process_running() {
    local pid=$1
    local name=$2
    local other_pids=$3  # PIDs of other processes to kill if this one dies

    if ps -p $pid > /dev/null; then
        echo "$name process with PID $pid is still running."
    else
        echo "$name process with PID $pid has terminated. Exiting script."
        if [ -n "$other_pids" ]; then  # Check if other_pids is not empty
            echo "Killing other processes: $other_pids"
            kill $other_pids 2>/dev/null
        fi
        exit 1
    fi
}

###################################################################
# conda init bash
source ~/.bashrc
conda activate mace

###################################################################

# print the date
echo
echo "Date: $(date +'%Y-%m-%d')"
echo "Time: $(date +'%H:%M:%S')"
echo
echo "which python: $(which python)"
echo "python --version: $(python --version)"
echo "conda env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo

if [ -n "$SLURM_JOB_ID" ]; then
  echo "This script is running under SLURM with job ID $SLURM_JOB_ID"
else
  echo "This script is running in a Bash shell"
fi


###################################################################
PROGRAMS_DIR="/u/elsto/programs"
source ${PROGRAMS_DIR}/i-pi/env.sh
IPI_PATH="${PROGRAMS_DIR}/i-pi/bin"
ipi_radix='i-pi'
ipi_input="input.xml"
port="20633"
PES_OUTPUT_FILE="mace.pes.out"
DIPOLE_OUTPUT_FILE="mace.dipole.out"

# IP address
HOST=$(hostname -i)

###################################################################
# create some folders
RESULTS_DIR="./results"
FILES_DIR="./files"
for DIR in ${RESULTS_DIR} ${FILES_DIR} ; do
    if test ! -d "${DIR}" ; then
        mkdir "${DIR}"
    fi
done

temp="EXIT"
if [[ -f ${temp} ]] ; then
    echo "removing '${temp}'"
    cmd="rm ${temp}"
    eval ${cmd}
fi

###################################################################
# chose the correct i-PI input file

if test -f "RESTART" ; then
    echo "Found RESTART file: it will be used as the input file for i-PI"
    ipi_input="RESTART"
fi
# 'ipi_input' is not defined
if [ -z ${ipi_input+x} ] ; then
    # echo "No input file for i-PI provided"
    if test -f "input.xml" ; then
        echo "Found 'input.xml' file: it will be used as the input file for i-PI"
        ipi_input="input.xml"
    else
        echo "No input file for i-PI provided"
        exit
    fi
fi
echo "using '${ipi_input}' as input file for i-PI"
# set the correct hostname
echo "Creating a copy of the input file '${ipi_input}' to 'START.xml'"
cp ${ipi_input} START.xml
ipi_input="START.xml"

###################################################################

# Host Name
sed -i "s/<address>.*<\/address>/<address>${HOST}<\/address>/g" ${ipi_input}
sed -i "s/<port>.*<\/port>/<port>${port}<\/port>/g" ${ipi_input}
echo "Set the correct address for the driver in '${ipi_input}': ${HOST}"

###################################################################
# i-PI
IPI_COMMAND="python -u ${IPI_PATH}/i-pi ${ipi_input} &> ${ipi_radix}.out &"
echo "Running i-PI with the following command:"
echo "${IPI_COMMAND}"
eval "${IPI_COMMAND}"

# Get the PID of the last background command
IPI_PID=$!

# Sleep for a specified amount of time
sleep_sec="10"
sleep_cmd="sleep ${sleep_sec}"
echo
echo "${sleep_cmd}"
eval "${sleep_cmd}"

check_process_running $IPI_PID "i-PI"

###################################################################
# create-mace-model.py -m "../MACE-ni=2-max_ell=2.n=10.model" -d cuda -o ../LiNbO3.PES.cuda.pickle

# PES
script="macelia.py"
model_type="MACE"
model="../../../MACE-ni=2-max_ell=2.n=10.model"
device="cuda"
args="-s start.extxyz"
args="${args} -a ${HOST}  -u false -p ${port}"
args="${args} -mt ${model_type} -m ${model}"
args="${args} -d ${device} -dt float32 -sc ase"
NN_COMMAND="${script} ${args} &> ${PES_OUTPUT_FILE} &"
echo "${NN_COMMAND}"
eval "${NN_COMMAND}"

# Get the PID of the PES process
PES_PID=$!
echo "PES ID: ${PES_PID}"

# # Dipole
# script="macelia.py"
# model_type="eslib"
# model="../../../LiNbO3.dipole.pickle"
# device="cuda"
# args="-s start.extxyz"
# args="${args} -a ${HOST}  -u false -p 20603"
# args="${args} -mt ${model_type} -m ${model}"
# args="${args} -d ${device} -dt float64 -sc eslib"
# args="${args} -sp ['delta-dipole','baseline','atomic_dipoles','delta-atomic_dipoles','baseline-atomic_dipoles','BEC','BECx','BECy','BECz']"
# NN_COMMAND="${script} ${args} &> ${DIPOLE_OUTPUT_FILE} &"
# echo "${NN_COMMAND}"
# eval "${NN_COMMAND}"

# DIP_PID=$!
# echo "Dipole ID: ${DIP_PID}"


###################################################################
# Sleep to let PES and Dipole processes start and check their status
# Sleep for a specified amount of time
sleep_sec="20"
sleep_cmd="sleep ${sleep_sec}"
echo
echo "${sleep_cmd}"
eval "${sleep_cmd}"

###################################################################
# Check if each process is still running
check_process_running $IPI_PID "i-PI"   "$PES_PID" # $DIP_PID"
check_process_running $PES_PID "PES"    "$IPI_PID" # $DIP_PID"
# check_process_running $DIP_PID "Dipole" "$IPI_PID $PES_PID"

###################################################################
# Wait for all processes to complete
echo "All processes are running. Waiting for completion..."
wait


###################################################################
# save the calculation number and create the zip file
echo "Updating '${FILES_DIR}/count.txt'"
cfile="${FILES_DIR}/count.txt"
echo '1' >> ${cfile}
l=$(wc -l < ${cfile})

###################################################################
# remove temporary files
cmd="rm \#*"
echo "${cmd}"
eval "${cmd}"

if [ -e "EXIT" ]; then
  cmd="rm EXIT"
  echo "${cmd}"
  eval "${cmd}"
fi

if [ -e "input_tmp.in" ]; then
  cmd="rm input_tmp.in"
  echo "${cmd}"
  eval "${cmd}"
fi

echo
echo "Job done :)"
echo

echo
echo "Date: $(date +'%Y-%m-%d')"
echo "Time: $(date +'%H:%M:%S')"
echo

