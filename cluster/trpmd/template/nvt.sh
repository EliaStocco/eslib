rm log.out

###################################################################
# Redefine the echo command to include a tab character
echo() {
  command echo -e "\t$@"
}

###################################################################
check_process_running() {
    local pid=$1
    local name=$2
    local other_pids=$3

    echo "DEBUG: Checking $name with PID $pid" >> log.out

    if ps -p "$pid" > /dev/null 2>&1; then
        echo "$name process with PID $pid is still running." >> log.out
    else
        echo "$name process with PID $pid has terminated. Exiting script." >> log.out
        if [ -n "$other_pids" ]; then
            for opid in $other_pids; do
                if [[ "$opid" =~ ^[0-9]+$ ]]; then
                    if ps -p "$opid" > /dev/null 2>&1; then
                        echo "Killing other process PID $opid" >> log.out
                        kill "$opid" 2>/dev/null
                    else
                        echo "Other process PID $opid is already dead" >> log.out
                    fi
                else
                    echo "Skipping invalid PID: $opid" >> log.out
                fi
            done
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
echo >> log.out
echo "Date: $(date +'%Y-%m-%d')" >> log.out
echo "Time: $(date +'%H:%M:%S')" >> log.out
echo >> log.out
echo "which python: $(which python)" >> log.out
echo "python --version: $(python --version)" >> log.out
echo "conda env: $(conda info --envs | grep '*' | awk '{print $1}')" >> log.out
echo  >> log.out

if [ -n "$SLURM_JOB_ID" ]; then
  echo "This script is running under SLURM with job ID $SLURM_JOB_ID" >> log.out
else
  echo "This script is running in a Bash shell" >> log.out
fi

###################################################################



###################################################################
PROGRAMS_DIR="/u/elsto/programs"
source ${PROGRAMS_DIR}/i-pi/env.sh
IPI_PATH="${PROGRAMS_DIR}/i-pi/bin"
ipi_radix='i-pi'
ipi_input="input.xml"

# IP address
export HOST_PES="HOSTNAME_PES"
export HOST_DIP="HOSTNAME_DIP"

###################################################################
# # create some folders
# RESULTS_DIR="./results"
# FILES_DIR="./files"
# for DIR in ${RESULTS_DIR} ${FILES_DIR} ; do
#     if test ! -d "${DIR}" ; then
#         mkdir "${DIR}"
#     fi
# done

temp="EXIT"
if [[ -f ${temp} ]] ; then
    echo "removing '${temp}'" >> log.out
    cmd="rm ${temp}"
    eval ${cmd}
fi

###################################################################
# chose the correct i-PI input file
	
if test -f "RESTART" ; then
    echo "Found RESTART file: it will be used as the input file for i-PI" >> log.out
    ipi_input="RESTART"
fi
# 'ipi_input' is not defined
if [ -z ${ipi_input+x} ] ; then
    # echo "No input file for i-PI provided"
    if test -f "input.xml" ; then
        echo "Found 'input.xml' file: it will be used as the input file for i-PI" >> log.out
        ipi_input="input.xml"
    else
        echo "No input file for i-PI provided" >> log.out
        exit
    fi
fi
echo "using '${ipi_input}' as input file for i-PI" >> log.out
# set the correct hostname
echo "Creating a copy of the input file '${ipi_input}' to 'START.xml'" >> log.out
cp ${ipi_input} START.xml
ipi_input="START.xml"

###################################################################

# Host Name
# sed -i "s/<address>.*<\/address>/<address>${HOST}<\/address>/g" ${ipi_input}
# sed -i "s/<port>PES_PORT<\/port>/<port>${PES_PORT}<\/port>/g" ${ipi_input}
# sed -i "s/<port>DIP_PORT<\/port>/<port>${DIP_PORT}<\/port>/g" ${ipi_input}
# echo "Set the correct address and port for the driver in '${ipi_input}': ${HOST}"

###################################################################
# i-PI
IPI_COMMAND="python -u ${IPI_PATH}/i-pi ${ipi_input} &> ${ipi_radix}.out &"
echo "Running i-PI with the following command:" >> log.out
echo "${IPI_COMMAND}" >> log.out
eval "${IPI_COMMAND}"

# Get the PID of the last background command
IPI_PID=$!
echo "i-PI ID: ${IPI_PID}" >> log.out

sleep_sec="10"
sleep_cmd="sleep ${sleep_sec}"
echo >> log.out
echo "${sleep_cmd}" >> log.out
eval "${sleep_cmd}"

check_process_running $IPI_PID "i-PI"

${MACE_PREFIX} run.py

# ###################################################################
# # PES
# echo "Preparing PES " >> log.out
# export OUTPUT_SCRIPT_NAME="pes"
# script="macelia-gpu.py"
# model_type="MACE"
# model="../../../../64_bulk_watter_swa.model"
# device="cuda"
# args="-s start.extxyz"
# args="${args} -a ${HOST_PES}  -u true " 
# args="${args} -mt ${model_type} -m ${model}"
# args="${args} -d ${device} -dt float32 -sc ase"

# # Declare an array to hold PIDs
# NN_COMMAND="${MACE_PREFIX} ${script} ${args} "
# echo "${NN_COMMAND}" >> log.out
# eval "${NN_COMMAND}" &

# # Get the PID of the PES process
# PES_PID=$!
# echo "PES ID: ${PES_PID}" >> log.out

# ###################################################################
# # Dipole
# echo "Preparing DIPOLE " >> log.out
# export OUTPUT_SCRIPT_NAME="dipole"
# script="macelia-gpu.py"
# model_type="eslib"
# model="../../../../model.cuda.bec.pickle"
# device="cuda"
# args="-s start.extxyz"
# args="${args} -a ${HOST_DIP}  -u true "
# args="${args} -mt ${model_type} -m ${model}"
# args="${args} -d ${device} -dt float64 -sc eslib"

# # Declare an array to hold PIDs
# NN_COMMAND="${MACE_PREFIX} ${script} ${args} "
# echo "${NN_COMMAND}" >> log.out
# eval "${NN_COMMAND}" &

# # Get the PID of the PES process
# DIP_PID=$!
# echo "Dipole ID: ${DIP_PID}" >> log.out

###################################################################
# Sleep to let PES and Dipole processes start and check their status
# Sleep for a specified amount of time
sleep_sec="20"
sleep_cmd="sleep ${sleep_sec}"
echo >> log.out
echo "${sleep_cmd}" >> log.out
eval "${sleep_cmd}"

###################################################################
# Check if each process is still running
check_process_running $IPI_PID "i-PI"   "${DIP_PID} ${PES_PID}"

###################################################################

wait

###################################################################
# remove temporary files
# cmd="rm \#*"
# echo "${cmd}" >> log.out
# eval "${cmd}"

if [ -e "EXIT" ]; then
  cmd="rm EXIT"
  echo "${cmd}" >> log.out
  eval "${cmd}"
fi

if [ -e "input_tmp.in" ]; then
  cmd="rm input_tmp.in"
  echo "${cmd}" >> log.out
  eval "${cmd}"
fi

# mv i-pi.chk_* chk/.

echo >> log.out
echo "Job done :)" >> log.out
echo >> log.out

echo  >> log.out
echo "Date: $(date +'%Y-%m-%d')" >> log.out
echo "Time: $(date +'%H:%M:%S')" >> log.out
echo >> log.out