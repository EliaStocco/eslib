# 0.00 0.02 0.05 0.10
for EFIELD in  0.15 0.10 0.05 0.02 ; do 
    name="E=${EFIELD}"
    mkdir -p "${name}"
    # mkdir -p "${name}/slurm"

    for n in {0..3..1} ; do
        folder="${name}/run-${n}"
        mkdir -p "${folder}"
        mkdir -p "${folder}/slurm"
        
        cp template/main.sh "${folder}/."
        # cp template/mace.sh "${folder}/."
        cp template/nvt.sh "${folder}/."
        cp template/run.py "${folder}/."
        cp template/input.xml "${folder}/input.xml"

        chmod +x ${folder}/*.sh
        chmod +x ${folder}/run.py
        
        cp start/E=${EFIELD}/start.n=${n}.extxyz "${folder}/start.extxyz"
        cp start/E=${EFIELD}/start.n=${n}.chk "${folder}/start.chk"

        seed=$RANDOM
        sed -i "s/SEED/${seed}/g" "${folder}/input.xml"
        
        sed -i "s/EFIELD/${EFIELD}/g" "${folder}/input.xml"

        HOSTNAME_PES=$RANDOM
        sed -i "s/HOSTNAME_PES/${HOSTNAME_PES}/g" "${folder}/input.xml"
        sed -i "s/HOSTNAME_PES/${HOSTNAME_PES}/g" "${folder}/nvt.sh"

        HOSTNAME_DIP=$RANDOM
        sed -i "s/HOSTNAME_DIP/${HOSTNAME_DIP}/g" "${folder}/input.xml"
        sed -i "s/HOSTNAME_DIP/${HOSTNAME_DIP}/g" "${folder}/nvt.sh"

        # break 
        
    done

done



