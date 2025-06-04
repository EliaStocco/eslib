rm slurm/*
for EFIELDS in  0.10 ; do
 # 0.02 0.05 0.10 0.15
    for n in {0..3..1}; do
    
        cd E=${EFIELDS}/run-${n}
        submit.py -n 12 -e 301650
        # -n 10
        # -n 8
        cd ../..
        sleep 2
        # rm ada.sh
        # break
    done
    # break
done
