mkdir -p pos
for EFIELDS in "0.00"  ; do
    mkdir -p pos/E=${EFIELDS}
    mkdir -p pos/E=${EFIELDS}/indices
    for n in {0..7..1}; do
        folder="E=${EFIELDS}/run-${n}"

        # get-steps.py -i ${folder}/i-pi.xc.extxyz -o step.txt
        # unique-indices.py -i step.txt -o pos/E=${EFIELDS}/indices/xc.run-${n}.txt

        # get-steps.py -i ${folder}/i-pi.vc.extxyz -o step.txt
        # unique-indices.py -i step.txt -o pos/E=${EFIELDS}/indices/vc.run-${n}.txt

        # subsample.py -i ${folder}/i-pi.xc.extxyz -o pos/E=${EFIELDS}/xc.run-${n}.extxyz -n pos/E=${EFIELDS}/indices/xc.run-${n}.txt
        # subsample.py -i ${folder}/i-pi.vc.extxyz -o pos/E=${EFIELDS}/vc.run-${n}.extxyz -n pos/E=${EFIELDS}/indices/vc.run-${n}.txt
        

        # rename-field.py -i pos/E=${EFIELDS}/xc.run-${n}.extxyz -o pos/E=${EFIELDS}/xc.run-${n}.extxyz -on x_centroid -nn positions
        # rename-field.py -i pos/E=${EFIELDS}/vc.run-${n}.extxyz -o pos/E=${EFIELDS}/vc.run-${n}.extxyz -on v_centroid -nn positions

        add-charges2extxyz.py -i pos/E=${EFIELDS}/xc.run-${n}.extxyz -o pos/E=${EFIELDS}/xc.run-${n}.extxyz -c charges.json
        
    done

done