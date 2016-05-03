#!/bin/bash

set -x
set -e

# declare -a MRIQC_TESTS=(
# 	"-i data/ds003_downsampled -o outputs/ms-func/out -w outputs/ms-func/work --skip-anatomical" #
# 	"-i data/ds003_downsampled -o outputs/ms-anat/out -w outputs/ms-anat/work --skip-functional --use-plugin plugin.yml --save-memory" #
# # -i data/ds003_sub-01/ -o outputs/ss-all/out -w outputs/ss-all/work
# )


# i=0
# for test_params in "${MRIQC_TESTS[@]}"; do
# 	if [ $(($i % $CIRCLE_NODE_TOTAL)) -eq $CIRCLE_NODE_INDEX ]; then
# 	    docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch oesteban/mriqc ${test_params}
# 	fi
# 	((i=i+1))
# done

if [[ $CIRCLE_NODE_INDEX -eq 0 ]]; then
	docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch oesteban/mriqc -B /scratch/data/ds003_downsampled -S $(seq -f '%02g' 01 06) -d anat -o outputs/ms-anat/out -w outputs/ms-anat/work
	docker run -i -v /etc/localtime:/etc/localtime:ro --entrypoint="/usr/bin/run_tests" oesteban/mriqc
#	docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch --entrypoint="/usr/bin/run_dfcheck" oesteban/mriqc -i /scratch/outputs/ms-func/out/funcMRIQC.csv -r /root/src/mriqc/mriqc/data/tests/ds003_downsampled_fMRI.csv
fi

if [[ $CIRCLE_NODE_INDEX -eq 1 ]]; then
	docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch oesteban/mriqc -B /scratch/data/ds003_downsampled -S $(seq -f '%02g' 07 13) -d anat -o outputs/ms-anat/out -w outputs/ms-anat/work
	docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch --entrypoint="/usr/bin/run_mriqc_plot" oesteban/mriqc -d anat -o outputs/ms-anat/out -w outputs/ms-anat/work
	docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch --entrypoint="/usr/bin/run_dfcheck" oesteban/mriqc -i /scratch/outputs/ms-anat/out/anatMRIQC.csv -r /root/src/mriqc/mriqc/data/tests/ds003_downsampled_sMRI.csv
fi