#!/bin/bash

#NOTE: These should work with Slurm HPC systems, 
 # but these specific paramters  have only been tested on 
 # Stanford's Sherlock. Some parameters may need to be 
 # adjusted for other HPCs, specifically --partition.

#SBATCH --job-name mriqc
#SBATCH --partition normal #TODO: update for your HPC

#NOTE: The --array parameter allows multiple jobs to be launched at once, 
 # and is generally recommended to efficiently run several hundred jobs 
 # at once.

# TODO: adjust the range for your dataset; 1-n%j where n is the number of 
 # participants and j is the maximum number of concurrent jobs you'd like 
 # to run. 

#SBATCH --array=1-216%50
#SBATCH --time=1:00:00 #NOTE: likely longer than generally needed 
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
# Outputs ----------------------------------
#SBATCH --output log/%x-%A-%a.out
#SBATCH --error log/%x-%A-%a.err
#SBATCH --mail-user=%u@stanford.edu #TODO: update for your email domain
#SBATCH --mail-type=ALL
# ------------------------------------------

unset PYTHONPATH

MRIQC_VERSION="24.0.0" #TODO: update if using a different version
BIDS_DIR="ds002785" #TODO: replace with path to your dataset
OUTPUT_DIR="${BIDS_DIR}/derivatives/mriqc-${MRIQC_VERSION}"

SINGULARITY_CMD="singularity run -e mriqc-${MRIQC_VERSION}.simg"

#NOTE: The first clause in this line selects a row in participants.tsv 
 # using the system generated array index variable SLURM_ARRAY_TASK_ID. 
 # This is piped to grep to isolate the subject id. The regex should 
 # work for most subject naming conventions, but may need to be modified.    

subject=$( sed -n ${SLURM_ARRAY_TASK_ID}p ${BIDS_DIR}/participants.tsv \
 | grep -oP "sub-[A-Za-z0-9_]*" ) 

echo Subject $subject

cmd="${SINGULARITY_CMD} ${BIDS_DIR} ${OUTPUT_DIR} participant \
     --participant-label $subject \
     -w $PWD/work/ \
     --omp-nthreads 8 --mem 10 \
     --verbose-reports"

echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode
