#!/bin/bash
#
#SBATCH -J mriqc
#SBATCH --array=1-13
#SBATCH --time=136:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
# Outputs ----------------------------------
#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=%u@gmail.com
#SBATCH --mail-type=ALL
# ------------------------------------------

unset PYTHONPATH

BIDS_DIR="$STUDY/ds000030"
OUTPUT_DIR="${BIDS_DIR}/derivatives/mriqc-0.16.1"

SINGULARITY_CMD="singularity run -e $STUDY/mriqc-0.16.1.simg"

subject=$( sed -n -E \
    "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub (\S*)\>.*/\1/gp" \
    ${BIDS_DIR}/participants.tsv )

cmd="${SINGULARITY_CMD} ${BIDS_DIR} ${OUTPUT_DIR} participant \
     --participant-label $subject \
     -w $L_SCRATCH/work/ \
     --omp-nthreads 8 --mem 10"
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode
