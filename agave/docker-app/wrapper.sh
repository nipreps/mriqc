#set -x

# Run the container in docker, mounting the current job directory as /scratch in the container
# Note that here the docker container image must exist for the container to run. If it was
# not built using a passed in Dockerfile, then it will be downloaded here prior to
# invocation. Also note that all output is written to the mounted directory. This allows
# Agave to stage out the data after running.

docker run -i --rm -v `pwd`:/scratch -w /scratch oesteban/mriqc -i ${bidsFolder} -o outputs -w workdir 2>> ${AGAVE_JOB_NAME}.err 1>> ${AGAVE_JOB_NAME}.out

if [ ! $? ]; then
	echo "Docker process exited with an error status." >&2
	${AGAVE_JOB_CALLBACK_FAILURE}
	exit
fi

# Good practice would suggest that you clean up your image after running. For throughput
# you may want to leave it in place. iPlant's docker servers will clean up after themselves
# using a purge policy based on size, demand, and utilization.

#sudo docker rmi $DOCKER_IMAGE
