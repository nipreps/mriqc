HOST=127.0.0.1
TEST_PATH=./
VERSION := $(shell python version.py)
$( eval VERSION := $( shell python version.py ))

.PHONY: clean-pyc
clean-pyc:
		find . -name '__pycache__' -type d -exec rm -r {} +
		find . -name '*.pyc' -exec rm --force {} +
		find . -name '*.pyo' -exec rm --force {} +
		find . -name '*~' -exec rm --force  {} +

.PHONY: clean-build
clean-build:
		rm --force --recursive build/
		rm --force --recursive dist/
		rm --force --recursive *.egg-info
		rm --force --recursive src/

.PHONY: tag
tag:
		git tag -a $(VERSION) -m "Version ${VERSION}"
		git push origin $(VERSION)
		git push upstream $(VERSION)

lint:
		pylint ./mriqc/

.PHONY: test
test: clean-pyc
		py.test --ignore=src/ --verbose $(TEST_PATH)

dist: clean-build clean-pyc
		python setup.py sdist

.PHONY: docker-build
docker-build:
		docker build \
			-f ./docker/Dockerfile_py27 \
			-t poldracklab/mriqc:$(VERSION)-python27 -t poldracklab/mriqc:latest .
		docker build \
			-f ./docker/Dockerfile_py35 \
			-t poldracklab/mriqc:$(VERSION)-python35 .

.PHONY: docker
docker: docker-build
		docker push poldracklab/mriqc:$(VERSION)-python27
		docker push poldracklab/mriqc:latest
		docker push poldracklab/mriqc:$(VERSION)-python35

.PHONY: release
release: clean-build tag docker
		python setup.py sdist
		twine upload dist/*

singularity: release
	mkdir -p ./build/singularity
	docker run --privileged -ti --rm  \
    	-v /var/run/docker.sock:/var/run/docker.sock \
    	-v ./build/singularity:/output \
    	filo/docker2singularity \
    	bids/mriqc:$(VERSION)
