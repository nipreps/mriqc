# Basic actions

TEST_PATH = ./
MRIQC_VERSION = latest


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

.PHONY: docker
docker:
		docker build -t nipreps/mriqc:$(MRIQC_VERSION) \
			--build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
			--build-arg VCS_REF=`git rev-parse --short HEAD` \
			--build-arg VERSION=`python setup.py --version` .

.PHONY: release
release: clean-build tag docker
		python setup.py sdist
		twine upload dist/*

singularity: docker
	mkdir -p build/singularity
	docker run --privileged -ti --rm  \
    	-v /var/run/docker.sock:/var/run/docker.sock \
    	-v $(shell pwd)/build/singularity:/output \
    	singularityware/docker2singularity \
    	nipreps/mriqc:$(MRIQC_VERSION)

