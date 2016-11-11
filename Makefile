HOST=127.0.0.1
TEST_PATH=./
VERSION := $(shell python version.py)
$( eval VERSION := $( shell python version.py ))

.PHONY: clean-pyc clean-build tag docker-build docker-push

clean-pyc:
		find . -name '__pycache__' -type d -exec rm -r {} +
		find . -name '*.pyc' -exec rm --force {} +
		find . -name '*.pyo' -exec rm --force {} +
		find . -name '*~' -exec rm --force  {} +

clean-build:
		rm --force --recursive build/
		rm --force --recursive dist/
		rm --force --recursive *.egg-info

tag:
		git tag -a $(VERSION) -m "Version ${VERSION}"
		git push origin $(VERSION)
		git push upstream $(VERSION)

lint:
		pylint ./mriqc/

test: clean-pyc
		py.test --verbose $(TEST_PATH)

dist: clean-build clean-pyc
		python setup.py sdist

docker-build: tag
		docker build \
			-f ./docker/Dockerfile_py27 \
			-t poldracklab/mriqc:$(VERSION)-python27 .
		docker build \
			-f ./docker/Dockerfile_py35 \
			-t poldracklab/mriqc:$(VERSION)-python35 .
		docker build \
			-f ./docker/Dockerfile_py27 \
			-t poldracklab/mriqc:latest .

docker: docker-build
		docker push poldracklab/mriqc:$(VERSION)-python27
		docker push poldracklab/mriqc:latest
		docker push poldracklab/mriqc:$(VERSION)-python35
