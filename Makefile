VERSION := $(shell cat VERSION)
docker: test docker-main docker-per-model

docker-main:
	mkdir -p docker-build
	cp setup.py docker-build/
	cp requirements.txt docker-build/
	cp -r slaid docker-build/
	cp -r bin docker-build/
	cd docker-build &&	docker build . -f ../docker/Dockerfile -t slaid:$(VERSION)


install:
	pip install -e .

test: install
	cd tests; ./run_all_tests.sh

docker-per-model:
	mkdir -p docker-build
	cd docker; ./build_docker_per_model.py  -v $(VERSION)
clean:
	rm -r docker-build
