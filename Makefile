TAG := $(shell cd docker; ./get_docker_tag.sh)
docker: test docker-main docker-per-model

docker-main:
	mkdir -p docker-build
	cp setup.py docker-build/
	cp VERSION docker-build/
	cp requirements.txt docker-build/
	cp -r slaid docker-build/
	cp -r bin docker-build/
	cd docker-build &&	docker build . -f ../docker/Dockerfile -t slaid:$(TAG)
	docker tag slaid:$(TAG) slaid
	tests/docker/test_docker.sh

install:
	pip install -e .

test: install
	tests/run_all_tests.sh

docker-per-model:
	mkdir -p docker-build
	cd docker; ./docker_cmd.py -v $(TAG) build
clean:
	rm -r docker-build

docker-push: docker
	cd docker/; ./docker-push.sh 
