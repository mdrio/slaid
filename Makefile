TAG := $(shell cd docker; ./get_docker_tag.sh)

ifndef skip_test
	extra_dep_docker:= test

else
	extra_dep_docker :=
endif

docker: $(extra_dep_docker) docker-main docker-per-model

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
	touch install

test: install
	tests/run_all_tests.sh
	touch test

docker-per-model:
	mkdir -p docker-build
	cd docker; ./docker_cmd.py -v $(TAG) $(DOCKER_ARGS) build
clean:
	rm -f install
	rm -f test
	rm -rf docker-build

docker-push: docker
	cd docker/; ./docker-push.sh $(DOCKER_ARGS)
