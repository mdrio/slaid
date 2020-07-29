docker-build:
	cp setup.py docker/
	cp requirements.txt docker/
	cp -r slaid docker/
	cp -r bin docker/
	cd docker &&	docker build . -f ../Dockerfile -t slaid
