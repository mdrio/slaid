docker-build:
	cp setup.py docker/
	cp requirements.txt docker/
	cp -r slaid docker/
	cp -r bin docker/
	cd docker &&	docker build . -f ../Dockerfile -t slaid

clean:
	cd docker; rm setup.py
	cd docker; rm requirements.txt
	cd docker; rm -r slaid
	cd docker; rm -r bin
