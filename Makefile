docker-build:
	mkdir docker-build
	cp setup.py docker-build/
	cp requirements.txt docker-build/
	cp -r slaid docker-build/
	cp -r bin docker-build/
	cd docker-build &&	docker build . -f ../docker/Dockerfile -t slaid

clean:
	rm -r docker-build
