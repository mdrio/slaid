FROM python:3.6.8
RUN mkdir /opt/slide_classifier
COPY requirements.txt /opt/slide_classifier
COPY src/* opt/slide_classifier/
RUN apt update && apt install -y openslide-tools && \
    pip install -r /opt/slide_classifier/requirements.txt

ENTRYPOINT /opt/slide_classifier/runner.py

