FROM python:3.6.8
COPY . /tmp/slaid
WORKDIR /tmp/slaid
RUN useradd -ms  /bin/bash slaid && \
      chown slaid /tmp/slaid && \
      apt update && apt install -y openslide-tools
USER slaid
ENV PATH="/home/slaid/.local/bin:${PATH}"
RUN  pip install  --user -e . && \
  cp bin/extract_tissue.py  /home/slaid/.local/bin # FIXME




