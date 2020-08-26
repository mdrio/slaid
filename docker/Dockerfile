FROM pyecvl-gpu:0.4.2
COPY . /tmp/slaid
WORKDIR /tmp/slaid
RUN useradd -ms  /bin/bash slaid && \
    chown slaid /tmp/slaid && \
    apt update && apt install -y openslide-tools && \
    pip uninstall -y setuptools && \
    pip install setuptools==45
USER slaid
RUN  pip install  --user -e .
ENV PATH="/home/slaid/.local/bin:${PATH}"



