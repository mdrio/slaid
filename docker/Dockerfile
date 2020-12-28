FROM mobydick.crs4.it:5000/pyecvl-gpu:7c5d3e6
COPY . /tmp/slaid
WORKDIR /tmp/slaid
RUN useradd -ms  /bin/bash slaid && \
    chown slaid /tmp/slaid && \
    apt update && apt install -y openslide-tools ffmpeg libsm6 libxext6 && \
    pip uninstall -y setuptools && \
    pip install setuptools==45
USER slaid
RUN pip install --use-feature=2020-resolver --user -e . 
ENV PATH="/home/slaid/.local/bin:${PATH}"



