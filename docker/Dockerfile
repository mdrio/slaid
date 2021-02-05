FROM  local/pylibs-toolkit:a157c06
ARG MODEL
ARG FEATURE
ENV SLAID_MODEL=$MODEL
ENV SLAID_FEATURE=$FEATURE

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

ENTRYPOINT ["classify.py"]


