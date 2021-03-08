FROM  mobydick.crs4.it:5000/pylibs-toolkit:a157c06

# RUN useradd -ms  /bin/bash slaid && \
#     chown slaid /tmp/slaid && \
RUN apt update && apt install -y openslide-tools ffmpeg libsm6 libxext6 && \
    pip3 uninstall -y setuptools && \
    pip3 install setuptools==45
# USER slaid

COPY . /tmp/slaid
WORKDIR /tmp/slaid
RUN pip3 install -r requirements.txt && \ 
    python3 setup.py install
# ENV PATH="/home/slaid/.local/bin:${PATH}"
# ENV PATH="/root/.local/bin:${PATH}"

ARG MODEL
ENV SLAID_MODEL=$MODEL
ENTRYPOINT ["classify.py"]


