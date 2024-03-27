FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

MAINTAINER Stepan Kudin <sskudin@sberbank.ru>

COPY requirements.txt /tmp/requirements.txt

ENV USER=docker
ENV GROUP=docker
ENV WORKDIR=/app
ENV PYTHONPATH=$WORKDIR:$PYTHONPATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Etc/UTC

RUN mkdir ${WORKDIR}
WORKDIR ${WORKDIR}

RUN rm /etc/apt/sources.list.d/cuda.list || true
RUN rm /etc/apt/sources.list.d/nvidia-ml.list || true
RUN apt-key del 7fa2af80 || true
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update --fix-missing
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-utils \
        curl \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        python3 \
        python3-pip \
        python3-setuptools

RUN pip3 install -U pip
RUN python3 -m pip install -r /tmp/requirements.txt

RUN addgroup --gid 1000 ${GROUP} && \
    adduser --uid 1000 --ingroup ${GROUP} --home /home/${USER} --shell /bin/sh --disabled-password --gecos "" ${USER}

RUN USER=${USER} && \
    GROUP=${GROUP} && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.6.0/fixuid-0.6.0-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: ${USER}\ngroup: ${GROUP}\n" > /etc/fixuid/config.yml

USER ${USER}:${GROUP}

ENTRYPOINT ["fixuid"]
