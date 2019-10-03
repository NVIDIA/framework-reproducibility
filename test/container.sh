#!/bin/bash

IMAGE=${1}
ENTRYPOINT="--entrypoint /mnt/test/${2}"

docker pull ${IMAGE}

ENV_VARS="-e NVIDIA_VISIBLE_DEVICES=1" # Work around XLA issue in TF 1.15

docker run --runtime=nvidia -it        \
           -u $(id -u):$(id -g)        \
           -v ${PWD}/..:/mnt           \
           -w /mnt/test                \
           ${ENV_VARS}                 \
           --shm-size=1g               \
           --ulimit memlock=-1         \
           --ulimit stack=67108864     \
           ${ENTRYPOINT}               \
           ${IMAGE} bash
