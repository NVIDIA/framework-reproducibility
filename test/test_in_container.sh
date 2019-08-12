#! /bin/bash

IMAGE=tensorflow/tensorflow:1.14.0-gpu
ENTRYPOINT="--entrypoint /mnt/test/test.sh"

docker pull ${IMAGE}

docker run --runtime=nvidia -it     \
           -u $(id -u):$(id -g)     \
           -v ${PWD}/..:/mnt        \
           -w /mnt/test             \
           --shm-size=1g            \
           --ulimit memlock=-1      \
           --ulimit stack=67108864  \
           ${ENTRYPOINT}            \
           ${IMAGE} bash
