#!/bin/bash

# Copyright 2019 The TensorFlow-Determinism Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================

if [ "$1" == "--help" ]; then
	echo "Usage:"
	echo "  To run a program in a container:"
	echo "    ${0} <docker-image> <filename-in-current-directory>"
	echo "  To run an interactive bash session in the default container:"
	echo "    ${0}"
	exit 0
fi

if [ -z ${2+present} ]; then
  IMAGE=tensorflow/tensorflow:1.14.0-gpu
  ENTRYPOINT=""
else
  IMAGE=${1}
  ENTRYPOINT="--entrypoint /mnt/test/${2}"
fi

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
