#!/bin/bash

# Copyright 2019-2021 NVIDIA Corporation. All Rights Reserved
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

# set -x

if [ "$1" == "--help" ]; then
  echo "Usage:"
  echo
  echo "  To run a program in a container:"
  echo "    ${0} <executable> <arguments>"
  echo "    <executable> may be either python, mpirun, or a file in"
  echo "      the current directory"
  echo
  echo "  To run an interactive bash session in the default container:"
  echo "    ${0}"
  exit 0
fi

DIR=${PWD##*/}
PARENT="$(dirname "${PWD}")"

IMAGE=framework-determinism/${DIR}:latest
if [[ -z ${1+present} ]]; then
  ENTRYPOINT=""
  ENTRYPOINT_ARGUMENTS=bash
else
  # It seems kind of hacky to have to list each non-local-directory script or
  # program that I want to be able to run in the container as the entrypoint
  # script. I wonder if there is a more optimal way of doing all of this.
  if [[ "${1}" == "python" || "${1}" == "bash" ]]; then
    ENTRYPOINT="--entrypoint ${1}"
  else
    # The following specifies the entrypoint relative to the current directory
    # that this script is being run from. This enables it to use an entrypoint
    # script either from the current directory or from the ../utils directory.
    ENTRYPOINT="--entrypoint /mnt/test/${1}"
  fi
  ENTRYPOINT_ARGUMENTS="${@:2}"
fi

# cuDNN API Logging: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#api-logging
# ENV_VARS="--env CUDNN_LOGINFO_DBG=1 --env CUDNN_LOGDEST_DBG=/mnt/${DIR}/cudnn.log"
ENV_VARS=""

# --network=host is a work-around to enable the internet to be reached from
# inside the container when the host is using the Cisco VPN client. It also
# happens to make the in-container networking higher performance (if you happen
# to need that) but is less optimal from a security standpoint.
# Check if this is still needed in Ubuntu 18.04
# docker pull ${IMAGE}
docker run --runtime=nvidia -it        \
           -u $(id -u):$(id -g)        \
           -v ${PWD}/..:/mnt           \
           -w /mnt/test                \
           ${ENV_VARS}                 \
           --shm-size=1g               \
           --ulimit memlock=-1         \
           --ulimit stack=67108864     \
           --network=host              \
           ${ENTRYPOINT}               \
           ${IMAGE}                    \
           ${ENTRYPOINT_ARGUMENTS}
