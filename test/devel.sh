#!/bin/bash

set -e # If any test fails, this script will exit and forward the error code

IMAGE=tensorflow/tensorflow:2.3.0-gpu
#IMAGE=nvcr.io/nvidia/tensorflow:19.06-py3
#IMAGE=gitlab-master.nvidia.com:5005/dl/dgx/tensorflow:master-py3-devel

./container.sh ${IMAGE} python test_patch_softmax_xent.py
