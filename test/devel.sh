#!/bin/bash

set -e # If any test fails, this script will exit and forward the error code

# IMAGE=tensorflow/tensorflow:2.3.0-gpu
IMAGE=nvcr.io/nvidia/tensorflow:19.06-py3

./container.sh ${IMAGE} python test_patch_bias_add.py
