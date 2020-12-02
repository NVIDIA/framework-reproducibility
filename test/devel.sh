#!/bin/bash

set -e # If any test fails, this script will exit and forward the error code

./container.sh nvcr.io/nvidia/tensorflow:19.06-py3 python test_patch_bias_add.py

# The segment sum patch has been shown to pass on the following NGC containers:
#   19.06-py2/3
#   19.07-py2
#   19.09-py2/3
#   19.11-tf1/2-py3
#   19.12-tf1/2-py3
#   20.01-tf1/2-py3
#   20.06-tf1/2-py3
#   20.08-tf1/2-py3
#   20.09-tf2-py3
# and the following stock TensorFlow containers:
#   ?
