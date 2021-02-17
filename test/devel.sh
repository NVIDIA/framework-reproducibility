#!/bin/bash

set -e # If any test fails, this script will exit and forward the error code

./container.sh tensorflow/tensorflow:2.4.0-gpu python test_patch_softmax_xent.py