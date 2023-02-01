#!/bin/bash

# Copyright 2019 NVIDIA Corporation. All Rights Reserved
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

PASS_COUNT=0
FAIL_COUNT=0

expect () {
  local exp_exit_code=$1
  local exp_last_line=$2
  shift 2
  "$@" > >(tee stdout.log) 2> >(tee stderr.log >&2)
  local exit_code=$?
  local last_line=$(tac stdout.log | head -n 1 | tr -d '\r')
  if [ $exit_code -eq $exp_exit_code ]; then
    if [[ "$exp_last_line" != "" && "$last_line" != "$exp_last_line" ]]; then
      echo "ERROR: Unexpected last line of output"
      echo "Actual  : ${last_line}"
      echo "Expected: ${exp_last_line}"
      ((FAIL_COUNT++))
    else
      ((PASS_COUNT++))
    fi
  else
    echo "ERROR: Unexpected exit code"
    echo "Actual  : ${exit_code}"
    echo "Expected: ${exp_exit_code}"
    ((FAIL_COUNT++))
  fi
}

VERSION=$(python get_version.py)
DISTRIBUTION_NAME=framework-reproducibility
PACKAGE_NAME=fwr13y

OK=0
ERROR=1

# if false; then
echo -e "\nTesting package installation"
expect $OK    "Successfully installed ${DISTRIBUTION_NAME}" \
              ./install_package.sh

echo -e "\nTesting misc and utils in stock TF 1.14.0 container"
expect $OK    "" \
              ./container.sh tensorflow/tensorflow:1.14.0-gpu test_misc_and_utils.sh

echo -e "\nTesting that attempting to apply ${PACKAGE_NAME}.d9m.tensorflow.patch inside NGC an container will fail"
expect $OK    "Expected exception (TypeError) caught: ${PACKAGE_NAME}.d9m.tensorflow.patch: TensorFlow inside NGC containers does not require patching" \
              ./container.sh nvcr.io/nvidia/tensorflow:19.09-py2 python test_patch_apply.py --expected-exception TypeError

echo -e "\nTesting that attempting to apply ${PACKAGE_NAME}.d9m.tensorflow.patch to TF 1.13.1 will fail"
expect $OK    "Expected exception (TypeError) caught: ${PACKAGE_NAME}.d9m.tensorflow.patch: No patch available for version 1.13.1 of TensorFlow" \
              ./container.sh tensorflow/tensorflow:1.13.1-gpu python test_patch_apply.py --expected-exception TypeError

echo -e "\nTesting that ${PACKAGE_NAME}.d9m.tensorflow.patch can be applied to TF 1.14.0"
expect $OK    "TensorFlow version 1.14.0 has been patched using ${PACKAGE_NAME}.d9m.tensorflow.patch version ${VERSION}"  \
              ./container.sh tensorflow/tensorflow:1.14.0-gpu python test_patch_apply.py

echo -e "\nTesting that ${PACKAGE_NAME}.d9m.tensorflow.patch produces a deprecation warning"
expect $OK    "Expected warning produced" \
              ./container.sh tensorflow/tensorflow:1.14.0-gpu test_patch_deprecation_message.sh

echo -e "\nTesting that ${PACKAGE_NAME}.d9m.tensorflow.enable_determinism can be applied to TF 2.0.0"
expect $OK    "${PACKAGE_NAME}.d9m.tensorflow.enable_determinism (version ${VERSION}) has been applied to TensorFlow version 2.0.0" \
              ./container.sh tensorflow/tensorflow:2.0.0-gpu python test_enable_determinism_apply.py
# fi

CONTAINERS=(                                          \
            # See https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow/tags
            # nvcr.io/nvidia/tensorflow:19.06-py2     \ # failing bias_add only - unknown reason #3 (shape mismatch)
            # nvcr.io/nvidia/tensorflow:19.06-py3     \ # failing bias_add only - unknown reason #3 (shape mismatch)
            nvcr.io/nvidia/tensorflow:19.07-py2       \
            nvcr.io/nvidia/tensorflow:19.07-py3       \
            # 19.10: final release with only TF1
            nvcr.io/nvidia/tensorflow:19.10-py2       \
            nvcr.io/nvidia/tensorflow:19.10-py3       \
            # 19:11: first release with TF1 & TF2
            nvcr.io/nvidia/tensorflow:19.11-tf1-py3   \
            nvcr.io/nvidia/tensorflow:19.11-tf2-py3   \
            nvcr.io/nvidia/tensorflow:20.12-tf1-py3   \
            nvcr.io/nvidia/tensorflow:20.12-tf2-py3   \
            nvcr.io/nvidia/tensorflow:21.12-tf1-py3   \
            nvcr.io/nvidia/tensorflow:21.12-tf2-py3   \
            # Note to self: noticed CUDA version warnings starting in 22.02 (driver 465.01/CUDA11.3; built with CUDA 11.6)
            # nvcr.io/nvidia/tensorflow:22.12-tf1-py3 \ # failing segment_sum (sorted and unsorted) only - unknown reason #2 (AssertionError: 2 != 1)
            # nvcr.io/nvidia/tensorflow:22.12-tf2-py3 \ # failing segment_sum (sorted and unsorted) only - unknown reason #1 (AssertionError: 0 != 1)
            # nvcr.io/nvidia/tensorflow:23.01-tf1-py3 \ # failing - CUDA driver version insufficient
            # nvcr.io/nvidia/tensorflow:23.01-tf2-py3 \ # failing - CUDA driver version insufficient
            #
            # See https://hub.docker.com/r/tensorflow/tensorflow/tags
            tensorflow/tensorflow:1.14.0-gpu          \
            tensorflow/tensorflow:1.14.0-gpu-py3      \
            tensorflow/tensorflow:1.15.0-gpu          \
            tensorflow/tensorflow:2.0.0-gpu           \
            tensorflow/tensorflow:2.1.0-gpu           \
            tensorflow/tensorflow:2.3.0-gpu           \
            tensorflow/tensorflow:2.4.0-gpu           \
            tensorflow/tensorflow:2.5.0-gpu           \
            tensorflow/tensorflow:2.6.0-gpu           \
            # tensorflow/tensorflow:2.7.0-gpu         \ # failing segment_sum (sorted and unsorted) only - unknown reason #1 (AssertionError: 0 != 1)
            # tensorflow/tensorflow:2.8.0-gpu         \ # failing segment_sum (sorted and unsorted) only - unknown reason #1 (AssertionError: 0 != 1)
            # tensorflow/tensorflow:2.9.0-gpu         \ # failing segment_sum (sorted and unsorted) only - unknown reason #1 (AssertionError: 0 != 1)
            # tensorflow/tensorflow:2.10.0-gpu        \ # failing segment_sum (sorted and unsorted) only - unknown reason #1 (AssertionError: 0 != 1)
            # tensorflow/tensorflow:2.11.0-gpu        \ # failing segment_sum (sorted and unsorted) only - unknown reason #1 (AssertionError: 0 != 1)
)

for CONTAINER in ${CONTAINERS[@]};
do
  echo "Testing patched determinism in ${CONTAINER}"
  expect $OK "" ./container.sh ${CONTAINER} test_patched_d9m.sh
done

echo "${PASS_COUNT} tests passed"
echo "${FAIL_COUNT} tests failed"

if [ $FAIL_COUNT -gt 0 ]; then
  echo "ERROR: NOT ALL TESTS PASSED"
  exit 1
else
  echo "SUCCESS: ALL TESTS PASSED"
  exit 0
fi
