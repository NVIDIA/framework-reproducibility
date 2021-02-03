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
      FAIL_COUNT=$((FAIL_COUNT+1))
    else
      PASS_COUNT=$((PASS_COUNT+1))
    fi
  else
    echo "ERROR: Unexpected exit code"
    echo "Actual  : ${exit_code}"
    echo "Expected: ${exp_exit_code}"
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
  return $exit_code
}

VERSION=$(python get_version.py)
DISTRIBUTION_NAME=framework-determinism
PACKAGE_NAME=fwd9m

OK=0
ERROR=1


# expect $OK    "" \
#               ./container.sh nvcr.io/nvidia/tensorflow:19.10-py3 ./test_patch_deprecation_message.sh --expected-exception TypeError

# expect $OK    "Successfully installed ${DISTRIBUTION_NAME}" \
#               ./install_package.sh

# Testing the patch

# expect $OK    "Expected exception (TypeError) caught: ${PACKAGE_NAME}.tensorflow.patch: TensorFlow inside NGC containers does not require patching" \
#               ./container.sh nvcr.io/nvidia/tensorflow:19.09-py2 python test_patch_apply.py --expected-exception TypeError

# expect $OK    "Expected exception (TypeError) caught: ${PACKAGE_NAME}.tensorflow.patch: No patch available for version 1.13.1 of TensorFlow" \
#               ./container.sh tensorflow/tensorflow:1.13.1-gpu python test_patch_apply.py --expected-exception TypeError

# expect $OK    "TensorFlow version 1.14.0 has been patched using ${PACKAGE_NAME}.tensorflow.patch version ${VERSION}"  \
#               ./container.sh tensorflow/tensorflow:1.14.0-gpu python test_patch_apply.py


if [ $# -ne 1 ];then
   echo "Usage: $0 list-of-containers"
   exit
fi
if [ ! -f $1 ];then
   echo "the $1 is not a file"
   exit
fi

list_containers=()
pass_list=()
fail_list=()
mapfile list_containers <$1
for i in ${list_containers[@]}
do
   echo $i
   ./build-image.sh $i
   expect $OK    "" \
             ./container.sh python test_enable_determinism.py
   if [ $? -eq $OK ]; then
     pass_list+="$i\n"
   else
     fail_list+="$i\n"
   fi
done

echo "----------------------------------------------------------------------"
if [ ${#pass_list[@]} -gt 0 ];then
  echo "pass list:"
  for i in ${pass_list[@]}
  do
    echo -e "$i"
  done
  echo "${PASS_COUNT} tests passed"
  echo "----------------------------------------------------------------------"
fi

if [ ${#fail_list[@]} -gt 0 ];then
  echo "fail list:"
  for i in ${fail_list[@]}
  do
    echo -e $i;
  done
  echo "${FAIL_COUNT} tests failed"
  echo "----------------------------------------------------------------------"
fi

if [ $FAIL_COUNT -gt 0 ]; then
  echo "ERROR: NOT ALL TESTS PASSED"
  exit 1
else
  echo "SUCCESS: ALL TESTS PASSED"
  exit 0
fi
