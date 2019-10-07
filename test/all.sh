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

version=$(python get_version.py)

PASS=0
FAIL=1

expect $PASS "Successfully installed tensorflow-determinism" \
             ./install_package.sh

expect $FAIL "Exception: tfdeterminism: TensorFlow inside NGC containers does not require patching" \
             ./container.sh nvcr.io/nvidia/tensorflow:19.09-py2 test_patch_apply.sh

expect $FAIL "Exception: tfdeterminism: No patch available for version 1.13.1 of TensorFlow" \
             ./container.sh tensorflow/tensorflow:1.13.1-gpu test_patch_apply.sh

expect $PASS "TensorFlow version 1.14.0 has been patched using tfdeterminism version ${version}"  \
             ./container.sh tensorflow/tensorflow:1.14.0-gpu test_patch_apply.sh

expect $PASS "" \
             ./container.sh tensorflow/tensorflow:1.14.0-gpu test_patch.sh

expect $PASS "" \
             ./container.sh tensorflow/tensorflow:1.14.0-gpu-py3 test_patch.sh

expect $PASS "" \
             ./container.sh tensorflow/tensorflow:1.15.0rc2-gpu test_patch.sh

expect $FAIL "Exception: tfdeterminism: No patch available for version 2.0.0 of TensorFlow" \
             ./container.sh tensorflow/tensorflow:2.0.0-gpu test_patch.sh

echo "${PASS_COUNT} tests passed"
echo "${FAIL_COUNT} tests failed"

if [ $FAIL_COUNT -gt 0 ]; then
    echo "ERROR: NOT ALL TESTS PASSED"
    exit 1
else
    echo "SUCCESS: ALL TESTS PASSED"
    exit 0
fi
