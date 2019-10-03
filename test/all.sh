#!/bin/bash

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

PASS=0
FAIL=1

expect $FAIL "ValueError: No patch available for version 1.13.1 of TensorFlow" \
             ./container.sh tensorflow/tensorflow:1.13.1-gpu test.sh

expect $PASS "" \
             ./container.sh tensorflow/tensorflow:1.14.0-gpu test.sh
 
expect $PASS "" \
             ./container.sh tensorflow/tensorflow:1.15.0rc2-gpu test.sh

echo "${PASS_COUNT} tests passed"
echo "${FAIL_COUNT} tests failed"

if [ $FAIL_COUNT -gt 0 ]; then
    echo "ERROR: NOT ALL TESTS PASSED"
    exit 1
else
    echo "SUCCESS: ALL TESTS PASSED"
    exit 0
fi
