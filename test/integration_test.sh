#!/bin/bash

set -e

# TODO: Add TF2 support
TF_VERSION=$(python -c 'import tensorflow as tf; print(tf.__version__)')
if [ "${TF_VERSION}" == "2.0.0" ]; then
  echo "Integration test does not yet support TF2"
  exit 0
fi

SUMMARY="log/integration_test.summary"

FAIL_COUNT=0
TEST_COUNT=0

run_twice() {
  local expected_match=$1
  local flags=$2
  local executable="python integration_test.py ${flags}"
  echo "Running: ${executable}; expect match=${expected_match}"
  eval ${executable}
  SUMMARY1=$(cat ${SUMMARY})
  rm ${SUMMARY}
  eval ${executable}
  SUMMARY2=$(cat ${SUMMARY})
  rm ${SUMMARY}
  MATCH=false
  if [ "${SUMMARY1}" = "${SUMMARY2}" ]; then
    echo "match=true"
    MATCH=true
  else
    echo "match=false"
  fi
  if [ "${MATCH}" != "${expected_match}" ]; then
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
  TEST_COUNT=$((TEST_COUNT+1))
}

run_twice false "-s"
run_twice false "-s -l"
run_twice true ""
run_twice true "-l"

if [ ${FAIL_COUNT} -gt 0 ]; then
  echo "${FAIL_COUNT} of ${TEST_COUNT} integration tests did not run as expected"
  exit 1
else
  echo "All ${TEST_COUNT} integration tests ran as expected"
  exit 0
fi