#!/bin/bash

# Copyright 2020 NVIDIA Corporation. All Rights Reserved
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

# Expecting sub-scripts to complete successfully
set -e
set -o pipefail

WARNING=("WARNING: fwr13y.d9m.tensorflow.patch has been deprecated. "
         "Please use enable_determinism "
         "(which supports all versions of TensorFlow) xxx.")

echo "Testing that patch produces a deprecation warning"
if python test_patch_apply.py | tee grep "${WARNING}"; then
   echo "Expected warning produced"
else
   echo "Either expected warning NOT produced or exception"
   exit 1
fi