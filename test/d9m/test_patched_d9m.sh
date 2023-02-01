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

set -e # All the following tests should return a zero exit code

# One or more of the following tests should fail in containers where the
# associated deterministic functionality is not available.
python test_bias_add_d9m.py
python test_segment_sum_d9m.py
python test_unsorted_segment_sum_d9m.py
