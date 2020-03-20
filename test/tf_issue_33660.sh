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

printf "\n\n########## TensorFlow Version 2.0 (exhibits issue)\n\n"
./container.sh tensorflow/tensorflow:2.0.0-gpu tf_issue_33660_cases.sh
printf "\n\n########## TensorFlow Version 2.1 (issue resolved)\n\n"
./container.sh tensorflow/tensorflow:2.1.0-gpu tf_issue_33660_cases.sh
