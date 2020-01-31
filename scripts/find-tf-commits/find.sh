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

[[ -d "tensorflow" ]] || git clone git@github.com:tensorflow/tensorflow.git

cd tensorflow
git pull

IFS=$'\n'
# The initial commit to the TensorFlow repository was on 2015-11-06
# https://github.com/tensorflow/tensorflow/commit/f41959ccb2d9d4c722fe8fc3351401d53bcf4900
first="2015-11-06"
last="2020-01-30"
hashes=( $(git log --since="${first} 00:00" --until="${last} 23:59" --pretty=format:"%H") )
regex="determinis(tic|m)"
found_count=0
total_count=0
for hash in "${hashes[@]}"
do
  if [[ $(git show ${hash} | grep -i -E ${regex}) ]]; then
    date=$(git show -s --format=%ci ${hash})
    date=${date:0:10}
    echo "${date}: https://github.com/tensorflow/tensorflow/commit/${hash}"
    ((found_count++))
  fi
  ((total_count++))
done
echo "Found ${found_count} relevant commits out of a total of ${total_count} commits"
