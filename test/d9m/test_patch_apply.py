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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

sys.path.insert(0, '../..')

expected_exception = None
if len(sys.argv) > 2 and sys.argv[1] == "--expected-exception":
  expected_exception_string = sys.argv[2]
  if expected_exception_string == "TypeError":
    expected_exception = TypeError

from fwr13y.d9m.tensorflow import patch
try:
  patch()
except Exception as e:
  if type(e) == expected_exception:
    print("Expected exception (%s) caught: " % expected_exception_string + str(e))
    sys.exit(0)
  else:
    print("Unexpected exception: %s" % str(e))
    sys.exit(1)

if expected_exception is not None:
  print("Expected exception (%s) didn't occur!" % expected_exception_string)
  sys.exit(1)
