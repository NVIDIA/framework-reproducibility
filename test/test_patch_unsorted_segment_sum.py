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

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np
import os

import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test

import random
#@test_util.run_all_in_graph_and_eager_modes
class UnsortedSegSumTestDeterministic(test.TestCase):
  def _testDeterministicUnsortedSegmentSumCase(self, op_binding, data_type, sorted):
    seed = (hash(data_type) % 256)

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    num_segments = 64
    data = tf.random.normal([1024, 8, 8], dtype = data_type)
    data = tf.constant(data)
    # Note that the size of the segments vector should be equal to the size
    # of the first dimension of the data to be processed by segment_sum
    # see https://www.tensorflow.org/api_docs/python/tf/math#Segmentation
    segments_size = data.shape[0]
    segment_ids = np.random.randint(low=0, high=num_segments, size=segments_size)

    args = (data, segment_ids, num_segments)
    repeat_count = 5
    if sorted: 
      print('this a a sorted case')
      segments = np.sort(segments)
      args = (data, segments)
    op = op_binding
    
    
    if not context.executing_eagerly():
      data = ops.convert_to_tensor(data, name="input_data")
      segment_ids = ops.convert_to_tensor(segment_ids, name="segment_ids")
      num_segments = ops.convert_to_tensor(num_segments, name="num_segments")

    runs = []
    for i in range(repeat_count):
      segment_sums = op(*args)
      something = sum(map(lambda x: x.numpy().sum(), segment_sums))
      runs.append(something)
      print("%d, %.30f, %s, %s" % (i, something, op_binding.__name__, data_type))
    
    for i in range(1, repeat_count):
      self.assertAllEqual(runs[i], runs[0])


#  @test_util.run_in_graph_and_eager_modes
  @test_util.run_cuda_only
  def test_unsorted_segment_reduction(self):
    with self.session(force_gpu=True):
      for op_binding in (math_ops.unsorted_segment_sum,tf.math.unsorted_segment_sum):
        for data_type in (dtypes.float16, dtypes.float32): # float64 is still nondeterministic
          for sorted in [False]:
            self._testDeterministicUnsortedSegmentSumCase(op_binding, data_type, sorted)

if __name__ == "__main__":
  import sys
  sys.path.insert(0, '..')
  from fwd9m.tensorflow import patch
  patch()
  test.main()
