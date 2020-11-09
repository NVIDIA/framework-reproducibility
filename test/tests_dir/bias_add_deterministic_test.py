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

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test

sys.path.insert(0, '..')
import fwd9m.tensorflow as fwd9m_tensorflow
import utils

class BiasAddTestDeterministic(test.TestCase):

  def _makeShapeTuple(self, batch_size, channel_count, data_rank, data_dim,
                        data_layout):
    data_dims = data_rank * (data_dim,)
    if data_layout == 'channels_first':
      shape = (batch_size,) + (channel_count,) + data_dims
    elif data_layout == 'channels_last':
      shape = (batch_size,) + data_dims + (channel_count,)
    else:
      raise ValueError('Unknown data format')
    return shape

  def _dataFormatFromDataLayout(self, data_layout=None):
    if data_layout == 'channels_first':
      return 'NCHW'
    elif data_layout == 'channels_last':
      return 'NHWC'
    else:
      raise ValueError('Unknown data_layout')

  def _randomNDArray(self, shape):
    return 2 * np.random.random_sample(shape) - 1

  def _randomDataOp(self, shape, data_type):
    return constant_op.constant(self._randomNDArray(shape), dtype=data_type)

  def _testDeterministicGradientsCase(self, op_binding, data_layout, data_rank,
                                      data_type):
    seed = (
        hash(data_layout) % 256 + hash(data_rank) % 256 + hash(data_type) % 256)
    np.random.seed(seed)
    batch_size = 10
    channel_count = 8
    data_dim = 14
    input_shape = self._makeShapeTuple(batch_size, channel_count, data_rank,
                                       data_dim, data_layout)
    bias_shape = (channel_count,)
    output_shape = input_shape
    input_val = self._randomDataOp(input_shape, data_type)
    bias_val = self._randomDataOp(bias_shape, data_type)
    data_format = self._dataFormatFromDataLayout(data_layout)
    repeat_count = 5
    if context.executing_eagerly():

      def bias_gradients(local_seed):
        np.random.seed(local_seed)
        upstream_gradients = self._randomDataOp(output_shape, data_type)
        with backprop.GradientTape(persistent=True) as tape:
          tape.watch(bias_val)
          bias_add_output = op_binding(input_val, bias_val,
                                       data_format=data_format)
          gradient_injector_output = bias_add_output * upstream_gradients
        return tape.gradient(gradient_injector_output, bias_val)

      for i in range(repeat_count):
        local_seed = seed + i # select different upstream gradients
        result_a = bias_gradients(local_seed)
        result_b = bias_gradients(local_seed)
        self.assertAllEqual(result_a, result_b)
    else:
      upstream_gradients = array_ops.placeholder(data_type, shape=output_shape,
                                                 name='upstream_gradients')
      bias_add_output = op_binding(input_val, bias_val, data_format=data_format)
      gradient_injector_output = bias_add_output * upstream_gradients
      # The gradient function behaves as if grad_ys is multiplied by the op
      # gradient result, not passing the upstram gradients through the op's
      # gradient generation graph. This is the reason for using the
      # gradient injector
      bias_gradients = gradients_impl.gradients(
          gradient_injector_output,
          bias_val,
          grad_ys=None,
          colocate_gradients_with_ops=True)[0]
      for i in range(repeat_count):
        feed_dict = {upstream_gradients: self._randomNDArray(output_shape)}
        result_a = bias_gradients.eval(feed_dict=feed_dict)
        result_b = bias_gradients.eval(feed_dict=feed_dict)
        self.assertAllEqual(result_a, result_b)

  @test_util.run_in_graph_and_eager_modes
  def testDeterministicGradients(self):
    with utils.force_gpu_session(self):
      # There are problems with using force_gpu=True and cached_session with
      # both eager mode and graph mode in the same test. Using a non-cached
      # session and putting everything inside the same session context is
      # a compromise.
      for op_binding in (tf.nn.bias_add, nn.bias_add, nn_ops.bias_add):
        for data_layout in ('channels_first', 'channels_last'):
          # With the selected layer configuration, at least in TensorFlow
          # version 2.0, when data_layout='channels_last', bias_add operates
          # deterministically by default. I don't know if this is true for
          # all layer configurations. These cases are still being tested here,
          # for completeness.
          for data_rank in (1, 2, 3):
            for data_type in (dtypes.float16, dtypes.float32, dtypes.float64):
              self._testDeterministicGradientsCase(op_binding, data_layout,
                                                   data_rank, data_type)


if __name__ == "__main__":
  fwd9m_tensorflow.enable_determinism()
  test.main()
