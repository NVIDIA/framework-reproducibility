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
"""Functional tests for deterministic max pooling gradient functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

sys.path.insert(0, '..')
from fwd9m import tensorflow as fwd9m_tensorflow
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
import utils as tests_utils

class MaxPoolingOpDeterministicTest(test.TestCase):

  def _randomNDArray(self, shape):
    return 2 * np.random.random_sample(shape) - 1

  def _randomDataOp(self, shape, data_type):
    return constant_op.constant(self._randomNDArray(shape), dtype=data_type)

  @test_util.run_in_graph_and_eager_modes
  def testDeterministicGradients(self, data_type=dtypes.float32):
    with tests_utils.force_gpu_session(self):
      seed = (hash(data_type) % 256)
      np.random.seed(seed)
      N = 100
      image_size = 30
      nchannels = 3
      ksize = [1, 3, 3, 1]
      strides = (1, 2, 2, 1)
      input_shape = (N, image_size , image_size, nchannels)  # NHWC
      output_shape = (N, 15, 15, 1)
      input_image = self._randomDataOp(input_shape, data_type)
      repeat_count = 3
      if context.executing_eagerly():
        def maxpool_gradients(local_seed):
          np.random.seed(local_seed)
          upstream_gradients = self._randomDataOp(output_shape, dtypes.float32)
          with backprop.GradientTape(persistent=True) as tape:
            tape.watch(input_image)
            output_image = nn_ops.max_pool(input_image, ksize, strides,
                                           padding="SAME")
            gradient_injector_output = output_image * upstream_gradients
          return tape.gradient(gradient_injector_output, input_image)

        for i in range(repeat_count):
          local_seed = seed + i  # select different upstream gradients
          result_a = maxpool_gradients(local_seed)
          result_b = maxpool_gradients(local_seed)
          self.assertAllEqual(result_a, result_b)
      else:  # graph mode
        upstream_gradients = array_ops.placeholder(
            data_type, shape=output_shape, name='upstream_gradients')

        output_image = nn_ops.max_pool(input_image, ksize, strides,
                                       padding="SAME")

        gradient_injector_output = output_image * upstream_gradients
        # The gradient function behaves as if grad_ys is multiplied by the op
        # gradient result, not passing the upstram gradients through the op's
        # gradient generation graph. This is the reason for using the
        # gradient injector
        maxpool_gradients = gradients_impl.gradients(
            gradient_injector_output,
            input_image,
            grad_ys=None,
            colocate_gradients_with_ops=True)[0]
        for i in range(repeat_count):
          feed_dict = {upstream_gradients: self._randomNDArray(output_shape)}
          result_a = maxpool_gradients.eval(feed_dict=feed_dict)
          result_b = maxpool_gradients.eval(feed_dict=feed_dict)
          self.assertAllEqual(result_a, result_b)


if __name__ == '__main__':
  fwd9m_tensorflow.enable_determinism()
  test.main()
