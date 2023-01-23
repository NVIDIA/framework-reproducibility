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
# ==============================================================================
"""Functional tests for segment reduction ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import warnings

import numpy as np
import tensorflow as tf

from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segment_reduction_helper import SegmentReductionHelper

sys.path.insert(0, '..')
import fwrepro.tensorflow as fwrepro_tensorflow
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Simplifies logging

# The tests in the following class were originally copied from
# https://github.com/tensorflow/tensorflow/blob/1e9b9b1568d550e6779d2ddd5d193968254d3029/tensorflow/python/kernel_tests/segment_reduction_ops_test.py
# and were then enhanced.

# NOTE: Op `gen_math_ops.segment_sum` has GPU kernels for the following data
# types float16/32/64. The dynamic patch adopts a "super-accumulator" approach
# which does the operation in higher precision with necessary pre-conversion
# and post-conversion. Also note that integer operation generally has no issue
# with the non-associativity of floating-point rounding errors. Therefore the
# patch will not provide determinism for float64 or integer operands. For
# bfloat16, no GPU kernel is available for TF version less than(and equal to)
# 2.3. But it is likely that the patched ops will operate, in any given
# configuration, faster using float32 on GPU than using bfloat16 on a CPU.
# Therefore, we demonstrate a proof-of-concept for rapidly providing accelerated
# GPU support in frameworks for new data formats before they are implemented
# natively in hardware.

# Upstream class name: SegmentReductionOpTest
class SegmentSumTest(SegmentReductionHelper):

  def testValues(self):
    dtypes = [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int64,
        dtypes_lib.int32, dtypes_lib.complex64, dtypes_lib.complex128
    ]

    # Each item is np_op1, np_op2, tf_op
    ops_list = [(np.add, None, math_ops.segment_sum)]

    # A subset of ops has been enabled for complex numbers
    complex_ops_list = [(np.add, None, math_ops.segment_sum)]

    n = 10
    shape = [n, 2]
    indices = [i // 3 for i in range(n)]
    for dtype in dtypes:
      if dtype in (dtypes_lib.complex64, dtypes_lib.complex128):
        curr_ops_list = complex_ops_list
      else:
        curr_ops_list = ops_list
      for use_gpu in [True, False]:
        with self.cached_session(use_gpu=use_gpu):
          tf_x, np_x = self._input(shape, dtype=dtype)
          for np_op1, np_op2, tf_op in curr_ops_list:
            np_ans = self._segmentReduce(indices, np_x, np_op1, np_op2)
            s = tf_op(data=tf_x, segment_ids=indices)
            tf_ans = self.evaluate(s)
            self.assertAllClose(np_ans, tf_ans)
            # NOTE(mrry): The static shape inference that computes
            # `tf_ans.shape` can only infer that sizes from dimension 1
            # onwards, because the size of dimension 0 is data-dependent
            # and may therefore vary dynamically.
            self.assertAllEqual(np_ans.shape[1:], tf_ans.shape[1:])

  @test_util.run_deprecated_v1
  def testSegmentIdsShape(self):
    shape = [4, 4]
    tf_x, _ = self._input(shape)
    indices = constant_op.constant([0, 1, 2, 2], shape=[2, 2])
    with self.assertRaises(ValueError):
      math_ops.segment_sum(data=tf_x, segment_ids=indices)

  @test_util.run_deprecated_v1
  def testSegmentIdsSize(self):
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape)
        indices = [0, 1]
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        with self.assertRaisesOpError("segment_ids should be the same size"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsValid(self):
    # This is a baseline for the following SegmentIdsInvalid* tests.
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 0, 1]
        result = math_ops.segment_sum(data=tf_x, segment_ids=indices).eval()
        self.assertAllEqual([[15, 18, 21, 24], [13, 14, 15, 16]], result)

  def testSegmentIdsGreaterThanZero(self):
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float32)
        indices = [1, 1, 2, 2]
        np_ans = self._segmentReduce(indices, np_x, np.add)
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  def testSegmentIdsHole(self):
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 3, 3]
        np_ans = self._segmentReduce(indices, np_x, np.add)
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid1(self):
    shape = [4, 4]
    with self.cached_session():
      tf_x, _ = self._input(shape)
      indices = [-1, -1, 0, 0]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError(
          r"Segment id -1 out of range \[0, 1\), possibly because "
          "'segment_ids' input is not sorted."):
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid2(self):
    shape = [4, 4]
    with self.cached_session():
      tf_x, _ = self._input(shape)
      indices = [0, 1, 0, 1]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError("segment ids are not increasing"):
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid3(self):
    shape = [4, 4]
    with self.cached_session():
      tf_x, _ = self._input(shape)
      indices = [0, 1, 2, 0]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError(
          r"Segment id 1 out of range \[0, 1\), possibly "
          "because 'segment_ids' input is not sorted."):
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid4(self):
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 0, -1]
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid5(self):
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 0, -2]
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradient(self):
    shape = [4, 4]
    indices = [0, 1, 2, 2]
    for tf_op in [
        math_ops.segment_sum]:
      with self.cached_session():
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float64)
        s = tf_op(data=tf_x, segment_ids=indices)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s, [3, 4],
            x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n)

  # Method removed because it only tests math_ops.segment_mean
  # def testDataInvalid(self):
  #   ...


class SegmentSumDeterministicTest(SegmentReductionHelper):

  def __init__(self, methodName='runTest'):
  # Each item is np_op1, np_op2, tf_op, initial_value functor
    self.ops_list = [(np.add, None,
                      math_ops.segment_sum, lambda t: 0),
                     (np.add, None,
                      tf.math.segment_sum, lambda t: 0)]

    # A subset of ops has been enabled for complex numbers
    self.complex_ops_list = [(np.add, None,
                              math_ops.segment_sum, lambda t: 0),
                             (np.add, None,
                              tf.math.segment_sum, lambda t: 0)]

    self.differentiable_dtypes = [dtypes_lib.float16, dtypes_lib.float32]

    self.all_dtypes = (self.differentiable_dtypes + [dtypes_lib.bfloat16])
    self.repeat_count = 5
    super(SegmentSumDeterministicTest,
          self).__init__(methodName=methodName)

  def _testBackwardCase(self, dtype, indices, tf_op, shape):
    numpy_seed = 123

    input_val = self._randomDataOp(shape, dtype, seed=None)
    output_shape = [indices[-1]+1, shape[1]]
    if context.executing_eagerly():
      def op_gradients(local_seed):
        with backprop.GradientTape() as tape:
          tape.watch(input_val)
          op_output = tf_op(input_val, indices)
          upstream_gradients = self._randomDataOp(output_shape, dtype, local_seed)
          gradient_injector_output = op_output * upstream_gradients
        return tape.gradient(gradient_injector_output, input_val)

      for i in range(self.repeat_count):
        local_seed = numpy_seed + i # select different upstream gradients
        result_a = op_gradients(local_seed)
        result_b = op_gradients(local_seed)
        self.assertAllEqual(result_a, result_b)

    else:
      op_output = tf_op(input_val, indices)
      upstream_gradients = array_ops.placeholder(dtype, shape=output_shape,
                                                 name='upstream_gradients')
      gradient_injector_output = op_output * upstream_gradients
      op_gradients = gradients_impl.gradients(
            gradient_injector_output,
            input_val,
            grad_ys=None,
            colocate_gradients_with_ops=True)[0]

      for i in range(self.repeat_count):
        feed_dict = {upstream_gradients:np.random.random(output_shape)}
        result_a = op_gradients.eval(feed_dict=feed_dict)
        result_b = op_gradients.eval(feed_dict=feed_dict)
        self.assertAllEqual(result_a, result_b)

  @test_util.run_in_graph_and_eager_modes
  def testForward(self):
    num_cols = 8
    num_segments = 32
    segment_size = 256

    shape = [segment_size, num_cols]
    indices = np.random.randint(low=0, high=num_segments, size=(segment_size,))
    indices = np.sort(indices)

    with utils.force_gpu_session(self):
      for dtype in self.all_dtypes:#(dtypes_lib.complex64,)
        ops_list = self.complex_ops_list if dtype.is_complex \
            else self.ops_list
        tf_x, _ = self._random_input(shape, dtype=dtype)
        # have to use float to exec nond9m
        for _, _, tf_op, _ in ops_list:
          for _ in range(self.repeat_count):
            # pass
            result_a=tf_op(data=tf_x, segment_ids=indices)
            result_b=tf_op(data=tf_x, segment_ids=indices)
            self.assertAllEqual(result_a, result_b)

  # The backward operation is not known or expected to introduce nondeterminism
  # but we're testing it for completeness.
  @test_util.run_in_graph_and_eager_modes
  def testBackward(self):
    num_cols = 8
    num_segments = 32
    segment_size = 256
    shape = [segment_size, num_cols]
    indices = np.random.randint(low=0, high=num_segments, size=(segment_size,))
    indices = np.sort(indices)

    with utils.force_gpu_session(self):
    # with self.session(force_gpu=True):#force_gpu=True leads to XLA issue
      for dtype in self.differentiable_dtypes:
        ops_list = self.complex_ops_list if dtype.is_complex \
            else self.ops_list
        for _, _, tf_op, _ in ops_list:
          self._testBackwardCase(dtype, indices, tf_op, shape)

  # Op `gen_math_ops.segment_sum()` is not patched for data type float64 on GPU.
  # A warning will be thrown to indicate users float64 is still exposed to
  # GPU-nondeterminism.
  @test_util.run_in_graph_and_eager_modes
  def testNonSupportedDataTypes(self):
    shape = [10, 2]
    indices = [i // 3 for i in range(10)]
    non_supported_types = (dtypes_lib.float64,)
    with utils.force_gpu_session(self):
      for dtype in non_supported_types:
        ops_list = self.complex_ops_list if dtype.is_complex \
            else self.ops_list
        tf_x, _ = self._input(shape, dtype)
        for _, _, tf_op, _ in ops_list:
          with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = tf_op(data=tf_x, segment_ids=indices)
            self.evaluate(s)
            self.assertEqual(len(w), 1)
            self.assertIsInstance(w[0].message, UserWarning)
            self.assertTrue("GPU-determinism" in str(w[-1].message))

class SegmentReductionTestMisc(test.TestCase):
  def testSDocstring(self):
    op = tf.math.segment_sum
    docstring = op.__doc__

    if not docstring: # falsy (None or "")
        self.fail("The patched op %s has no docstring" % op.__name__)
    if docstring.startswith('ERROR'):
        self.fail("The docstring for the patched op %s has not been assigned"
                % op.__name__)


if __name__ == "__main__":
  fwrepro_tensorflow.enable_determinism()
  test.main()
