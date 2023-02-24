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
"""Functional tests for unsorted segment reduction ops."""
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
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segment_reduction_helper import SegmentReductionHelper

sys.path.insert(0, '../..')
import fwr13y.d9m.tensorflow as tf_determinism
import utils as local_test_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Simplifies logging

# The tests in the following class were originally copied from
# https://github.com/tensorflow/tensorflow/blob/1e9b9b1568d550e6779d2ddd5d193968254d3029/tensorflow/python/kernel_tests/segment_reduction_ops_test.py
# and were then enhanced.

# NOTE: gen_math_ops.unsorted_segment_sum has GPU kernels for the following
# data types, float16/32/64, complex64/128. The dynamic patch adopts a
# "super-accumulator" approach which does the operation in higher precision with
# necessary pre-conversion and post-conversion. Also note that integer operation
# generally has no issue with the non-associativity of floating-point rounding
# errors. Therefore the patch will not provide determinism for float64,
# complex128 or integer operands. For bfloat16, no GPU kernel is available for
# TF version less than(and equal to) 2.3. But it is likely that the patched ops
# will operate, in any given configuration, faster using float32 on GPU than
# using bfloat16 on a CPU. Therefore, we demonstrate a proof-of-concept for
# rapidly providing accelerated GPU support in frameworks for new data formats
# before they are implemented natively in hardware.

# Upstream class name: UnsortedSegmentTest
class UnsortedSegmentSumTest(SegmentReductionHelper):

  def __init__(self, methodName='runTest'):
    # Each item is np_op1, np_op2, tf_op, initial_value functor
    self.ops_list = [(np.add, None,
                      math_ops.unsorted_segment_sum, lambda t: 0)]

    # A subset of ops has been enabled for complex numbers
    self.complex_ops_list = [(np.add, None,
                              math_ops.unsorted_segment_sum, lambda t: 0)]
    self.differentiable_dtypes = [dtypes_lib.float16, dtypes_lib.float32,
                                  dtypes_lib.float64]
    self.all_dtypes = (self.differentiable_dtypes +
                       [dtypes_lib.bfloat16,
                        dtypes_lib.int64, dtypes_lib.int32,
                        dtypes_lib.complex64, dtypes_lib.complex128])
    super(UnsortedSegmentSumTest, self).__init__(methodName=methodName)

  def testValues(self):
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (2,)
      for dtype in self.all_dtypes:
        ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
        tf_x, np_x = self._input(shape, dtype=dtype)
        for use_gpu in [True, False]:
          with self.cached_session(use_gpu=True):
            for np_op1, np_op2, tf_op, init_op in ops_list:
              # sqrt_n doesn't support integers
              if (np_op2 == self._sqrt_n_reduce_op and dtype.is_integer):
                continue
              # todo(philjd): enable this test once real_div supports bfloat16
              if (np_op2 in [self._sqrt_n_reduce_op, self._mean_reduce_op] and
                  dtype == dtypes_lib.bfloat16):
                continue
              np_ans = self._segmentReduce(
                  indices, np_x, np_op1, np_op2, num_segments=num_segments,
                  initial_value=init_op(dtype))
              s = tf_op(tf_x, segment_ids=indices, num_segments=num_segments)
              tf_ans = self.evaluate(s)
              if dtype is dtypes_lib.bfloat16:
                tf_ans = tf_ans.astype(np.float32)
              self.assertAllCloseAccordingToType(np_ans, tf_ans)
              self.assertShapeEqual(np_ans, s)

  def testNumSegmentsTypes(self):
    dtypes = [dtypes_lib.int32, dtypes_lib.int64]
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (2,)
      for dtype in dtypes:
        with self.cached_session(use_gpu=True):
          tf_x, np_x = self._input(shape)
          num_segments_constant = constant_op.constant(
              num_segments, dtype=dtype)
          np_ans = self._segmentReduce(
              indices, np_x, np.add, op2=None, num_segments=num_segments)
          s = math_ops.unsorted_segment_sum(
              data=tf_x,
              segment_ids=indices,
              num_segments=num_segments_constant)
          tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, s)

  @tf_test_util.run_deprecated_v1
  def testGradientsTFGradients(self):
    num_cols = 2
    indices_flat = np.array([0, 4, 0, -1, 3, -1, 4, 7, 7, 3])
    num_segments = max(indices_flat) + 3
    for dtype in self.differentiable_dtypes:
      ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
      for indices in indices_flat, indices_flat.reshape(5, 2):
        shape = indices.shape + (num_cols,)
        # test CPU and GPU as tf.gather behaves differently on each device
        for use_gpu in [False, True]:
          with self.cached_session(use_gpu=use_gpu):
            for _, _, tf_op, _ in ops_list:
              tf_x, np_x = self._input(shape, dtype=dtype)
              s = tf_op(tf_x, indices, num_segments)
              jacob_t, jacob_n = gradient_checker.compute_gradient(
                  tf_x,
                  shape,
                  s, [num_segments, num_cols],
                  x_init_value=np_x,
                  delta=1.)
              self.assertAllCloseAccordingToType(jacob_t, jacob_n,
                                                 half_atol=1e-2)

  def _computeGradient(self, tf_op, indices, num_segments,
                       shape, num_cols, dtype):
    tf_x, np_x = self._input(shape, dtype=dtype)
    if context.executing_eagerly():
      def f(x):
        return tf_op(x, indices, num_segments)

      gradient_tape_jacob_t, jacob_n = gradient_checker_v2.compute_gradient(
          f, [tf_x], delta=1.0)
      self.assertAllClose(jacob_n, gradient_tape_jacob_t)
    else:
       with self.cached_session():
         s = tf_op(tf_x, indices, num_segments)
         jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s, [num_segments, num_cols],
            x_init_value=np_x,
            delta=1)
         self.assertAllClose(jacob_t, jacob_n)

  # This method has been enhanced to run on older versions of TensorFlow
  @tf_test_util.run_in_graph_and_eager_modes
  def testGradientsGradientTape(self):
    num_cols = 2
    indices_flat = np.array([0, 4, 0, -1, 3, -1, 4, 7, 7, 3])
    num_segments = max(indices_flat) + 3
    for dtype in self.differentiable_dtypes:
      ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
      for indices in indices_flat, indices_flat.reshape(5, 2):
        shape = indices.shape + (num_cols,)
        # test CPU and GPU as tf.gather behaves differently on each device
        # fwr13y.d9m note: the upstream test uses tf_test_util.use_gpu, which
        # seems to suffer from the same problem, and presumably does the same
        # thing, as self.session(force_gpu=true). So we replaced
        # tf_test_util.use_gpu with local_test_utils.force_gpu_session(self).
        for use_gpu in [local_test_utils.force_gpu_session(self),
                        tf_test_util.force_cpu()]:
          with use_gpu:
          # with local_test_utils.force_gpu_session(self):
            for _, _, tf_op, _ in ops_list:
              self._computeGradient(tf_op, indices, num_segments, shape,
                                    num_cols, dtype)

  # Method removed because it only tests math_ops.unsorted_segment_prod
  # def testProdGrad(self):
  #   ...

  @tf_test_util.run_deprecated_v1
  def testGradientMatchesSegmentSum(self):
    # Strategy: compute the gradient for UnsortedSegmentSum and SegmentSum
    # and compare the outputs, which should be identical.
    # NB: for this test to work, indices must be valid for SegmentSum, namely
    # it must be sorted, the indices must be contiguous, and num_segments
    # must be max(indices) + 1.
    indices = [0, 0, 1, 1, 1, 2, 3, 4, 5]
    n = len(indices)
    num_cols = 2
    shape = [n, num_cols]
    num_segments = max(indices) + 1
    for dtype in self.differentiable_dtypes:
      with self.cached_session(use_gpu=True):
        tf_x, np_x = self._input(shape, dtype=dtype)
        # Results from UnsortedSegmentSum
        unsorted_s = math_ops.unsorted_segment_sum(
            data=tf_x, segment_ids=indices, num_segments=num_segments)
        unsorted_jacob_t, unsorted_jacob_n = (
            gradient_checker.compute_gradient(tf_x, shape, unsorted_s,
                                              [num_segments, num_cols],
                                              x_init_value=np_x, delta=1))

        # Results from SegmentSum
        sorted_s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        sorted_jacob_t, sorted_jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            sorted_s, [num_segments, num_cols],
            x_init_value=np_x,
            delta=1)
      self.assertAllClose(unsorted_jacob_t, sorted_jacob_t)
      self.assertAllClose(unsorted_jacob_n, sorted_jacob_n)

  @tf_test_util.run_deprecated_v1
  def testBadIndices(self):
    # Note: GPU kernel does not return the out-of-range error needed for this
    # test, so this test is marked as cpu-only.
    # Note: With PR #13055 a negative index will be ignored silently.
    with self.session(use_gpu=False):
      for bad in [[2]], [[7]]:
        unsorted = math_ops.unsorted_segment_sum([[17]], bad, num_segments=2)
        with self.assertRaisesOpError(
            r"segment_ids\[0,0\] = %d is out of range \[0, 2\)" % bad[0][0]):
          self.evaluate(unsorted)

  @tf_test_util.run_deprecated_v1
  def testEmptySecondDimension(self):
    dtypes = [np.float16, np.float32, np.float64, np.int64, np.int32,
              np.complex64, np.complex128]
    with self.session(use_gpu=True):
      for dtype in dtypes:
        for itype in (np.int32, np.int64):
          data = np.zeros((2, 0), dtype=dtype)
          segment_ids = np.array([0, 1], dtype=itype)
          unsorted = math_ops.unsorted_segment_sum(data, segment_ids, 2)
          self.assertAllEqual(unsorted.eval(), np.zeros((2, 0), dtype=dtype))

  def testDropNegatives(self):
    # Note: the test is done by replacing segment_ids with 8 to -1
    # for index  and replace values generated by numpy with 0.
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (2,)
      for dtype in self.all_dtypes:
        with self.session(use_gpu=True):
          tf_x, np_x = self._input(shape, dtype=dtype)
          np_ans = self._segmentReduce(
              indices, np_x, np.add, op2=None, num_segments=num_segments)
          # Replace np_ans[8] with 0 for the value
          np_ans[8:] = 0
          # Replace 8 with -1 in indices
          np.place(indices, indices == 8, [-1])
          s = math_ops.unsorted_segment_sum(
              data=tf_x, segment_ids=indices, num_segments=num_segments)
          tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, s)


class UnsortedSegmentSumDeterministicTest(SegmentReductionHelper):

  def __init__(self, methodName='runTest'):
    # Each item is np_op1, np_op2, tf_op, initial_value functor
    self.ops_list = [(np.add, None,
                      math_ops.unsorted_segment_sum, lambda t: 0),
                     (np.add, None,
                      tf.math.unsorted_segment_sum, lambda t: 0)]

    # A subset of ops has been enabled for complex numbers
    self.complex_ops_list = [(np.add, None,
                              math_ops.unsorted_segment_sum, lambda t: 0),
                             (np.add, None,
                              tf.math.unsorted_segment_sum, lambda t: 0)]

    self.differentiable_dtypes = [dtypes_lib.float16, dtypes_lib.float32]

    self.all_dtypes = (self.differentiable_dtypes +
                       [dtypes_lib.complex64, dtypes_lib.bfloat16])
    self.repeat_count = 5
    super(
        UnsortedSegmentSumDeterministicTest, self).__init__(
            methodName=methodName)

  def _conditionally_skip_test(self):
    if local_test_utils.tf_version_at_least('2.7'):
      self.skipTest("Not testing this in TF 2.7 and onward")

  def _testBackwardCase(self, dtype, indices, num_segments, op_binding, shape):
    numpy_seed = 123
    _, _, tf_op, _ = op_binding

    input_val = self._randomDataOp(shape, dtype, seed=None)

    if context.executing_eagerly():
      def op_gradients(local_seed):
        with backprop.GradientTape() as tape:
          tape.watch(input_val)
          op_output = tf_op(input_val, indices, num_segments)
          upstream_gradients = self._randomDataOp(op_output.shape,
                                                  dtype, local_seed)
          gradient_injector_output = op_output * upstream_gradients
        return tape.gradient(gradient_injector_output, input_val)

      for i in range(self.repeat_count):
        local_seed = numpy_seed + i # select different upstream gradients
        result_a = op_gradients(local_seed)
        result_b = op_gradients(local_seed)
        self.assertAllEqual(result_a, result_b)

    else:
      op_output = tf_op(input_val, indices, num_segments)
      output_shape = op_output.shape
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

  # The backward operation is not known or expected to introduce nondeterminism
  # but we're testing it for completeness.
  @tf_test_util.run_in_graph_and_eager_modes
  def testBackward(self):
    num_cols = 2
    num_rows = 64
    num_segments = 64
    segment_size = num_cols * num_rows
    indices_flat = np.random.randint(low=-1, high=num_segments,
                                     size=(segment_size,))

    with local_test_utils.force_gpu_session(self):
      for dtype in self.differentiable_dtypes:
        for indices in indices_flat, indices_flat.reshape(num_rows, num_cols):
          ops_list = self.complex_ops_list if dtype.is_complex \
              else self.ops_list
          for op_binding in ops_list:
            shape = indices.shape + (num_cols,)
            self._testBackwardCase(dtype, indices, num_segments,
                                   op_binding, shape)

  @tf_test_util.run_in_graph_and_eager_modes
  def testForward(self):
    # We don't patch TF version 2.7 or later, so it's not imperative that we
    # test determinism of this op in those versions of TensorFlow. However,
    # this test should theoretically pass on TF 2.7+ and is currently failing
    # for unknown reasons.
    # TODO: Get this test working/passing on TF 2.7+
    self._conditionally_skip_test()
    num_cols = 2
    num_rows = 64
    num_segments = 64
    segment_size = num_cols * num_rows
    indices_flat = np.random.randint(low=-1, high=num_segments,
                                     size=(segment_size,))
    with local_test_utils.force_gpu_session(self):
      for dtype in self.all_dtypes:
        for indices in indices_flat, indices_flat.reshape(num_rows, num_cols):
          shape = indices.shape + (num_cols,)
          ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
          x, _  = self._random_input(shape, dtype=dtype)

          for _, _, tf_op, _ in ops_list:
            for _ in range(self.repeat_count):
              result_a = self.evaluate(tf_op(x, indices, num_segments))
              result_b = self.evaluate(tf_op(x, indices, num_segments))
              self.assertAllEqual(result_a, result_b)


  # Prior to TF 2.7 (when we patch), op `gen_math_ops.segment_sum()` is not
  # patched for data type float64 and complex128 on GPU. A warning will be
  # thrown to indicate to users float64/complex128 is still exposed to
  # GPU-nondeterminism.
  @tf_test_util.run_deprecated_v1
  def testNonSupportedDataTypes(self):
    self._conditionally_skip_test()
    non_supported_types = (dtypes_lib.float64, dtypes_lib.complex128)
    indices = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    shape = indices.shape + (2,)
    with local_test_utils.force_gpu_session(self):
      for dtype in non_supported_types:
        ops_list = self.complex_ops_list if dtype.is_complex \
            else self.ops_list
        tf_x, _ = self._input(shape, dtype)

        for _, _, tf_op, _ in ops_list:
          with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = tf_op(tf_x, indices, num_segments)
            self.evaluate(s)
            # In NGC TF1 containers from 22.03 onwards, this op generates an
            # extra warning ["tostring() is deprecated"]. In all other
            # containers, only the expected warning is generated.
            self.assertGreater(len(w), 0)
            self.assertIsInstance(w[-1].message, UserWarning)
            self.assertTrue("GPU-determinism" in str(w[-1].message))


class SegmentReductionTestMisc(test.TestCase):

  def testSDocstring(self):
    op = tf.math.unsorted_segment_sum
    docstring = op.__doc__

    if not docstring: # falsy (None or "")
        self.fail("The patched op %s has no docstring" % op.__name__)
    if docstring.startswith('ERROR'):
        self.fail("The docstring for the patched op %s has not been assigned"
                % op.__name__)

if __name__ == "__main__":
  tf_determinism.enable_determinism()
  test.main()
