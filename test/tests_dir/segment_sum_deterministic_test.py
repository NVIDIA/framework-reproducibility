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

import itertools
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.eager import backprop
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
import unittest

sys.path.insert(0, '..')

from fwd9m.tensorflow import enable_determinism
from fwd9m.tensorflow import fwd9m_tfsession

enable_determinism()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
# NOTE: 
# 1. Op `segment_sum` is expected to be deterministic on CPU and it indeed 
#    behaves like that if forcely pinned to CPU with tf.device('/device:CPU:0').
#    What is not fully understood is that without explicitly pinned to CPU but
#    with `Session(use_gpu=False)`, it seems to run on GPU still and produces 
#    nondeterminism. But by setting `log_device_placement=True` configuration,
#    command line outputs indicate the op is pinned to CPU.  
# 2. To capture nondeterminism, random input data is necessary.
# 3. Nondeterminism of dtypes_lib.float64, dtypes_lib.complex128 cannot be fixed 
#    by this patch, so they're not tested. 
class SegmentReductionHelper(test.TestCase):
  def _random_input(self, input_shape, dtype=dtypes_lib.int32):
    seed =  (hash(dtype) % 256)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    np_values = np.random.random(input_shape).astype(dtype.as_numpy_dtype)
    # Add a non-zero imaginary component to complex types.
    if dtype.is_complex:
      np_values -= 1j * np_values
    return constant_op.constant(
        np_values, shape=input_shape, dtype=dtype), np_values

  def _input(self, input_shape, dtype=dtypes_lib.int32):
    num_elem = 1
    for x in input_shape:
      num_elem *= x
    values = np.arange(1, num_elem + 1)
    np_values = values.reshape(input_shape).astype(dtype.as_numpy_dtype)
    # Add a non-zero imaginary component to complex types.
    if dtype.is_complex:
      np_values -= 1j * np_values
    return constant_op.constant(
        np_values, shape=input_shape, dtype=dtype), np_values

  def _segmentReduce(self, indices, x, op1, op2=None, num_segments=None,
                     initial_value=0):
    if not x.size:
      return np.array([])
    indices = np.asarray(indices)
    if num_segments is None:
      num_segments = indices[-1] + 1
    output = [None] * num_segments
    slice_shape = x.shape[indices.ndim:]
    x_flat = x.reshape((indices.size,) + slice_shape)
    for i, index in enumerate(indices.ravel()):
      if (output[index] is not None) and op1 == np.max:
        for j in range(0, output[index].shape[0]):
          output[index][j] = op1([output[index][j], x_flat[i][j]])
      elif output[index] is not None:
        output[index] = op1(output[index], x_flat[i])
      else:
        output[index] = x_flat[i]
    # zero initialize values that are still uncalcuated.
    initial_value_slice = np.ones(slice_shape) * initial_value
    output = [o if o is not None else initial_value_slice for o in output]
    if op2 is not None:
      output = [op2(o) for o in output]
    output = [o.reshape(slice_shape) for o in output]
    return np.array(output)

  def _mean_cum_op(self, x, y):
    return (x[0] + y, x[1] + 1) if isinstance(x, tuple) else (x + y, 2)

  def _mean_reduce_op(self, x):
    return x[0] / x[1] if isinstance(x, tuple) else x

  def _sqrt_n_reduce_op(self, x):
    return x[0] / np.sqrt(x[1]) if isinstance(x, tuple) else x

class UnsortedSegmentTestDeterministic(SegmentReductionHelper):
  def __init__(self, methodName='runTest'):
    # Each item is np_op1, np_op2, tf_op, initial_value functor
    self.ops_list = [(np.add, None,
                      math_ops.unsorted_segment_sum, lambda t: 0),
                    (np.add, None,
                      tf.math.unsorted_segment_sum, lambda t: 0)]

    # A subset of ops has been enabled for complex numbers
    self.complex_ops_list = [(np.add, None,
                              math_ops.unsorted_segment_sum, lambda t: 0)]

    self.differentiable_dtypes = [dtypes_lib.float16, dtypes_lib.float32]
    self.all_dtypes = (self.differentiable_dtypes +
                       [dtypes_lib.bfloat16,
                        dtypes_lib.int64, dtypes_lib.int32,
                        dtypes_lib.complex64])
    self.repeat_count = 5
    super(UnsortedSegmentTestDeterministic, self).__init__(methodName=methodName)

  @test_util.run_deprecated_v1
  def _testDeterministicUnsortedSegmentSumCase(self, dtype, indices, 
                                               num_segments, num_cols, ops_list,
                                               shape, use_gpu):
    x, _  = self._random_input(shape, dtype=dtype)
    def forward(tf_op):
      s = tf_op(x, indices, num_segments)
      tf_ans = self.evaluate(s)
      return tf_ans

    with self.cached_session(use_gpu=use_gpu):
      for _, _, tf_op, _ in ops_list:
        run_ref = forward(tf_op)
        for i in range(self.repeat_count):
          self.assertAllEqual(forward(tf_op), run_ref)


  def _testDeterministicGradientsCase(self, dtype, indices, num_segments,
                                      op_binding, shape):
    numpy_seed = 123
    _, _, tf_op, _ = op_binding

    def _randomDataOp(shape, data_type, seed):
      if seed is not None:
        np.random.seed(seed)
      return constant_op.constant(np.random.random_sample(shape), dtype=data_type)
    
    input_val = _randomDataOp(shape, dtype, seed=None)

    if context.executing_eagerly():
      def op_gradients(local_seed):
        with backprop.GradientTape() as tape:
          tape.watch(input_val)
          op_output = tf_op(input_val, indices, num_segments)
          upstream_gradients = _randomDataOp(op_output.shape, dtype, local_seed)
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


  def testDeterministicUnsortedSegmentSum(self):
    num_cols = 2
    num_rows = 64
    num_segments = 64
    segment_size = num_cols * num_rows
    indices_flat = np.random.randint(low=-1, high=num_segments,
                                     size=(segment_size,))
    
    for dtype in self.all_dtypes:
      ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
      for indices in indices_flat, indices_flat.reshape(num_rows, num_cols):
        shape = indices.shape + (num_cols,)
        # For completeness of testing, cpu behavior is also tested
        for use_gpu in [False, True]:
          self._testDeterministicUnsortedSegmentSumCase(
              dtype, indices, num_segments, num_cols, ops_list, shape, use_gpu)

  @test_util.run_in_graph_and_eager_modes
  def testDeterministicGradients(self):
    num_cols = 2
    num_rows = 64
    num_segments = 64
    segment_size = num_cols * num_rows
    indices_flat = np.random.randint(low=-1, high=num_segments, size=(segment_size,))
    
    with fwd9m_tfsession(self, force_gpu=True):
    # with self.session(force_gpu=True):
      for dtype in self.differentiable_dtypes:
        ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
        for op_binding in ops_list:
          for indices in indices_flat, indices_flat.reshape(num_rows, num_cols):
            shape = indices.shape + (num_cols,)
            self._testDeterministicGradientsCase(dtype, indices, num_segments,
                                                 op_binding, shape)
    

class SegmentReductionOpTestDeterministic(SegmentReductionHelper):

  def __init__(self, methodName='runTest'):
  # Each item is np_op1, np_op2, tf_op, initial_value functor
    self.ops_list = [(np.add, None,
                      math_ops.segment_sum, lambda t: 0)]

    # A subset of ops has been enabled for complex numbers
    self.complex_ops_list = [(np.add, None,
                              math_ops.segment_sum, lambda t: 0)]
    
    self.differentiable_dtypes = [dtypes_lib.float16, dtypes_lib.float32]

    self.all_dtypes = (self.differentiable_dtypes +
                      [dtypes_lib.bfloat16,
                        dtypes_lib.int64, dtypes_lib.int32,
                        dtypes_lib.complex64])
    self.repeat_count = 5
    super(SegmentReductionOpTestDeterministic, 
          self).__init__(methodName=methodName)

  def _testDeterministicGradientsCase(self, dtype, indices, tf_op, shape):
    numpy_seed = 123

    def _randomDataOp(shape, data_type, seed):
      if seed is not None:
        np.random.seed(seed)
      return constant_op.constant(np.random.random_sample(shape), dtype=data_type)
    
    input_val = _randomDataOp(shape, dtype, seed=None)
    output_shape = [indices[-1]+1, shape[1]]
    if context.executing_eagerly():
      def op_gradients(local_seed):
        with backprop.GradientTape() as tape:
          tape.watch(input_val)
          op_output = tf_op(input_val, indices)
          upstream_gradients = _randomDataOp(output_shape, dtype, local_seed)
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

  def _testDeterministicSegmentSumCase(self, dtype, indices, ops_list, 
                                       shape, use_gpu):
    # have to use float to exec nond9m
    tf_x, _ = self._random_input(shape, dtype=dtype)
    if use_gpu == False:
      config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, 
                                        inter_op_parallelism_threads=1,
                                        log_device_placement=True)
    
    with self.session(use_gpu=use_gpu, config=None if use_gpu else config):
      for _, _, tf_op, _ in ops_list:
        run_ref = tf_op(data=tf_x, segment_ids=indices, name="tf_op_output")
        for i in range(self.repeat_count):
          self.assertAllEqual(tf_op(data=tf_x, segment_ids=indices), run_ref)
  
  @test_util.run_in_graph_and_eager_modes
  def testDeterministicGradient(self):
    gradient_test = True
    num_cols = 8
    num_segments = 32
    segment_size = 256
    shape = [segment_size, num_cols]
    indices = np.random.randint(low=0, high=num_segments, size=(segment_size,))
    indices = np.sort(indices)
    
    with fwd9m_tfsession(self, force_gpu=True):
    # with self.session(force_gpu=True):#force_gpu=True leads to XLA issue
      for dtype in self.differentiable_dtypes:
        ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
        for _, _, tf_op, _ in ops_list:
          self._testDeterministicGradientsCase(dtype, indices, tf_op, shape)

  @test_util.run_in_graph_and_eager_modes
  def testDeterministicSegmentSum(self):
    num_cols = 8
    num_segments = 32
    segment_size = 256

    shape = [segment_size, num_cols]
    indices = np.random.randint(low=0, high=num_segments, size=(segment_size,))
    indices = np.sort(indices)
    
    for dtype in self.all_dtypes:
      ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
      # For completeness of testing, cpu behavior is also tested
      for use_gpu in [True]:
        self._testDeterministicSegmentSumCase(dtype, indices, ops_list, 
                                              shape, use_gpu)
            

if __name__ == "__main__":
  enable_determinism()
  test.main()