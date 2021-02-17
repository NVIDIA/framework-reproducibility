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
import os
import sys
import unittest

import numpy as np
import tensorflow as tf

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
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

sys.path.insert(0, '..')
import fwd9m.tensorflow as fwd9m_tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Simplifies logging

# Notes:
# 0. These notes are relevant to this current file and also
#    test_patch_segment_sum.py and test_patch_unsorted_segment_sum.py
# 1. The ops were expected to operate deterministically on the CPU and they do
#    indeed operate deterministically if forcely pinned to the CPU with
#    tf.device('/device:CPU:0'). What is not fully understood is why when they
#    are placed on the CPU using self.session(use_gpu=False), the ops still
#    introduce nondeterminism. By setting the log_device_placement parameter in
#    the session config to True under these conditions, we are able to confirm
#    that the ops are running on the CPU.
# 2. To capture nondeterminism, random input data is necessary.
# 3. The nondeterminism of dtypes_lib.float64, dtypes_lib.complex128 cannot be
#    removed by this patch, so they are not tested.
# 4. The regular op tests below, represented by all the test classes except the
#    final two, which have names ending in "Deterministic", were taken from
#    tensorflow/python/kernel_tests/segment_reduction_ops_test.py
#    (as of 2020-08-02); URL to file-at-commit:
#    https://github.com/tensorflow/tensorflow/blob/6371d4a38cfb122a8d9b2a03d5f56444e95462b0/tensorflow/python/kernel_tests/segment_reduction_ops_test.py
# 5. The names of most of the upstream test classes are confusing (even more so
#    in the context of their limited use here), so the names have been changed
#    here, as appropriate, along with comments to indicate the original test
#    class names.


class SegmentReductionHelper(test.TestCase):

  def _random_input(self, input_shape, dtype=dtypes_lib.int32):
    np.random.seed(hash(dtype) % 256)

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

  def _randomDataOp(self, shape, data_type, seed):
    if seed is not None:
      np.random.seed(seed)
    return constant_op.constant(np.random.random_sample(shape), dtype=data_type)

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