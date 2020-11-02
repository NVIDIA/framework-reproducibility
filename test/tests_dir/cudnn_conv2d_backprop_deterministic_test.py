# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for deterministic cuDNN functionality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import os
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from fwd9m.tensorflow import enable_determinism
from fwd9m.tensorflow import fwd9m_tfsession

enable_determinism()
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# Notes:
#
# Deterministic cuDNN operation is selected by setting either of the two
# environment variables TF_CUDNN_DETERMINISTIC or TF_DETERMINISTIC_OPS to 'true'
# or '1' while also not setting the environment variable TF_CUDNN_USE_AUTOTUNE
# to 'false' or '0'.
#
# Where both deterministic and non-deterministic cuDNN algorithms are available,
# selecting determinitic operation will lead to only the deterministic
# algorithms being chosen. Additionally, selecting deterministic operation will
# result in a deterministic, or reproducible, selection of algorithms (for any
# given layer configuration) for each of the forward and the two backward paths.
#
# These tests intend to confirm that deterministic algorithms are chosen (for
# the back-prop paths) when desterministic operation is selected. The tested
# configurations were first confirmed to produce non-deterministic results when
# the above-mentioned environment variables are not set.
#
# Even though selecting determinitic operation should ensure that the same
# algorithms, for a given layer configuration, are always used (i.e. that
# algorithm selection is deterministic / reproducible), this is not tested.


LayerShapeNHWC = collections.namedtuple('LayerShapeNHWC',
                                        'batch, height, width, channels')
FilterShape2D = collections.namedtuple(
    'FilterShape2D', 'height, width, in_channels, out_channels')

LayerShapeNCDHW = collections.namedtuple(
    'LayerShapeNCDHW', 'batch, channels, depth, height, width')
FilterShape3D = collections.namedtuple(
    'FilterShape3D', 'depth, height, width, in_channels, out_channels')


class ConvolutionTest(test.TestCase):

  def _random_data_op(self, shape):
    # np.random.random_sample can properly interpret either tf.TensorShape or
    # namedtuple as a list.
    return constant_op.constant(
        2 * np.random.random_sample(shape) - 1, dtype=dtypes.float32)

  def _random_out_op(self, in_shape, filter_shape, strides, padding):
    # Choosing not to use array_op.zeros() to prevent possible removal by
    # optimization
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    # Use the forward op's shape-inference
    conv_op = nn_ops.conv2d(in_op, filter_op, strides=strides, padding=padding)
    out_shape = conv_op.get_shape()
    out_op = self._random_data_op(out_shape)
    return out_op

  def _assert_reproducible(self, operation):
    with fwd9m_tfsession(self, force_gpu=True):
      result_1 = self.evaluate(operation)
      result_2 = self.evaluate(operation)
    self.assertAllEqual(result_1, result_2)

  # @test_util.run_cuda_only
  @test_util.run_in_graph_and_eager_modes
  def testBackwardFilterGradient(self):
    np.random.seed(1)
    in_shape = LayerShapeNHWC(batch=8, height=128, width=128, channels=8)
    filter_shape = FilterShape2D(
        height=3, width=3, in_channels=8, out_channels=8)
    in_op = self._random_data_op(in_shape)
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    out_op = self._random_out_op(in_shape, filter_shape, strides, padding)
    filter_gradient_op = nn_ops.conv2d_backprop_filter(
        in_op, filter_shape, out_op, strides=strides, padding=padding)
    self._assert_reproducible(filter_gradient_op)

  # @test_util.run_cuda_only
  @test_util.run_in_graph_and_eager_modes
  def testBackwardInputGradient(self):
    np.random.seed(2)
    in_shape = LayerShapeNHWC(batch=8, height=32, width=32, channels=8)
    filter_shape = FilterShape2D(
        height=7, width=7, in_channels=8, out_channels=128)
    filter_op = self._random_data_op(filter_shape)
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    out_op = self._random_out_op(in_shape, filter_shape, strides, padding)
    input_gradient_op = nn_ops.conv2d_backprop_input(
        in_shape, filter_op, out_op, strides=strides, padding=padding)
    self._assert_reproducible(input_gradient_op)