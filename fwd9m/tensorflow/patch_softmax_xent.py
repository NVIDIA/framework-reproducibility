# Copyright 2021 NVIDIA Corporation. All Rights Reserved
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

import tensorflow as tf

import functools
import numbers
import os

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export

# NOTE: This patch provides GPU-determinism for
# `tf.nn.softmax_cross_entropy_with_logits` via overriding the fused op
# `gen_nn_ops.softmax_cross_entropy_with_logit` with sequential calling of
# softmax, logarithm and reduce_sum which are known deterministic.

def _patch_softmax_xent():
  _new_softmax_xent_with_logits.__doc__ = \
      tf.nn.softmax_cross_entropy_with_logits.__doc__
  _new_softmax_cross_entropy_with_logits_v2_helper.__doc__ = \
      nn_ops.softmax_cross_entropy_with_logits_v2_helper.__doc__
  tf.nn.softmax_cross_entropy_with_logits = \
      _new_softmax_xent_with_logits # access via public API
  nn.softmax_cross_entropy_with_logits = _new_softmax_xent_with_logits
  nn_ops.softmax_cross_entropy_with_logits = _new_softmax_xent_with_logits

# The original, pre-patched python wrapper can be viewed at
# https://github.com/tensorflow/tensorflow/blob/0c95acca049a05756f63bec731dbe9a11f9d8382/tensorflow/python/ops/nn_ops.py#L3998

def _core_op(labels, logits):
  """Internal only. The shape should be checked equal eariler."""
  dim = -1
  softmax = tf.nn.softmax(logits=logits, axis=dim)
  epsilon_ = constant_op.constant(K.epsilon(), dtype=softmax.dtype.base_dtype)
  softmax = clip_ops.clip_by_value(softmax, epsilon_, 1. - epsilon_)
  # ??? * needs the data type to be the same
  return -tf.reduce_sum(tf.math.log(softmax) * labels, axis=dim)

_XENT_DEPRECATION = """
Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.
See `tf.nn.softmax_cross_entropy_with_logits_v2`.
"""
def _flatten_outer_dims(logits):
  """Flattens logits' outer dimensions and keep its last dimension."""
  rank = array_ops.rank(logits)
  last_dim_size = array_ops.slice(
      array_ops.shape(logits), [math_ops.subtract(rank, 1)], [1])
  output = array_ops.reshape(logits, array_ops.concat([[-1], last_dim_size], 0))

  # Set output shape if known.
  if not context.executing_eagerly():
    shape = logits.get_shape()
    if shape is not None and shape.dims is not None:
      shape = shape.as_list()
      product = 1
      product_valid = True
      for d in shape[:-1]:
        if d is None:
          product_valid = False
          break
        else:
          product *= d
      if product_valid:
        output_shape = [product, shape[-1]]
        output.set_shape(output_shape)

  return output

def _ensure_xent_args(name, sentinel, labels, logits):
  # Make sure that all arguments were passed as named arguments.
  if sentinel is not None:
    raise ValueError("Only call `%s` with "
                     "named arguments (labels=..., logits=..., ...)" % name)
  if labels is None or logits is None:
    raise ValueError("Both labels and logits must be provided.")


@tf_export(v1=["nn.softmax_cross_entropy_with_logits"])
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions=_XENT_DEPRECATION)
def _new_softmax_xent_with_logits(
    _sentinel=None,  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    dim=-1,
    name=None,
    axis=None):
  """ERROR: docstring should have been added programatically. """
  dim = deprecated_argument_lookup("axis", axis, "dim", dim)
  _ensure_xent_args("softmax_cross_entropy_with_logits", _sentinel, labels,
                    logits)

  with ops.name_scope(name, "softmax_cross_entropy_with_logits_sg",
                      [logits, labels]) as name:
    labels = array_ops.stop_gradient(labels, name="labels_stop_gradient")

  return softmax_cross_entropy_with_logits_v2(
      labels=labels, logits=logits, axis=dim, name=name)

@tf_export("nn.softmax_cross_entropy_with_logits", v1=[])
@dispatch.add_dispatch_support
def softmax_cross_entropy_with_logits_v2(labels, logits, axis=-1, name=None):
  return _new_softmax_cross_entropy_with_logits_v2_helper(
      labels=labels, logits=logits, axis=axis, name=name)

@tf_export(v1=["nn.softmax_cross_entropy_with_logits_v2"])
@dispatch.add_dispatch_support
@deprecated_args(None, "dim is deprecated, use axis instead", "dim")
def _new_softmax_cross_entropy_with_logits_v2_helper(
    labels, logits, axis=None, name=None, dim=None):
  # TODO(pcmurray) Raise an error when the labels do not sum to 1. Note: This
  # could break users who call this with bad labels, but disregard the bad
  # results.
  axis = deprecated_argument_lookup("axis", axis, "dim", dim)
  del dim
  if axis is None:
    axis = -1

  with ops.name_scope(name, "softmax_cross_entropy_with_logits",
                      [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    convert_to_float32 = (
        logits.dtype == dtypes.float16 or logits.dtype == dtypes.bfloat16)
    precise_logits = math_ops.cast(
        logits, dtypes.float32) if convert_to_float32 else logits
    # labels and logits must be of the same type
    labels = math_ops.cast(labels, precise_logits.dtype)
    input_rank = array_ops.rank(precise_logits)
    # For shape inference.
    shape = logits.get_shape()

    # Move the dim to the end if dim is not the last dimension.
    if axis != -1:

      def _move_dim_to_end(tensor, dim_index, rank):
        return array_ops.transpose(
            tensor,
            array_ops.concat([
                math_ops.range(dim_index),
                math_ops.range(dim_index + 1, rank), [dim_index]
            ], 0))

      precise_logits = _move_dim_to_end(precise_logits, axis, input_rank)
      labels = _move_dim_to_end(labels, axis, input_rank)

    input_shape = array_ops.shape(precise_logits)

    # Make precise_logits and labels into matrices.
    precise_logits = _flatten_outer_dims(precise_logits)
    labels = _flatten_outer_dims(labels)

    # Do the actual op computation.
    # The second output tensor contains the gradients.  We use it in
    # CrossEntropyGrad() in nn_grad but not here.
    # cost, unused_backprop = gen_nn_ops.softmax_cross_entropy_with_logits(
    #     precise_logits, labels, name=name)
    cost = _core_op(labels=labels, logits=precise_logits)

    # The output cost shape should be the input minus axis.
    output_shape = array_ops.slice(input_shape, [0],
                                   [math_ops.subtract(input_rank, 1)])
    cost = array_ops.reshape(cost, output_shape)

    # Make shape inference work since reshape and transpose may erase its static
    # shape.
    if not context.executing_eagerly() and shape is not None \
       and shape.dims is not None:
      shape = shape.as_list()
      del shape[axis]
      cost.set_shape(shape)

    if convert_to_float32:
      return math_ops.cast(cost, logits.dtype)
    else:
      return cost