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
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


# NOTE: This patch provides GPU-determinism for
# `tf.nn.sparse_softmax_cross_entropy_with_logits` via overriding the fused op
# `gen_nn_ops.sparse_softmax_cross_entropy_with_logit` with turning labels into
# one_hot encoding and calling patched
# gen__nn_ops.softmax_cross_entropy_with_logits.

def _patch_sparse_softmax_xent():
  _new_sparse_softmax_xent_with_logits.__doc__ = \
      tf.nn.sparse_softmax_cross_entropy_with_logits.__doc__
  tf.nn.sparse_softmax_cross_entropy_with_logits = \
      _new_sparse_softmax_xent_with_logits  # access via public API
  nn.sparse_softmax_cross_entropy_with_logits = \
      _new_sparse_softmax_xent_with_logits
  nn_ops.sparse_softmax_cross_entropy_with_logits = \
      _new_sparse_softmax_xent_with_logits
  # NOTE: Since enable_determinism
  # patches gen_nn_ops.softmax_cross_entropy_with_logits and other ops
  # universally, there is no need to patch here.

# The original, pre-patched python wrapper
# `nn.sparse_softmax_cross_entropy_with_logits` can be found at
# https://github.com/tensorflow/tensorflow/blob/0c95acca049a05756f63bec731dbe9a11f9d8382/tensorflow/python/ops/nn_ops.py#L4066
# The fused op `gen_nn_ops.sparse_softmax_cross_entropy_with_logit` is
# automatically-generated. Therefore, we cannot provide a URL to its location in
# the source repository.

@tf_export("nn.sparse_softmax_cross_entropy_with_logits", v1=[])
@dispatch.add_dispatch_support
def sparse_softmax_cross_entropy_with_logits_v2(labels, logits, name=None):
  return nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name=name)

def _ensure_xent_args(name, sentinel, labels, logits):
  # Make sure that all arguments were passed as named arguments.
  if sentinel is not None:
    raise ValueError("Only call `%s` with "
                     "named arguments (labels=..., logits=..., ...)" % name)
  if labels is None or logits is None:
    raise ValueError("Both labels and logits must be provided.")

@tf_export(v1=["nn.sparse_softmax_cross_entropy_with_logits"])
@dispatch.add_dispatch_support
def _new_sparse_softmax_xent_with_logits(
    _sentinel=None,  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    name=None):
  _ensure_xent_args("sparse_softmax_cross_entropy_with_logits", _sentinel,
                    labels, logits)

  # TODO(pcmurray) Raise an error when the label is not an index in
  # [0, num_classes). Note: This could break users who call this with bad
  # labels, but disregard the bad results.

  # Reshape logits and labels to rank 2.
  with ops.name_scope(name, "SparseSoftmaxCrossEntropyWithLogits",
                      [labels, logits]):
    labels = ops.convert_to_tensor(labels)
    logits = ops.convert_to_tensor(logits)
    precise_logits = math_ops.cast(logits, dtypes.float32) if (dtypes.as_dtype(
        logits.dtype) == dtypes.float16) else logits

    # Store label shape for result later.
    labels_static_shape = labels.get_shape()
    labels_shape = array_ops.shape(labels)
    static_shapes_fully_defined = (
        labels_static_shape.is_fully_defined() and
        logits.get_shape()[:-1].is_fully_defined())
    if logits.get_shape().ndims is not None and logits.get_shape().ndims == 0:
      raise ValueError(
          "Logits cannot be scalars - received shape %s." % logits.get_shape())
    if logits.get_shape().ndims is not None and (
        labels_static_shape.ndims is not None and
        labels_static_shape.ndims != logits.get_shape().ndims - 1):
      raise ValueError("Rank mismatch: Rank of labels (received %s) should "
                       "equal rank of logits minus 1 (received %s)." %
                       (labels_static_shape.ndims, logits.get_shape().ndims))
    if (static_shapes_fully_defined and
        labels_static_shape != logits.get_shape()[:-1]):
      raise ValueError("Shape mismatch: The shape of labels (received %s) "
                       "should equal the shape of logits except for the last "
                       "dimension (received %s)." % (labels_static_shape,
                                                     logits.get_shape()))

    # Check if no reshapes are required.
    if logits.get_shape().ndims == 2:
      # Override of `gen_nn_ops.sparse_xent_with_logit`
      if labels.get_shape().ndims is None:
        raise errors_impl.InvalidArgumentError(
            None, None, ".*labels must be 1-D.*")
        # raise errors_impl.OpError(None, None, "labels must be 1-D", errors_impl.OpError)
      onehot_encoding = tf.one_hot(labels, precise_logits.shape[-1],
                                   dtype=dtypes.as_dtype(precise_logits.dtype))
#      cost = _core_op(labels=onehot_encoding, logits=precise_logits)

      cost, _ = gen_nn_ops.softmax_cross_entropy_with_logits(
          precise_logits, onehot_encoding, name=name)

      if precise_logits.dtype == dtypes.float16:
        return math_ops.cast(cost, dtypes.float16)
      else:
        return cost

    # Perform a check of the dynamic shapes if the static shapes are not fully
    # defined.
    shape_checks = []
    if not static_shapes_fully_defined:
      shape_checks.append(
          check_ops.assert_equal(
              array_ops.shape(labels),
              array_ops.shape(logits)[:-1]))
    with ops.control_dependencies(shape_checks):
      # Reshape logits to 2 dim, labels to 1 dim.
      num_classes = array_ops.shape(logits)[array_ops.rank(logits) - 1]
      precise_logits = array_ops.reshape(precise_logits, [-1, num_classes])
      labels = array_ops.reshape(labels, [-1])
      if labels.get_shape().ndims is None:
        raise errors_impl.InvalidArgumentError(None, None,
                                               ".*labels must be 1-D.*")
      # The second output tensor of `gen_nn_ops.sparse_xent_with_logits`
      # contains the gradients. But it's used in _CrossEntropyGrad() in nn_grad
      # but not here.
  #    cost, _ = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
   #       precise_logits, labels, name=name)

      onehot_encoding = tf.one_hot(labels, num_classes)
      cost, _ = gen_nn_ops.softmax_cross_entropy_with_logits(precise_logits, onehot_encoding,name=name)

      cost = array_ops.reshape(cost, labels_shape)
      cost.set_shape(labels_static_shape)

      if logits.dtype == dtypes.float16:
        return math_ops.cast(cost, dtypes.float16)
      else:
        return cost

