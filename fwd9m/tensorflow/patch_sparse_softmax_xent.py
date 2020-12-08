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
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_lib

from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args

from tensorflow.python.util.tf_export import tf_export

def _core_op(labels, logits):
  """Internal only. The shape should be checked equal eariler."""
  dim = -1
  softmax = tf.nn.softmax(logits=logits, axis=dim)
  epsilon_ = constant_op.constant(K.epsilon(), dtype=softmax.dtype.base_dtype)
  softmax = clip_ops.clip_by_value(softmax, epsilon_, 1. - epsilon_)
  print("HERE", labels, softmax)
  # labels = math_ops.cast(labels, softmax.dtype.base_dtype)
  return -tf.reduce_sum(tf.math.log(softmax) * labels, axis=dim)

@tf_export("nn.sparse_softmax_cross_entropy_with_logits", v1=[])
@dispatch.add_dispatch_support
def sparse_softmax_cross_entropy_with_logits_v2(labels, logits, name=None):
  return sparse_softmax_cross_entropy_with_logits(
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
def _new_sparse_softmax_cross_entropy_with_logits(
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
      # Has to be here, because it really tests gen_nn_ops.sparse_xent
      if labels.get_shape().ndims is None:
        raise errors_impl.InvalidArgumentError(None, None,
                                               ".*labels must be 1-D.*")
        # raise errors_impl.OpError(None, None, "labels must be 1-D", errors_impl.OpError)
      onehot_encoding = tf.one_hot(labels, precise_logits.shape[-1],
                                   dtype=dtypes.as_dtype(precise_logits.dtype))
      print("onehot_encoding"*100, onehot_encoding, precise_logits)
      cost = _core_op(labels=onehot_encoding, logits=precise_logits)

      if precise_logits.dtype == dtypes.float16:
        return math_ops.cast(cost, dtypes.float16)
      else:
        return cost

    # if logits.get_shape().ndims == 2:
    #   cost, _ = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
    #       precise_logits, labels, name=name)
    #   if logits.dtype == dtypes.float16:
    #     return math_ops.cast(cost, dtypes.float16)
    #   else:
    #     return cost

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
      # The second output tensor contains the gradients.  We use it in
      # _CrossEntropyGrad() in nn_grad but not here.
      # cost, _ = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
      #     precise_logits, labels, name=name)
      print("##"*1000)
      onehot_encoding = tf.one_hot(labels, num_classes)
      cost = _core_op(logits=precise_logits, labels=onehot_encoding)

      cost = array_ops.reshape(cost, labels_shape)
      cost.set_shape(labels_static_shape)
      if logits.dtype == dtypes.float16:
        return math_ops.cast(cost, dtypes.float16)
      else:
        return cost
