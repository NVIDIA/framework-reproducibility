from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# from tensorflow.python.eager import context
# from tensorflow.python.framework import config
# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import ops
# from tensorflow.python.keras import backend as K
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import clip_ops
# from tensorflow.python.ops import gen_math_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import nn
# from tensorflow.python.ops import nn_ops



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
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_lib

from tensorflow.python.platform import device_context
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup

from tensorflow.python.util.tf_export import tf_export


# The original, pre-patched method can be viewed at
# https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/ops/nn_ops.py#L3182
def _core_op(labels, logits):
  """Internal only. The shape should be checked equal eariler."""
  dim = -1
  softmax = tf.nn.softmax(logits=logits, axis=dim)
  epsilon_ = constant_op.constant(K.epsilon(), dtype=softmax.dtype.base_dtype)
  softmax = clip_ops.clip_by_value(softmax, epsilon_, 1. - epsilon_)
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
def _new_softmax_cross_entropy_with_logits(
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
  """Computes softmax cross entropy between `logits` and `labels`.
  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.
  **NOTE:**  While the classes are mutually exclusive, their probabilities
  need not be.  All that is required is that each row of `labels` is
  a valid probability distribution.  If they are not, the computation of the
  gradient will be incorrect.
  If using exclusive `labels` (wherein one and only
  one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.
  Usage:
  >>> logits = [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]]
  >>> labels = [[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]
  >>> tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  <tf.Tensor: shape=(2,), dtype=float32,
  numpy=array([0.16984604, 0.82474494], dtype=float32)>
  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.
  A common use case is to have logits and labels of shape
  `[batch_size, num_classes]`, but higher dimensions are supported, with
  the `axis` argument specifying the class dimension.
  `logits` and `labels` must have the same dtype (either `float16`, `float32`,
  or `float64`).
  Backpropagation will happen into both `logits` and `labels`.  To disallow
  backpropagation into `labels`, pass label tensors through `tf.stop_gradient`
  before feeding it to this function.
  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**
  Args:
    labels: Each vector along the class dimension should hold a valid
      probability distribution e.g. for the case in which labels are of shape
      `[batch_size, num_classes]`, each row of `labels[i]` must be a valid
      probability distribution.
    logits: Per-label activations, typically a linear output. These activation
      energies are interpreted as unnormalized log probabilities.
    axis: The class dimension. Defaulted to -1 which is the last dimension.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` that contains the softmax cross entropy loss. Its type is the
    same as `logits` and its shape is the same as `labels` except that it does
    not have the last dimension of `labels`.
  """
  return softmax_cross_entropy_with_logits_v2_helper(
      labels=labels, logits=logits, axis=axis, name=name)


@tf_export(v1=["nn.softmax_cross_entropy_with_logits_v2"])
@dispatch.add_dispatch_support
@deprecated_args(None, "dim is deprecated, use axis instead", "dim")
def softmax_cross_entropy_with_logits_v2_helper(
    labels, logits, axis=None, name=None, dim=None):
  """Computes softmax cross entropy between `logits` and `labels`.
  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.
  **NOTE:**  While the classes are mutually exclusive, their probabilities
  need not be.  All that is required is that each row of `labels` is
  a valid probability distribution.  If they are not, the computation of the
  gradient will be incorrect.
  If using exclusive `labels` (wherein one and only
  one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.
  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.
  A common use case is to have logits and labels of shape
  `[batch_size, num_classes]`, but higher dimensions are supported, with
  the `axis` argument specifying the class dimension.
  `logits` and `labels` must have the same dtype (either `float16`, `float32`,
  or `float64`).
  Backpropagation will happen into both `logits` and `labels`.  To disallow
  backpropagation into `labels`, pass label tensors through `tf.stop_gradient`
  before feeding it to this function.
  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**
  Args:
    labels: Each vector along the class dimension should hold a valid
      probability distribution e.g. for the case in which labels are of shape
      `[batch_size, num_classes]`, each row of `labels[i]` must be a valid
      probability distribution.
    logits: Unscaled log probabilities.
    axis: The class dimension. Defaulted to -1 which is the last dimension.
    name: A name for the operation (optional).
    dim: Deprecated alias for axis.
  Returns:
    A `Tensor` that contains the softmax cross entropy loss. Its type is the
    same as `logits` and its shape is the same as `labels` except that it does
    not have the last dimension of `labels`.
  """
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
    if not context.executing_eagerly(
    ) and shape is not None and shape.dims is not None:
      shape = shape.as_list()
      del shape[axis]
      cost.set_shape(shape)

    if convert_to_float32:
      return math_ops.cast(cost, logits.dtype)
    else:
      return cost