# Copyright 2019 NVIDIA Corporation. All Rights Reserved
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
# ========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops

from .version import __version__

def _patch():
  """Patches TensorFlow to increase determinism when running on GPUs.

    Calling this method either before or after explicitly importing TensorFlow,
    but always before constructing any graphs, will increase the determinsism
    when running on GPUs.

    Returns: nothing

    Raises:
      TypeError (1) if a patch is not available for the installed version of
      TensorFlow (either because it doesn't need one or because one has not
      yet been implemented), or (2) if there is an attempt to apply the patch
      inside an NGC TF container (where it should not be needed).
  """
  if os.environ.get('NVIDIA_TENSORFLOW_VERSION'):
    raise TypeError("tfdeterminism: TensorFlow inside NGC containers does not "
                    "require patching")
  tf_version = tf.version.VERSION
  if re.match("(1\.(14|15)|2\.0)", tf_version):
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    _patch_bias_add()
    _patch_fused_softmax_cross_entropy()
    print("TensorFlow version %s has been patched "
          "using tfdeterminism version %s" %
          (tf_version, __version__), file=sys.stderr)
  elif re.match("2\.1|2\.2"):
    _patch_fused_softmax_cross_entropy()
    print("TensorFlow version %s has been patched "
          "using tfdeterminism version %s" %
          (tf_version, __version__), file=sys.stderr)
  else:
    raise TypeError("tfdeterminism: No patch available "
                    "for version %s of TensorFlow" % tf_version)


def _patch_bias_add():
  tf.nn.bias_add = _new_bias_add_1_14 # access via public API
  nn.bias_add = _new_bias_add_1_14 # called from tf.keras.layers.convolutional.Conv
  nn_ops.bias_add = _new_bias_add_1_14 # called from tests

# The original, pre-patched method can be viewed at
# https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/ops/nn_ops.py#L2628
def _new_bias_add_1_14(value, bias, data_format=None, name=None):
  """Adds `bias` to `value`.

  This is (mostly) a special case of `tf.add` where `bias` is restricted to 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.
  Unlike `tf.add`, the type of `bias` is allowed to differ from `value` in the
  case where both types are quantized.

  Args:
    value: A `Tensor` with type `float`, `double`, `int64`, `int32`, `uint8`,
      `int16`, `int8`, `complex64`, or `complex128`.
    bias: A 1-D `Tensor` with size matching the channel dimension of `value`.
      Must be the same type as `value` unless `value` is a quantized type,
      in which case a different quantized type may be used.
    data_format: A string. 'N...C' and 'NC...' are supported. If `None` (the
      default) is specified then 'N..C' is assumed.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError if data format is unrecognized, if `value` has less than two
    dimensions when `data_format` is 'N..C'/`None` or `value` has less
    then three dimensions when `data_format` is `NC..`, if `bias` does not
    have exactly one dimension (is a vector), or if the size of `bias`
    does not match the size of the channel dimension of `value`.
  """
  with ops.name_scope(name, "BiasAdd", [value, bias]) as name:
    if data_format is not None:
      if data_format.startswith("NC"):
        data_format = "NCHW"
      elif data_format.startswith("N") and data_format.endswith("C"):
        data_format = "NHWC"
      else:
        raise ValueError("data_format must be of the form `N...C` or `NC...`")

    if not context.executing_eagerly():
      value = ops.convert_to_tensor(value, name="input")
      bias = ops.convert_to_tensor(bias, dtype=value.dtype, name="bias")

    if data_format == 'NCHW':
      broadcast_shape_head = [1, array_ops.size(bias)]
      broadcast_shape_tail = array_ops.ones(array_ops.rank(value) - 2,
                                            dtype=dtypes.int32)
      broadcast_shape = array_ops.concat(
          [broadcast_shape_head, broadcast_shape_tail], 0)
      return math_ops.add(
          value, array_ops.reshape(bias, broadcast_shape), name=name)
    else: # data_format == 'NHWC' or data_format == None
      return math_ops.add(value, bias, name=name)


def _patch_fused_softmax_cross_entropy():
  # Sparse
  tf.nn.softmax_cross_entropy_with_logits = _new_softmax_cross_entropy_with_logits  # access via public API
  nn.softmax_cross_entropy_with_logits = _new_softmax_cross_entropy_with_logits  # called from tf.keras.layers.convolutional.Conv
  nn_ops.softmax_cross_entropy_with_logits = _new_softmax_cross_entropy_with_logits  # called from tests

  # Non-sparse
  tf.nn.sparse_softmax_cross_entropy_with_logits = _new_sparse_softmax_cross_entropy_with_logits  # access via public API
  nn.sparse_softmax_cross_entropy_with_logits = _new_sparse_softmax_cross_entropy_with_logits  # called from tf.keras.layers.convolutional.Conv
  nn_ops.sparse_softmax_cross_entropy_with_logits = _new_sparse_softmax_cross_entropy_with_logits  # called from tests


# The original, pre-patched method can be viewed at
# https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/ops/nn_ops.py#L2628
def _new_softmax_cross_entropy_with_logits(labels, logits, axis=-1, name=None):
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
  the `dim` argument specifying the class dimension.
  Backpropagation will happen only into `logits`.  To calculate a cross entropy
  loss that allows backpropagation into both `logits` and `labels`, see
  `tf.nn.softmax_cross_entropy_with_logits_v2`.
  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**
  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: Each vector along the class dimension should hold a valid
      probability distribution e.g. for the case in which labels are of shape
      `[batch_size, num_classes]`, each row of `labels[i]` must be a valid
      probability distribution.
    logits: Per-label activations, typically a linear output. These activation
      energies are interpreted as unnormalized log probabilities.
    dim: The class dimension. Defaulted to -1 which is the last dimension.
    name: A name for the operation (optional).
    axis: Alias for dim.
  Returns:
    A `Tensor` that contains the softmax cross entropy loss. Its type is the
    same as `logits` and its shape is the same as `labels` except that it does
    not have the last dimension of `labels`.
  """
  raise NotImplementedError()


# The original, pre-patched method can be viewed at
# https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/ops/nn_ops.py#L2628
def _new_sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    name=None):
  """Computes sparse softmax cross entropy between `logits` and `labels`.
  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.
  **NOTE:**  For this operation, the probability of a given label is considered
  exclusive.  That is, soft classes are not allowed, and the `labels` vector
  must provide a single specific index for the true class for each row of
  `logits` (each minibatch entry).  For soft softmax classification with
  a probability distribution for each entry, see
  `softmax_cross_entropy_with_logits_v2`.
  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.
  A common use case is to have logits of shape
  `[batch_size, num_classes]` and have labels of shape
  `[batch_size]`, but higher dimensions are supported, in which
  case the `dim`-th dimension is assumed to be of size `num_classes`.
  `logits` must have the dtype of `float16`, `float32`, or `float64`, and
  `labels` must have the dtype of `int32` or `int64`.
  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**
  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
      `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
      must be an index in `[0, num_classes)`. Other values will raise an
      exception when this op is run on CPU, and return `NaN` for corresponding
      loss and gradient rows on GPU.
    logits: Per-label activations (typically a linear output) of shape
      `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float16`, `float32`, or
      `float64`. These activation energies are interpreted as unnormalized log
      probabilities.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of the same shape as `labels` and of the same type as `logits`
    with the softmax cross entropy loss.
  Raises:
    ValueError: If logits are scalars (need to have rank >= 1) or if the rank
      of the labels is not equal to the rank of the logits minus one.
  """
  raise NotImplementedError()
