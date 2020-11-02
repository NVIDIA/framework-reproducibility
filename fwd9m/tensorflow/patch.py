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
from tensorflow.python.ops import gen_math_ops

from ..utils import _Version as Version
from ..version import __version__ as package_version

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
  # NOTE: Figure out later
  print("WARNING: %s has been deprecated. Please use enable_determinism (which "
        "supports all versions of TensorFlow)." % __name__)
  if os.environ.get('NVIDIA_TENSORFLOW_VERSION'):
    raise TypeError("%s: TensorFlow inside NGC containers does not "
                    "require patching" % __name__)
  tf_vers = Version(tf.version.VERSION)
  if tf_vers.between('1.14', '2.3'):
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    _patch_bias_add()
    _patch_unsorted_segment_sum()
    _patch_segment_sum()
    # Apply the fused softmax/cross-entropy patch here
    print("TensorFlow version %s has been patched using %s version %s" %
          (tf_vers.original_version_string, __name__,
           package_version))
  else:
    raise TypeError("%s: No patch available for version %s of TensorFlow" %
                    (__name__, tf_vers.original_version_string))

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


def _patch_unsorted_segment_sum():
  math_ops.unsorted_segment_sum = _new_unsorted_segment_sum_2_3 # access via public API
  tf.math.unsorted_segment_sum = _new_unsorted_segment_sum_2_3 # access via public API

def _patch_segment_sum():
  math_ops.segment_sum = _new_segment_sum_2_3 # access via public API
  tf.math.segment_sum = _new_segment_sum_2_3 # access via public API

# The original, pre-patched method is a self-generated function. 
def _new_unsorted_segment_sum_2_3(data, segment_ids, num_segments, name=None):
  """ Computes the sum along segments of a tensor
  Args:
    data: A `Tensor`. Must be one of the following types: float32, float64, 
      int32, uint8, int16, int8, complex64, int64, qint8, quint8, qint32, 
      bfloat16, uint16, complex128, half, uint32, uint64.
    segment_ids: A `Tensor`. Must be one of the following types: int32, int64. 
      A 1-D tensor whose size is equal to the size of data's first dimension. 
      Values should be sorted and can be repeated.
    num_segments: A `Tensor`. Must be one of the following types: int32, int64.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as data.
  """
  with ops.name_scope(name, "UnsortedSegSum", [data, segment_ids, num_segments]) as name:
    data = tf.convert_to_tensor(data)
    
    orig_dtype = data.dtype
    if 'float' in str(orig_dtype):
      data = tf.cast(data, dtype=tf.float64)
    elif 'complex' in str(orig_dtype):
      data = tf.cast(data, dtype=tf.complex128)

    if not context.executing_eagerly():
      data = ops.convert_to_tensor(data, name="input_data")
      segment_ids = ops.convert_to_tensor(segment_ids, name="segment_ids")
      num_segments = ops.convert_to_tensor(num_segments, name="num_segments")
    
    result = gen_math_ops.unsorted_segment_sum(data, segment_ids, num_segments)
    return tf.cast(result, dtype=orig_dtype)

# The original, pre-patched method is a self-generated function. 
def _new_segment_sum_2_3(data, segment_ids, name=None):
    """Computes the sum along segments of a tensor.
    Args:
      data: A `Tensor`. Must be one of the following types: float32, float64, 
        int32, uint8, int16, int8, complex64, int64, qint8, quint8, qint32, 
        bfloat16, uint16, complex128, half, uint32, uint64.
      segment_ids: A `Tensor`. Must be one of the following types: int32, int64. 
        A 1-D tensor whose size is equal to the size of data's first dimension. 
        Values should be sorted and can be repeated.
      name: A name for the operation (optional).
    Returns:
      A `Tensor`. Has the same type as data.
    """
    with ops.name_scope(name, "SortedSegSum", [data, segment_ids]) as name:
      # Note: data can be a list. To quest the type of data, convert to tensor
      # first.  
      data = tf.convert_to_tensor(data)
      orig_dtype = data.dtype
      if 'float' in str(orig_dtype):
        data = tf.cast(data, dtype=tf.float64)
      elif 'complex' in str(orig_dtype):
        data = tf.cast(data, dtype=tf.complex128)

      if not context.executing_eagerly():
        data = ops.convert_to_tensor(data, name="input_data")
        segment_ids = ops.convert_to_tensor(segment_ids, name="segment_ids")
      
      result = gen_math_ops.segment_sum(data, segment_ids)
      return tf.cast(result, dtype=orig_dtype)