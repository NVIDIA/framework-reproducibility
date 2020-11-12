from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops

# The original, pre-patched function is automatically-generated. Therefore, we
# cannot provide a URL to its location in the source repository.
# For the history of this patch, please refer to
# https://github.com/tensorflow/tensorflow/issues/39751
def _new_unsorted_segment_sum(data, segment_ids, num_segments, name=None):
  """ERROR: docstring should have been added programatically. """
  with ops.name_scope(
      name, "UnsortedSegmentSum", [data, segment_ids, num_segments]) as name:
    # Note that data can be a vector-like list (or an n-dimensional
    # tensor-like list of lists). We convert to tensor here to replicate the
    # behavior of the pre-existing op.
    data = tf.convert_to_tensor(data)

    # Note that this patch does not provide determinism when the dtype of the
    # data argument is tf.float64 or tf.complex128.
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

# The original, pre-patched function is automatically-generated. Therefore, we
# cannot provide a URL to its location in the source repository.
# For the history of this patch, please refer to
# https://github.com/tensorflow/tensorflow/issues/39751
def _new_segment_sum(data, segment_ids, name=None):
  """ERROR: docstring should have been added programatically. """
  with ops.name_scope(name, "SegmentSum", [data, segment_ids]) as name:
    # Note that data can be a vector-like list (or an n-dimensional
    # tensor-like list of lists). We convert to tensor here to replicate the
    # behavior of the pre-existing op.
    data = tf.convert_to_tensor(data)

    # Note that this patch does not provide determinism when the dtype of the
    # data argument is tf.float64 or tf.complex128.
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
