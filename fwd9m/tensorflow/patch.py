# Copyright 2019-2020 NVIDIA Corporation. All Rights Reserved
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
  print("WARNING: %s has been deprecated. Please use enable_determinism (which "
        "supports all versions of TensorFlow)." % __name__)
  if os.environ.get('NVIDIA_TENSORFLOW_VERSION'):
    raise TypeError("%s: TensorFlow inside NGC containers does not "
                    "require patching" % __name__)
  tf_vers = Version(tf.version.VERSION)
  if tf_vers.between('1.14', '2.0'):
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    _patch_bias_add()
    # Apply the fused softmax/cross-entropy patch here
    print("TensorFlow version %s has been patched using %s version %s" %
          (tf_vers.original_version_string, __name__,
           package_version))
  else:
    raise TypeError("%s: No patch available for version %s of TensorFlow" %
                    (__name__, tf_vers.original_version_string))

def _patch_bias_add():
  _new_bias_add.__doc__ = tf.nn.bias_add.__doc__
  tf.nn.bias_add = _new_bias_add # access via public API
  nn.bias_add = _new_bias_add # called from tf.keras.layers.convolutional.Conv
  nn_ops.bias_add = _new_bias_add # called from tests

# The original, pre-patched method can be viewed at
# https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/ops/nn_ops.py#L2628
#
# This patched version of bias_add does not implement some of the error checks
# provided by the original op. For more information, see the list of test cases
# excluded from the testing of the patched op functionality.
def _new_bias_add(value, bias, data_format=None, name=None):
  """ERROR: docstring should have been added programatically. """
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
