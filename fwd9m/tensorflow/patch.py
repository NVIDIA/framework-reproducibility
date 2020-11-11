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

from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops

from ..utils import _Version as Version
from ..version import __version__ as package_version

from .patch_bias_add import _new_bias_add
from .patch_segment_reduction import _new_segment_sum
from .patch_segment_reduction import _new_unsorted_segment_sum
from .patch_softmax_xent import _new_softmax_cross_entropy_with_logits
from .patch_sparse_softmax_xent import _new_sparse_softmax_cross_entropy_with_logits

# This function was used to patch tf.nn.bias_add in a limited range of stock
# TensorFlow versions. It is now deprecated and we are no longer developing it.
# enable_determinism should be used.
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

def _patch_unsorted_segment_sum():
  _new_unsorted_segment_sum.__doc__ = tf.math.unsorted_segment_sum.__doc__
  math_ops.unsorted_segment_sum = _new_unsorted_segment_sum # access via public API
  tf.math.unsorted_segment_sum = _new_unsorted_segment_sum # access via public API

def _patch_segment_sum():
  _new_segment_sum.__doc__ = tf.math.segment_sum.__doc__
  math_ops.segment_sum = _new_segment_sum # access via public API
  tf.math.segment_sum = _new_segment_sum # access via public API

def _patch_fused_softmax_cross_entropy():
  # Non-sparse
  _new_softmax_cross_entropy_with_logits.__doc__ = tf.nn.softmax_cross_entropy_with_logits.__doc__
  tf.nn.softmax_cross_entropy_with_logits = _new_softmax_cross_entropy_with_logits  # access via public API
  nn.softmax_cross_entropy_with_logits = _new_softmax_cross_entropy_with_logits  # called from tf.keras.layers.convolutional.Conv
  nn_ops.softmax_cross_entropy_with_logits = _new_softmax_cross_entropy_with_logits  # called from tests

  # tf.nn.softmax_cross_entropy_with_logits_v2 = _new_softmax_cross_entropy_with_logits
  # softmax_cross_entropy_with_logits_v2 # maybe tensorflow/python/ops/nn_ops.py

  # Sparse TO-DO
  # tf.nn.sparse_softmax_cross_entropy_with_logits = _new_sparse_softmax_cross_entropy_with_logits_1_14  # access via public API
  # nn.sparse_softmax_cross_entropy_with_logits = _new_sparse_softmax_cross_entropy_with_logits_1_14  # called from tf.keras.layers.convolutional.Conv
  # nn_ops.sparse_softmax_cross_entropy_with_logits = _new_sparse_softmax_cross_entropy_with_logits_1_14

def _patch_fused_sparse_softmax_cross_entropy():
  # sparse
  _new_sparse_softmax_cross_entropy_with_logits.__doc__ = tf.nn.sparse_softmax_cross_entropy_with_logits.__doc__
  tf.nn.sparse_softmax_cross_entropy_with_logits = _new_sparse_softmax_cross_entropy_with_logits  # access via public API
  nn.sparse_softmax_cross_entropy_with_logits = _new_sparse_softmax_cross_entropy_with_logits  # called from tf.keras.layers.convolutional.Conv
  nn_ops.sparse_softmax_cross_entropy_with_logits = _new_sparse_softmax_cross_entropy_with_logits  # called from tests