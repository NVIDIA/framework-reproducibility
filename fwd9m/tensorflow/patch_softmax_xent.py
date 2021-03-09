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
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export

# NOTE: This patch provides GPU-determinism for
# `tf.nn.softmax_cross_entropy_with_logits` via patching the op
# `gen_nn_ops.softmax_cross_entropy_with_logit` with sequential calling of
# softmax, logarithm and reduce_sum which are known deterministic.

def _patch_softmax_xent():
  gen_nn_ops.softmax_cross_entropy_with_logits = _new_soft_xent_op

# The original, pre-patched python wrapper can be viewed at
# gen_nn_ops.py which is a auto-generated code and the c++ code implementation
# is \core\kernels\xent_op.cc.

def _new_soft_xent_op(features, labels, name=None):

  if not context.executing_eagerly():
    features = ops.convert_to_tensor(features)
    labels = ops.convert_to_tensor(labels)
    features_rank = array_ops.shape(features).shape
    labels_rank = array_ops.shape(labels).shape
  else:
    features_rank = array_ops.rank(features)
    labels_rank = array_ops.rank(labels)

  if features_rank == 1 or labels_rank == 1:
    raise ValueError("must be 2d")
  elif features_rank == 3 or labels_rank == 3:
    raise ValueError("rank 2, but is rank 3")
      
  softmax = tf.nn.softmax(logits=features, axis=-1)
  epsilon_ = constant_op.constant(K.epsilon(), dtype=softmax.dtype.base_dtype)
  softmax = clip_ops.clip_by_value(softmax, epsilon_, 1. - epsilon_)
  # ??? * needs the data type to be the same
  bp = (softmax - labels)
  return -tf.reduce_sum(tf.math.log(softmax) * labels, axis=-1), bp

