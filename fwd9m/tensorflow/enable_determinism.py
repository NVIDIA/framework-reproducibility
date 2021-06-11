# Copyright 2020 NVIDIA Corporation. All Rights Reserved
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

import inspect
import os
import re

import tensorflow as tf

# By calling the deprecated patch API here, we continue to test its effect
# without having to test it explicitly. Note that this form of import
# necessarily breaks the Google Python Style Guide rule to import packages
# and modules only (and not individual functions).
from ..tensorflow import patch as patch_bias_add
from . import patch_segment_sum
from . import patch_unsorted_segment_sum
from .. import utils
from .. import version

def _enable_determinism(seed=None):
  """Provides a best-effort recipe to increase framework determinism when
    running on GPUs.

    Call this method either before or after explicitly importing TensorFlow,
    but always before constructing any graphs.

    This function cannot address all possible sources of non-determinism. Please
    see further instructions at https://github.com/NVIDIA/framework-determinism
    to understand how to use it in a larger deterministic context.

    Arguments:
      seed: <fill in>

    Returns: None
  """
  tf_vers = utils._Version(tf.version.VERSION)
  ngc_tf_container_version_string = os.environ.get('NVIDIA_TENSORFLOW_VERSION')
  if ngc_tf_container_version_string:
    in_ngc_cont = True
    ngc_vers = utils._Version(ngc_tf_container_version_string)
  else:
    in_ngc_cont = False
  if not in_ngc_cont and tf_vers.between('1.14', '2.0'):
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    patch_bias_add._patch(_silent=True)
  if in_ngc_cont and ngc_vers.at_least('19.06') or tf_vers.at_least('2.1'):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
  if in_ngc_cont and ngc_vers.at_least('19.06') or tf_vers.at_least('1.14'):
    patch_segment_sum._patch_segment_sum()
    patch_unsorted_segment_sum._patch_unsorted_segment_sum()
    # Apply the fused softmax/cross-entropy patch here
    pass
  # TODO: Add other recipe items (e.g. seed)
  print("%s (version %s) has been applied to TensorFlow "
        "version %s" % (__name__, version.__version__,
                        tf_vers.original_version_string))
