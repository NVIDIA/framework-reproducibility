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
from . import patch as patch_api_module
from . import patch_segment_sum
from . import patch_unsorted_segment_sum
from .. import utils
from ... import version

def _enable_determinism():
  """Provides a best-effort recipe to increase framework determinism when
    running on GPUs.

    Call this method after importing TensorFlow and before constructing any
    graphs.

    This function cannot address all possible sources of nondeterminism. Please
    see further instructions at
    https://github.com/NVIDIA/framework-reproducibility to understand how to use
    it in a larger deterministic context.

    Arguments:
      None

    Returns: None
  """
  tf_vers = utils._Version(tf.version.VERSION)
  ngc_tf_container_version_string = os.environ.get('NVIDIA_TENSORFLOW_VERSION')
  if ngc_tf_container_version_string:
    in_ngc_cont = True
    ngc_vers = utils._Version(ngc_tf_container_version_string)
  else:
    in_ngc_cont = False

  if tf_vers.at_least('2.8'):
    tf.config.experimental.enable_op_determinism()
  elif in_ngc_cont and ngc_vers.at_least('19.06') or tf_vers.at_least('2.1'):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
  elif not in_ngc_cont and tf_vers.between('1.14', '2.0'):
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    patch_api_module._patch(_silent=True)

  if (in_ngc_cont and ngc_vers.between('19.06', '21.12') or
      tf_vers.between('1.14', '2.6')):
    patch_segment_sum._patch_segment_sum()
    patch_unsorted_segment_sum._patch_unsorted_segment_sum()
    if tf_vers.at_least('2.5'):
      os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'

  print("%s (version %s) has been applied to TensorFlow "
        "version %s" % (__name__, version.__version__,
                        tf_vers.original_version_string))
