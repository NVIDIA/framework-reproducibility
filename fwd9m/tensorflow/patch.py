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

from . import patch_bias_add
from .. import utils
from .. import version

# This function was used to patch tf.nn.bias_add in a limited range of stock
# TensorFlow versions. It is now deprecated and we are no longer developing it.
# enable_determinism should be used.
def _patch(_silent=False):
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
  if not _silent:
    print("WARNING: %s has been deprecated. Please use enable_determinism "
          "(which supports all versions of TensorFlow)." % __name__)
  if os.environ.get('NVIDIA_TENSORFLOW_VERSION'):
    raise TypeError("%s: TensorFlow inside NGC containers does not "
                    "require patching" % __name__)
  tf_vers = utils._Version(tf.version.VERSION)
  if tf_vers.between('1.14', '2.0'):
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    patch_bias_add._patch_bias_add()
    if not _silent:
      print("TensorFlow version %s has been patched using %s version %s" %
            (tf_vers.original_version_string, __name__,
             version.__version__))
  else:
    raise TypeError("%s: No patch available for version %s of TensorFlow" %
                    (__name__, tf_vers.original_version_string))
