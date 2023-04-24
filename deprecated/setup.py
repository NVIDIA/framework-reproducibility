# Copyright 2020-2023 NVIDIA Corporation. All Rights Reserved
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

import importlib
import setuptools

# package_name = 'fwd9m'
package_name = 'tfdeterminism'

print("PACKAGE IMPORT WARNING (expected):")
package = importlib.import_module(package_name)

description = ("Providing reproducibility in deep learning frameworks")
url = "https://github.com/NVIDIA/%s" % package.distribution_name

print("Now running setuptools.setup()")

# Note that using python 3.6 (i.e. via the `python3.6` executable) results in
# the long_description_content_type being ignored.
# For more info, see https://github.com/di/markdown-description-example/issues/4

setuptools.setup(
  name                          = package.distribution_name,
  version                       = package.version,
  packages                      = [package_name],
  url                           = url,
  license                       = 'Apache 2.0',
  author                        = 'NVIDIA',
  author_email                  = 'duncan@nvidia.com',
  description                   = description,
  long_description              = package.long_description,
  long_description_content_type = 'text/markdown',
  install_requires              = [],
  classifiers                   = [],
  keywords                      = [],
  platforms                     = []
)
