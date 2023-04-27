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

from setuptools import setup, find_packages
import os

distribution_name = 'framework-reproducibility'
package_name = 'fwr13y'

# This file needs to be executed during installation. It's not possible to
# import the full package during installation because it will fail to import if
# all the supported frameworks have not been installed. By temporarility
# appending to sys.path, it's possible to just import from the version module.
import sys
sys.path.append(package_name)
from version import __version__ as version
sys.path.remove(package_name)

readme = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                      "pypi_description.md")
with open(readme, "r") as fp:
  long_description = fp.read()

description = ("Providing reproducibility in deep learning frameworks")
url = "https://github.com/NVIDIA/%s" % distribution_name
install_requires = [] # intentionally not including the framework packages

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python'
]

keywords = ("framework tensorflow gpu deep-learning determinism "
            "reproducibility pytorch seed seeder noise noise-reduction "
            "variance-reduction atomics ngc gpu-determinism deterministic-ops "
            "frameworks gpu-support d9m r13y fwr13y")

setup(
  name                          = distribution_name,
  version                       = version,
  packages                      = ['fwr13y'],
  url                           = url,
  license                       = 'Apache 2.0',
  author                        = 'NVIDIA',
  author_email                  = 'duncan@nvidia.com',
  description                   = description,
  long_description              = long_description,
  long_description_content_type = 'text/markdown',
  install_requires              = install_requires,
  classifiers                   = classifiers,
  keywords                      = keywords,
  platforms                     = ['TensorFlow', 'PyTorch', 'PaddlePaddle']
)
