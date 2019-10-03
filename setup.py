# Copyright 2019 The TensorFlow-Determinism Authors. All Rights Reserved
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

from setuptools import setup
import os

# This file needs to be executed during installation. It's not possible to
# import the full tfdeterminism package during installation because it will
# fail to import if TensorFlow has not yet been installed. By temporarility
# appending 'tfdeterminism' to sys.path, it's possible to just import the
# version module.
import sys
sys.path.append('tfdeterminism')
from version import __version__
sys.path.remove('tfdeterminism')

readme = os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md")
with open(readme, "r") as fp:
  long_description = fp.read()

description = "Tracking, debugging, and patching non-determinism in TensorFlow"
url = "https://github.com/NVIDIA/tensorflow-determinism"
install_requires = [] # intentionally not including tensorflow-gpu

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python'
]

setup(
  name                          = 'tensorflow-determinism',
  version                       = __version__,
  packages                      = ['tfdeterminism'],
  url                           = url,
  license                       = 'Apache 2.0',
  author                        = 'NVIDIA',
  author_email                  = 'duncan@nvidia.com',
  description                   = description,
  long_description              = long_description,
  long_description_content_type = 'text/markdown',
  install_requires              = install_requires,
  classifiers                   = classifiers,
  keywords                      = "tensorflow gpu deep-learning determinism"
)
