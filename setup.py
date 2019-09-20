from setuptools import setup
import os
# TODO: Get version from package files. Normal package import is problematic
#       here because TensorFlow may not be installed yet during installation
#       and this file must be executable during installation. 
__version__ = '0.1.0'

readme = os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md")
with open(readme, "r") as fp:
  long_description = fp.read()

description = "Debugging and patching non-determinism in TensorFlow"
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
