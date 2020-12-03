# Copyright 2019 The TensorFlow-Determinism Authors. All Rights Reserved
#
# Some code in this file was derived from the TensorFlow project and/or
# has been, or will be, contributed to the TensorFlow project.
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

import collections
import tensorflow as tf
import yaml
import os
import re
import importlib
import sys
import unittest
sys.path.insert(0, '..')

from fwd9m.utils import _Version as Version
SUCCESS = 0
FAILURE = 1

def get_container():
  ngc_tf_container_version_string = os.environ.get('NVIDIA_TENSORFLOW_VERSION')
  if ngc_tf_container_version_string:
    container_id = "NGC_" + ngc_tf_container_version_string
  else:
    container_id = "TF_" + tf.version.VERSION
  return container_id

def read_tests_yml(container_id, yml_file, tests_dir):
  file = open(yml_file, 'r')
  file_data = file.read()
  file.close()
  yml_info = yaml.load(file_data)

  tests_data = {}

  for key, val in yml_info.items():
    file_path = tests_dir+'/'+val['filename']
    print(file_path)
    if not os.path.isfile(file_path):
      raise Exception("Test file %s does not exist under folder %s!"
                      % (val['filename'], tests_dir) )

    if 'stock' not in val and 'ngc' not in val:
      tests_data[val['filename']] = False
    else:
      stock_vers_onwards = str(val['stock'])
      ngc_vers_onwards = str(val['ngc'])

      if container_id[:3]=="NGC":
        in_ngc_cont = True
        ngc_vers = Version(container_id[4:])
      else:
        in_ngc_cont = False
        tf_vers = Version(tf.version.VERSION)

      if not in_ngc_cont:
        tests_data[val['filename']] = False if tf_vers.at_least(stock_vers_onwards) else True
      else:
        tests_data[val['filename']] = False if ngc_vers.at_least(ngc_vers_onwards) else True

  return tests_data

class IntegrationTest():
  _tests_data = {}
  _tests_dir = None
  _runner = None

  @classmethod
  def initialize(cls, yml="enable_determinism.yml", tests_dir='./tests_dir'):
    cls._container_id = get_container()
    cls._tests_dir = tests_dir
    cls._tests_data = read_tests_yml(cls._container_id, yml, tests_dir)
    cls._runner = unittest.TextTestRunner()

  @classmethod
  def run_tests(cls):
    _TestResult = collections.namedtuple("_TestResult", ["status", "message"])
    test_status = SUCCESS
    total_failure_counts = 0
    ret_summary = []
    for file_name, skip in cls._tests_data.items():
      if not skip:
        discover = unittest.defaultTestLoader.discover(cls._tests_dir, pattern=file_name)
        result = cls._runner.run(discover)
        local_failures = result.failures + result.expectedFailures + \
                         result.errors
        total_failure_counts += len(local_failures)
        if not result.wasSuccessful():
          test_status = FAILURE

        if local_failures:
          ret = _TestResult(status="failure", message="%d failues %d errors at %s" %(len(result.failures), len(result.errors) ,file_name))
        else:
          ret = _TestResult(status="ok", message="OK at %s" % file_name)
      else:
        ret = _TestResult(status="skipped", message="Skipped at %s" % file_name)

      ret_summary.append(ret)

    print("Summary for container %s:" % cls._container_id)
    for ret in ret_summary:
      if ret.message:
        print(ret.message)
    return test_status

if __name__=="__main__":
  IntegrationTest.initialize()
  sys.exit(IntegrationTest.run_tests())
