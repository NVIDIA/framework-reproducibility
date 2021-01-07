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

import sys
import unittest

sys.path.insert(0, '..')
from fwd9m.utils import _Version as Version

class TestUtils(unittest.TestCase):

  def test_version_attributes(self):
    major = 1
    minor = 2
    version_string = "%d.%d.3" % (major, minor)
    v = Version(version_string)
    self.assertEqual(v.original_version_string, version_string)
    self.assertEqual(v.major_minor_version_string, version_string[0:3])
    self.assertEqual(v.major, major)
    self.assertEqual(v.minor, minor)

  def test_version_class(self):
    v = Version('23.45.26')
    self.assertTrue (v.in_list(['1.2', '2.8', '23.45']))
    self.assertFalse(v.in_list(['4.5', '2.9', '99.4']))
    self.assertTrue (v.at_least('23.45'))
    self.assertFalse(v.at_least('23.46'))
    self.assertTrue (v.at_most('23.45'))
    self.assertFalse(v.at_most('23.44'))
    self.assertTrue (v.between('23.44', '23.47'))
    self.assertFalse(v.between('1.2', '2.4'))
    self.assertFalse(v.between('100.2', '2.4'))
    self.assertTrue (v.between('1.0', '200.4'))
    v = Version('1.2')
    self.assertTrue (v.between('0.9', '1.4'))
    v = Version('10.09-tf3')
    self.assertTrue (v.in_list(['10.02', '10.09', '09.12']))
    self.assertTrue (v.at_least('10.09'))
    self.assertFalse(v.at_least('10.10'))
    self.assertTrue (v.at_most('10.09'))
    self.assertFalse(v.at_most('10.08'))

  def test_version_class_exceptions(self):
    self.assertRaises(ValueError, Version, '10')
    self.assertRaises(TypeError, Version, None)
    v = Version('2.3')
    self.assertRaises(ValueError, v.at_least, '1')
    self.assertRaises(ValueError, v.at_least, '1.2.3')
    self.assertRaises(ValueError, v.at_most, '012')
    self.assertRaises(ValueError, v.at_most, '012.004.435')
    self.assertRaises(ValueError, v.between, '10', '2.2')
    self.assertRaises(ValueError, v.between, '1.3', '20')
    # self.assertRaises(ValueError, v.between, '10.2', '2.26.2') # short-circuit
    self.assertRaises(ValueError, v.between, '1.2', '2.26.2')
    self.assertRaises(ValueError, v.between, '1.3.6', '20.5')

if __name__ == '__main__':
  unittest.main()
