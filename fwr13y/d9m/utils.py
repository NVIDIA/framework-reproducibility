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

import re

def _confirm_string(input):
  if not isinstance(input, str): raise TypeError("Only strings are allowed!")

class _Version:
  """Provides version comparison functionality"""
  def __init__(self, version_string):
    """Provide a version string containing at least a major and minor version"""
    _confirm_string(version_string)
    version_pieces = re.split('\.|-', version_string)
    if len(version_pieces) < 2:
      raise ValueError("The version string must contain at least a major and a minor version")
    major = version_pieces[0]
    minor = version_pieces[1]
    self.original_version_string = version_string
    self.major_minor_version_string = major + '.' + minor
    self.major = int(major)
    self.minor = int(minor)

  def __str__(self):
    string = (
        "Version object: " +
        "original_version_string: %s; " % self.original_version_string +
        "major_minor_version_string: %s; " % self.major_minor_version_string +
        "major: %i; " % self.major +
        "minor: %i" % self.minor)
    return string

  def in_list(self, list_of_versions):
    """Is the version in the list of version provided?"""
    for version in list_of_versions: _confirm_string(version)
    return self.major_minor_version_string in list_of_versions

  def _only_major_and_minor(self, version):
    version_pieces = version.split('.')
    if len(version_pieces) != 2:
      raise ValueError("The version string must contain a major and a minor version (only)")
    major = int(version_pieces[0])
    minor = int(version_pieces[1])
    return major, minor

  def at_least(self, oldest_version):
    """Is the version at least the oldest_version provided?"""
    _confirm_string(oldest_version)
    oldest_major, oldest_minor = self._only_major_and_minor(oldest_version)
    if (self.major > oldest_major or
        self.major == oldest_major and self.minor >= oldest_minor):
      return True
    else:
      return False

  def at_most(self, newest_version):
    """Is the version at most the newest version provided?"""
    _confirm_string(newest_version)
    newest_major, newest_minor = self._only_major_and_minor(newest_version)
    if (self.major < newest_major or
        self.major == newest_major and self.minor <= newest_minor):
      return True
    else:
      return False

  def between(self, oldest_version, newest_version):
    """Is the version between the oldest and newest versions
    provided (inclusive)?"""
    _confirm_string(oldest_version)
    _confirm_string(newest_version)
    if self.at_least(oldest_version) and self.at_most(newest_version):
      return True
    else:
      return False

  def equals(self, target_version):
    """Is the version equal to the version provided?"""
    _confirm_string(target_version)
    target_major, target_minor = self._only_major_and_minor(target_version)
    if (self.major == target_major and self.minor == target_minor):
      return True
    else:
      return False
