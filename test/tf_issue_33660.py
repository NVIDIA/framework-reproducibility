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

# Reported at https://github.com/tensorflow/tensorflow/issues/33660

import numpy as np
import sys
import tensorflow as tf

def section(msg):
  print("=======================================================")
  print(msg)
  print("-------------------------------------------------------")

def empty(rank):
  shape = (0,) * rank
  return np.array([]).reshape(shape)

def eager_repro_bias_add():
  section("EAGER REPRO with tf.nn.biad_add")
  tf.test.compute_gradient(tf.nn.bias_add, [empty(3), empty(1)])

def eager_repro_matmul():
  section("EAGER RERPO with tf.linalg.matmul")
  tf.test.compute_gradient(tf.linalg.matmul, [empty(2), empty(3)])


def eager_work_around():
  section("EAGER WORK-AROUND")
  input_val = empty(3)
  bias_val = empty(1)

  def bias_add_1(input_val):
    return tf.nn.bias_add(input_val, bias_val)

  def bias_add_2(bias_val):
    return tf.nn.bias_add(input_val, bias_val)

  input_jacobians = tf.test.compute_gradient(bias_add_1, [input_val])
  bias_jacobians = tf.test.compute_gradient(bias_add_2, [bias_val])

def empty_tensor(shape):
  return tf.constant([], shape=shape)

def graph_repro():
  section("GRAPH MODE REPRO")
  tf.compat.v1.disable_eager_execution()
  input_shape = output_shape = (0, 0, 0)
  bias_shape = (0,)
  input_tensor = empty_tensor(input_shape)
  bias_tensor = empty_tensor(bias_shape)
  output_tensor = tf.nn.bias_add(input_tensor, bias_tensor)
  with tf.compat.v1.Session() as sess:
    jacobians = tf.compat.v1.test.compute_gradient(
        [input_tensor, bias_tensor], [input_shape, bias_shape], output_tensor,
        output_shape)

def random_tensor(shape):
  return tf.constant(np.random.random_sample(shape))

# Taken from tensorflow/python/ops/gradient_checker_v2_test.py / testEmptyMatMul
def existing_test_repro():
  section("EXISTING TEST REPRO")
  def f(x, y):
    return tf.linalg.matmul(x, y)

  x = random_tensor((0, 3))
  y = random_tensor((3, 4))

  jacobians = tf.test.compute_gradient(f, [x, y])

if __name__ == "__main__":
  if not tf.__version__.startswith("2.0"):
    raise("This is designed to run with TensorFlow version 2.0")
  exec("%s()" % sys.argv[1])
