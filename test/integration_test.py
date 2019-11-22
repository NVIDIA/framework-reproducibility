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

import argparse
import getopt
import os
import random
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv3D, MaxPooling3D, Lambda
from tensorflow.python.keras.models import Model

def summarize_weights(model, message = None):
  weights = model.get_weights()
  summary = sum(map(lambda x: x.sum(), weights))
  if message:
    print("Summary of weights (%s): %.13f" % (message, summary))
  return summary

def train(data_format="channels_first"):
  print("data_format: %s" % data_format)
  SEED = 123
  random.seed(SEED)
  np.random.seed(SEED)
  tf.set_random_seed(SEED)

  input_sz = 30
  if data_format == "channels_first":
    img_shape = (1, None, None, None)
  else:
    img_shape = (None, None, None, 1)
  img_input = Input(shape=img_shape)
  pool_size = (4, 4, 4)

  filters = 8
  kernel_size = (3, 3, 3)
  padding_mode = 'valid'
  f = 'relu'
  use_bias = True
  x = Conv3D(filters, kernel_size, use_bias=use_bias, activation=f,
             padding=padding_mode, data_format=data_format,
             name='conv1')(img_input)
  x = Conv3D(filters, kernel_size, use_bias=use_bias, activation=f,
             padding=padding_mode, data_format=data_format,
             name='conv2')(x)
  x = MaxPooling3D(pool_size, strides=(1, 2, 2), data_format=data_format,
                   name='pool')(x)
  x = Conv3D(filters*2, kernel_size, use_bias=use_bias, dilation_rate=(1,1,1),
             activation=f, padding=padding_mode, data_format=data_format,
             name='conv3')(x)
  x = Conv3D(1, (1,1,1), use_bias=use_bias, activation='sigmoid',
             data_format=data_format, name='predictions')(x)
  model = Model(img_input, x, name='mymodel')

  model.compile(optimizer='Adam',loss='binary_crossentropy',
                metrics=["binary_accuracy"])

  batch_size = 10
  N = 100
  if data_format == "channels_first":
    input_shape = (N, 1, input_sz, input_sz, input_sz)
  else:
    input_shape = (N, input_sz, input_sz, input_sz, 1)
  x_train = np.random.random_sample(input_shape)
  y_train = np.round(np.random.rand(N,1,1,1,1))

  summarize_weights(model, "before training")
  for i in range(N):
    if y_train[i]==1:
      x_train[i,:,:,:,:]=x_train[i,:,:,:,:]-0.2
  history = model.fit(x_train, y_train, epochs=4, batch_size=batch_size,
                      shuffle=False)
  loss_history = history.history['loss']
  print("loss history: %r" % loss_history)
  return summarize_weights(model, "after training"), loss_history[-1]

def main(argv):
  parser = argparse.ArgumentParser()
  # parser.add_argument('--skip_fixes', dest='skip_fixes', action=)
  try:
    opts, _ = getopt.getopt(argv, 'sl')
  except getopt.GetopError:
    print("%s [-s][-l]" % sys.argv[0])
    sys.exit(2)
  skip_fixes = False
  data_format = 'channels_first'
  for opt, _ in opts:
    if opt == '-s': skip_fixes = True
    if opt == '-l': data_format = 'channels_last'
  if not skip_fixes:
    if os.environ.get('NVIDIA_TENSORFLOW_VERSION'):
      # inside an NGC TF container
      os.environ['TF_DETERMINISTIC_OPS'] = 'true'
    else: # running on stock TensorFlow
      sys.path.insert(0, '..')
      from tfdeterminism import patch
      patch()
  summary, final_loss = train(data_format)
  f = open('log/integration_test.summary', 'w')
  f.write("%.16f" % summary)
  f.close()

if __name__=="__main__":
  main(sys.argv[1:])
