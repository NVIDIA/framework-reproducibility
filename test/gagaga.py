import argparse
import numpy as np
import os
import random
import sys
import tensorflow as tf
from tensorflow.python.ops import gen_sparse_ops
sys.path.insert(0, '..')
import fwd9m.tensorflow as fwd9m_tensorflow
import utils

fwd9m_tensorflow.enable_determinism()
os.environ['TF_DETERMINISTIC_OPS']='1'

random.seed(123)
tf.random.set_seed(123)
parser = argparse.ArgumentParser(description='TensorFlow entry point')
parser.add_argument('--precision', type=int, default=32)
args = parser.parse_args()

if args.precision == 32:
  dtype = tf.float32
elif args.precision == 64:
  dtype = tf.float64
else:  
  print('Precision argument must be 32 or 64')
  sys.exit()

m = 10
k = 20
n = 100
sparse_input_dense_shape = [m, k]
dense_input_shape = [k, n]
indices = []
prob_of_index=0.3
for row in range(m):
  for col in range(k):
    if random.uniform(0, 1) < prob_of_index:
      indices.append([row, col])
dest=tf.float16
values = tf.random.normal(
    shape=[len(indices)], mean=0.0, stddev=1.0, dtype=dtype, seed=123)


values = tf.cast(values, dtype=dest)
sparse_input = tf.SparseTensor(indices, values, sparse_input_dense_shape)

dense_input = tf.random.normal(
    dense_input_shape, mean=0.0, stddev=1.0, dtype=dtype, seed=123)
dense_input = tf.cast(dense_input, dtype=dest)

with tf.device('/gpu:0'):
  result_1 = tf.sparse.sparse_dense_matmul(sparse_input, dense_input)
# result_1 = tf.cast(result_1, dtype)

  result_2 = tf.sparse.sparse_dense_matmul(sparse_input, dense_input)
# result_2 = tf.cast(result_2, dtype)

# result_1 = gen_sparse_ops.sparse_tensor_dense_mat_mul(
#           a_indices=indices,
#           a_values=values,
#           a_shape=sparse_input_dense_shape,
#           b=dense_input,
#           adjoint_a=False,
#           adjoint_b=False)
# result_2 = gen_sparse_ops.sparse_tensor_dense_mat_mul(
#           a_indices=indices,
#           a_values=values,
#           a_shape=sparse_input_dense_shape,
#           b=dense_input,
#           adjoint_a=False,
#           adjoint_b=False)

# diff = result_1 - result_2
# print("Sum of difference is %e" % (np.sum(diff))) 
# print("Difference matrix is ", diff)
