import numpy as np
import tensorflow as tf

def empty(rank):
  shape = (0,) * rank
  return np.array([], dtype=np.float32).reshape(shape)

def empty_gradient():
  try:
    tf.test.compute_gradient(tf.nn.bias_add, [empty(3), empty(1)])
  except ValueError as e:
    print(e)
  try:
    tf.test.compute_gradient(tf.linalg.matmul, [empty(2), empty(3)])
  except ValueError as e:
    print(e)

if __name__ == "__main__":
  empty_gradient()
