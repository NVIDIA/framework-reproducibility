import tensorflow as tf
import numpy as np

class DeterministicTest(tf.test.TestCase):

  def _randomInts(self, shape, high, dtype):
    return tf.constant(
        np.random.randint(low=0, high=high, size=shape).astype(dtype))

  def _randomFloats(self, shape, dtype, normalized_rows=False):
    a = (2 * np.random.random_sample(shape) - 1).astype(dtype)

    if normalized_rows:

      def normalize(row):
        return row / row.sum()

      a = np.apply_along_axis(normalize, 1, a)

    return tf.constant(a)

  def _testDeterministicGradients(self, exclusive_labels):
    with self.session(force_gpu=True):
      batch_size = 1024
      classes_count = 1000
      logits_shape = (batch_size, classes_count)
      logits_dtype = np.float32
      logits = self._randomFloats(logits_shape, logits_dtype)
      if exclusive_labels:
        labels_shape = (batch_size)
        labels_dtype = np.int32
        labels = self._randomInts(labels_shape, classes_count, labels_dtype)
      else:
        labels_shape = logits_shape
        labels_dtype = logits_dtype
        labels = self._randomFloats(labels_shape, labels_dtype,
                                    normalized_rows=True)
      output_shape = (batch_size)
      output_dtype = logits_dtype

      def gradients(local_seed):
        np.random.seed(local_seed)
        upstream_gradients = self._randomFloats(output_shape, output_dtype)
        with tf.GradientTape(persistent=True) as tape:
          tape.watch(logits)
          if exclusive_labels:
            tested_op = tf.nn.sparse_softmax_cross_entropy_with_logits
          else:
            tested_op = tf.nn.softmax_cross_entropy_with_logits
          op_output = tested_op(labels=labels, logits=logits)
          gradient_injector_output = op_output * upstream_gradients
        return tape.gradient(gradient_injector_output, logits)

      repeat_count = 5
      for seed in range(repeat_count):
        result_a = gradients(seed)
        result_b = gradients(seed)
        self.assertAllEqual(result_a, result_b)

  # def testExclusiveLabelsDeterministicGradients(self):
  #   self._testDeterministicGradients(exclusive_labels=True)

  def testDistributionLabelsDeterministicGradients(self):
    self._testDeterministicGradients(exclusive_labels=False)

if __name__ == '__main__':
  tf.test.main()