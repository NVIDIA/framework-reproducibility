import tensorflow as tf
from tensorflow.python.platform import test

from fwd9m.utils import _Version as Version


# Notes about force_gpu_session:
#
# In TF1.15 and TF2.0, an apparent bug in tf.test.TestCase::session prevents us
# from throwing an exception when a GPU is not available (by setting the
# force_gpu argument of session to True). On those versions of TensorFlow, if
# force_gpu is set to True when there are two GPUs in the machine, including
# a GPU such as a GeForce GT 710 (for display only), the TensorFlow code that
# checks for the presence of GPUs produces an exception (that includes a
# mention of XLA). We assume that this bug went unnoticed in the TensorFlow CI
# test suite at least partly because those CI machines are, presumably,
# headless, single GPU machines. Therefore, the responsibility, if tests
# were only run on those TensorFlow versions, would be on the user to make
# sure to run the tests on a machine that contains a GPU. Luckly, our test
# suite runs on other versions of TensorFlow, and will throw an exception if
# there is no GPU present.
#
# In TF1.15 and TF2.0, Setting the config keyword argument of
# tf.test.TestCase::session also activates the above bug related to detecting
# the presence of a GPU. So the config keyword argument must not be set on those
# versions of TensorFlow.
def force_gpu_session(test_object):
  """A work-around for a bug in TensorFlow versions 1.15 and 2.0

  If you want to use tf.test.TestCase::session with force_gpu=True on versions
  of TensorFlow including version 1.15 or 2.0, then call this function instead
  to gracefully work-around a bug in which an XLA-related exception is thrown
  (when there are GPUs available) on some machines. On TensorFlow 1.15 and 2.0
  it will return tf.test.TestCase::session(use_gpu=True). On other versions of
  TensorFlow, it will return tf.test.TestCase::session(force_gpu=True).

  Typical usage is as follows:

  with force_gpu_session(test_object):
    # ... eager-mode or graph-mode code

  Args:
    test_object:
      A reference to the test object, an instance of tf.test.TestCase.

  Returns:
    None

  Raises:
    ValueError if test_object is not an instance of tf.test.TestCase.
    If a GPU is not available, this function is expected to raise an exception
    on all versions of TensorFlow except 1.15 and 2.0.
  """
  if not isinstance(test_object, tf.test.TestCase):
    raise ValueError("test_object must be an instance of tf.test.TestCase")
  tf_version = Version(tf.version.VERSION)
  if tf_version.in_list(['1.15', '2.0']):
    print("WARNING:"
          "an exception will not be thrown if there is no GPU present.")
    # The same bug that makes force_gpu=True throw an exception on some machines
    # containing more than one GPU also prevents us from checking for the
    # presence of a GPU using tf.test.is_gpu_available so that we can throw
    # an exception if one isn't.
    return test_object.session(use_gpu=True)
  else:
    return test_object.session(force_gpu=True)

def is_gpu_available_xla():
  tf_version = Version(tf.version.VERSION)
  if tf_version.in_list(['1.15', '2.0']):
    print("WARNING:"
          "an exception will not be thrown if there is no GPU present.")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus)>0:
      return True
    else:
      print("WARNING: no GPU present.")
      return False
  else:
    return test.is_gpu_available()