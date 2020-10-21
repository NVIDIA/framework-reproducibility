import tensorflow as tf
from fwd9m.utils import _Version as Version

  # 1. On TF1.15 and TF2.0, bug in tf.testcase prevents us from throwing an 
  #    exception when GPU is not available. So the responsibility is on the user 
  #    to run the test with machine with GPU.
  # 2. This patch cannot capture the case when `force_gpu=True` is used together
  #    with `config=` within `session()`. For all the avoided and working 
  #    scenarios, see the following:
  
  # To avoided cases:
  # session(force_gpu=True/False, config=....)
  # session(use_gpu=True/False, config=....)
  # session(config=....)
  # session(force_gpu=True) $$

  # Working cases:
  # session(force_gpu=False)
  # session(use_gpu=True/False)
  # session() $$def _tf_session(test_cls, **kw)

def _tf_session(test_cls, **kw):
  """Get around force_gpu bug version TF2.0 and TF1.15
      All other versions are OK
  """
  tf_version = Version(tf.version.VERSION)
  if not tf_version.in_list(tf_version.tf_force_gpu_bug_version):
    return test_cls.session(**kw)

  if 'force_gpu' not in kw:
    return test_cls.session(**kw)
  
  force_gpu = kw['force_gpu']

  def replace_force_gpu_key(d):
    new_dict = {}
    for k, v in d.items():
      if k!='force_gpu':
        new_dict[k] = v
      else:
        new_dict['use_gpu'] = v
    return new_dict

  if force_gpu:
    print("WARNING: The bug in TF1.15 and TF2.0 prevents from throwing \
          exception at `tf.test.is_gpu_available` if no GPU is present")
    print("WARNING: substitute `force_gpu` with `use_gpu` as a work-around")
    new_kw = replace_force_gpu_key(kw)
    return test_cls.session(**new_kw)
  else:
    return test_cls.session(**kw)