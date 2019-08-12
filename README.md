# TensorFlow Determinism

The TensorFlow determinism debug tool described in the GTC 2019 talk
[_Determinism in Deep Learning_][1] will be released here. Note that the
features of this tool have not yet been released.

Updates on the status of determinism in deep learning will also be conveyed here
along with dynamic patches for TensorFlow.

## Deterministic TensorFlow Solutions

There are currently two main ways to access GPU-deterministc functionality in
TensorFlow for most deep learning applications. The first way is to use an
NVIDIA NGC TensorFlow container. The second way is to use version 1.14.0 of
pip-installed TensorFlow with the addition of a patch supplied in this repo.

### NVIDIA NGC TensorFlow Containers

NGC TensorFlow container versions 19.06 (based on TensorFlow 1.13.1) and 19.07
(based on TensorFlow 1.14.0) both implement GPU-deterministic TensorFlow
functionality. In Python code running inside the container, this can be enabled
as follows:


```
import tensorflow as tf
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# Now build your graph and train it
```

For information about pulling and running the NVIDIA NGC containers, see [these
instructions][2].

### PyPI (pip-installed) TensorFlow

Version 1.14.0 of standard TensorFlow implements a reduced form of GPU
determinism, which must be supplemented with a patch provided in this repo.
The following Python code is running on a machine in which pip package
tensorflow-gpu=1.14.0 has been installed correctly. The Python code is also
run in a location from which the `tfdeterminism` package from this repo is
accessible (either include the `tfdeterminism` package in your project
directory or add its path using `sys.path.append()`).

```
import tensorflow as tf
from tfdeterminism import patch
patch()
# build your graph and train it
```

Tensorflow can be easily installed as follows:

```
pip install tensorflow-gpu=1.14.0
```

The TensorFlow project includes [detailed instructions][3] for installing
TensorFlow with GPU support.

[1]: http://bit.ly/determinism-in-deep-learning
[2]: https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow
[3]: https://www.tensorflow.org/install/gpu