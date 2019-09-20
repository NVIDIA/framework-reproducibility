# TensorFlow Determinism

The TensorFlow determinism debug tool described in the GTC 2019 talk
[_Determinism in Deep Learning_][1] will be released here. Note that the
features of this tool have not yet been released.

Updates on the status of determinism in deep learning will also be conveyed here
along with dynamic patches for TensorFlow.

## Installation

Use `pip` to install:

```
pip install tensorflow-determinism
```

This will install a package that can be imported as `tfdeterminism`. The
installation of `tensorflow-determinism` will not automatically install
TensorFlow. The intention of this is to allow you to install your chosen
version of TensorFlow. You will need to install your chosen version of
TensorFlow before you can import and use `tfdeterminism`.

## Deterministic TensorFlow Solutions

There are currently two main ways to access GPU-deterministc functionality in
TensorFlow for most deep learning applications. The first way is to use an
NVIDIA NGC TensorFlow container. The second way is to use version 1.14.0 of
pip-installed TensorFlow with GPU support plus the application of a patch
supplied in this repo.

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

Version 1.14.0 of stock TensorFlow implements a reduced form of GPU
determinism, which must be supplemented with a patch provided in this repo.
The following Python code is running on a machine in which `pip` package
`tensorflow-gpu=1.14.0` has been installed correctly and on which
`tensorflow-determinism` has also been installed (as shown in the
[installation](#installation) section above).

```
import tensorflow as tf
from tfdeterminism import patch
patch()
# build your graph and train it
```

Tensorflow with GPU support can be installed as follows:

```
pip install tensorflow-gpu=1.14.0
```

The TensorFlow project includes [detailed instructions][3] for installing
TensorFlow with GPU support.

[1]: http://bit.ly/determinism-in-deep-learning
[2]: https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow
[3]: https://www.tensorflow.org/install/gpu