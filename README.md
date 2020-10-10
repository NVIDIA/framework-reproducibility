# TensorFlow Determinism

This main README is currently focused on GPU-determinism with TensorFlow.
For GPU-determinism with PyTorch, see the [pytorch page](./pytorch.md).

## Announcement

In the next release of this package (version 0.4.0), the distribution name will
be changed from `tensorflow-determinism` to `framework-determinism` and the
package name will be changed from `tfdeterminism` to `fwd9m`. These changes
reflect an intention going forward for this repo to increasingly support
determinism in multiple deep learning frameworks.

Users of `tfdeterminism.patch` will need to use the to-be-deprecated
`fwd9m.tensorflow.patch` and will be encouraged to migrate to using
`fwd9m.tensorflow.enable_determinism` instead, which is intended to provide
compatibility, indefinitely, with future versions of TensorFlow.

## Introduction

This repository serves three purposes:

  1. Provide up-to-date information (in this file) about non-determinism
     sources and solutions in TensorFlow and beyond, with a focus on determinism
     when running on GPUs.
  2. Provide a patch to attain various levels of GPU-specific determinism in
     stock TensorFlow, via the installation of the `tensorflow-determinism` pip
     package.
  3. Be the location where a TensorFlow determinism debug tool will be released
     as part of the `tensorflow-determinism` pip package.

For more information, please watch the video of the GTC 2019 talk
[_Determinism in Deep Learning_][1]. The desciption under that video also
includes links to the slides from the talk and to a poster presentation on this
topic.

## Installation

Note that, currently, you only need to install and use this package if you're
using a version of TensorFlow for which there is a determinism patch available.
There is currently no patch available for TensorFlow versions 2.1 or greater
because the effect of the patch that was developed for earlier versions has been
upstreamed into these newer versions.

The next release of this package (version 0.4.0) will include an
`enable_determinism` function that can be applied to any version of TensorFlow
to obtain the latest and best solutions, including any new patches (including
for earlier versions of TensorFlow).

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

There are currently three ways to access GPU-deterministic op functionality in
TensorFlow for most deep learning applications:

1. Use the lastest version of stock TensorFlow (currently version 2.3), which
   implements all of the currently-available deterministic op solutions. It does
   not require patching.
2. Use an NVIDIA NGC TensorFlow Docker image (version >= 19.06).
3. Use version 1.14, 1.15, or 2.0 of stock TensorFlow with GPU support, plus the
   application of `tfdeterminism.patch`. Version 2.1 of stock TensorFlow
   does not have a patch avaliable (and does not require earlier patching) and
   includes many of the deterministic op solutions.

For the latest status of GPU-determinism for different versions of TensorFlow,
see the [tables](#confirmed-current-gpu-specific-sources-of-non-determinism-with-solutions)
below. Although in the short-term, solutions may be deployed as patches for
stock TensorFlow or via the NGC TensorFlow container images, the long-term
intention and plan is to continue upstreaming all solutions into stock
TensorFlow.

### Stock TensorFlow Version 2.3

Stock TensorFlow version 2.3 implements most of the currently-available
GPU-deterministic op solutions. It is missing deterministic backprop for
bilinear resizing, which is provided by
[NGC TF Docker image](#nvidia-gpu-cloud-ngc-tensorflow-docker-images) version
20.03+.

The following Python code is running on a machine in which `pip` package
`tensorflow=2.3.0` has been installed correctly.

```
import tensorflow as tf
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# Now build your graph and train it
```

Stock TensorFlow version 2.3 with GPU support can be installed as follows:

```
pip install tensorflow=2.3.0
```

The TensorFlow project includes [detailed instructions][3] for installing
TensorFlow with GPU support.

### NVIDIA GPU Cloud (NGC) TensorFlow Docker Images

NGC TensorFlow Docker images, starting with version 19.06, implement
GPU-deterministic op functionality. Version 19.12 (and beyond) also implements
[multi-algorithm deterministic cuDNN convolutions][1003], which solves the
problem of some layer configurations causing an exception to be thrown with the
message "No algorithm worked!". Version 20.03 (and beyond) also implements
deterministic backprop for bilinear resizing.

In Python code running inside the container, deterministic ops can be enabled
as follows:

```
import tensorflow as tf
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# Now build your graph and train it
```

The following table shows which version of TensorFlow each NGC Docker image
version is based on:

 NGC TF Image Version | TensorFlow Version |
:---------------------|:-------------------|
 19.06                | 1.13               |
 19.07 - 19.10        | 1.14               |
 19.11 - 20.01        | 1.15 / 2.0         |
 20.02 - 20.03        | 1.15 / 2.1         |
 20.06 - 20.07        | 1.15 / 2.2         |

For information about pulling and running the NVIDIA NGC Docker images, see
[these instructions][2].

### Stock TensorFlow Version < 2.1

Versions 1.14, 1.15, and 2.0 of stock TensorFlow implement a reduced form of
GPU-deterministic op functionality, which must be supplemented with a patch
provided in this repo. The following Python code is running on a machine in
which `pip` package `tensorflow-gpu=2.0.0` has been installed correctly and on
which `tensorflow-determinism` has also been installed (as shown in the
[installation](#installation) section above).

```
import tensorflow as tf
from tfdeterminism import patch
patch()
# use tf as normal
```

Stock TensorFlow with GPU support can be installed as follows:

```
pip install tensorflow-gpu=2.0.1
```

The TensorFlow project includes [detailed instructions][3] for installing
TensorFlow with GPU support.

### Additional Ingredients in the Determinism Recipe

Deterministic op functionality, such as that enabled by
`TF_DETERMINISTIC_OPS=1`, can only contribute to fully-deterministic operation
of a model or training regime in the context of a deterministic system. The
following are notes on various other items that must be addressed in order to
ensure that your model trains or infers with prefect reproducibility.

#### Seeds ####

You'll also need to set any and all appropriate random seeds:

```
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

You also may need to set the environment variable `PYTHONHASHSEED` to a
reproducible value before you start the python process.

In the TensorFlow version 1 API, `tf.random.set_seed` was `tf.set_random_seed`.

In most models, the effect of setting `tf.random.set_seed` is to ensure that the
trainable variables (the weights and biases) in your model are pseudorandomly
initialized the same way each time. Every time `tf.random.set_seed` is called,
with a particular seed value, the pseudorandom number generator that TensorFlow
uses to initialize the trainable variables is reset ("seeded") deterministically
according to that seed.

If you're using dropout, which introduces a pseudorandom dropout sequence during
training, then to achieve deterministic results you will need to reset
the pseudorandom number generator that is used to produce those dropout
sequences. The pseudorandom dropout sequence introduced by `tf.keras.Dropout`
layers _will_ be reset by `tf.random.set_seed`.

If you would like to control the seed used for dropout independently of the seed
used for trainable variable initialization, then you can call
`tf.random.set_seed` just before training (e.g. just before calling `model.fit`
but after constructing the model and running `model.compile`). However, if you
would like to explicitly control the seed used for the dropout sequence, then
you can specify it using the `seed` argument of the `tf.keras.layers.Dropout`
constructor.

Note that the `shuffle` argument of the Keras model `fit` method defaults to
`True` (enabled). This pseudorandom shuffling is controlled by a pseudorandom
number generator that is also reset reproducibly by `tf.random.set_seed`,
probably the same pseudorandom number generator that TensorFlow uses throughout.

You may also need to provide a seed for any other ops you use that rely on
pseudorandom number generators, such as
`tf.image.sample_distorted_bounding_box`, otherwise they may use a random seed
and their operation will not be repducible.

#### Dataset Sharding ####

~~If you're using `tf.data.Dataset`, you should not shard the dataset. This
is achieved by either not calling the `shard()` method, or by setting its
`num_shards` parameter to 1.~~

From at least TensorFlow 2.2 onward, if not earlier, `tf.data.Dataset::shard`
appears to operate deterministically.

#### Data-Loader Parallelism ####

When the data-loader pipeline is stateful and is replicated into multiple
asynchronous threads, the threads can interact with each other, resulting in
non-deterministic operation of the overall data-loader functionality. The
most common example of this is when pseudorandom number genration is used in
the data-loader pipeline, such as for data augmentation. When the same
underlying pseudorandom number generator state is used in all the threads, it
will result in non-deterministic functionality. This happens when the default
pseudorandom number generator (e.g. from numpy) is used in the data-loader.
There are two solution to this:

  1. Make sure that the instances of the objects operating in the parallel
     threads have their own pseudorandom number generator state. For example,
     see [numpy.random.Generator][5].
  2. Run the data-loader code in only one thread.

How you run the data-loader code in only one thread depends on how you're
running the data-loader. If you're using `tf.keras.Model::fit()` or
`tf.keras.Model::fit_generator()` then `workers` should be set to not more than
`1`.

Since the validation process runs in the main thread, if the validation
process uses the same stateful data pipeline as the training data-loader then
these two processes will also run in separate threads and you'll wind up with
the same problem. In this case, you need to set `workers` to `0` (zero), which
will cause the data-loader to be run in the main thread.

#### While Loop Parallelism ####

The use of `tf.while_loop` when `parallel_iterations` is greater than 1 (note
that 10 is the default) may introduce non-determinism into model functionality.
Additionally, the [AutoGraph Transformations][6], that operate while compiling
code into a graph when (TF2 API) `tf.function` (or the use of the @tf.function
decorator) is used, may lead to [loops][7] being implemented using
`tf.while_loop` and, therefore, parallelized.

The current work-around, to prevent this non-determinism, is to use
`tf.autograph.experimental.set_loop_options` inside the `for` loop, with
`parallel_iterations=1`.

It has not yet been determined if this non-determinism is specific to operation
on a GPU or if it is a general issue in TensorFlow.

#### Gradient Gating ####

In some TensorFlow API interfaces, it is possible to limit the amount of
paralellism that is allowed during back-propagation calculations.

If used, `tf.gradients` (not supported in eager execution) should have its
`gate_gradients` parameter set to `True` (the default is `False`).

The non-Keras, TF1 API optimizers, based on the `tf.compat.v1.train.Optimizer`,
such as `tf.compat.v1.train.AdamOptimizer`, accept a `gate_gradients` parameter
in their `minimize` and `compute_gradient` methods. If this is set to
`tf.compat.v1.train.Optimizer.GATE_NONE` then there is an increased probability
of introducing nondeterminism in the backprop. If not specified,`gate_gradients`
defaults to `tf.compat.v1.train.Optimizer.GATE_OP`, which theoretically could
lead to the introduction of nondeterminism, although I have not yet seen this
happen in a real application. The setting of this parameter that minimizes
parallelism in the backprop calculation (and leads to the lowest performance)
is `tf.compat.v1.train.Optimizer.GATE_GRAPH`. If you've removed all other
sources of nondeterminism and nondeterminism is still being introducted
somewhere, then you could try this setting, if it's available to you. I don't
recommend changing `gate_gradients` to `GATE_GRAPH` as a standard practice.

The Keras optimizers, such as `tf.keras.optimizers.SGD`, which are now the
standard optimizers in the TF2 API, do not offer a `gate_gradients` parameter
in the `minimize` or `get_gradients` methods. There is also no ability to
control gradient gating on `tf.GradientTape` for calculation of gradients in
eager execution.

The reduced availablilty of control of gradient gating in TF2, with eager
execution and an increased reliance on the (often high-level) Keras interface,
doesn't seem to be a real problem with respect to GPU-determinism.

#### Multi-GPU using Horovod ####

If you're using Horovod for multi-GPU training, you may need to disable Tensor
Fusion (assuming that the non-determinism associated with Tensor Fusion has not
yet been resolved, see Horovod [PR 1130][503]):

```
os.environ['HOROVOD_FUSION_THRESHOLD']='0'
```

#### Multi-GPU using MirroredStrategy ####

Prior to TensorFlow version 2.3, when using `tf.data.Dataset::shuffle` with
`tf.distribute.MirroredStrategy` (or perhaps any `tf.distribute` strategy),
setting `reshuffle_each_iteration=True` introduces nondeterminism. This
appears to have been fixed in TensorFlow version 2.3. See TF issue
[38197](https://github.com/tensorflow/tensorflow/issues/38197) for more
information.

Also, note that the `seed` parameter of the `shuffle` method should always be
set in order to obtain determinism.

#### CPU ####

If you want to obtain determinism when your ops are running on the CPU, you may
need to limit the number of CPU threads used. In the TF1 API, this can be
acheived as follows:

```
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
with tf.compat.v1.Session(config=config):
  ...
```

In the TF2 API, it can be achieved like this:

```
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
```

It should not be necessary to limit the number of CPU threads used when your
ops are not running on the CPU (e.g. when they're running on a GPU).

## Detailed Status of Determinism in TensorFlow and Beyond

Confirmed and likely sources of non-determinism, along with any existing
solutions, are being tracked here.

### GPU-Specific Sources of Non-Determinism

#### Historic GPU-Specific Sources of Non-Determinism

In the past, `tf.math.reduce_sum` and `tf.math.reduce_mean` operated
nondeterministically when running on a GPU. This was resolved before
TensorFlow version 1.12. These ops now function deterministically
by default when running on a GPU.

#### Confirmed Current GPU-Specific Sources of Non-Determinism (With Solutions)

Note | Source                                                                        | TF 1.14, 1.15,<br>2.0  | NGC 19.06+ /<br>TF 2.1 | TF 2.2     | TF 2.3     |
----:|:------------------------------------------------------------------------------|:-----------------------|:-----------------------|:-----------|:-----------|
   1 | TF auto-tuning of cuDNN convolution<br>algorithms                             | TCD or TDP             | TDO                    | TDO        | TDO        |
   2 | `tf.nn.conv*d` and<br>`tf.keras.layers.Conv*D`<br>backprop                    | TCD or TDP             | TDO                    | TDO        | TDO        |
   3 | `tf.nn.max_pool*d` and<br>`tf.keras.layers.MaxPool*D` backprop                | TCD or TDP             | TDO                    | TDO        | TDO        |
   4 | `tf.nn.bias_add` backprop                                                     | TDP                    | TDO                    | TDO        | TDO        |
   5 | XLA reductions on GPU                                                         | NS                     | NS                     | TDO        | TDO        |
   6 | `tf.nn.ctc_loss` backprop                                                     | NS                     | NS                     | NS         | TDO        |
   7 | Fused sofmax/crossentropy:<br>`tf.nn.*_cross_entropy_with_logits`<br>backprop | NS                     | NS                     | NS         | NS         |

Note | Source                                                                                                                                  | TF < 2.4  | NGC 20.03+ | TF 2.4 |
----:|:----------------------------------------------------------------------------------------------------------------------------------------|:----------|:-----------|:-------|
   8 | `tf.image.resize` with `method=ResizeMethod.BILINEAR`<br>and `tf.keras.layers.UpSampling2D` with<br>`interpolation='bilinear'` backprop | NS        | TDO        | TDO    |
   9 | `tf.image.resize` with `method=ResizeMethod.NEAREST`<br>and `tf.keras.layers.UpSampling2D` with<br>`interpolation='nearest'` backprop   | NS        | NS         | NS     |
  10 | `tf.math.segment_sum` and `tf.math.unsorted_segment_sum`<br>forward, and `tf.gather` and `tfa.image.dense_image_warp`<br>backprop       | NS        | NS         | NS     |
  11 | `tf.image.crop_and_resize` backprop to `image` (on CPU<br>or GPU) and backprop to `boxes`                                               | NS        | NS         | NS     |
  12 | `tf.sparse.sparse_dense_matmul` forward                                                                                                 | NS        | NS         | NS     |

##### Key to the Solutions Referenced Above

 Solution | Description                                                                                                                                                                                     |
:---------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 TCD      | Set environment variable `TF_CUDNN_DETERMINISTIC` to '1' or 'true'. Also *do not* set environment variable `TF_USE_CUDNN_AUTOTUNE` at all (and particularly *do not* set it to '0' or 'false'). |
 TDO      | Set environment variable `TF_DETERMINISTIC_OPS` to '1' or 'true'. Also *do not* set environment variable `TF_USE_CUDNN_AUTOTUNE` at all (and particularly *do not* set it to '0' or 'false').   |
 TDP      | Apply `tfdeterminism.patch`. Note that solution TDO is in stock TensorFlow v2.1 (see [PR 31465](https://github.com/tensorflow/tensorflow/pull/31465)), which makes patching unnecessary.        |
 NS       | There is no solution in the specified version, but there may be a solution in other versions (as shown)                                                                                         |

##### Notes

  1. From NGC TF 19.12 onwards and stock TensorFlow 2.2 onwards, the
     cuDNN forward and backward convolution algorithms are selected
     deterministically from several deterministic algorithms. Prior to this (i.e.
     NGC 19.11 and earlier, and stock TensorFlow 2.1 and earlier), there is only
     one deterministic algorithm selected for each of the forward and two
     backward paths. In those versions of TensorFlow, some layer configurations
     are not supported (resulting in an exception being thrown with the message
     "No algorithm worked!").
  2. cuDNN convolution backprop algorithms (backprop to both input feature
     maps and to trainable variables) are exposed through `tf.nn.conv1d`,
     `tf.nn.conv2d`, and `tf.nn.conv3d`. Functionality that is built on top is
     also affected, such as `tf.keras.layers.Conv1D`, `tf.keras.layers.Conv2D`,
     and `tf.keras.layers.Conv3D`
  3. cuDNN max pooling is exposed through `tf.nn.max_pool1d`,
     `tf.nn.max_pool2d`, and `tf.nn.max_pool3d`. Functionality that is built on
     top is also affected, such as `tf.keras.layers.MaxPool1D`,
     `tf.keras.layers.MaxPool2D`, and `tf.keras.layers.MaxPool3D`. This solution
     does not currently have unit tests in the TensorFlow repo but has been
     confirmed to work in production models.
  4. Note that various Keras layers, including the Keras convolution layers
     (`tf.keras.layers.Conv*D`), are built on top of `tf.nn.bias_add`.
     Therefore, when `use_bias=True` the deterministic functionality of the
     layer is dependent on the deterministic functionality of `tf.nn.bias_add`.
     Prior to TensorFlow version 2.2, this deterministic `tf.nn.bias_add`
     backprop solution will not work when XLA JIT compilation is enabled due to
     XLA reductions on GPU not being deterministic (see
     [this comment](https://github.com/tensorflow/tensorflow/pull/34887#discussion_r355610837)
     on PR 34887). This is resolved in TensorFlow version 2.2 and NGC TF Docker
     images based on that version of TensorFlow.
  5. Without this solution, when XLA JIT compilation is enabled, any op that
     relies on XLA reductions, whether in the forward or backward direction,
     will introduce nondeterministic noise.
  6. I had previously assumed that deterministic cuDNN CTC loss was exposed, via
     `tf.nn.ctc_loss`, by changes that ended up appearing in stock TensorFlow
     version 2.3 (see TF issue
     [38151](https://github.com/tensorflow/tensorflow/issues/38151)), but more
     recent experiments, the results of which I am in the process of publishing,
     suggest that this is not true and that `tf.nn.ctc_loss` has
     nondeterministic backprop when operating on either CPU or GPU.
  7. Fused softmax/cross-entropy ops `tf.nn.softmax_cross_entropy_with_logits`
     and `tf.nn.sparse_softmax_cross_entropy_with_logits`, accessed via
     `tf.keras.losses.categorical_crossentropy`,
     `tf.keras.losses.CategoricalCrossentropy`,
     `tf.keras.losses.sparse_categorical_crossentropy`, and
     `tf.keras.losses.SparseCategoricalCrossentropy`), are known to inject
     nondetermininsm into backprop. See TensorFlow
     [issue 38185](https://github.com/tensorflow/tensorflow/issues/38185).
     A confirmed work-around is to use separate non-fused softmax and
     cross-entropy ops. For example, assuming you're using `tf.keras`, select
     the activation on the final layer (e.g. a `Dense` layer) to be 'softmax'
     (which chooses `tf.nn.softmax`) and then, for the loss function, continue
     to use `tf.keras.losses.categorical_crossentropy` (possibly by using its
     wrapper class `tf.keras.losses.CategoricalCrossentropy`) or
     `tf.keras.losses.sparse_categorical_crossentropy` (possibly by using its
     wrapper class `tf.keras.losses.SparseCategoricalCrossentropy`). Since it
     uses non-fused kernels, the work-around will be lower performance.
     Theoretically, you should ensure that the loss function parameter
     `from_logits` is set to `False` (the default), perhaps only for performance
     reasons since setting it to `True` is a no-op arithmetically and does not
     appear to contribute to nondeterminism. A patch is
     [in development](https://github.com/NVIDIA/framework-determinism/pull/21).
  8. `tf.image.resize` with `method=ResizeMethod.BILINEAR` (TF2 API):
     `BILINEAR` is the default `method` setting. In the TF1 API, this
     functionality is accessed via `tf.image.resize_bilinear`. It is also exposed
     through `tf.keras.layers.UpSampling2D` with `interpolation='bilinear'`
     (which is not the default `interpolation` setting).
  9. `tf.image.resize` with `method=ResizeMethod.NEAREST` (TF2 API): In the TF1
     API, this functionality is accessed via `tf.image.resize_nearest_neighbor`.
     It is also exposed through `tf.keras.layers.UpSampling2D` with
     `interpolation='nearest'` (which is the default `interpolation` setting). A
     potential work-around is to use `tf.keras.layers.Conv2DTranspose` (see
     issues [#12](https://github.com/NVIDIA/framework-determinism/issues/12) and
     [#24](https://github.com/NVIDIA/framework-determinism/issues/24))
  10. Segment reduction ops `tf.math.segment_sum` and
     `tf.math.unsorted_segment_sum` have nondeterministic forward operation on
     GPU. Other ops that are dependent on these ops, including `tf.gather` and
     `tfa.image.dense_image_warp` (both in backprop), therefore also operate
     nondeterministically. See
     [Issue 39751](https://github.com/tensorflow/tensorflow/issues/39751).
  11. Backprop to `image` on `tf.image.crop_and_resize` introduces
      nondeterministic noise when running on either CPU or GPU. Backprop to
      `boxes` introduces nondeterministic noise when running on GPU. See
      TensorFlow
      [Issue 42033](https://github.com/tensorflow/tensorflow/issues/42033) for
      more information.
  12. The forward path of `tf.sparse.sparse_dense_matmul` introduces
      nondeterminism for `tf.float32` and (allegedly) for `tf.float64`. See
      TF [Issue 18037](https://github.com/tensorflow/tensorflow/issues/18037).

#### Other Possible GPU-Specific Sources of Non-Determinism

Going beyond the above-mentioned sources, in version 1.12 of TensorFlow (and
also in the master branch on 2019-03-03, afer release 1.31.1), the following
files call CUDA `atomicAdd` either directly or indirectly. This makes them
candidates for the injection of non-determinism.

* `scatter_functor_gpu.cu.h`
* `scatter_nd_op_gpu.cu.cc`
* `dilation_ops_gpu.cu.cc`
* `maxpooling_op_gpu.cu.cc`
* `svd_op_gpu.cu.cc`
* `cuda_kernel_helper_test.cu.cc`
* `depthwise_conv_op_gpu.h`
* `resampler_ops_gpu.cu.cc`
* `histogram_op_gpu.cu.cc`
* `stateful_random_ops_gpu.cu.cc`

Unless you are using TensorFlow ops that depend on these files (i.e. ops with
similar names), then your model will not be affected by these potential sources
of non-determinism.

Beyond `atomicAdd`, there are ten other CUDA [atomic functions][4] whose use
could lead to the injection of non-determinism, such as `atomicCAS` (the most
generic, atomic compare and swap). Note also that the word 'atomic' was present
in 167 files in the TensorFlow repo and some of these may be related to the use
of CUDA atomic operations. It's important to remember that it's possible to use
CUDA atomic operations without injecting non-determinism, and that, therefore,
when CUDA atomic operations are present in op code, it doesn't guarantee that
the op injects non-determinism into the computation.

### Sources of Non-Determinism in TensorFlow Unrelated to GPU

* [Issue 29101](https://github.com/tensorflow/tensorflow/issues/29101): Random
  seed not set in graph context of `Dataset#map`. This may have been resolved
  in version 1.14 of TensorFlow.
* `tf.data.Dataset` with more than one shard (aka worker). The work-around is to
  use only one shard.
* `tf.while_loop` with `parallel_iterations` > 1 (default is 10). See
  [While Loop Parallelism](#while-loop-parallelism)

### Sources of Non-Determinism Beyond TensorFlow

* TensorRT timing-based kernel schedule. Each time an inference engine is
  generated, it could be slightly different, particularly if there is varying
  load on the machine used to run TensorRT. There is a solution planned for
  this.
* Horovod Tensor Fusion. Work-around: disable Tensor Fusion by setting the
  environment variable `HOROVOD_FUSION_THRESHOLD` to '0'. See Horovod
  [PR 1130][503].
* Before this project started (in 2018), PyTorch was widely considered to have a
  more complete and coherent GPU determinism story than TensorFlow. At the time
  of writing (2020-02-25), it is no longer clear that one framework is superior
  to the other in this regard. For more information about determinism in
  PyTorch, see the
  [reproducibility documentation](http://bit.ly/pytorch-determinism) in the
  PyTorch repo.

## Relevant Links

This section catalogs relevant links.

### TensorFlow Issues

GitHiub issues in the TensorFlow project:

Number                                                         | Title                                                                                    | Date Opened | Status |
--------------------------------------------------------------:|:-----------------------------------------------------------------------------------------|:------------|:-------|
 [2652](https://github.com/tensorflow/tensorflow/issues/2652)  | Backward pass of broadcasting on GPU is non-deterministic                                | 2016-06-03  | Closed |
 [2732](https://github.com/tensorflow/tensorflow/issues/2732)  | Mention that GPU reductions are nondeterministic in docs                                 | 2016-06-08  | Closed |
[13932](https://github.com/tensorflow/tensorflow/issues/13932) | Non-determinism from `tf.data.Dataset.map` with random ops                               | 2017-10-23  | Closed |
[16889](https://github.com/tensorflow/tensorflow/issues/16889) | Problems Getting TensorFlow to behave Deterministically                                  | 2018-02-09  | Open   |
[18037](https://github.com/tensorflow/tensorflow/issues/18037) | tf.sparse_tensor_dense_matmul makes small errors with<br>tf.float32 matrices on GPU      | 2018-03-27  | Open   |
[18096](https://github.com/tensorflow/tensorflow/issues/18096) | Feature Request: Support for configuring deterministic<br>options of cuDNN conv ...      | 2018-03-29  | Open   |
[22398](https://github.com/tensorflow/tensorflow/issues/22398) | CUDA implementation of BiasAddGrad op is non-determinstic                                | 2018-09-19  | Closed |
[29101](https://github.com/tensorflow/tensorflow/issues/29101) | Random seed not set in graph context of `Dataset#map`                                    | 2019-05-28  | Open   |
[38151](https://github.com/tensorflow/tensorflow/issues/38151) | Test deterministic cuDNN CTC loss                                                        | 2020-04-01  | Open   |
[38185](https://github.com/tensorflow/tensorflow/issues/38185) | Add GPU-deterministic back-prop for fused<br>softmax/cross-entropy ops                   | 2020-04-02  | Open   |
[38197](https://github.com/tensorflow/tensorflow/issues/38197) | Model not deterministic, even though<br>os.environ['TF_DETERMINISTIC_OPS'] = '1' set     | 2020-04-03  | Closed |
[39751](https://github.com/tensorflow/tensorflow/issues/39751) | Non-deterministic behaviour: tf.math.unsorted_segment_sum<br>uses CUDA Atomic Operations | 2020-05-21  | Open   |
[40514](https://github.com/tensorflow/tensorflow/issues/40514) | TFBertForSequenceClassification: Non-deterministic when<br>training on GPU ...           | 2020-06-16  | Closed |
[42033](https://github.com/tensorflow/tensorflow/issues/42033) | Add deterministic tf.image.crop_and_resize backprop                                      | 2020-08-04  | Open   |

### Related Project Issues

GitHub issues in dependent or related projects:

 Project      | Number                                                          | Title                                                                 | Date Opened | Status |
:-------------|----------------------------------------------------------------:|:----------------------------------------------------------------------|:------------|:-------|
 Keras        | [12800](https://github.com/keras-team/keras/issues/12800)       | Unable to get reproducible results using Keras / TF on GPU            | 2019-05-07  | Closed |
 Tensorpack   | [902](https://github.com/tensorpack/tensorpack/issues/902)      | How to run Tensorpack training with deterministic behavior            | 2018-09-20  | Closed |
 transformers | [5603](https://github.com/huggingface/transformers/issues/5063) | Non-deterministic training issue on GPU: TF-BERT                      | 2020-06-16  | Open   |

### TensorFlow Pull Requests

The following pull requests (and some inidividual commits) are those in the
TensorFlow GitHub repo that are directly related to this project. As we have
[discovered](scripts/README.md#find-tensorflow-commits), 1.8% of all commits
seem to reference, or have some relationship with, "determinism" or
"deterministic". As of 2020-01-30, that was 1,391 commits.

ID                                                           | Title                                                            | Status | Date Merged | Version |
------------------------------------------------------------:|:-----------------------------------------------------------------|:-------|:------------|:--------|
[24747](https://github.com/tensorflow/tensorflow/pull/24747) | Add cuDNN deterministic env variable (only<br>for convolution).  | merged | 2019-01-15  | 1.14    |
[25269](https://github.com/tensorflow/tensorflow/pull/25269) | Add deterministic cuDNN max-pooling                              | merged | 2019-01-30  | 1.14    |
[25796](https://github.com/tensorflow/tensorflow/pull/25796) | Added tests for `TF_CUDNN_DETERMINISTIC`                         | merged | 2019-02-22  | 1.14    |
[c2790][1001]<sup>1</sup>                                    | Add a decorator to disable autotuning during<br>test executions. | merged | 2019-03-13  | 1.14    |
[29667](https://github.com/tensorflow/tensorflow/pull/29667) | Add release note about `TF_CUDNN_DETERMINISTIC`                  | merged | 2019-08-06  | 1.14    |
[31389](https://github.com/tensorflow/tensorflow/pull/31389) | Enhance release notes related to<br>`TF_CUDNN_DETERMINISTIC`     | merged | 2019-08-07  | 1.14    |
[31465](https://github.com/tensorflow/tensorflow/pull/31465) | Add GPU-deterministic `tf.nn.bias_add`                           | merged | 2019-10-17  | 2.1     |
[32979](https://github.com/tensorflow/tensorflow/pull/32979) | Fix typo in release note                                         | closed |             |         |
[33483](https://github.com/tensorflow/tensorflow/pull/33483) | Fix small typo in v2.0.0 release note                            | merged | 2019-10-25  | 2.1     |
[33803](https://github.com/tensorflow/tensorflow/pull/33803) | Enable tf.nn.bias_add python op tests<br>to work in eager mode   | merged | 2020-02-12  | 2.2     |
[33900](https://github.com/tensorflow/tensorflow/pull/33900) | Address problems with use_deterministic_cudnn<br>test decorator  | merged | 2020-01-09  | 2.2     |
[34887](https://github.com/tensorflow/tensorflow/pull/34887) | Add info about `TF_DETERMINISTIC_OPS` to v2.1<br>release notes   | merged | 2019-12-09  | 2.1     |
[34951][1003]                                                | Add multi-algorithm deterministic cuDNN<br>convolutions          | merged | 2020-01-27  | 2.2     |
[35006](https://github.com/tensorflow/tensorflow/pull/35006) | Fix version 2.1 release note regarding<br>TF_DETERMINISTIC_OPS   | merged | 2019-12-20  | 2.1     |
[e3195][1002]<sup>1</sup>                                    | [XLA/GPU] Convert reduction into tree reduction<br>using padding | merged | 2020-01-07  | 2.2     |
[8b7a3][1004]<sup>1</sup>                                    | [XLA] Respect TF_DETERMINISTIC_OPS env variable<br>for reductions| merged | 2020-02-19  | 2.2     |
[37377](https://github.com/tensorflow/tensorflow/pull/37377) | [XLA] follow-up on GPU-deterministic reductions                  | merged | 2020-03-09  | 2.3     |
[9e096][1005]<sup>1</sup>                                    | Use the CUDNN_CTC_LOSS_ALGO_DETERMINISTIC<br>algorithm ...       | merged | 2020-03-10  | 2.3     |
[38089](https://github.com/tensorflow/tensorflow/pull/38089) | Add reminder to test deterministic cuDNN CTC loss                | closed |             |         |
[38509](https://github.com/tensorflow/tensorflow/pull/38509) | List deterministic op func bug fixes in v2.2<br>release notes    | merged | 2020-04-15  | 2.2     |
[39243](https://github.com/tensorflow/tensorflow/pull/39243) | GPU-deterministic tf.image.resize (bilinear)                     | merged | 2020-09-22  | 2.4     |
 
Notes:
  1. These are individual commits.

[1001]: https://github.com/tensorflow/tensorflow/commit/c27909ea80e8823dbf4f7176ab69991a630356a1
[1002]: https://github.com/tensorflow/tensorflow/commit/e31955d9fb34ae7273354dc2347ba99eea8c5280
[1003]: https://github.com/tensorflow/tensorflow/pull/34951
[1004]: https://github.com/tensorflow/tensorflow/commit/8b7a3db0b6e09415b5640be4986fb4d7c6e5209a
[1005]: https://github.com/tensorflow/tensorflow/commit/9e096debc4a0909deb69970f38bee7b77e5e5f7d

### PyTorch Pull Requests

ID                                                     | Title                                                         | Status | Date Merged | Version |
------------------------------------------------------:|:--------------------------------------------------------------|:-------|:------------|:--------|
[33795](https://github.com/pytorch/pytorch/pull/33795) | Enhance reproducibility documentation                         | merged | 2020-03-06  | 1.5     |

### Horovod Pull Requests

ID                                                     | Title                                                         | Status | Date Merged | Version |
------------------------------------------------------:|:--------------------------------------------------------------|:-------|:------------|:--------|
 [1130][503]                                           | Add grouped allreduce feature                                 | Open   |             |         |

### Miscellaneous

* [Gradient injection][505] in the testing of op backprop determinism in
  TensorFlow tests.
* Two Sigma: [A Workaround for Non-Determinism in
  TensorFlow](http://bit.ly/two-sigma-determinism)
* Chainer [PR 2710](https://github.com/chainer/chainer/pull/2710): cuDNN
  Deterministic mode
* SE / Stack Overflow: [Tensorflow: Different results with the same random seed][501]
* SE / Stack Overflow: [Are tensorflow random values guaranteed to be the same inside a single run? (comment)][502]
* SE / Data Science: [Making Keras + Tensorflow code execution deterministic on a GPU][504]

[501]: https://stackoverflow.com/questions/54047654/tensorflow-different-results-with-the-same-random-seed
[502]: https://stackoverflow.com/questions/52213325/are-tensorflow-random-values-guaranteed-to-be-the-same-inside-a-single-run#comment91376212_52213325
[503]: https://github.com/horovod/horovod/pull/1130
[504]: https://datascience.stackexchange.com/questions/14812/making-keras-tensorflow-code-execution-deterministic-on-a-gpu/
[505]: ./test/gradient_injection.md

## Credits

Here are the names of some of the people who have helped out with this project.
If any names are missing, then please let us know.

Christoph Angerer,
Ben Barsdell,
Kevin Brown,
Carl Case,
Bryan Catanzaro,
Sharan Chetlur,
Joey Conway,
Emilio Coutinho,
Sanjoy Das,
Timo Denk,
Luke Durant,
Marc Edgar,
Adam Ellsworth,
Mostafa Hagog,
Kaixi Hou,
George Karpenkov,
Tero Karras,
Bob Keating,
Andrew Kerr,
Xiang Bo Kong,
Nicolas Koumchatzky,
Jorge Albericio Latorre,
Simon Layton,
Ned Letcher,
Jose Alvarez Lopez,
Nathan Luehr,
Conrado Silva Miranda,
John Montrym,
Michael O'Connor,
Lauri Peltonen,
Rakesh Ranjan,
Jussi Rasanen,
Duncan Riach (PIC),
Josh Romero,
Mikko Ronkainen,
Gavin Seegoolam,
Dilip Sequeria,
Hao Shen,
Matthijs De Smedt,
Valentina Taviani,
Phil Teare,
Amirhossein Tebbifakhr,
Matthijs Van keirsbilck,
Kevin Vincent,
Stephen Warren,
Shu Wang,
Hao Wu,
Yifang Xu,
Tim Zaman,
William Zhang.

[1]: http://bit.ly/determinism-in-deep-learning
[2]: https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow
[3]: https://www.tensorflow.org/install/gpu
[4]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
[5]: https://numpy.org/doc/stable/reference/random/generator.html
[6]: https://www.tensorflow.org/guide/function#autograph_transformations
[7]: https://www.tensorflow.org/guide/function#loops
