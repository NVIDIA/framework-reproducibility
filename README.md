# TensorFlow Determinism

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

1. Use stock TensorFlow version 2.2, which implements most of the
   currently-available deterministic op solutions. It does not require patching.
2. Use an NVIDIA NGC TensorFlow Docker image (version >= 19.06).
   Note that version 20.03+ implements deterministic backprop for bilinear
   resizing, which has not yet been released in stock TensorFlow.
3. Use version 1.14, 1.15, or 2.0 of stock TensorFlow with GPU support, plus the
   application of `tfdeterminism.patch`. Version 2.1 of stock TensorFlow
   does not require patching and includes almost all of the available
   deterministic op solitions, except for
   [multi-algorithm deterministic cuDNN convolutions][1003].

The long-term intention and plan is to continue upstreaming all solutions into
stock TensorFlow.

### Stock TensorFlow Version 2.2

Stock TensorFlow version 2.2 implements most of the currently-available
GPU-deterministic op solutions. It is missing deterministic backprop for
bilinear resizing, which is provided by
[NGC TF Docker image](#nvidia-gpu-cloud-ngc-tensorflow-docker-images) version
20.03+.

The following Python code is running on a machine in which `pip` package
`tensorflow=2.2.0` has been installed correctly.

```
import tensorflow as tf
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# Now build your graph and train it
```

Stock TensorFlow version 2.2 with GPU support can be installed as follows:

```
pip install tensorflow=2.2.0
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
 20.06                | 1.15 / 2.2         |

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
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

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

For deterministic functionality, some types of models may require
`gate_gradients=tf.train.Optimizer.GATE_OP` in the session config. I have never
actually seen this be required though.

#### Multi-GPU using Horovod ####

If you're using Horovod for multi-GPU training, you may need to disable Tensor
Fusion (assuming that the non-determinism associated with Tensor Fusion has not
yet been resolved, see Horovod [PR 1130][503]):

```
os.environ['HOROVOD_FUSION_THRESHOLD']='0'
```

#### CPU ####

If you want to obtain determinism when your ops are running on the CPU, you may
need to limit the number of CPU threads used:

```
session_config.intra_op_parallelism_threads = 1
session_config.inter_op_parallelism_threads = 1
```

It should not be necessary to do this when your ops are not running on the CPU
(e.g. when they're running on a GPU).

## Detailed Status of Determinism in TensorFlow and Beyond

Confirmed and likely sources of non-determinism, along with any existing
solutions, are being tracked here.

### GPU-Specific Sources of Non-Determinism

#### Historic GPU-Specific Sources of Non-Determinism

In the past, `tf.math.reduce_sum` and `tf.math.reduce_mean` operated
non-deterministically when running on a GPU. This was resolved before
TensorFlow version 1.12. These ops now function deterministically
by default when running on a GPU.

#### Confirmed Current GPU-Specific Sources of Non-Determinism (With Solutions)

 Source                                                               | TF 1.14, 1.15,<br>2.0  | NGC 19.06+ /<br>TF 2.1 | TF 2.2     | TF 2.3     |
:---------------------------------------------------------------------|:-----------------------|:-----------------------|:-----------|:-----------|
 TF auto-tuning of cuDNN convolution algorithms (see multi-algo note) | TCD or TDP             | TDO                    | TDO        | TDO        |
 cuDNN convolution backprop to weight gradients                       | TCD or TDP             | TDO                    | TDO        | TDO        |
 cuDNN convolution backprop to data gradients                         | TCD or TDP             | TDO                    | TDO        | TDO        |
 cuDNN max-pooling backprop                                           | TCD or TDP             | TDO                    | TDO        | TDO        |
 cuDNN CTC loss                                                       | NS                     | NS                     | NS         | TDO        |
 `tf.nn.bias_add` backprop (see XLA note)                             | TDP                    | TDO                    | TDO        | TDO        |
 XLA reductions on GPU                                                | NS                     | NS                     | TDO        | TDO        |
 Fused softmax/cross-entropy ops backprop (see note)                  | NS                     | NS                     | NS         | NS         |

 Source                                                               | TF < 2.4  | NGC 20.03+ | TF 2.4 ? |
:---------------------------------------------------------------------|:----------|:-----------|:---------|
 `tf.image.resize_bilinear` backprop (see note)                       | NS        | TDO        | TDO      |


Key to the solutions refenced above:

 Solution | Description                                                                                                                                                                                     |
:---------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 TCD      | Set environment variable `TF_CUDNN_DETERMINISTIC` to '1' or 'true'. Also *do not* set environment variable `TF_USE_CUDNN_AUTOTUNE` at all (and particularly *do not* set it to '0' or 'false'). |
 TDO      | Set environment variable `TF_DETERMINISTIC_OPS` to '1' or 'true'. Also *do not* set environment variable `TF_USE_CUDNN_AUTOTUNE` at all (and particularly *do not* set it to '0' or 'false').   |
 TDP      | Apply `tfdeterminism.patch`. Note that solution TDO is in stock TensorFlow v2.1 (see [PR 31465](https://github.com/tensorflow/tensorflow/pull/31465)), which makes patching unnecessary.        |
 NS       | There is no solution in the specified version, but there may be a solution in other versions (as shown)                                                                                         |

Notes:
  * multi-algo: From NGC TF 19.12 onwards and stock TensorFlow 2.2 onwards, the
    cuDNN forward and backward convolution algorithms are selected
    deterministically from several deterministic algorithms. Prior to this (i.e.
    NGC 19.11 and earlier, and stock TensorFlow 2.1 and earlier), there is only
    one deterministic algorithm selected for each of the forward and two
    backward paths. In those versions of TensorFlow, some layer configurations
    are not supported (resulting in an exception being thrown with the message
    "No algorithm worked!").
  * XLA: Prior to TensorFlow version 2.2, these solutions will not work when
    XLA JIT compilation is enabled due to XLA reductions on GPU not being
    deterministic (see
    [this comment](https://github.com/tensorflow/tensorflow/pull/34887#discussion_r355610837)
    on PR 34887). This will be resolved in TensorFlow version 2.2 and NGC TF
    Docker images based on that version of TensorFlow.
  * Fused softmax/cross-entropy ops refers to
    `tf.nn.softmax_cross_entropy_with_logits` and
    `tf.nn.sparse_softmax_cross_entropy_with_logits`. See TensorFlow
    [issue 38185](https://github.com/tensorflow/tensorflow/issues/38185).
    Work-around: use non-fused softmax and cross-entropy. For example, assuming
    you're using `tf.keras`, select the activation on the final `Dense` layer to
    be 'softmax' and then select `tf.keras.losses.categorical_crossentropy` for
    the loss function.
  * `tf.image.resize_bilinear` (TF 1 API): In the TF 2 API, this functionality
    is accessed via `tf.image.resize` with `method=ResizeMethod.BILINEAR` (which
    is the default `method` setting). It is also exposed through
    `tf.keras.layers.UpSampling2D` with `interpolation='bilinear'` (which is not
    the default `interpolation` setting). The solution in TF 2.3 depends upon
    [PR 39243](https://github.com/tensorflow/tensorflow/pull/39243) getting
    approved and merged before that version snaps.

#### Other Possible GPU-Specific Sources of Non-Determinism

Going beyond the above-mentioned sources, in version 1.12 of TensorFlow (and
also in the master branch on 2019-03-03, afer release 1.31.1), the following
files call CUDA `atomicAdd` either directly or indirectly. This makes them
candidates for the injection of non-determinism.

* `crop_and_resize_op_gpu.cu.cc`
* `scatter_functor_gpu.cu.h`
* `scatter_nd_op_gpu.cu.cc`
* `sparse_tensor_dense_matmul_op_gpu.cu.cc`
* `resize_nearest_neighbor_op_gpu.cu.cc`
* `segment_reduction_ops.h`
* `segment_reduction_ops_gpu.cu.cc`
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

Number                                                         | Title                                                                                 | Date Opened | Status |
--------------------------------------------------------------:|:--------------------------------------------------------------------------------------|:------------|:-------|
 [2652](https://github.com/tensorflow/tensorflow/issues/2652)  | Backward pass of broadcasting on GPU is non-deterministic                             | 2016-06-03  | Closed |
 [2732](https://github.com/tensorflow/tensorflow/issues/2732)  | Mention that GPU reductions are nondeterministic in docs                              | 2016-06-08  | Closed |
[13932](https://github.com/tensorflow/tensorflow/issues/13932) | Non-determinism from `tf.data.Dataset.map` with random ops                            | 2017-10-23  | Closed |
[16889](https://github.com/tensorflow/tensorflow/issues/16889) | Problems Getting TensorFlow to behave Deterministically                               | 2018-02-09  | Open   |
[18096](https://github.com/tensorflow/tensorflow/issues/18096) | Feature Request: Support for configuring deterministic options of cuDNN conv ...      | 2018-03-29  | Open   |
[22398](https://github.com/tensorflow/tensorflow/issues/22398) | CUDA implementation of BiasAddGrad op is non-determinstic                             | 2018-09-19  | Closed |
[29101](https://github.com/tensorflow/tensorflow/issues/29101) | Random seed not set in graph context of `Dataset#map`                                 | 2019-05-28  | Open   |
[38151](https://github.com/tensorflow/tensorflow/issues/38151) | Test deterministic cuDNN CTC loss                                                     | 2020-04-01  | Open   |
[38185](https://github.com/tensorflow/tensorflow/issues/38185) | Add GPU-deterministic back-prop for fused softmax/cross-entropy ops                   | 2020-04-02  | Open   |
[38197](https://github.com/tensorflow/tensorflow/issues/38197) | Model not deterministic ...                                                           | 2020-04-03  | Open   |
[40514](https://github.com/tensorflow/tensorflow/issues/40514) | BERT: Non-deterministic on GPU ...                                                    | 2020-06-16  | Closed |

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

ID                                                           | Title                                                         | Status | Date Merged | Version |
------------------------------------------------------------:|:--------------------------------------------------------------|:-------|:------------|:--------|
[24747](https://github.com/tensorflow/tensorflow/pull/24747) | Add cuDNN deterministic env variable (only for convolution).  | merged | 2019-01-15  | 1.14    |
[25269](https://github.com/tensorflow/tensorflow/pull/25269) | Add deterministic cuDNN max-pooling                           | merged | 2019-01-30  | 1.14    |
[25796](https://github.com/tensorflow/tensorflow/pull/25796) | Added tests for `TF_CUDNN_DETERMINISTIC`                      | merged | 2019-02-22  | 1.14    |
[c2790][1001]<sup>1</sup>                                    | Add a decorator to disable autotuning during test executions. | merged | 2019-03-13  | 1.14    |
[29667](https://github.com/tensorflow/tensorflow/pull/29667) | Add release note about `TF_CUDNN_DETERMINISTIC`               | merged | 2019-08-06  | 1.14    |
[31389](https://github.com/tensorflow/tensorflow/pull/31389) | Enhance release notes related to `TF_CUDNN_DETERMINISTIC`     | merged | 2019-08-07  | 1.14    |
[31465](https://github.com/tensorflow/tensorflow/pull/31465) | Add GPU-deterministic `tf.nn.bias_add`                        | merged | 2019-10-17  | 2.1     |
[32979](https://github.com/tensorflow/tensorflow/pull/32979) | Fix typo in release note                                      | closed |             |         |
[33483](https://github.com/tensorflow/tensorflow/pull/33483) | Fix small typo in v2.0.0 release note                         | merged | 2019-10-25  | 2.1     |
[33803](https://github.com/tensorflow/tensorflow/pull/33803) | Enable tf.nn.bias_add python op tests to work in eager mode   | merged | 2020-02-12  | 2.2     |
[33900](https://github.com/tensorflow/tensorflow/pull/33900) | Address problems with use_deterministic_cudnn test decorator  | merged | 2020-01-09  | 2.2     |
[34887](https://github.com/tensorflow/tensorflow/pull/34887) | Add info about `TF_DETERMINISTIC_OPS` to v2.1 release notes   | merged | 2019-12-09  | 2.1     |
[34951][1003]                                                | Add multi-algorithm deterministic cuDNN convolutions          | merged | 2020-01-27  | 2.2     |
[35006](https://github.com/tensorflow/tensorflow/pull/35006) | Fix version 2.1 release note regarding TF_DETERMINISTIC_OPS   | merged | 2019-12-20  | 2.1     |
[e3195][1002]<sup>1</sup>                                    | [XLA/GPU] Convert reduction into tree reduction using padding | merged | 2020-01-07  | 2.2     |
[8b7a3][1004]<sup>1</sup>                                    | [XLA] Respect TF_DETERMINISTIC_OPS env variable for reductions| merged | 2020-02-19  | 2.2     |
[37377](https://github.com/tensorflow/tensorflow/pull/37377) | [XLA] follow-up on GPU-deterministic reductions               | merged | 2020-03-09  | 2.3     |
[9e096][1005]<sup>1</sup>                                    | Use the CUDNN_CTC_LOSS_ALGO_DETERMINISTIC algorithm ...       | merged | 2020-03-10  | 2.3     |
[38089](https://github.com/tensorflow/tensorflow/pull/38089) | Add reminder to test deterministic cuDNN CTC loss             | closed |             |         |
[38509](https://github.com/tensorflow/tensorflow/pull/38509) | List deterministic op func bug fixes in v2.2 release notes    | merged | 2020-04-15  | 2.2     |
[39243](https://github.com/tensorflow/tensorflow/pull/39243) | GPU-deterministic tf.image.resize (bilinear)                  | open   |             |         |
 
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
Matthijs De Smedt,
Valentina Taviani,
Amirhossein Tebbifakhr,
Kevin Vincent,
Stephen Warren,
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
