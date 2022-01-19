# Status of GPU-Determinism in TensorFlow

## Introduction

This page provides an up-to-date view of the status of GPU-related sources of
nondeterminism in TensorFlow. This is almost exclusively focused on the
determinsitic functionality of ops running on a GPU.

For a broader view, see the [TensorFlow Determinism](./README.md) page.

## Summary

The following table indicates whether a solution is available for each source.
For further information, see the later detailed notes, which are linked to
from the "Solution Available" column.

 Source                                                                  | Solution Available           |
-------------------------------------------------------------------------|:-----------------------------|
 Auto-tuning of cuDNN convolution algorithms                             | [YES](#auto-tuning)          |
 `tfa.image.dense_image_warp` backprop                                   | [YES](#dense-image-warp)     |
 `tf.compat.v1.nn.fused_batch_norm` backrop                              | [NO](#fused-batch-norm)      |
 `tf.convert_to_tensor` forward, for `tf.IndexedSlices`                  | [YES](#convert-to-tensor)    |
 `tf.gather` backprop                                                    | [YES](#gather)               |
 `tf.keras.layers.BatchNormalization` backprop                           | [NO](#fused-batch-norm)      |
 `tf.keras.layers.Conv1D` backprop                                       | [YES](#cudnn-conv)           |
 `tf.keras.layers.Conv2D` backprop                                       | [YES](#cudnn-conv)           |
 `tf.keras.layers.Conv3D` backprop                                       | [YES](#cudnn-conv)           |
 `tf.keras.layers.DepthwiseConv2D` backprop to `filter`                  | [NO](#depthwise-conv)        |
 `tf.keras.layers.MaxPool1D` backprop                                    | [YES](#max-pool)             |
 `tf.keras.layers.MaxPool2D` backprop                                    | [YES](#max-pool)             |
 `tf.keras.layers.MaxPool3D` backprop                                    | [YES](#max-pool)             |
 `tf.keras.layers.UpSampling2D` backprop when `interpolation='bilinear'` | [YES](#resize-bilinear)      |
 `tf.keras.layers.UpSampling2D` backprop with `interpolation='nearest'`  | [NO](#resize-nearest)        |
 `tf.keras.losses.categorical_crossentropy` forward and backprop         | [YES](#softmax-xent)         |
 `tf.keras.losses.CategoricalCrossentropy` forward and backprop          | [YES](#softmax-xent)         |
 `tf.keras.losses.sparse_categorical_crossentropy` forward and backprop  | [YES](#softmax-xent)         |
 `tf.keras.losses.SparseCategoricalCrossentropy` forward and backprop    | [YES](#softmax-xent)         |
 `tf.image.adjust_contrast` forward                                      | [NO](#adjust-contrast)       |
 `tf.image.crop_and_resize` backprop                                     | [NO](#crop-and-resize)       |
 `tf.image.resize` backprop when `method=ResizeMethod.BILINEAR`          | [YES](#resize-bilinear)      |
 `tf.image.resize` backprop when `method=ResizeMethod.NEAREST`           | [NO](#resize-nearest)        |
 `tf.math.segment_prod` forward                                          | [YES](#segment-reduction)    |
 `tf.math.segment_sum` forward                                           | [YES](#segment-reduction)    |
 `tf.math.unsorted_segment_mean`                                         | [YES](#segment-reduction)    |
 `tf.math.unsorted_segment_prod`                                         | [YES](#segment-reduction)    |
 `tf.math.unsorted_segment_sqrt_n`                                       | [YES](#segment-reduction)    |
 `tf.math.unsorted_segment_sum`                                          | [YES](#segment-reduction)    |
 `tf.nn.bias_add` backprop                                               | [YES](#bias-addition)        |
 `tf.nn.conv1d` backprop                                                 | [YES](#cudnn-conv)           |
 `tf.nn.conv2d` backprop                                                 | [YES](#cudnn-conv)           |
 `tf.nn.conv3d` backprop                                                 | [YES](#cudnn-conv)           |
 `tf.nn.ctc_loss` backprop                                               | [YES](#ctc-loss)             |
 `tf.nn.depthwise_conv2d` backprop to `filter`                           | [NO](#depthwise-conv)        |
 `tf.nn.max_pool1d` backprop                                             | [YES](#max-pool)             |
 `tf.nn.max_pool2d` backprop                                             | [YES](#max-pool)             |
 `tf.nn.max_pool3d` backprop                                             | [YES](#max-pool)             |
 `tf.nn.softmax_cross_entropy_with_logits`                               | [YES](#softmax-xent)         |
 `tf.nn.sparse_softmax_cross_entropy_with_logits`                        | [YES](#softmax-xent)         |
 `tf.sparse.sparse_dense_matmul` forward                                 | [NO](#sparse-dense-matmul)   |
 XLA reductions on GPU                                                   | [YES](#xla-reductions)       |

Information for each source is listed below. To reduce repetition, the following
abbreviations have been used throughout:

  * <a name="TF_DETERMINISTIC_OPS">**TF_DETERMINISTIC_OPS**</a>: Set environment
    variable `TF_DETERMINISTIC_OPS` to `'1'` or `'true'`. Also *do not* set
    environment variable `TF_USE_CUDNN_AUTOTUNE` at all (and particularly
    *do not* set it to `'0'` or `'false'`).
  * <a name="TF_CUDNN_DETERMINISTIC">**TF_CUDNN_DETERMINISTIC**</a>: Set
    environment variable `TF_CUDNN_DETERMINISTIC` to `'1'` or `'true'`. Also
    *do not* set environment variable `TF_USE_CUDNN_AUTOTUNE` at all (and
    particularly *do not* set it to `'0'` or `'false'`).
  * <a name="PATCH">**PATCH**</a>: Apply `tfdeterminism.patch`. Note that
    this solution, enabled by [`TF_DETERMINISTIC_OPS`](#TF_DETERMINISTIC_OPS),
    is in stock TensorFlow version 2.1 (see github/tensorflow/tensorflow pull
    request [31465][31465], which makes patching unnecessary.
  * <a name="NO_SOLUTION">**NO SOLUTION**</a>: There is no solution in the
    specified version, but there may be a solution in other versions (as shown).

Where it is indicated that solutions are available in NGC TensorFlow container
images, it can be assumed, unless stated otherwise, that the solutions are
available in both TensorFlow API version 1 and TensorFlow API version 2 variants
of those container images.

If an NGC TF container version is not mentioned in the list of solutions for an
op, any NGC TF container image version that is based on a version of stock
TensorFlow that contains a solution probably also contains the same solution.

---

<a name="auto-tuning"></a>
## Auto-Tuning of cuDNN Convolution Algorithms

### Problem

TensorFlow normally tries different cuDNN convolution algorithms for a given
layer configuration to find the one that runs fastest. The functionality is
inherently nondeterministic because different algorithms can be selected on each
run.

### Solution

  * TF 1.14, 1.15, 2.0: [TF_CUDNN_DETERMINISTIC](#TF_CUDNN_DETERMINISTIC) or
    [PATCH](#PATCH)
  * NGC 19.06+, TF 2.1+: [TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS)

From TensorFlow 2.5 onwards, the environment variable `TF_CUDNN_USE_FRONTEND`
must also be set to `1`, to work-around the issue described in
github/tensorflow/tensorflow issue [53771][53771].

There is an additional issue related to nondeterministic out-of-memory events
when selecting algorithms, which could result in nondeterministic functionality.
However, it's relatively unlikely that this issue will be encountered. See
[this comment](https://github.com/tensorflow/tensorflow/issues/53771#issuecomment-1016028174)
on github/tensorflow/tensorflow issue [53771][53771]. Hopefully, this issue will
be addressed in TensorFlow version 2.8.

### Additional Information

From NGC TF 19.12 onwards and stock TensorFlow 2.2 onwards, the
cuDNN forward and backward convolution algorithms are selected
deterministically from several deterministic algorithms. Prior to this (i.e.
NGC 19.11 and earlier, and stock TensorFlow 2.1 and earlier), there is only
one deterministic algorithm selected for each of the forward and two
backward paths. In those versions of TensorFlow, some layer configurations
are not supported (resulting in an exception being thrown with the message
"No algorithm worked!").

---

<a name="dense-image-warp"></a>
## Dense Image Warp

### Problem

The backprop for `tfa.image.dense_image_warp` may introduce truly random noise
because it uses the nondeterministic [`tf.gather`](#gather) functionality.

### Solution

See the solution status for [`tf.gather`](#gather).

---

<a name="fused-batch-norm"></a>
## Fused Batch-Norm

### Problem

`tf.compat.v1.nn.fused_batch_norm` introduces truly random noise in the backrop
to `offset` when `is_training=False`, when running on a GPU.

Backprop through `tf.compat.v1.nn.fused_batch_norm` when `training=False` is
used for fine-tuning. See github/tensorflow/tensorflow issue [10857][10857] for
more information.

This nondeterminism is also exposed through `tf.keras.layers.BatchNormalization`
when the model/layer attribute `trainable` is set to `True` but the layer is
called with `training=False`. In fine-tuning the
`tf.compat.v1.nn.fused_batch_norm` `offset` input
(`tf.keras.layers.BatchNormalization` `beta` parameter) will be updated via
back-propagation, even while the running mean and variance are held static (and
used), in "inference mode". For more information, see the TensorFlow guide on
[transfer learning and fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning).

### Solution

There is currently no available solution

### Additional Information

Stock TensorFlow version 2.7+ will throw a `tf.errors.UnimplementedError` if the
nondeterministic path through `tf.compat.v1.nn.fused_batch_norm` is
traversed with the expectation of determinism (i.e. with `TF_DETERMINISTIC_OPS`
set to `"true"` or `"1"`). See github/tensorflow/tensorflow pull request
[50505][50505].

---

<a name="convert-to-tensor"></a>
## Convert to Tensor

### Problem

`tf.convert_to_tensor`, when fed with (sparse) `tf.IndexedSlices`, uses the
potentially nondeterministic behavior of
[`tf.math.unsorted_segment_sum`](#segment-reduction) in its forward direction
and therefore may introduce truly random noise into its output when a slice
index is represented more than twice in its input (such as when reducing the
word embedding gradients from multiple instances of the same word in a sentence
or across a batch of sentences).

### Solution

See the solution status for
[`tf.math.unsorted_segment_sum`](#segment-reduction).

---

<a name="gather"></a>
## Gather

### Problem

`tf.gather` is often used to select word embeddings from an embedding matrix in
a model's forward direction and `tf.gather`'s backprop generates sparse
gradients conveyed as `tf.IndexedSlices`. The reduction of the back-propagated
sparse gradients from `tf.gather` by
[`tf.convert_to_tensor`](#convert-to-tensor) can therefore introduce truly
random noise into an embedding trainable variable.

### Solution

See the solution status for
[`tf.convert-to-tensor`](#convert-to-tensor).

A lower-performance work-around for this nondeterminism related to the use of
`tf.gather` is to use `tf.linalg.matmul` instead:

```
# inputs_embeds = tf.gather(embeddings, input_ids)
input_embeds = tf.dtypes.cast(
    tf.one_hot(input_ids, embeddings.shape[0]),
    embeddings.dtype) @ embeddings
```

Note that the backward (and forward) functionality of `tf.gather` itself _is_
deterministic.

---

<a name="cudnn-conv"></a>
## cuDNN Convolution

### Problem

cuDNN convolution backprop algorithms are exposed through `tf.nn.conv1d`,
`tf.nn.conv2d`, and `tf.nn.conv3d`. For the backprop (to both `input` and
`filters`) of these ops to function deterministically, TensorFlow must expose
the relevant deterministic cuDNN convolution algorithms.

Functionality that is built on top of these ops is also affected, such as
`tf.keras.layers.Conv1D`, `tf.keras.layers.Conv2D`, and
`tf.keras.layers.Conv3D`. See also notes on [bias addition](#bias-addition)

### Solution

  * TF 1.14, 1.15, 2.0: [TF_CUDNN_DETERMINISTIC](#TF_CUDNN_DETERMINISTIC) or
    [PATCH](#PATCH)
  * NGC 19.06+, TF 2.1+: [TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS)

---

<a name="depthwise-conv"></a>
## Depthwise Convolution

### Problem

`tf.nn.depthwise_conv2d` and `tf.keras.layers.DepthwiseConv2D` do not operate
fully deterministically when running on GPU, even when the solutions for
[cuDNN convolution](#cudnn-conv) are applied. The relatively complex story is
explicated in the docstring of
[this gist](https://gist.github.com/duncanriach/4c18cb07a73510c5fcb2deb52adbffaa).

### Solution

As described in the aforementioned
[gist](https://gist.github.com/duncanriach/4c18cb07a73510c5fcb2deb52adbffaa),
for a single input channel the functionality may be determinstic using the
solutions for [cuDNN convolution](#cudnn-conv). For other configurations, there
is currently no solutiuon.

However, it may be possible to implement the required functionality, with
reasonable performance, using multiple instances of regular convolution
followed by an appropiate splicing of their outputs.

### Additional Information

Stock TensorFlow version 2.7+ will throw a `tf.errors.UnimplementedError` if the
non-cuDNN nondeterministic path through `tf.nn.depthwise_conv2d` is
traversed with the expectation of determinism (i.e. with `TF_DETERMINISTIC_OPS`
set to `"true"` or `"1"`). See github/tensorflow/tensorflow pull request
[51920][51920].

See these issues:

  * github/tensorflow/tensorflow issue [47174][47174]
  * github/NVIDIA/framework-determinism issue
    [26](https://github.com/NVIDIA/framework-determinism/issues/26)

---

<a name="max-pool"></a>
## cuDNN Max-Pooling

### Problem

cuDNN max pooling is exposed through `tf.nn.max_pool1d`, `tf.nn.max_pool2d`, and
`tf.nn.max_pool3d`. For the backprop of these ops to function deterministically,
TensorFlow must expose the relevant deterministic cuDNN convolution algorithms.

Functionality that is built on top of these ops is also affected, such as
`tf.keras.layers.MaxPool1D`, `tf.keras.layers.MaxPool2D`, and
`tf.keras.layers.MaxPool3D`.

### Solution

  * TF 1.14, 1.15, 2.0: [TF_CUDNN_DETERMINISTIC](#TF_CUDNN_DETERMINISTIC) or
    [PATCH](#PATCH)
  * NGC 19.06+, TF 2.1+: [TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS)

### Additional Information

This solution does not currently have unit tests in the TensorFlow repo but has
been confirmed to work in production models. Here is the
[TODO](https://github.com/tensorflow/tensorflow/blob/6269e15ade2b6b56cd5128afc46d7886da962571/tensorflow/python/kernel_tests/cudnn_deterministic_base.py#L53)
comment to add those tests.

---

<a name="resize-bilinear"></a>
## Bilinear Image Resizing

### Problem

`tf.image.resize` with `method=ResizeMethod.BILINEAR` (TF2 API) introduces truly
random noise into the backprop path. Note that `BILINEAR` is the default
`method` setting. In the TF1 API, this functionality is accessed via
`tf.image.resize_bilinear` (`tf.compat.v1.image.resize_bilinear` in TF 2.x). It
is also exposed through `tf.keras.layers.UpSampling2D` with
`interpolation='bilinear'` (which is not the default `interpolation` setting).

### Solution

  * NGC 20.03+, TF 2.4+: [TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS)

---

<a name="resize-nearest"></a>
## Nearest-Neighbor Image Resizing

### Problem

`tf.image.resize` with `method=ResizeMethod.NEAREST` (TF2 API) introduces truly
random noise in the backprop path. Note that `BILINEAR` is the default `method`
setting. In the TF1 API, this functionality is accessed via
`tf.image.resize_nearest_neighbor` (`tf.compat.v1.image.resize_nearest_neighbor`
in TF 2.x). It is also exposed through `tf.keras.layers.UpSampling2D` with
`interpolation='nearest'` (which is the default `interpolation` setting).

### Solution

There is currently no solution available. Use bilinear image resizing, if
possible.

A potential work-around is to use `tf.keras.layers.Conv2DTranspose` (see
issues [#12](https://github.com/NVIDIA/framework-determinism/issues/12) and
[#24](https://github.com/NVIDIA/framework-determinism/issues/24) for this
current repository).

### Additional Information

Stock TensorFlow version 2.7+ will throw a `tf.errors.UnimplementedError` if the
nondeterministic paths described above are used with the expectation of
determinism (i.e. with `TF_DETERMINISTIC_OPS` set to `"true"` or `"1"`). See
github/tensorflow/tensorflow pull request
[51023][51023].

---

<a name="softmax-xent"></a>
## Fused Softmax/Cross-Entropy

### Problem

The fused softmax/cross-entropy ops `tf.nn.softmax_cross_entropy_with_logits`
and `tf.nn.sparse_softmax_cross_entropy_with_logits` (accessed via
`tf.keras.losses.categorical_crossentropy`,
`tf.keras.losses.CategoricalCrossentropy`,
`tf.keras.losses.sparse_categorical_crossentropy`, and
`tf.keras.losses.SparseCategoricalCrossentropy`) are known to inject
nondeterminism into both the backward and forward paths. See
github/tensorflow/tensorflow issue [38185][38185].

### Solutions

In TF2.7+, both `tf.nn.softmax_cross_entropy_with_logits` and
`tf.nn.sparse_softmax_cross_entropy_with_logits` (and the Keras
layers based on them) can be made to operate deterministically on GPU
using `tf.config.experimental.enable_deterministic_ops(True)`. See
github/tensorflow/tensorflow pull request [50070][50070].

In TF2.6, `tf.nn.softmax_cross_entropy_with_logits` (and the Keras layers based
on it) can be made to operate deterministically on GPU using
[TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS). See github/tensorflow/tensorflow
pull requests [49178][49178].

A confirmed work-around for older versions of TensorFlow is to use separate
non-fused softmax and cross-entropy ops. For example, assuming you're using
`tf.keras`, select the activation on the final layer (e.g. a `Dense` layer) to
be 'softmax' (which chooses `tf.nn.softmax`) and then, for the loss function,
continue to use `tf.keras.losses.categorical_crossentropy` (possibly by using
its wrapper class `tf.keras.losses.CategoricalCrossentropy`) or
`tf.keras.losses.sparse_categorical_crossentropy` (possibly by using its wrapper
class `tf.keras.losses.SparseCategoricalCrossentropy`). Since it uses non-fused
kernels, the work-around will be lower performance. Theoretically, you should
ensure that the loss function parameter `from_logits` is set to `False` (the
default), perhaps only for performance reasons since setting it to `True` is a
no-op arithmetically and does not appear to contribute to nondeterminism.

### Additional Information

Stock TensorFlow version 2.6 will throw a `tf.errors.UnimplementedError` if the
nondeterministic paths of `tf.nn.sparse_softmax_cross_entropy_with_logits` are
used with the expectation of determinism (i.e. with `TF_DETERMINISTIC_OPS` set
to `"true"` or `"1"`). See github/tensorflow/tensorflow pull request
[47925][47925].

---

<a name="adjust-contrast"></a>
## Adjust Contrast

### Problem

`tf.image.adjust_contrast` injects truly random noise in the forward direction
when running on a GPU.

### Solution

There is currently no available solution.

### Additional Information

Stock TensorFlow version 2.7+ will throw a `tf.errors.UnimplementedError` if the
nondeterministic paths through `tf.image.adjust_contrast` are
used with the expectation of determinism (i.e. with `TF_DETERMINISTIC_OPS` set
to `"true"` or `"1"`). See github/tensorflow/tensorflow pull request
[51140][51140].

---

<a name="crop-and-resize"></a>
## Crop and Resize

### Problem

Backprop to `image` on `tf.image.crop_and_resize` introduces nondeterministic
noise when running on either CPU or GPU. Backprop to `boxes` introduces
nondeterministic noise when running on GPU. See github/tensorflow/tensorflow
issue [42033][42033] for more information.

### Solution

There is currently no available solution for GPU nondeterminism. For stock
TensorFlow version 2.6+, [TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS) makes
CPU backprop to `image` deterministic (see github/tensorflow/tensorflow pull
request [48905][48905]).

### Additional Information

Stock TensorFlow version 2.6+ will throw a `tf.errors.UnimplementedError` if the
nondeterministic paths through this op are used with the expectation of
determinism (i.e. with `TF_DETERMINISTIC_OPS` set to `"true"` or `"1"`). See
github/tensorflow/tensorflow pull request [48905][48905].

---

<a name="segment-reduction"></a>
## Segment Reduction

### Problem

The following ops have been shown to introduce truly random noise in the forward
path:

  * `tf.math.segment_sum`
  * `tf.math.unsorted_segment_prod`
  * `tf.math.unsorted_segment_sum` (see github/tensorflow/tensorflow issue
    [39751][39751])

The souce code that implements `tf.math.segment_prod` seems as though it should
introduce truly random noise in the forward path, although we have not be able
to produce it.

The following ops are implemented on top of `tf.math_unsorted_segment_sum` and
therefore also introduce truly random noise in the forward path:

  * `tf.math.unsorted_segment_sqrt_n`
  * `tf.math.unsroted_segment_mean`

### Solution

TF 2.7+ adds (non-sparse) GPU-deterministic segment reduction ops, enabled by
[TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS). See github/tensorflow/tensorflow
pull requests [51861][51861] and [51392][51392]. Included is also a
GPU-deterministic implementation of `tf.math.segment_mean`.

github/tensorflow/tensorflow pull request [47974][47974] adds GPU sparse segment
reduction ops, in TF 2.6+, which are also deterministic by default.

### Additional Information

We **do** have an unreleased patch for `tf.math.segment_sum` and
`tf.math.unsorted_segment_sum` that can be used by cloning this current
repository, installing the `fwd9m` package, and calling
`fwd9m.tensorflow.enable_determinism`. However, this patch will not provide
robust deterministic functionality and should not be relied upon. For more
information, see github/tensorflow/tensorflow pull request [47749][47749], in
which the approach used in this patch was discovered to be flawed.

Stock TensorFlow version 2.5+ will throw a `tf.errors.UnimplementedError` if the
nondeterministic paths of these ops are used with the expectation of determinism
(i.e. with `TF_DETERMINISTIC_OPS` set to `"true"` or `"1"`). See
github/tensorflow/tensorflow pull request [47772][47772].

Prior to TF 2.7, `tf.math.segment_mean` is not implemented on the GPU and the
CPU implementation operates deterministically.

See also:
  * Issue [31](https://github.com/NVIDIA/framework-determinism/issues/31) in
    this current repository.

---

<a name="bias-addition"></a>
## Bias Addition

### Problem

The backprop of `tf.nn.bias_add` performs large, structured reductions using
CUDA `atomicAdd`, thereby capturing the truly random alignment between
asynchronous compute engines into truly randomly varying floating-point rounding
error in the gradients.

Note that various Keras layers, including the Keras convolution layers
(i.e. `tf.keras.layers.Conv1D`, `tf.keras.layers.Conv2D`, and
`tf.keras.layers.Conv3D`), are built on top of `tf.nn.bias_add`. Therefore,
when `use_bias=True` the deterministic functionality of the layer is dependent
on the deterministic functionality of `tf.nn.bias_add`.

### Solution

  * TF 1.14, 1.15, 2.0: [PATCH](#PATCH)
  * NGC 19.06+, TF 2.1+: [TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS)

### Additional Information

Prior to TensorFlow version 2.2, this deterministic `tf.nn.bias_add` backprop
solution will not work when XLA JIT compilation is enabled due to XLA reductions
on GPU not being deterministic (see
[this comment](https://github.com/tensorflow/tensorflow/pull/34887#discussion_r355610837)
on github/tensorflow/tensorflow pull request 34887). This is resolved in stock
TensorFlow version 2.2 and NGC TF container images based on that version of
TensorFlow; see the [notes on that](#xla-reductions) elsewhere in this current
file.

---

<a name="ctc-loss"></a>
## CTC Loss

### Problem

Deterministic cuDNN CTC loss was exposed, via `tf.nn.ctc_loss`, by changes that
ended up appearing in stock TensorFlow version 2.3
(see github/tensorflow/tensorflow issue [38151][38151]). However, there was a
bug that was preventing deterministic operation. This bug was resolved in
version 2.6, if not earlier, as confirmed by testing added in version 2.6 (see
github/tensorflow/tensorflow pull request [52227][52227]).

### Solution

Resolved in TF 2.6 (and possibly earlier versions). Use
[TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS).

---

<a name="sparse-dense-matmul"></a>
## Sparse-Dense Matmul

### Problem

In TF 2.4 and earlier, the forward path of `tf.sparse.sparse_dense_matmul`
introduces nondeterminism for `tf.float32`, but not for `tf.float64` (for which
there is no GPU implementation). See github/tensorflow/tensorflow issue
[18037][18037].

GPU support for other floating-point types (`tf.float16`, `tf.float64`,
`tf.complex64`, and `tf.complex128`) was added in TF 2.5
(see github/tensorflow/tensorlow pull request [47419][47419]) and NGC TF 21.05
(even though it's based on stock TF 2.4). In TF 2.5 (and NGC TF 21.05) onwards,
if you were relying on the determinism of the `tf.float64` CPU implementation
being automatically selected because of an absence of the `tf.float64` GPU
implementation, you will need to force the op to run on the CPU or use a
different data type.

### Solutions

No solution has yet been released.

### Additional Information

A more deterministic GPU implementation of `tf.sparse.sparse_dense_matmul` when
the data type of the input tensors is `tf.float32`, for both TF1 and TF2
variants of the NGC TF container image is available in version 21.04 onwards
(based on stock TF 2.4; enabled by
[TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS)), and will still only support
`tf.float32` on GPU.

A more deterministic GPU implementation of `tf.sparse.sparse_dense_matmul` when
the data type is `tf.float16`, `tf.float32`, or `tf.complex64` is available in
NGC TF 21.05 onwards (enabled by
[TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS)). In this solution, an attempt to
use the `tf.sparse.sparse_dense_matmul` GPU implementation with data of type
`tf.float64` or `tf.complex128`, when deterministic op functionality is enabled
(currently via `TF_DETERMINISTIC_OPS` being set to `"true"` or `"1"`), will
cause a `tf.errors.UnimplementedError` to be thrown.

The above-mentioned GPU implementations are flawed (and can introduce
nondeterminism) and will not appear in stock TensorFlow (see
github/tensorflow/tensorflow pull request [47749][47749])

Stock TensorFlow version 2.6+ will throw a `tf.errors.UnimplementedError` if the
nondeterministic path through this op is used with the expectation of
determinism (i.e. with `TF_DETERMINISTIC_OPS` set to `"true"` or `"1"`). See
github/tensorflow/tensorflow pull request [50355][50355].

---

<a name="xla-reductions"></a>
## XLA reductions on GPU

### Problem

Without this solution, when XLA JIT compilation is enabled, any op that relies
on XLA reductions, whether in the forward or backward direction, will introduce
nondeterministic noise.

### Solution

  * TF 2.2+: [TF_DETERMINISTIC_OPS](#TF_DETERMINISTIC_OPS)

---

[10857]: https://github.com/tensorflow/tensorflow/issues/10857
[18037]: https://github.com/tensorflow/tensorflow/issues/18037
[38151]: https://github.com/tensorflow/tensorflow/issues/38151
[38185]: https://github.com/tensorflow/tensorflow/issues/38185
[39751]: https://github.com/tensorflow/tensorflow/issues/39751
[42033]: https://github.com/tensorflow/tensorflow/issues/42033
[47174]: https://github.com/tensorflow/tensorflow/issues/47174
[47419]: https://github.com/tensorflow/tensorflow/pull/47419
[47749]: https://github.com/tensorflow/tensorflow/pull/47749
[47772]: https://github.com/tensorflow/tensorflow/pull/47772
[47925]: https://github.com/tensorflow/tensorflow/pull/47925
[47974]: https://github.com/tensorflow/tensorflow/pull/47974
[48905]: https://github.com/tensorflow/tensorflow/pull/48905
[49178]: https://github.com/tensorflow/tensorflow/pull/49178
[50070]: https://github.com/tensorflow/tensorflow/pull/50070
[50355]: https://github.com/tensorflow/tensorflow/pull/50355
[50505]: https://github.com/tensorflow/tensorflow/pull/50505
[51023]: https://github.com/tensorflow/tensorflow/pull/51023
[51140]: https://github.com/tensorflow/tensorflow/pull/51140
[51392]: https://github.com/tensorflow/tensorflow/pull/51392
[51861]: https://github.com/tensorflow/tensorflow/pull/51861
[51920]: https://github.com/tensorflow/tensorflow/pull/51920
[52227]: https://github.com/tensorflow/tensorflow/pull/52227
[53771]: https://github.com/tensorflow/tensorflow/issues/53771
