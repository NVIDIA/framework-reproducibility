# Gradient Injection
by [@duncanriach][5], 2020-10-08

## Introduction

This document explains how and why we use gradient injection in the testing of
op backprop determinism in TensorFlow (in both eager and graph modes). This
document is necessary because the approach is somewhat non-intuitive and can be
challenging to both understand and explain.

Gradient injection can be seen [in the testing][1] of the backprop determinism
of the dynamic patch for `tf.nn.bias_add` in this current repo and in
determinism tests in stock TensorFlow, such as for [`tf.nn.bias_add`][2] and
[`tf.image.resize` (with `method='bilinear'`)][3].

I expect that we'll be using this technique to ensure the determinism of the
backprop of an increasing number of ops in TensorFlow. The approach may also be
applicable to other deep learning frameworks.

## How to Inject Gradients

In this section, I will provide the core, simplified elements of the TensorFlow
test code used to implement gradient injection for testing of backprop
determinism.

The following functions will be used below:
```
import numpy as np
import tensorflow as tf

def random_nd_array(shape):
    return 2 * np.random.random_sample(shape) - 1

def random_data_op(shape, dtype):
    return tf.constant(random_nd_array(shape), dtype=dtype)
```

The basic form is as follows for eager mode (when
`tf.executing_eagerly()` returns `True`). The variable `op_binding` can
be assumed to contain a reference to the op under test, and in this example it
happens to have two inputs (`input_a` and `input_b`). Also in this example,
`input_a` is the one for which we are testing the reproducibility of the
gradients. `dtype` and `output_shape` are assumed to have been assigned
appropriately elsewhere. `repeat_count` is the number of times to re-run the
test to reduce the probability of sporadic nondeterminim being missed.
```
def calculate_gradients(local_seed):
  np.random.seed(local_seed)
  upstream_gradients = random_data_op(output_shape, dtype)
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(input_a)
    op_output = op_binding(input_a, input_b, dtype=data_type)
    gradient_injector_output = op_output * upstream_gradients
  return tape.gradient(gradient_injector_output, input_a)

for i in range(repeat_count):
  local_seed = seed + i # select different upstream gradients
  result_a = calculate_gradients(local_seed)
  result_b = calculate_gradients(local_seed)
  tf.debugging.assert_equal(result_a, result_b)
```

For graph mode (when `tf.executing_eagerly()` returns `False`), the equivalent
looks like this:
```
upstream_gradients = tf.compat.v1.placeholder(dtype=dtype, shape=output_shape,
                                              name='upstream_gradients')
op_output = op_binding(input_a, input_b, dtype=dtype)
gradient_injector_output = op_output * upstream_gradients
gradients_op = tf.gradients(gradient_injector_output, input_a, grad_ys=None,
                            colocate_gradients_with_ops=True)[0]
for i in range(repeat_count):
  feed_dict = {upstream_gradients: self._random_nd_array(output_shape)}
  result_a = gradients_op.eval(feed_dict=feed_dict)
  result_b = gradients_op.eval(feed_dict=feed_dict)
  tf.debugging.assert_equal(result_a, result_b)
```

## Why Inject Gradients

This section will explain why it's necessary to inject gradients in order to
test the deterministic operation of the backprop computation of an op.

To test that an op's backprop is determinismtic, it's necessary to get random
upstream gradients into the backprop function so that nondeterminism in the
computation can be exposed. If the upstream gradients were all the same value
then it would not expose random noise introduced by variying the associativity
of any floating-point reductions, for example.

When I started testing op backprop determinism in the TF1 API, I found that it
was not possible to inject gradients into the backprop computation of an op
directly by specifying the gradients to inject via the `grad_ys` parameter
of the `tf.gradients` function such that it exposed the nondeterministic
functionality of the backprop computation. I'm not sure that I ever fully
understood why this was so. It could be due to some kind of optimization that is
arithmetically correct, and performant, but which masks the nondeterminism in
the backprop computation. It could just be a bug.

In any case, it seemed that my need was very specialized, so I developed the
gradient injection approach. I then extended it to work with the TF2 API,
with gradient tape.

I don't know if the `output_gradients` parameter of the `gradient` function of
`tf.GradientTape` now provides the functionality needed; we should probably test
that. I also don't know if the functionality of the `grad_ys` parameter of
`tf.gradients` now works as needed; that might be worth exploring.

For now, the gradient injection pattern show above works really well, and for
now this seems to be an optimal approach.

## How It Works

In the above code, whether you evaluate the `gradients_op` (returned by
`tf.gradients`) in the current session (in graph mode) or call
`calcuate_gradients` (which returns a gradient function from
`tf.GradientTape::gradient` in eager mode) the gradient at `input_a` will be
evaluated as if the gradient flowing into the node of the backprop graph
associated with `gradient_injector_output` in the forward graph is a
constant with all elements equal to each other (and, I believe, such that they
sum to 1.0).

In the forward direction, the equation looks like this:
```
gradient_injector_output = op(input_a, input_b) * upstream_gradients
```

The backprop graph for this, from `gradient_injector_output` to `input_a` can be
written mathematically as:

```
g = d_gradient_injector_output / d_input_a
  = d_(op * upstream_gradients) / d_input_a
```

where `g` contains an element associated with each element of `input_a`, and
where the gradient in that element is actually the sum of the gradients for all
output elements with respect to that input element.

By the product rule:
```
g = (d_op / d_input_a) * upstream_gradients +
    op * (d_upstream_gradients / d_input_a)
```

and then, since `d_upstream_gradients / d_input_a == 0`,

```
g = upstream_gradients * (d_op / d_input_a)
```

But this math doesn't capture that the gradient tensor (`g`), shaped the same as
`input_a`, is generated by a computational process that takes
`upstream_gradients` as an input. In other words,

```
g = f(upstream_gradients)
```

The gradient values that are calculated are a function of the
`upstream_gradients` and that function is a computational process that
implements `d_op / d_input_a`.

## Conclusion

Gradient injection is a practical method that allows directed test data to be
injected into the input of an op's backprop computation such that
nondeterministic functionality in that computation can be detected.

## Credits

Thanks to [@wenscarl][4] for being patient with my one-on-one explanation of
this and also for, in the process, helping me to formalize the math presented
above.

[1]: https://github.com/NVIDIA/framework-determinism/blob/1c5450f167fd0d75393df82ea8b5896b3a438d0b/test/test_patch_bias_add.py#L325
[2]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/kernel_tests/bias_op_deterministic_test.py#L79
[3]: https://github.com/tensorflow/tensorflow/blob/2d83c3230fa879fcccad836cc3849d8101c70fb4/tensorflow/python/ops/image_grad_deterministic_test.py#L70
[4]: https://github.com/wenscarl
[5]: https://github.com/duncanriach