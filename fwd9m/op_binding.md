# Op Binding in Python

One of the functionalities that we provide in this package is temporary
dynamic patching of framework ops so that they operate deterministically;
temporary because we always intend to upstream solutions into the stock
frameworks.

In order to effectively patch ops at the python level, using dynamic
programming, we potentially have to rebind the op in a number of places.
An example of this is the `bias_add` op in TensorFlow. If you look at the
[patch code][1] for that, you'll see that we bind the new function at

  1. `tf.nn.bias_add`, which is how it's accessed via the public API,
  2. `tensorflow.python.ops.nn.bias_add` which is how it's accessed from
     `tf.keras.layers.convolutional.Conv`, and
  3. `tensorflow.python.ops.nn_ops.bias_add`, which is how it's called from
     within tests.

A reasonable amount of searching in the TensorFlow repo gave us confidence that
these are all the paths through which this op is accessed.

One question that arises is "Why is it not possible to rebind one of those
three points, and have that cause the others to be rebound correctly?" The
answer requires understanding how function pointers work in Python. When you
define a function, it creates a function object and a variable which contains a
reference to that object:

```
>>> def a():
...   print("Running a")
...
>>> a()
Running a
>>> a
<function a at 0x7fe640bb38c0>
>>> type(a)
<class 'function'>
```

This means that the variable `a` contains a reference to an object that is an
instance of the 'function' class. We can make a copy of the reference like this:

```
>>> b = a
>>> b()
Running a
>>> b
<function a at 0x7fe640bb38c0>
>>> type(b)
<class 'function'>
```

I can show you that `a` is just a variable that contains a reference to an
object by sticking a reference to an instance of the integer class into it:

```
>>> a = 5
>>> print(a)
5
>>> type(a)
<class 'int'>
```

Now that we've changed what `a` points to, what does `b` point to?

```
>>> b()
Running a
>>> b
<function a at 0x7fe640bb38c0>
>>> type(b)
<class 'function'>
```

The variable `b` still contains a pointer (address `0x7fe640bb38c0`) to the
instance of the function class that was originally defined with the name `a`.
If I was to store a reference to something else in the `b` variable, then that
object at address `0x7fe640bb38c0` would no longer have a reference. Next time
the garbage collector came by, it would notice that the object was no longer
referenced anywhere and deallocate the memory associated with it.

Now let's return to those various bindings in TensorFlow and put it all
together. After the op function is defined, the reference to the underlying
instance of the function class gets copied into variables in the places from
which it can be accessed. This often happens in the `__init__.py` files in
package directories. The following shows what happens if you only rebind one of
these points. I've chosen to replace the original binding because that's the
binding we might expect is the only one that needs to be changed.
```
>>> def original_op():
...   print("original op")
...
>>> alternative_route = original_op
>>> original_op()
original op
>>> alternative_route()
original op
>>> def new_op():
...   print("new op")
...
>>> new_op()
new op
>>> original_op = new_op
>>> original_op()
new op
>>> alternative_route()
original op
```

Uh oh! That's not what we want, so, finally, we rebind `alternative_route`
and all is well.

```
>>> alternative_route = new_op
>>> alternative_route()
new op
```

[1]: ./tensorflow/patch.py
