# Testing Notes (d9m)

## How to run tests

Requirements:

  * Installed GPU that supports CUDA
  * A driver for the GPU (a sufficiently recent version)
  * Access to the NGC Container Registry (nvcr.io) (see [documentation][2])
  * docker
  * nvidia-docker
  * pip install -r requirements.txt

Then you should be able to run `./all.sh` and observe the final message.

## Documentation

See the notes about [gradient injection][1] for testing backprop determinism
of TensorFlow ops.

[1]: ./gradient_injection.md
[2]: https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#nvcrio
