# Testing Notes (d9m)

## How to run tests

Requirements:

  * Installed GPU that supports CUDA
  * GPU driver that supports at least CUDA 10
  * Access to the NGC Container Registry (nvcr.io) (see [documentation][2])
  * docker
  * nvidia-docker
  * pip install -r requirements.txt

Then you should be able to run `./all.sh` and observe the final message.

## Things to do before next release

  * Add integration tests for enable_determinism
  * Factor integration test run/test mechanism for others to use
  * Update documentation organization to reflect multiple frameworks
  * Update documentation to cover `enable_determinism`
  * Potentially add translation from GitHub to PyPI documentation.

## Documentation

See the notes about [gradient injection][1] for testing backprop determinism
of TensorFlow ops.

[1]: ./gradient_injection.md
[2]: https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#nvcrio
