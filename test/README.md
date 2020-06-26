# Testing of tensorflow-determinism

## How to run tests

Requirements:

  * Installed GPU that supports CUDA
  * GPU driver that supports CUDA 10
  * docker
  * nvidia-docker
  * pip install -r requirements.txt

Then you should be able to run `./all.sh` and observe the final message.

## Things to do before next release

  * Add integration tests for enable_determinism
  * Move TensorFlow-related functionality into tensorflow subdirectory
  * Change repo URL and description
  * Factor integration test run/test mechanism for others to use
  * Update documentation
