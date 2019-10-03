# Release 0.2.0

* Add patch availability on TensorFlow version 1.15
* Print the version of tensorflow-determinism when patch is applied
* Test that patch will throw exception on non-supported versions of TF
* Test that patch will throw exception in NGC containers

# Release 0.1.0

This release includes a patch for standard TF 1.14.0 that enables most deep
learning TF models to train deterministically on GPUs. GPU-determinism support
in the NVIDIA NGC TF containers is also described.
