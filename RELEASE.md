# Release 0.3.0

Add patch availability on TensorFlow version 2.0

# Release 0.2.0

## New Functionality

* Add patch availability on TensorFlow version 1.15
* Print the version of tensorflow-determinism when patch is applied

## Enhanced Testing / Higher Quality

* Test that patch will throw exception on non-supported versions of TF
* Test that patch will throw exception in NGC containers
* Test that patch works in Python 3
* Test that package will install when TensorFlow is not yet installed

Developed by Duncan Riach with thanks to Nathan Luehr for review.

# Release 0.1.0

This release includes a patch for standard TF 1.14.0 that enables most deep
learning TF models to train deterministically on GPUs. GPU-determinism support
in the NVIDIA NGC TF containers is also described.

Developed by Duncan Riach with thanks to Nathan Luehr for review.
