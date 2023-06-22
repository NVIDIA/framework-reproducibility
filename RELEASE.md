# Release 0.5.0

In Seeder, add the ability to reseed generators with either
  1. a different seed for each worker (the existing functionality) or
  2. the same seed for each worker (added functionality).

# Release 0.4.0

## Enhanced Functionality

  * Add the Seeder tool for variance reduction. This is an experimental feature.
  * Rename the distribution from tensorflow-determinism to
    framework-reproducibility and rename the package from tfdeterminism to
    fwr13y.
  * Add `fwr13y.d9m.tensorflow.enable_determinism`, which makes a best-effort
    to enable determinism in whichever version of TensorFlow is being used.
  * Add a script to find commits in the TensorFlow repo related to determinism.
  * `fwr13y.d9m.tensorflow.patch` throws more specific exceptions.

## Enhanced Testing / Higher Quality

  * Test patched determinism over a wider range of stock TensorFlow and NGC
    TensorFlow versions.

# Release 0.3.0

Add patch availability for stock TensorFlow version 2.0, and test in eager mode.

Developed by Duncan Riach with thanks to Nathan Luehr for review.

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
