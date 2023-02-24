# Might-Do Menu

  * Search for `TODO` in the files in this repo.
  * Add integration testing, which would confirm that
    `fwr13y.d9m.tensorflow.enable_determinism` makes TensorFlow operate
    deterministically when training models containing all of the previously
    nondeterministic ops. This effort was started in the `integration-testing`
    branch.
  * Enhance the test-running mechanism so that `all.sh` is written in Python
    and reads test specifications from a yaml file. This effort was started in
    the `integration-testing` branch.
  * Factor integration test running mechanism for others to use (inclduing
    Seeder).
