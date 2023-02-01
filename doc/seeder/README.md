# Framework Reproducibility: Seeder

## Introduction

Seeder (implemented in the `seeder` sub-module) generates deterministic seeds to
reduce variance across training (and inference) runs. This reduces the number
of runs needed to catch regressions without changing the underlying algorithms
used (from nondeterministic to deterministic). It supports suspend/resume
functionality.

Seeder is an experimental feature.

## Installation

pip install framework-reproducibility --upgrade

## Frameworks Supported

See the instructions specific to the framework you're using:

* [PyTorch](./seeder_pyt.md)
* [TensorFlow2](./seeder_tf2.md)
