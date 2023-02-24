# Framework Reproducibility (fwr13y)

## Repository Name Change

The name of this GitHub repository was changed to
`framework-reproducibility` on 2023-02-14. Prior to this, it was named
`framework-determinism`. Before that, it was named `tensorflow-determinism`.

"In addition to redirecting all web traffic, all `git clone`, `git fetch`, or
`git push` operations targetting the previous location[s] will continue to
function as if made to the new location. However, to reduce confusion, we
strongly recommend updating any existing local clones to point to the new
repository URL." -- [GitHub documentation][1]


## Repository Intention

This repository is intended to:
* provide documentation, status, patches, and tools related to
  [determinism][2] (bit-accurate, run-to-run reproducibility) in deep learning
  frameworks, with a focus on determinism when running on GPUs, and
* provide a tool, and related guidelines, for reducing variance
  ([Seeder][3]]) in deep learning frameworks.

[1]: https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository
[2]: ./doc/d9m/README.md
[3]: ./doc/seeder/README.md
