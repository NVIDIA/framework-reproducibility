# Framework Determinism

## Introduction

This repository is intended to provide documentation, status, patches, and
tools related to determinism (bit-accurate, run-to-run reproducibility) in deep
learning frameworks, with a focus on determinism when running on GPUs.

Determinism is important when deep learning is used in applications in which
processes must be reproducible, such as robotics (including autonomous
vehicles), heathcare, and finance. Determinism also simplifies and accelerates
both experimentation (by increasing the signal-to-noise ratio) and debugging.

Currently, this repository primarily provides documentation and guidance on how
to obtain deterministic functionality when using deep learning frameworks, and
also, in some cases, the status of that functionality. In some cases, patches
have been provided via installation of the associated PyPI package.

For background, you may want to watch the video of the 2019 GTC talk
[_Determinism in Deep Learning_][1]. The description under that video also
includes links to the slides from the talk and to a poster presentation on this
topic.

## Frameworks

  * [PyTorch](doc/pytorch.md)
  * [TensorFlow](doc/tensorflow.md)

## Future

The nondeterminism debug tool mentioned in the GTC 2019 talk
[_Determinism in Deep Learning_][1] may be released via the associated PyPI
package at some point in the future.

[1]: http://bit.ly/determinism-in-deep-learning

## Credits

Here are the names of some of the people who have helped out with this project.
If any names are missing, then please let us know.

Christoph Angerer,
Ben Barsdell,
Kevin Brown,
Carl Case,
Bryan Catanzaro,
Sharan Chetlur,
Joey Conway,
Emilio Coutinho,
Sanjoy Das,
Timo Denk,
Luke Durant,
Marc Edgar,
Adam Ellsworth,
Mostafa Hagog,
Kaixi Hou,
Pankaj Kanwar,
George Karpenkov,
Tero Karras,
Bob Keating,
Andrew Kerr,
Xiang Bo Kong,
Nicolas Koumchatzky,
Jorge Albericio Latorre,
Lin Lan,
Simon Layton,
Ned Letcher,
Jose Alvarez Lopez,
Nathan Luehr,
Conrado Silva Miranda,
John Montrym,
Michael O'Connor,
Lauri Peltonen,
Rakesh Ranjan,
Jussi Rasanen,
Duncan Riach (PIC),
Josh Romero,
Mikko Ronkainen,
Gavin Seegoolam,
Dilip Sequeria,
Hao Shen,
Bhupinder Singh,
Matthijs De Smedt,
Valentina Taviani,
Phil Teare,
Amirhossein Tebbifakhr,
Matthijs Van keirsbilck,
Kevin Vincent,
Reed Wanderman-Milne,
Stephen Warren,
Shu Wang,
Hao Wu,
Yifang Xu,
Tim Zaman,
William Zhang.
