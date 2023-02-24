# Framework Reproducibility: Determinism (d9m)

## Introduction

Determinism is important when deep learning is used in applications in which
processes must be reproducible, such as robotics (including autonomous
vehicles), heathcare, and finance. Determinism also simplifies and accelerates
both experimentation (by increasing the signal-to-noise ratio) and debugging.

When determinism is optional, choosing it usually leads to slighltly lower
run-to-run performance. However, when nondeterminism would lead to multiple runs
being necessary, determimism can provide an overall increase in performance
by making a single run sufficient.

Determinism, often used as a short-hand for run-to-run reproducibility,
automatically yields between-implementations determinism (possible differences
when running on different hardware-software stacks are exactly and controllably
reproducible), and is a foundation for guaranteed between-implementations
reproducibility (results on any two particular and different hardware-software
stacks are bit-exactly the same). However, between-implementations
reproducibility is currently outside the scope of this project. In other words,
if you change anything about the hardware-software stack, the results from
training and/or inference may be different.

Currently, this repository primarily provides documentation and guidance on how
to obtain deterministic functionality when using deep learning frameworks, and
also, in some cases, the status of that functionality. In some cases
(historically), patches have been provided via installation of the associated
PyPI package.

For background, you may want to watch the video of the 2019 GTC talk
[_Determinism in Deep Learning_][1]. The description under that video also
includes links to the slides from the talk and to a poster presentation on this
topic.

## Myths About Determinism

MYTH: "You just have to set the seeds"

FACT: Setting the seeds (which means reproducibly resetting) the various
pseudo-random number generators in a software system is necessary but not
necessarily sufficient to obtain deterministic functionality. In deep learning
frameworks, it is often also necessary to have available and selectable
determinstic versions of all the underlying algorithms that are utilized by the
model and other parts of the program (such as input datapaths). While there are
many other potential sources of nondeterminism in a deep learning program, this
is the most commonly occuring and often the hardest to address. It has required
significant additional software development work in the deep learning frameworks
to provide this functionality.

MYTH: "Setting the seeds is not enough, so there must be a seed somewhere
      that's not getting set."

FACT: This is an extension of the previous assumption that obtaining
determinism is only about setting seeds. After setting all the seeds,
remaining nondeterminism could be due to a missing seed, but probably not.

MYTH: "Deterministic ops are slower."

FACT: For most of the ops in the deep learning frameworks, the default and
fastest known implementation of the op is deterministic. For a few ops, the
fastest implemented version is, or was, nondeterminsitic. In those cases, a
potentially slower implementation has been (or will be) developed that provides
deterministic functionality. In many cases, when the implementation of an op has
been revisisted in depth, the newly crafted determinsitic implementation has
been faster than the preexisting, and potentially hastily-implemented,
nondeterminsitic implementation. In these cases, as with most DL ops, the
determinsitic implementation has been set to always-selected.

MYTH: "The uncontrollable nondeterminism in the DL frameworks is necessary or
       useful for deep learning."

FACT: While this is stictly true, it's based on, and supports, the presumption
that nondeterminism cannot, and should not, be addressed. In fact, stochastic
gradient descent (and other similar methods) do benefit from randomness, but
that randomness need not be uncontrollable. Pseudo-random number generators can
be, and are, used to introduce noise and variance into training data to increase
the generalization of models, and controllable randomness is more than
sufficient. While the uncontrollable randomness introduced by things like
nondeterministic ops might happen to compensate for a lack of intentionally
introduced controllable noise, there is no reason, other than effort, not to
replace uncontrollable (or unreprodcible) noise with controllable (or
reproducible) noise. Once we have determinism, we can choose exactly where to
introduce reproducible noise into our system, exactly what type, and exactly how
much. Also, if we cannot disable the noise, it's very hard to determine whether
it is truly beneficial or not.

MYTH: "GPUs are inherently nondeterministic"

FACT: Any system that utilizes asynchronous parallelism can, theoretically, be
configured to convert asynchronous parallelism into truly random noise. GPUs
*can* introduce truly random noise, as *can* many other parallel computing
systems. That doesn't mean that it has to be, or should be, done; it's optional.

### Determinism in Specific Frameworks

  * [PyTorch][2]
  * [TensorFlow][3]

### Future Directions for Determinism

The nondeterminism debug tool mentioned in the GTC 2019 talk
[_Determinism in Deep Learning_][1] may be released via the associated PyPI
package at some point in the future.

[1]: http://bit.ly/determinism-in-deep-learning
[2]: ./pytorch.md
[3]: ./tensorflow.md

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
