# PyTorch Determinism

## Introduction

We have some experience debugging and removing nondeterminism in PyTorch-based
models, but our level of experience, so far, is not as extensive as for
TensorFlow.

PyTorch documentation includes some guidance for attaining GPU-determinism on
its [reproducibility page][1], which we have contributed to. Please refer to
that page also because it probably has different or additional information to
this current one.

Getting reproducible functionality on a single GPU, as with other frameworks,
involves several considerations:

## Seeds

Random seeds need to be set to ensure that the various psuedorandom number
generators (PRNGs) are reset reproducibily, this incudes ensuring that the
traininable variables are initialized reproducibly before training starts and
random processes, such as dropout, progress through a reproducible sequence.

Before starting the Python process itself, the environment variable
`PYTHONHASHSEED` should be set. Then, inside the Python process, the following
seeds should be set:

```
SEED = 123 # or whatever you choose
random.seed(SEED) # if you're using random
np.random.seed(SEED) # if you're using numpy
torch.manual_seed(SEED) # torch.cuda.manual_seed_all(SEED) is not required
```

It's often worth confirming that the trainable variables are being reproducibly
initialized by creating and printing some kind of digest of all the trainable
variables before beginning to train. Appropriate digests include a sum or a
hash.

## Data Loader

You'll need to make sure that your data loader process is reproducible, so that
the sequence of examples or batches of examples delivered to your model are
perfectly reproducible. If you have a mutlithreaded data loader, then it's
important not to share PRNG state between threads. There may be other
data loader restrictions that I'm not yet aware of.

Reproducible inter-epoch re-shuffling can be attained by creating
an instance (`self.g`) of `torch.Generator` in your
`torch.utils.data.DataLoader` and using it as follows:

```
def set_epoch(self, epoch):
  self.epoch = epoch
  if self.shuffle:
    # We want every epoch to shuffle differently, but in a reproducible way.
	# Therefore, reset the generator differently but reproducibly on each
	# epoch. It is recommended for the seed to have a good balance of zero and
	# one bits.
	# See https://pytorch.org/docs/stable/generated/torch.Generator.html
    self.g.manual_seed(5728479885 + self.epoch)
```

Then call `set_epoch` at the start of each epoch.

## Deterministic Ops

Once the trainable variables are initializing reproducibly and training
examples are being delivered reproducibly, the next step is to maximally enable
deterministic ops. The way you do this in versions of PyTorch earlier than 1.7
is a follows:

```
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The first line causes the deterministic algorithms to be selected in the NVIDIA
libraries, when they are available for functionality that PyTorch uses in those
libraries: convolution, max pooling, and CTC loss (all three from cuDNN), and
batch matrix-matrix product (from cuBLAS).

The second line disables dynamic selection of cuDNN convolution algorithms
and ensures that the algorithm selection itself is reproducible.

The [reproducibilty page][1] contains a reasonable but non-comprehensive list of
ops the are nondeterminsitic on GPU. Using these will cause nondeterminism to
be injected into the operation of your model which will, as is typical, lead to
the entire operation of the model to become nondeterministic. Those ops should
be avoided for now.

The non-cuDNN implementation of `torch.nn.CTCLoss` operates
nondeterministically. In order to use the cuDNN implementation (so that its
deterministic functionality can be selected via
`torch.backends.cudnn.deterministic = True`), certain
criteria must be met, as described in the PyTorch [documentation][4] for
`torch.nn.CTCLoss`. Another way of obtaining determinsitic CTC functionality
is to use [WarpCTC][2].

PyTorch 1.7 includes a new function, [`torch.set_deterministic`][5], which
precludes the need to set either `torch.backends.cudnn.deterministic` or
`torch.backends.cudnn.benchmark`. An additional advantage of using this function
is that it will cause an exception to be thrown if you try to use an op that
could inject nondeterminism into your model. It's impossible for an exception to
be thrown in all circumstances when nondeterminism could be introduced by an op,
let alone by the many other possible sources, but this feature will reduce the
amount of time spent isolating sources of nondeterminism coming from ops that
have already been identified as currently not able to operate deterministically
on a GPU.

## Reproducible Checkpointing

To save state and later resume reproducibly (ending the training process
exactly as if it had not been interrupted) you should `torch.save` and
`torch.load` the following state (as needed) using [the approach][6] given in
the PyTorch documentation, including the [guidance][7] for saving and loading
GPU state:

  * data loader state,
  * `model.state_dict()`,
  * `optimizer.state_dict()`, which includes the current learning rate and any
    other learning rate scheduler state,
  * epoch / iteration counter,
  * `torch.cuda.GradScaler` statistics,
  * `torch.get_rng_state()`,
  * `torch.cuda.get_rng_state()`,
  * `np.random.get_state()`, and
  * `random.getstate()`

## Multi-GPU

As with any deep learning framework, you must first get your model training
deterministically on a single GPU, which means getting exactly the same
trainable variables at the end of training on each run. Then, you can attempt
to extend that to multiple GPUs.

If you're using Horovod then you should set the environment variable
`HOROVOD_FUSION_THRESHOLD=0`. Other than that, I'm not aware of sources of
nondeterminism in other multi-GPU training regimes used with PyTorch.

## CPU Determinism

I currently don't know of any tricks to resolve nondeterminism when PyTorch ops
are running on a CPU, nor am I currently aware of the extent of such issues.

## Credit

Matthijs Van Keirsbilk of NVIDIA Research provided a significant amount of
the content on this page.

[1]: https://pytorch.org/docs/stable/notes/randomness.html
[2]: https://github.com/SeanNaren/warp-ctc
[3]: https://pytorch.org/tutorials/beginner/saving_loading_models.html
[4]: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
[5]: https://pytorch.org/docs/stable/generated/torch.set_deterministic.html
[6]: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
[7]: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-on-gpu-load-on-gpu
