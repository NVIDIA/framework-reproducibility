# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================

import random
import csv
import numpy as np
import torch

from os.path import abspath, dirname
import sys

sys.path.append(dirname(abspath(__file__)) + "/../../")
import fwr13y.seeder.pyt as seeder


class GetSeed:
    def __init__(self):
        self.seed = None

    def get_seed(self, seed):
        self.seed = seed


def call_torch_rand():
    return torch.rand(1).item()


def call_np_rand():
    return np.random.rand(1)[0]


def call_random():
    return random.random()


def generate_random_numbers(
    seeds, csv_filename, num_iterations_per_seed, seed_fn, random_fn
):
    csv_data = []
    # header
    if len(seeds) == 1:
        csv_data.append(["random value"])
    else:
        csv_data.append(["seed", "random value"])
        
    for s in seeds:
        seed_fn(s)
        for i in range(num_iterations_per_seed):
            if len(seeds) > 1:
                csv_data.append([s, random_fn()])
            else:
                csv_data.append([random_fn()])
    csvfile = csv_filename
    with open(csvfile, "w") as f:
        writer = csv.writer(f)
        for row in csv_data:
            writer.writerow(row)


def test_randomness(master_seed):

    # seedeing functions for random number generators
    seed_functions = (torch.manual_seed, np.random.seed, random.seed)
    random_functions = (call_torch_rand, call_np_rand, call_random)
    rng_names = ("torch", "numpy", "random")

    # num_seeds corresponds to the number of epochs

    # testing random number generation for one seed
    num_iters = 1000000
    for seed_fn, random_fn, name in zip(seed_functions, random_functions, rng_names):
        csv_filename = (
            "rand_"
            + name
            + "_one_seed_"
            + str(master_seed)
            + "_"
            + str(num_iters)
            + "_"
            + str(master_seed)
            + ".csv"
        )
        generate_random_numbers(
            [master_seed], csv_filename, num_iters, seed_fn, random_fn
        )

    # testing random number generation for 1000 seeds
    num_seeds = 1000
    num_iters_per_seed = 1000

    # seeds are generated:
    # - from sequence 0 .. num_seeds-1
    seeds_from_seq = []
    for s in range(num_seeds):
        seeds_from_seq.append(s)

    for seed_fn, random_fn, name in zip(seed_functions, random_functions, rng_names):
        filename = (
            "rand_"
            + name
            + "_1000seeds_from_seq_1000iters_"
            + str(master_seed)
            + ".csv"
        )

        generate_random_numbers(
            seeds_from_seq, filename, num_iters_per_seed, seed_fn, random_fn
        )

    # - from seedGen in seeder.py
    seeds_from_seedGen = []
    sg = seeder.SeedGen(master_seed=master_seed, ngpus=1, local_rank=0)
    for s in range(num_seeds):
        seeds_from_seedGen.append(sg(1, s))

    for seed_fn, random_fn, name in zip(seed_functions, random_functions, rng_names):
        csv_filename = (
            "rand_"
            + name
            + "_1000seeds_from_seedGen_1000iters_"
            + str(master_seed)
            + ".csv"
        )
        generate_random_numbers(
            seeds_from_seedGen, csv_filename, num_iters_per_seed, seed_fn, random_fn
        )


def main():
    test_randomness(int(sys.argv[1]))


if __name__ == "__main__":
    main()
