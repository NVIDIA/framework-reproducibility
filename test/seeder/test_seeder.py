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

import csv
import numpy as np

from os.path import abspath, dirname
import sys
sys.path.append(dirname(abspath(__file__))+'/../../')
import fwr13y.seeder.pyt as seeder

class GetSeed:
    def __init__(self):
        self.seed = None

    def get_seed(self, seed):
        self.seed = seed


def test_seeder():

    csv_data = []
    # header
    csv_data.append(["master_seed", "local_rank", "epoch", "task", "gen seed"])

    def generate_seeds():
        for local_rank in range(8):
            seeder.init(master_seed=master_seed, ngpus=8, local_rank=local_rank)
            gs = GetSeed()
            seeder.register_generator(gs.get_seed)

            seeder.reseed(0, 0)
            seed = gs.seed
            csv_data.append([master_seed, local_rank, 0, 0, seed])
            for epoch in range(10):
                seeder.reseed(1, epoch)
                seed = gs.seed
                csv_data.append([master_seed, local_rank, 0, 0, seed])

    for master_seed in np.random.randint(low=0, high=100000, size=10):
        generate_seeds()
    for master_seed in range(10):
        generate_seeds()

    with open("generated_seeds.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for row in csv_data:
            writer.writerow(row)


def main():
    test_seeder()


if __name__ == "__main__":
    main()
