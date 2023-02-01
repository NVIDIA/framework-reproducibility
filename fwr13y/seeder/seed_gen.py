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
import time
import sys

_MAX_INT32 = 2147483647


class SeedGen:
    def __init__(self, master_seed, ngpus, local_rank):
        self.master_seed = master_seed
        self.ngpus = ngpus
        self.local_rank = local_rank
        self.ntasks = 2
        self._used_seeds = set()
        self._rng = random.Random(0)

    def __call__(self, task, epoch):
        seed = (
            self.master_seed + (epoch * self.ngpus + self.local_rank)
        ) * self.ntasks + task
        if seed in self._used_seeds:
            print(
                "Warning!!! seed has been generated more than once!!!", file=sys.stderr
            )
        self._used_seeds.add(seed)
        self._rng.seed(seed)
        return self._rng.randint(0, _MAX_INT32)


def generate_master_seed_randomly():
    to_micro = 10 ** 6
    # for compatibility with older python version we can't use time_ns()
    seed = int(time.time() * to_micro) % _MAX_INT32
    return seed
