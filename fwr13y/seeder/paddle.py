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
import paddle
import numpy as np
from .seed_gen import SeedGen, generate_master_seed_randomly


class Seeder:
    def __init__(self, master_seed, ngpus, local_rank):

        self.master_seed_was_none = master_seed is None
        if master_seed is None and local_rank == 0:
            print('INFO: master_seed is None in seeder.init, random '
                  'master_seed will be generated (different one for '
                  'each worker).')

        self.master_seed = (master_seed if master_seed is not None
                            else generate_master_seed_randomly())
        self.seed_gen = SeedGen(self.master_seed, ngpus, local_rank)
        self._ext_generators = []
        self._ext_generators_shared = []

    def register_generator(self, gen, shared=False):
        if shared:
            if self.master_seed_was_none:
                raise Exception('master_seed was None during seeder.init, '
                                'seeds shared among workers cannot be used.')
            self._ext_generators_shared.append(gen)
        else:
            self._ext_generators.append(gen)

    def unregister_generator(self, gen):
        try:
            self._ext_generators.remove(gen)
        except ValueError:
            self._ext_generators_shared.remove(gen)

    def reseed(self, task, epoch):
        seed = self.seed_gen(task, epoch)
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        for generator in self._ext_generators:
            generator(seed)

        if self._ext_generators_shared:
            shared_seed = self.seed_gen(task, epoch, shared_seed=True)
            for generator in self._ext_generators_shared:
                generator(shared_seed)


_seeder_run = None


def init(master_seed, ngpus, local_rank):
    global _seeder_run
    _seeder_run = Seeder(master_seed, ngpus, local_rank)


def reseed(task, epoch=0):
    global _seeder_run
    _seeder_run.reseed(task, epoch)


def get_master_seed():
    global _seeder_run
    return _seeder_run.master_seed


def register_generator(gen, shared=False):
    global _seeder_run
    _seeder_run.register_generator(gen, shared)


def unregister_generator(gen):
    global _seeder_run
    _seeder_run.unregister_generator(gen)
