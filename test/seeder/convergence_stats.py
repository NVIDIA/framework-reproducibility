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

import numpy as np
from scipy import stats
import pandas as pd


def get_convergence_stats(pdata : pd.DataFrame):

    # assuming the first columnt contains hopper val loss and
    # columns 2,3... contain ampere val loss
    ds = pdata.iloc[:, 0] - np.mean(pdata.iloc[:, 1:], axis=1)

    mean_loss = np.mean(ds)
    std_loss = np.std(ds)

    # ll - lower_limit; ul - upper_limit
    ll95, ul95 = stats.t.interval(0.95, len(ds) - 1, mean_loss, std_loss/np.sqrt(len(ds)))
    print(f"Range for 95% CI: ( {ll95:.8f}, {ul95:.8f} )")
    print(f"95% CI contains 0:", "yes" if ll95 < 0 < ul95 else "no")

    ll99, ul99 = stats.t.interval(0.99, len(ds) - 1, mean_loss, std_loss/np.sqrt(len(ds)))
    print(f"Range for 99% CI: ( {ll99:.8f}, {ul99:.8f} )")
    print(f"99% CI contains 0:", "yes" if ll99 < 0 < ul99 else "no")
