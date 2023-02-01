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

import argparse
import pandas as pd
from print_stats import print_stats

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-file",
        type=str,
        help="Input log file",
    )
    return parser


def main():

    parser = parse_args()
    args = parser.parse_args()

    pdata = pd.read_csv(args.log_file, index_col=0)
    print_stats(pdata)


if __name__ == "__main__":
    main()
