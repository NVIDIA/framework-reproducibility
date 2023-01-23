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

from pathlib import Path
import json
import argparse
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def get_train_losses(logs, log_data, limits):
    metric = "train_loss"
    for logfile in logs:
        with Path(logfile).open() as f:
            data = []
            global_steps = []
            lines = f.readlines()
            for line in lines:
                # discarding the "DLL " prefix from each line gives us a valid json string
                record = json.loads(line[len("DLL  ") :])
                # step with format [x,y,z] is a training step
                if "step" in record and len(record["step"]) == 3:
                    s = record["step"]
                    if metric in record["data"]:
                        m = record["data"][metric]
                        data.append(m)
                        # record global step
                        gs = (s[0] - 1) * s[2] + s[1]
                        global_steps.append(gs)
                        if not limits[0] or np.min(gs) < limits[0]:
                            limits[0] = np.min(gs)
                        if not limits[1] or np.max(gs) > limits[1]:
                            limits[1] = np.max(gs)
                        if not limits[2] or np.min(m) < limits[2]:
                            limits[2] = np.min(m)
                        if not limits[3] or np.max(m) > limits[3]:
                            limits[3] = np.max(m)
            log_data.append((global_steps, data))


def plot_losses_from_logs(all_logs):
    train_losses = {}
    limits = [None, None, None, None]
    for run_type, file_pattern in all_logs.items():
        log_filelist = sorted(glob.glob(file_pattern))
        train_losses[run_type] = []
        get_train_losses(log_filelist, train_losses[run_type], limits)

    plt.figure(1, figsize=(8, 10), dpi=300)
    idx = 1
    for run_type, loss_data in train_losses.items():
        ax = plt.subplot(2, 2, idx)
        ax.set_title(run_type)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_xlabel("training steps")
        ax.set_ylabel("training loss")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        for steps, losses in loss_data:
            losses = np.array(losses)
            plt.plot(steps, losses)

        idx += 1

    train_stats = {}
    for run_type, loss_data in train_losses.items():
        train_stats[run_type] = {
            "steps": loss_data[0][0],
            "mean": np.mean(loss_data, axis=0)[1],
            "std": np.std(loss_data, axis=0)[1],
        }

    plt.figure(figsize=(8, 4), dpi=300)
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel("training steps")
    ax.set_ylabel("training loss")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    colors = [(0, 0, 1, 0.2), (0, 1, 0, 0.2), (0, 0.5, 0, 0.2), (1, 0, 0, 0.2)]
    idx = 0
    for run_type, loss_data in train_stats.items():
        plt.plot(loss_data["steps"], loss_data["mean"], label=run_type)
        s = loss_data["std"]
        m = loss_data["mean"]
        plt.fill_between(
            loss_data["steps"], y1=m - s, y2=m + s, color=colors[idx], label=f"σ error"
        )
        idx += 1

    ax.legend()
    plt.show()
