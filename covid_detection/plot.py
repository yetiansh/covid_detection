# Copyright Ye Tian (yetiansh@gmail.com)
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


import os

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(context="poster", style="whitegrid")
sns.set(font_scale=0.8)

plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plot"))
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


size_0 = (15, 3)
size_1 = (15, 6)


def plot_pr_auc(precision_list, recall_list, threshold_list, keys, title, prefix):
    axes = None
    fig = None
    if len(keys) == 2:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(*size_0)
        axes = axes.flatten()
    elif len(keys) == 7:
        fig, axes = plt.subplots(nrows=2, ncols=4)
        fig.set_size_inches(*size_1)
        axes = axes.flatten()

    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    for precision, recall, threshold, key, axis in zip(
        precision_list, recall_list, threshold_list, keys, axes
    ):
        key = key.split(",")
        key = [key_ if len(key_) > 0 else "None" for key_ in key]
        key = ",".join(key)
        axis.plot(
            recall,
            precision,
        )
        axis.set_title(key)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_ylim([-0.05, 1.05])
        axis.set_xlim([-0.05, 1.05])

    plt.suptitle(title)

    prefix = prefix.replace(" ", "_")
    plt.savefig(os.path.join(plot_dir, f"{prefix}_pr_auc.pdf"), bbox_inches="tight")


def plot_roc_auc(fpr_list, tpr_list, threshold_list, keys, title, prefix):
    axes = None
    fig = None
    if len(keys) == 2:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(*size_0)
        axes = axes.flatten()
    elif len(keys) == 7:
        fig, axes = plt.subplots(nrows=2, ncols=4)
        fig.set_size_inches(*size_1)
        axes = axes.flatten()

    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    for fpr, tpr, threshold, key, axis in zip(fpr_list, tpr_list, threshold_list, keys, axes):
        key = key.split(",")
        key = [key_ if len(key_) > 0 else "None" for key_ in key]
        key = ",".join(key)
        axis.plot(
            fpr,
            tpr,
        )
        axis.set_title(key)
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_ylim([-0.05, 1.05])
        axis.set_xlim([-0.05, 1.05])

    plt.suptitle(title)

    prefix = prefix.replace(" ", "_")
    plt.savefig(os.path.join(plot_dir, f"{prefix}_roc_auc.pdf"), bbox_inches="tight")
