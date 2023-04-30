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


import argparse
import numpy as np

import torch

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from covid_detection.model import (
    load_dataset,
    load_model,
    device,
    label_one_hot_encoding_1,
    label_one_hot_encoding_2,
)
from covid_detection.plot import plot_roc_auc, plot_pr_auc


def micro_avg_roc_auc(labels, outputs):
    roc_auc = roc_auc_score(labels, outputs, average="micro")
    return roc_auc


def micro_avg_pr_auc(labels, outputs):
    pr_auc = average_precision_score(labels, outputs, average="micro")
    return pr_auc


def macro_avg_roc_auc(labels, outputs):
    roc_auc_list = []
    fpr_list, tpr_list, thresholds_list = [], [], []
    for i in range(labels.shape[1]):
        labels_flat = labels[:, i]
        outputs_flat = outputs[:, i]
        if labels_flat.sum() > 0:
            roc_auc = roc_auc_score(labels_flat, outputs_flat)
            fpr, tpr, thresholds = roc_curve(labels_flat, outputs_flat)
            roc_auc_list.append(roc_auc)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            thresholds_list.append(thresholds)

    macro_roc_auc = np.mean(roc_auc_list)
    return macro_roc_auc, fpr_list, tpr_list, thresholds_list


def macro_avg_pr_auc(labels, outputs):
    pr_auc_list = []
    precision_list, recall_list, thresholds_list = [], [], []
    for i in range(labels.shape[1]):
        labels_flat = labels[:, i]
        outputs_flat = outputs[:, i]
        if labels_flat.sum() > 0:
            pr_auc = average_precision_score(labels_flat, outputs_flat)
            precision, recall, thresholds = precision_recall_curve(labels_flat, outputs_flat)
            pr_auc_list.append(pr_auc)
            precision_list.append(precision)
            recall_list.append(recall)
            thresholds_list.append(thresholds)

    macro_pr_auc = np.mean(pr_auc_list)
    return macro_pr_auc, precision_list, recall_list, thresholds_list


def apply_model(dataloader, model):
    model.eval()
    labels_1 = list()
    labels_2 = list()
    outputs_1 = list()
    outputs_2 = list()
    with torch.no_grad():
        for idx, (inputs, labels1, labels2) in enumerate(dataloader):
            outputs_1_, outputs_2_ = model(inputs.to(device))
            labels_1.extend(labels1.numpy())
            outputs_1.extend(outputs_1_.cpu().numpy())
            labels_2.extend(labels2.numpy())
            outputs_2.extend(outputs_2_.cpu().numpy())

    labels_1.extend([e.cpu().numpy() for e in label_one_hot_encoding_1.values()])
    outputs_1.extend([e.cpu().numpy() for e in label_one_hot_encoding_1.values()])
    labels_2.extend([e.cpu().numpy() for e in label_one_hot_encoding_2.values()])
    outputs_2.extend([e.cpu().numpy() for e in label_one_hot_encoding_2.values()])

    labels_1 = np.array(labels_1)
    outputs_1 = np.array(outputs_1)
    labels_2 = np.array(labels_2)
    outputs_2 = np.array(outputs_2)

    return labels_1, outputs_1, labels_2, outputs_2


def compute_accuracy(dataloader, model):
    correct_1 = 0
    correct_2 = 0
    total_1 = 0

    labels_1, outputs_1, labels_2, outputs_2 = apply_model(dataloader, model)
    for label_1, output_1 in zip(labels_1, outputs_1):
        if np.argmax(label_1) == np.argmax(output_1):
            correct_1 += 1
        total_1 += 1

    total_2 = 0
    for label_2, output_2 in zip(labels_2, outputs_2):
        if np.argmax(label_2) == np.argmax(output_2):
            correct_2 += 1
        
        total_2 += 1

    return correct_1 / total_1, correct_2 / total_2


def compute_micro_roc_auc(dataloader, model):
    labels_1, outputs_1, labels_2, outputs_2 = apply_model(dataloader, model)
    roc_auc_1 = micro_avg_roc_auc(labels_1, outputs_1)
    roc_auc_2 = micro_avg_roc_auc(labels_2, outputs_2)
    return roc_auc_1, roc_auc_2


def compute_micro_pr_auc(dataloader, model):
    labels_1, outputs_1, labels_2, outputs_2 = apply_model(dataloader, model)
    pr_auc_1 = micro_avg_pr_auc(labels_1, outputs_1)
    pr_auc_2 = micro_avg_pr_auc(labels_2, outputs_2)
    return (
        pr_auc_1,
        pr_auc_2,
    )


def compute_macro_roc_auc(dataloader, model):
    labels_1, outputs_1, labels_2, outputs_2 = apply_model(dataloader, model)
    roc_auc_1, fpr_1, tpr_1, threshold_1 = macro_avg_roc_auc(labels_1, outputs_1)
    roc_auc_2, fpr_2, tpr_2, threshold_2 = macro_avg_roc_auc(labels_2, outputs_2)
    return roc_auc_1, roc_auc_2, fpr_1, tpr_1, threshold_1, fpr_2, tpr_2, threshold_2


def compute_macro_pr_auc(dataloader, model):
    labels_1, outputs_1, labels_2, outputs_2 = apply_model(dataloader, model)
    pr_auc_1, precision_1, recall_1, thresholds_1 = macro_avg_pr_auc(labels_1, outputs_1)
    pr_auc_2, precision_2, recall_2, thresholds_2 = macro_avg_pr_auc(labels_2, outputs_2)
    return (
        pr_auc_1,
        pr_auc_2,
        precision_1,
        recall_1,
        thresholds_1,
        precision_2,
        recall_2,
        thresholds_2,
    )


def test_model(
    backbone_model_type="resnet18",
    num_workers=4,
    metrics=None,
):
    metrics = metrics.split(",")

    train_loader, test_loader = load_dataset(64, num_workers)

    model, epoch = load_model(backbone_model_type)
    for metric in metrics:
        if metric == "accuracy":
            print("Computing accuracy")
            test_acc_1, test_acc_2 = compute_accuracy(test_loader, model)
            print(f"Test accuracy: {test_acc_1}, {test_acc_2}")

            train_acc_1, train_acc_2 = compute_accuracy(train_loader, model)
            print(f"Train accuracy: {train_acc_1}, {train_acc_2}")

        elif metric == "micro_roc_auc":
            print("Computing micro ROC AUC")
            (
                roc_auc_1,
                roc_auc_2,
            ) = compute_micro_roc_auc(test_loader, model)
            print(f"Micro ROC AUC: {roc_auc_1}, {roc_auc_2}")

            (
                roc_auc_1,
                roc_auc_2,
            ) = compute_micro_roc_auc(train_loader, model)
            print(f"Micro ROC AUC: {roc_auc_1}, {roc_auc_2}")

        elif metric == "micro_pr_auc":
            print("Computing micro PR AUC")
            (
                pr_auc_1,
                pr_auc_2,
            ) = compute_micro_pr_auc(test_loader, model)
            print(f"Micro PR AUC: {pr_auc_1}, {pr_auc_2}")

            (
                pr_auc_1,
                pr_auc_2,
            ) = compute_micro_pr_auc(train_loader, model)
            print(f"Micro PR AUC: {pr_auc_1}, {pr_auc_2}")

        elif metric == "macro_roc_auc":
            print("Computing macro ROC AUC")
            (
                roc_auc_1,
                roc_auc_2,
                fpr_1,
                tpr_1,
                thresholds_1,
                fpr_2,
                tpr_2,
                thresholds_2,
            ) = compute_macro_roc_auc(test_loader, model)
            print(f"Macro ROC AUC: {roc_auc_1}, {roc_auc_2}")
            plot_roc_auc(
                fpr_1,
                tpr_1,
                thresholds_1,
                label_one_hot_encoding_1.keys(),
                f"ROC AUC for label 1 on test set of {backbone_model_type}",
                f"{backbone_model_type}_{epoch}_macro_roc_1_test",
            )
            plot_roc_auc(
                fpr_2,
                tpr_2,
                thresholds_2,
                label_one_hot_encoding_2.keys(),
                f"ROC AUC for label 2 on test set of {backbone_model_type}",
                f"{backbone_model_type}_{epoch}_macro_roc_2_test",
            )

            (
                roc_auc_1,
                roc_auc_2,
                fpr_1,
                tpr_1,
                thresholds_1,
                fpr_2,
                tpr_2,
                thresholds_2,
            ) = compute_macro_roc_auc(train_loader, model)
            print(f"Macro ROC AUC: {roc_auc_1}, {roc_auc_2}")
            plot_roc_auc(
                fpr_1,
                tpr_1,
                thresholds_1,
                label_one_hot_encoding_1.keys(),
                f"ROC AUC for label 1 of {backbone_model_type}",
                f"{backbone_model_type}_macro_roc_1_train",
            )
            plot_roc_auc(
                fpr_2,
                tpr_2,
                thresholds_2,
                label_one_hot_encoding_2.keys(),
                f"ROC AUC for label 2 of {backbone_model_type}",
                f"{backbone_model_type}_{epoch}_macro_roc_2_train",
            )

        elif metric == "macro_pr_auc":
            print("Computing macro PR AUC")
            (
                pr_auc_1,
                pr_auc_2,
                precision_1,
                recall_1,
                thresholds_1,
                precision_2,
                recall_2,
                thresholds_2,
            ) = compute_macro_pr_auc(test_loader, model)
            print(f"Macro PR AUC: {pr_auc_1}, {pr_auc_2}")
            plot_pr_auc(
                precision_1,
                recall_1,
                thresholds_1,
                label_one_hot_encoding_1.keys(),
                f"PR AUC for label 1 of {backbone_model_type}",
                f"{backbone_model_type}_{epoch}_macro_pr_1_test",
            )
            plot_pr_auc(
                precision_2,
                recall_2,
                thresholds_2,
                label_one_hot_encoding_2.keys(),
                f"PR AUC for label 2 of {backbone_model_type}",
                f"{backbone_model_type}_{epoch}_macro_pr_2_test",
            )

            (
                pr_auc_1,
                pr_auc_2,
                precision_1,
                recall_1,
                thresholds_1,
                precision_2,
                recall_2,
                thresholds_2,
            ) = compute_macro_pr_auc(train_loader, model)
            print(f"Macro PR AUC: {pr_auc_1}, {pr_auc_2}")
            plot_pr_auc(
                precision_1,
                recall_1,
                thresholds_1,
                label_one_hot_encoding_1.keys(),
                f"PR AUC for label 1 of {backbone_model_type}",
                f"{backbone_model_type}_{epoch}_macro_pr_1_train",
            )
            plot_pr_auc(
                precision_2,
                recall_2,
                thresholds_2,
                label_one_hot_encoding_2.keys(),
                f"PR AUC for label 2 of {backbone_model_type}",
                f"{backbone_model_type}_{epoch}_macro_pr_2_train",
            )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model for ChronoHack challenge")
    parser.add_argument(
        "--model",
        type=str,
        default="vgg11",
        help="Backbone model type (vgg11, vgg16, resnet18, resnet34, resnet50)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=64, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--metrics", type=str, default="accuracy", help="Metrics to use for evaluation"
    )
    args = parser.parse_args()

    # Train the model
    test_model(
        backbone_model_type=args.model,
        num_workers=args.num_workers,
        metrics=args.metrics,
    )
