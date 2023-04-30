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
import glob
from typing import Any, Mapping

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import pandas as pd
from PIL import Image
import numpy as np


labels_1 = [
    "Normal",
    "Pnemonia",
]
labels_2 = [
    ",",
    "Stress-Smoking,ARDS",
    "Virus,",
    "Virus,COVID-19",
    "Virus,SARS",
    "bacteria,",
    "bacteria,Streptococcus",
]
label_one_hot_encoding_1 = {
    "Normal": torch.Tensor([1, 0]),
    "Pnemonia": torch.Tensor([0, 1]),
}
label_one_hot_encoding_2 = {
    ",": torch.Tensor([1, 0, 0, 0, 0, 0, 0]),
    "Stress-Smoking,ARDS": torch.Tensor([0, 1, 0, 0, 0, 0, 0]),
    "Virus,": torch.Tensor([0, 0, 1, 0, 0, 0, 0]),
    "Virus,COVID-19": torch.Tensor([0, 0, 0, 1, 0, 0, 0]),
    "Virus,SARS": torch.Tensor([0, 0, 0, 0, 1, 0, 0]),
    "bacteria,": torch.Tensor([0, 0, 0, 0, 0, 1, 0]),
    "bacteria,Streptococcus": torch.Tensor([0, 0, 0, 0, 0, 0, 1]),
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, backbone, fc1, fc2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x):
        x = self.backbone(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2

    def compute_loss(self, out1, out2, label1, label2):
        loss1 = nn.BCELoss()(out1, label1)
        loss2 = nn.BCELoss()(out2, label2)
        return loss1 + loss2

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.backbone.load_state_dict(state_dict["backbone"], strict=strict)
        self.fc1.load_state_dict(state_dict["fc1"], strict=strict)
        self.fc2.load_state_dict(state_dict["fc2"], strict=strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = {}
        state_dict["backbone"] = self.backbone.state_dict(destination, prefix, keep_vars)
        state_dict["fc1"] = self.fc1.state_dict(destination, prefix, keep_vars)
        state_dict["fc2"] = self.fc2.state_dict(destination, prefix, keep_vars)
        return state_dict


class CTScanDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row["X_ray_image_name"]
        label_1 = row["Label"]
        label_2 = row["Label_1_Virus_category"]
        label_3 = row["Label_2_Virus_category"]

        if isinstance(label_2, float) and np.isnan(label_2):
            label_2 = ""
        if isinstance(label_3, float) and np.isnan(label_3):
            label_3 = ""

        label_2 = label_2 + "," + label_3
        label_1 = label_one_hot_encoding_1[label_1].float()
        label_2 = label_one_hot_encoding_2[label_2].float()

        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label_1, label_2


def load_dataset(batch_size, num_workers):
    # Define the transformations to apply to the input images
    transform = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the dataset
    train_dataset = CTScanDataset(
        "./data/chronohack/train.csv", "./data/chronohack/train", transform
    )
    test_dataset = CTScanDataset("./data/chronohack/test.csv", "./data/chronohack/test", transform)

    # Create the data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def load_model(backbone_model_type):
    # Load checkpoint if it exists
    checkpoint = None
    start_epoch = 0
    files = glob.glob(f"checkpoints/{backbone_model_type}_*.pth")
    if len(files) > 0:
        files.sort(key=os.path.getmtime)
        checkpoint = torch.load(files[-1])
        # Extract the epoch number from the checkpoint file name
        start_epoch = int(files[-1].split("_")[-1].split(".")[0]) + 1

    # Define the backbone model
    if backbone_model_type == "resnet18":
        backbone = models.resnet18(pretrained=True)
    elif backbone_model_type == "resnet34":
        backbone = models.resnet34(pretrained=True)
    elif backbone_model_type == "resnet50":
        backbone = models.resnet50(pretrained=True)
    else:
        raise ValueError("Invalid backbone model type")

    fc1 = nn.Sequential(
        nn.Linear(backbone.fc.out_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, len(label_one_hot_encoding_1)),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Softmax(dim=1),
    )
    fc2 = nn.Sequential(
        nn.Linear(backbone.fc.out_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, len(label_one_hot_encoding_2)),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Softmax(dim=1),
    )

    model = Model(backbone, fc1, fc2)
    model.to(device)

    # Load the checkpointed model parameters
    if checkpoint is not None:
        print(f"Loading checkpoint: {files[-1]}")
        model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found")

    return model, start_epoch
