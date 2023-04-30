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
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from covid_detection.model import device, load_model, load_dataset


def train_model(
    backbone_model_type="resnet18",
    batch_size=32,
    num_workers=4,
    learning_rate=0.001,
    num_epochs=10,
    checkpoint_interval=1,
):
    writer = SummaryWriter("runs/" + backbone_model_type)

    # Load dataset
    train_loader, test_loader = load_dataset(batch_size, num_workers)

    # Load the pretrained model
    model, start_epoch = load_model(backbone_model_type)

    # Define the optimizer
    optimizer = optim.Adam(
        [
            {"params": model.fc1.parameters()},
            {"params": model.fc2.parameters()},
        ],
        lr=learning_rate,
    )

    # Train the model
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for i, (inputs, labels_1, labels_2) in enumerate(train_loader):
            # Move the inputs and labels to the GPU (if available)
            inputs = inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            out1, out2 = model(inputs)

            labels_1 = labels_1.to(device)
            labels_2 = labels_2.to(device)

            # Compute the loss
            loss = model.compute_loss(out1, out2, labels_1, labels_2)

            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)

            # Print statistics
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}",
                    flush=True,
                )

        # Print epoch loss
        epoch_loss = running_loss / len(train_loader)
        print(f"Train loss of epoch [{epoch + 1}/{num_epochs}]: {epoch_loss:.4f}", flush=True)

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_filename = f"{backbone_model_type}_checkpoint_{epoch + 1}.pth"
            checkpoint_path = os.path.join("./checkpoints", checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_path)

        # Test the model
        model.eval()
        with torch.no_grad():
            for inputs, labels_1, labels_2 in test_loader:
                # Move the inputs and labels to the GPU (if available)
                inputs = inputs.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                out1, out2 = model(inputs)

                labels_1 = labels_1.to(device)
                labels_2 = labels_2.to(device)

                # Compute the loss
                loss = model.compute_loss(out1, out2, labels_1, labels_2)

                # Compute accuracy
                running_loss += loss.item() * inputs.size(0)

            average_loss = running_loss / len(test_loader.dataset)

            writer.add_scalar("Loss/test", average_loss, epoch)

        print(f"Test loss of epoch [{epoch + 1}/{num_epochs}]: {average_loss}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model on COVID-19 dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Backbone model type (resnet18, resnet34, resnet50)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=32, help="Number of workers for data loading"
    )
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--checkpoint_interval", type=int, default=5, help="Interval between saving checkpoints"
    )
    args = parser.parse_args()

    # Train the model
    train_model(
        args.model,
        args.batch_size,
        args.num_workers,
        args.learning_rate,
        args.num_epochs,
        args.checkpoint_interval,
    )
