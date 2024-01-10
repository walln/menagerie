"""Train ResNet."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from tqdm import tqdm

from datasets import load_dataset
from menagerie.resnet.model import ResNet18


def train_resnet():
    """Train ResNet."""
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 128

    logger.info("Hyperparameters:")
    logger.info(f"Num epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")

    dataset = "fcakyon/pokemon-classification"
    logger.info(f"Loading {dataset} dataset...")

    # Transformation for dataset
    def transform(batch):
        # Convert image to numpy array and scale to [0, 1]
        image = batch["img"]
        label = batch["label"]
        image = np.array(image) / 255.0

        # Normalize the image
        mean = np.array((0.5, 0.5, 0.5))
        std = np.array((0.5, 0.5, 0.5))
        normalized_image = (image - mean) / std

        # print("Shape before transpose:", normalized_image.shape)
        # PyTorch expects the color channel to be the first dimension but it's the third dimension in our images
        normalized_image = normalized_image.transpose((2, 0, 1))
        return {"image": normalized_image, "label": label}

    # Load dataset from Hugging Face datasets
    train_datset = (
        load_dataset("cifar10", split="train").with_format("numpy").map(transform)
    )
    test_dataset = (
        load_dataset("cifar10", split="test").with_format("numpy").map(transform)
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_datset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    logger.info("Creating ResNet-18 model...")

    # Model
    model = ResNet18(classes=110).to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)

    # summary(model, input_size=train_datset[0]["image"].shape)

    logger.info("Training ResNet-18 model...")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # TODO: Checkpointing

    # Train the model
    for epoch in range(num_epochs):
        model.train()

        max_steps = len(train_loader)
        train_loop = tqdm(enumerate(train_loader), total=max_steps, leave=False)

        for step, data in train_loop:
            images = data["image"].to(device)
            labels = data["label"].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO: Checkpointing

            train_loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            train_loop.set_postfix(loss=loss.item())

            if step == max_steps - 1:
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}]: Training Loss = {loss.item()}"
                )

    # Test the model
    model.eval()
    test_loop = tqdm(test_loader, total=len(test_loader), leave=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loop:
            images = data["image"].to(device)
            labels = data["label"].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        logger.info(
            f"Accuracy of the model on the 10000 test images: {100 * correct / total} %"
        )


if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)  # type: ignore

    train_resnet()
