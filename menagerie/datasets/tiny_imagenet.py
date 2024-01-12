"""Tiny imagenet dataset [https://huggingface.co/datasets/zh-plus/tiny-imagenet]."""
from typing import Dict

import torch
from menagerie.datasets.image_classification import create_standard_image_transforms

import datasets
from datasets import load_dataset

_image_transform = create_standard_image_transforms()


def _filter(row: Dict):
    img = row["image"]
    return len(img.shape) == 3 and img.shape[2] == 3


def _transform(row: Dict):
    row["image"] = [_image_transform(image=img)["image"] for img in row["image"]]
    return row


def create_train_dataset():
    """Create the train dataset."""
    train_dataset = (
        load_dataset("zh-plus/tiny-imagenet", split="train", streaming=False)
        .with_format("numpy")
        .filter(_filter)
    )

    if not isinstance(train_dataset, datasets.Dataset):
        raise ValueError("Expected a Dataset.")

    train_dataset.set_transform(_transform)
    train_dataset.set_format("torch")

    return train_dataset


def create_validation_dataset():
    """Create the validation dataset."""
    test_dataset = (
        load_dataset("zh-plus/tiny-imagenet", split="valid", streaming=False)
        .with_format("numpy")
        .filter(_filter)
    )
    if not isinstance(test_dataset, datasets.Dataset):
        raise ValueError("Expected a Dataset.")

    test_dataset.set_transform(_transform)
    test_dataset.set_format("torch")

    return test_dataset


class _TinyImagenetCollator:
    def __call__(self, batch):
        # Extract the images and labels from the batch
        images = [example["image"] for example in batch]
        labels = [example["label"] for example in batch]

        # Stack the images into a single tensor
        images = torch.stack(images)
        labels = torch.tensor(labels)

        # Permute the images to be channels first
        images = images.permute(0, 3, 1, 2)

        # Convert the images to float
        images = images.float()

        return images, labels


tiny_imagenet_collator = _TinyImagenetCollator()
