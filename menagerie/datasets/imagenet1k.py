"""Imagenet1k Image Classification."""
import logging
from typing import Literal

import albumentations as A
import lightning as L
import numpy as np
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from datasets import load_dataset

# Assuming logger is already configured elsewhere in your code
logger = logging.getLogger("menagerie")


class Imagenet1kDataModule(L.LightningDataModule):
    """Imagenet1k image classification dataset [https://huggingface.co/datasets/imagenet-1k]."""

    dataset = "imagenet-1k"
    num_classes = 1000

    def __init__(self, batch_size=32, num_workers=1):
        """Create a new Imagenet1kDataModule instance.

        Args:
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 1.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pipeline = A.Compose(
            [
                A.PadIfNeeded(min_height=224, min_width=224),
                A.CenterCrop(height=224, width=224),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

        logger.info(f"Created Imagenet1kDataModule with batch_size={batch_size}")

    def _transform_image(self, element):
        transformed = self.pipeline(image=np.array(element["image"]))
        image = transformed["image"]
        return {"image": image, "label": element["label"]}

    def _filter_images(self, example):
        img = np.array(example["image"])
        if len(img.shape) != 3:
            return False
        w, h, c = img.shape
        return w >= 224 and h >= 224 and c == 3

    def setup(self, stage=None):
        """Setup the dataset."""
        logger.info(f"Loading {self.dataset} dataset for {stage}")
        if stage == TrainerFn.FITTING or stage == TrainerFn.VALIDATING or stage is None:
            self.train_dataset = self._create_dataset("train")
            self.val_dataset = self._create_dataset("validation")
        if stage == TrainerFn.TESTING:
            self.test_dataset = self._create_dataset("test")
        if stage == TrainerFn.PREDICTING:
            self.predict_dataset = self._create_dataset("test")

    def _create_dataset(self, split: Literal["train", "validation", "test"]):
        dataset = (
            load_dataset(
                self.dataset,
                "full",
                split=split,
                streaming=True,
                trust_remote_code=True,
                cache_dir="data",
            )
            .select_columns(["image", "label"])
            .shuffle()
            .filter(self._filter_images)
            .map(self._transform_image, batched=False)
        )

        return dataset

    def train_dataloader(self):
        """Create a training data loader."""
        return DataLoader(
            self.train_dataset,  # type: ignore [HF Dataset]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Create a validation data loader."""
        return DataLoader(
            self.val_dataset,  # type: ignore [HF Dataset]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Create a test data loader."""
        return DataLoader(
            self.test_dataset,  # type: ignore [HF Dataset]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """Create a prediction data loader."""
        return DataLoader(
            self.predict_dataset,  # type: ignore [HF Dataset]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
