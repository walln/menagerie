"""Train a ResNet18 model."""

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from menagerie.datasets.imagenet1k import Imagenet1kDataModule
from menagerie.resnet.model import ResNet18
from menagerie.utils.logging import create_logger

logger = create_logger()


def main():
    """Main function."""
    torch.set_float32_matmul_precision("medium")

    # setup data
    datamodule = Imagenet1kDataModule(batch_size=16, num_workers=2)

    logger.info("Starting training...")
    model = ResNet18(classes=datamodule.num_classes)

    trainer = L.Trainer(
        max_epochs=10,
        logger=CSVLogger("logs"),
        callbacks=[ModelCheckpoint(monitor="val_accuracy")],
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
