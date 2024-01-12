"""Train Resnet."""

import time
import warnings

import torch
import typer
import wandb
from lightning import Fabric, seed_everything
from rich.progress import Progress
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.utils.data.dataloader import DataLoader
from torchmetrics import Accuracy
from typing_extensions import Annotated

from menagerie.datasets.tiny_imagenet import (
    create_train_dataset,
    create_validation_dataset,
    tiny_imagenet_collator,
)
from menagerie.datasets.utils import sanity_check_data
from menagerie.resnet.model import ResNet18
from menagerie.utils.console import console, progress_columns

# Ignore deprecation warnings this will be fixed in torch 2.2
warnings.filterwarnings(
    "ignore",
    message="'has_cuda' is deprecated, please use 'torch.backends.cuda.is_built()'",
)
warnings.filterwarnings(
    "ignore",
    message="'has_cudnn' is deprecated, please use 'torch.backends.cudnn.is_available()'",
)
warnings.filterwarnings(
    "ignore",
    message="'has_mps' is deprecated, please use 'torch.backends.mps.is_built()'",
)
warnings.filterwarnings(
    "ignore",
    message="'has_mkldnn' is deprecated, please use 'torch.backends.mkldnn.is_available()'",
)


num_classes = 200
num_workers = 4


def train(
    fabric: Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LRScheduler,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    epochs: int,
    *,
    log_wandb: bool = True,
):
    """Train the model."""
    for epoch in range(epochs):
        start_time = time.perf_counter()
        train_epoch(fabric, model, optimizer, train_dataloader)
        lr_scheduler.step()
        end_time = time.perf_counter()
        console.log(
            f"Completed Training Epoch: {epoch} in {(end_time - start_time):.4f}s"
        )
        if log_wandb:
            wandb.log({"lr": lr_scheduler.get_last_lr()[0]})

        # Validate an epoch
        start_time = time.perf_counter()
        validate_epoch(fabric, model, valid_dataloader, log_wandb=log_wandb)
        end_time = time.perf_counter()
        console.log(
            f"Completed Validation Epoch: {epoch} in {(end_time - start_time):.4f}s"
        )


def train_epoch(
    fabric: Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
):
    """Train an epoch."""
    model.train()
    with Progress(*progress_columns, transient=True) as progress:
        train_acc = Accuracy(task="multiclass", num_classes=num_classes).to(
            fabric.device
        )
        for batch in progress.track(
            train_dataloader,
            description="[purple]Training",
            total=len(train_dataloader),
        ):
            image, target = batch

            optimizer.zero_grad()
            logits = model(image)
            loss = cross_entropy(logits, target)
            fabric.backward(loss)
            optimizer.step()

            with torch.no_grad():
                batch_acc = train_acc(logits, target)

            fabric.log("train_loss", loss)
            fabric.log("train_acc", batch_acc)
        train_acc.reset()


@torch.no_grad()
def validate_epoch(
    fabric: Fabric,
    model: torch.nn.Module,
    valid_dataloader: DataLoader,
    *,
    log_wandb: bool = True,
):
    """Validate an epoch."""
    model.eval()
    with Progress(*progress_columns, transient=True) as progress:
        valid_acc = Accuracy(task="multiclass", num_classes=num_classes).to(
            fabric.device
        )
        for batch in progress.track(
            valid_dataloader,
            description="[purple]Validating",
            total=len(valid_dataloader),
        ):
            image, target = batch

            logits = model(image)
            loss = cross_entropy(logits, target)
            batch_acc = valid_acc(logits, target)

            fabric.log("valid_loss", loss)
            fabric.log("valid_acc", batch_acc)

            if log_wandb:
                wandb.log({"valid_loss": loss, "valid_acc": batch_acc})

        valid_acc.reset()


def main(
    epochs: Annotated[int, typer.Option(help="number of epochs to train")] = 150,
    batch_size: Annotated[
        int, typer.Option(help="input batch size for training")
    ] = 128,
    lr: Annotated[float, typer.Option(help="learning rate")] = 1e-1,
    seed: Annotated[int, typer.Option(help="random seed")] = 42,
    save_model: Annotated[
        bool, typer.Option(help="Save the model after training")
    ] = False,
    log_wandb: Annotated[bool, typer.Option(help="Log to wandb")] = True,
):
    """Train Resnet18 on Tiny Imagenet."""
    if log_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="menagerie",
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "architecture": "ResNet18",
                "dataset": "tiny-imagenet",
                "epochs": epochs,
                "batch_size": batch_size,
            },
        )
    console.rule("[bold purple]Configuring training run for ResNet18 on Tiny Imagenet")
    print("epochs:", epochs)
    print("batch_size:", batch_size)
    print("lr:", lr)
    print("seed:", seed)
    print("save_model:", save_model)
    print("log_wandb:", log_wandb)

    torch.set_float32_matmul_precision("medium")
    seed_everything(seed)

    torch.set_num_threads(8)
    print(f"CPU count: {torch.get_num_threads()}")

    fabric = Fabric(precision="16-mixed")
    fabric.launch()

    console.rule("[bold purple]Creating fabric environment")
    console.log("Creating model")
    with fabric.init_module():
        model = ResNet18(classes=num_classes)

    optimizer = Adam(model.parameters(), lr=lr)

    # Fabric currently has an issue with wrapping compiled models and strategy
    # ordering [https://github.com/Lightning-AI/pytorch-lightning/pull/19192]
    # model = torch.compile(model)
    model, optimizer = fabric.setup(model, optimizer)

    console.log("Loading dataset")
    with fabric.rank_zero_first(local=False):
        train_dataset = create_train_dataset()
        valid_dataset = create_validation_dataset()

    console.log("Creating dataloaders")
    train_dataloader = DataLoader(
        train_dataset,  # type: ignore [HF Dataset]
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=tiny_imagenet_collator,
        pin_memory=torch.cuda.is_available(),
    )
    valid_dataloader = DataLoader(
        valid_dataset,  # type: ignore [HF Dataset]
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=tiny_imagenet_collator,
        pin_memory=torch.cuda.is_available(),
    )

    train_dataloader, valid_dataloader = fabric.setup_dataloaders(
        train_dataloader, valid_dataloader
    )

    console.log("Creating LR scheduler")
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_dataloader),
    )

    console.log("Sanity checking data")
    sanity_check_data(train_dataloader, valid_dataloader)

    # TODO: gradient accumulation / multi-device training
    console.rule("[bold purple]Beginning training run")
    start_time = time.perf_counter()
    train(
        fabric,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        valid_dataloader,
        epochs,
        log_wandb=log_wandb,
    )
    end_time = time.perf_counter()

    console.rule(f"[bold purple]Training complete in {(end_time - start_time):.4f}s")
    if save_model:
        console.log("Saving model")
        fabric.save(
            "checkpoints/resnet18.ckpt",
            {"model": model, "optimizer": optimizer, "scheduler": scheduler},
        )
        console.rule()

    wandb.finish()


if __name__ == "__main__":
    typer.run(main)
