"""Pretrain a transfomer model on wikitext."""

import time
from typing import Annotated

import torch
import torch.nn.functional as F
import typer
import wandb
from lightning import Fabric, seed_everything
from menagerie.datasets.code_parrot import create_datasets
from menagerie.transfomer_pretrain.model import Transformer
from menagerie.utils.console import console, progress_columns
from rich.progress import Progress
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.utils.data.dataloader import DataLoader

num_workers = 1


# TODO: do global steps and validate every instead of epochs.
def main(
    epochs: Annotated[int, typer.Option(help="number of epochs to train")] = 1,
    batch_size: Annotated[int, typer.Option(help="input batch size for training")] = 1,
    lr: Annotated[float, typer.Option(help="learning rate")] = 1e-1,
    seed: Annotated[int, typer.Option(help="random seed")] = 42,
    save_model: Annotated[
        bool, typer.Option(help="Save the model after training")
    ] = False,
    log_wandb: Annotated[bool, typer.Option(help="Log to wandb")] = True,
    accumulation_steps: Annotated[
        int, typer.Option(help="Accumulate gradients across batches")
    ] = 8,
):
    """Train a simple transformer model on wikitext."""
    if log_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="menagerie",
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "architecture": "transformer_pretrain",
                "dataset": "wikitext",
                "epochs": epochs,
                "batch_size": batch_size,
            },
        )
    console.rule(
        "[bold purple]Configuring training run for a simple Transformer on Wikitext"
    )
    context_length = 128
    print("context_length:", context_length)
    print("epochs:", epochs)
    print("batch_size:", batch_size)
    print("lr:", lr)
    print("seed:", seed)
    print("save_model:", save_model)
    print("log_wandb:", log_wandb)
    print("accumulation_steps:", accumulation_steps)

    torch.set_float32_matmul_precision("medium")
    seed_everything(seed)

    fabric = Fabric(precision="bf16-mixed")
    fabric.launch()
    console.rule("[bold purple]Creating fabric environment")
    console.log("Loading dataset")
    with fabric.rank_zero_first(local=False):
        datasets, tokenizer, data_collator = create_datasets(
            seed=seed, context_length=context_length
        )
        train_dataset = datasets["train"]
        valid_dataset = datasets["valid"]

    console.log("Creating dataloaders")
    train_dataloader = DataLoader(
        train_dataset,  # type: ignore [HF Dataset]
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )
    valid_dataloader = DataLoader(
        valid_dataset,  # type: ignore [HF Dataset]
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    train_dataloader, valid_dataloader = fabric.setup_dataloaders(
        train_dataloader, valid_dataloader
    )
    console.log("Utilizing", fabric.device.type, "device")
    console.log("Creating model")
    with fabric.init_module():
        model = Transformer(vocab_size=len(tokenizer))

    # Display parameter count and estimate memory usage
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.log(f"Model has {params:,} trainable parameters")
    console.log(f"Estimated memory usage: {params * 4 / 1024 / 1024:.2f} MB")

    optimizer = Adam(model.parameters(), lr=lr)

    # Fabric currently has an issue with wrapping compiled models and strategy
    # ordering [https://github.com/Lightning-AI/pytorch-lightning/pull/19192]
    # model = torch.compile(model)
    model, optimizer = fabric.setup(model, optimizer)

    console.log("Creating LR scheduler")
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_dataloader),
    )

    console.log("Sanity checking data")
    # sanity_check_data(train_dataloader, valid_dataloader)

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
        accumulation_steps=accumulation_steps,
    )
    end_time = time.perf_counter()

    console.rule(f"[bold purple]Training complete in {(end_time - start_time):.4f}s")
    if save_model:
        console.log("Saving model")
        fabric.save(
            "checkpoints/pretrained_transformer.ckpt",
            {"model": model, "optimizer": optimizer, "scheduler": scheduler},
        )
        console.rule()

    wandb.finish()


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
    accumulation_steps: int = 8,
):
    """Train the model."""
    for epoch in range(epochs):
        start_time = time.perf_counter()
        train_epoch(
            fabric,
            model,
            optimizer,
            train_dataloader,
            lr_scheduler,
            log_wandb=log_wandb,
            accumulation_steps=accumulation_steps,
        )
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
    scheduler: LRScheduler,
    *,
    log_wandb: bool = True,
    accumulation_steps: int = 8,
):
    """Train an epoch."""
    model.train()
    average_train_loss = 0.0
    with Progress(*progress_columns, transient=True) as progress:
        for iteration, batch in progress.track(
            enumerate(train_dataloader),
            description="[purple]Training",
            total=len(train_dataloader),
        ):
            is_accumulating = iteration % accumulation_steps != 0

            input, target = batch["input_ids"], batch["labels"]
            input = input.to(fabric.device)
            target = target.to(fabric.device)

            with fabric.no_backward_sync(model, enabled=is_accumulating):  # type: ignore [wrapped module]
                logits = model(input, target)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), target.view(-1)
                )
                if log_wandb:
                    wandb.log({"train_loss": loss.item()})
                fabric.log("train_loss", loss)
                average_train_loss += loss.item()
                fabric.backward(loss)

            if not is_accumulating:
                # step the optimizer after accumulating gradients
                optimizer.step()
                optimizer.zero_grad()

                fabric.log("train_loss", average_train_loss / accumulation_steps)
                average_train_loss = 0.0

            scheduler.step()


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
    losses = torch.zeros(len(valid_dataloader))
    with Progress(*progress_columns, transient=True) as progress:
        for i, batch in progress.track(
            enumerate(valid_dataloader),
            description="[purple]Validating",
            total=len(valid_dataloader),
        ):
            input, target = batch["input_ids"], batch["labels"]
            input = input.to(fabric.device)
            target = target.to(fabric.device)

            logits = model(input, target)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            losses[i] = loss.item()

            fabric.log("valid_loss", loss)

            if log_wandb:
                wandb.log({"valid_loss": loss})

    valid_loss = losses.mean()
    return valid_loss


if __name__ == "__main__":
    typer.run(main)
