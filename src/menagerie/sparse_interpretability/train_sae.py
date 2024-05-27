"""Script to train the Sparse Autoencoder on transformer activations."""

import random

import numpy as np
import torch
import torch.optim as optim
from rich.progress import Progress
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset
from menagerie.sparse_interpretability.config import TrainConfig
from menagerie.sparse_interpretability.sparse_autoencoder import (
    SparseAutoEncoder,
    SparseAutoEncoderConfig,
)

train_config = TrainConfig()

device = "cuda"


torch.manual_seed(train_config.seed)
torch.cuda.manual_seed(train_config.seed)
np.random.seed(train_config.seed)
random.seed(train_config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

activation_size = 512

act_name = "transformer.h.0.mlp.c_proj"
buffer_multiplier = 384
buffer_size = train_config.batch_size * buffer_multiplier
buffer_batches = buffer_size // train_config.seq_len
model_batch_size = 64


sae_config = SparseAutoEncoderConfig(
    act_size=activation_size,
    dict_size=32 * activation_size,
    l1_coef=3e-4,
    encoder_dtype="fp32",
    seed=49,
    device="cuda",
)

model = SparseAutoEncoder(sae_config)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

dataset = load_dataset(
    "walln/c4-code-20k-gpt-2-activations", train_config.activation_layer_name
)
assert isinstance(dataset, Dataset)


def _collate_fn(batch):
    activations = torch.stack([item["activations"] for item in batch])
    return activations


dataloader = DataLoader(
    dataset=dataset,  # type: ignore
    batch_size=train_config.batch_size,
    collate_fn=_collate_fn,
    shuffle=True,
)


num_batches = train_config.num_tokens // train_config.batch_size
num_epochs = 3

with Progress() as progress:
    training_loop_task = progress.add_task("Training Epoch", total=10)

    for _ in range(num_epochs):
        sae_training_progress = progress.add_task("Training SAE", total=num_batches)

        for activations in dataloader:
            # Compute the forward and backward pass
            activations = activations.to(device)
            loss, x_reconstruct, mid_acts, l2_loss, l1_loss = model.forward(activations)
            loss.backward()

            ## We make the decoder weights and gradients to be unit norm
            model.unit_norm()

            ## Step the optimizer and zero the gradients
            optimizer.step()
            optimizer.zero_grad()

            loggables = {
                "loss": loss.item(),
                "l2_loss": l2_loss.item(),
                "l1_loss": l1_loss.item(),
            }

            ## Clear tracked tensors
            del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, activations

            # TODO: Logging and visualization

            progress.update(sae_training_progress, advance=1)

        progress.update(training_loop_task, advance=1)
