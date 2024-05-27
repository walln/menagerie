"""Sparse autoencoder model for learning interpretable representations."""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from menagerie.sparse_interpretability.utils import DTYPES


@dataclass
class SparseAutoEncoderConfig:
    """Configuration object for the sparse autoencoder.

    Attributes:
        act_size: Size of the input data
        dict_size: Size of the intermediate representation
        l1_coef: Coefficient for the L1 loss
        encoder_dtype: Data type for the encoder weights
        seed: Random seed
        device: Device to run the model on
    """

    act_size: int
    dict_size: int
    l1_coef: float
    encoder_dtype: Literal["fp32", "fp16", "bf16"]
    seed: int
    device: str

    def dtype_mapping(self):
        """Get the torch dtype corresponding to the encoder_dtype.

        Returns:
            dtype: Torch dtype
        """
        if self.encoder_dtype not in DTYPES:
            raise ValueError(f"Invalid dtype {self.encoder_dtype}")
        return DTYPES[self.encoder_dtype]


class SparseAutoEncoder(nn.Module):
    """Sparse autoencoder model.

    Learns an intermediate representation of the input data that is sparse
    and then reconstructs the input data from the intermediate representation. The
    model is trained to minimize the L2 loss between the input and the reconstructed
    data, with an additional L1 loss on the activations to encourage sparsity.
    """

    def __init__(self, config: SparseAutoEncoderConfig):
        """Construct the SAE model.

        Args:
            config: Configuration object
        """
        super().__init__()

        self.d_hidden = config.dict_size
        self.d_output = config.act_size
        self.l1_coef = config.l1_coef

        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(self.d_output, self.d_output, dtype=config.dtype_mapping())
            )
        )
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(self.d_output, self.d_output, dtype=config.dtype_mapping())
            )
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_output, dtype=config.dtype_mapping())
        )
        self.b_dec = nn.Parameter(
            torch.zeros(self.d_output, dtype=config.dtype_mapping())
        )

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.to(config.device)

    def forward(self, x) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, act_size)

        Returns:
            loss: Scalar loss value
            x_reconstructed: Reconstructed input tensor
            activations: Activations of the encoder
            l2_loss: L2 loss
            l1_loss: L1 loss
        """
        x_cent = x - self.b_enc
        activations = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstructed = activations @ self.W_dec + self.b_dec
        l2_loss = (x_reconstructed.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coef * (activations.float().abs().sum())
        loss = l2_loss + l1_loss

        return loss, x_reconstructed, activations, l2_loss, l1_loss

    @torch.no_grad()
    def unit_norm(self):
        """Enforce unit norm on the decoder weights and the gradients."""
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    # TODO: Loading and saving the model
