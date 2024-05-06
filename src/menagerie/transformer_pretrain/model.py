"""Simple transformer model for pretraining."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Transformer(nn.Module):
    """A simple transformer model."""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 768,  # Bert hidden size is 768
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.2,
    ):
        """Initialize the model."""
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_dim * 4
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=self.hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, inputs: Tensor, target: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass of the model."""
        _, t = inputs.shape

        # Assume the target has been shifted with respect to the inputs
        if mask is None:
            mask = torch.tril(torch.ones(t, t, device=inputs.device)) == 1
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, 0.0)
            )

        src = self.pos_encoder(self.embedding(inputs) * math.sqrt(self.hidden_dim))
        target = self.pos_encoder(self.embedding(target) * math.sqrt(self.hidden_dim))
        output = self.transformer(src, target, tgt_mask=mask)
        output = self.decoder(output)
        output = F.log_softmax(output, dim=-1)
        output = output.view(-1, self.vocab_size)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    encoding: Tensor | None = None

    def __init__(self, dim: int, dropout: float, max_len: int = 5000):
        """Initialize the model."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.max_len = max_len

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the positional encoding."""
        if self.encoding is None:
            encoding = torch.zeros(self.max_len, self.dim, device=x.device)
            position = torch.arange(
                0, self.max_len, dtype=torch.float, device=x.device
            ).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, self.dim, 2, device=x.device).float()
                * (-math.log(10000.0) / self.dim)
            )

            encoding[:, 0::2] = torch.sin(position * div)
            encoding[:, 1::2] = torch.cos(position * div)
            encoding = encoding.unsqueeze(0).transpose(0, 1)
            self.encoding = encoding

        x = x + self.encoding[: x.size(0), :]
        return self.dropout(x)
