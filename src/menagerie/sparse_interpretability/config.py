"""Configuration for sparse interpretability experiments."""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Training arguments."""

    num_tokens: int = int(2e9)
    batch_size: int = 64
    seq_len: int = 128
    seed: int = 42
    activation_size: int = 512
    activation_layer_name: str = "transformer.h.0.mlp.c_proj"
    buffer_multiplier = 384
    buffer_size: int = field(init=False)
    buffer_batches: int = field(init=False)
    model_batch_size = 64

    def __post_init__(self):
        """Derive buffer size and buffer batches."""
        self.buffer_size = self.batch_size * self.buffer_multiplier
        self.buffer_batches = self.buffer_size // self.seq_len
